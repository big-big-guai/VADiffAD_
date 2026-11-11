"""SMAP main-method execution: normal path bank, five scores, calibration, test."""
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from base.dataset import load_prepared_smap, standardize, windows, fold_windows
from base.metrics import evaluate_pointwise
from vadiffad.diffusion import variable_coefficients, VariableAwareSchedule
from vadiffad.path import DiffusionPathExtractor
from vadiffad.reference import NormalPathBank
from vadiffad.scoring import (
    EmpiricalUpperTail, causal_history_residual, fuse_local_scores,
    local_top_q, warmup_thresholds,
)
from vadiffad.trainer import NormalTrainer
from utils.proxy import inject_validation_proxies, select_rejectable_gate


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def build_normal_bank(cfg, train_windows, extractor, dimensions, device, seed):
    """At most 30,000 local time-variable signatures, distributed by variable."""
    rng = np.random.default_rng(seed+908)
    max_positions = max(1, cfg.bank_max_signatures//dimensions)
    window_count = min(len(train_windows), int(np.ceil(max_positions/cfg.window)))
    ids = np.sort(rng.choice(len(train_windows), size=window_count, replace=False))
    signatures = []
    print('Building Normal Path Bank from normal training windows', flush=True)
    for start in range(0, len(ids), cfg.batch_size):
        b = torch.as_tensor(train_windows[ids[start:start+cfg.batch_size]], device=device)
        _, _, signature, _ = extractor(b)
        signatures.append(signature.cpu().numpy().reshape(-1, dimensions, signature.shape[-1]))
    h = np.concatenate(signatures, axis=0)
    if len(h) > max_positions:
        sample = np.sort(rng.choice(len(h), max_positions, replace=False))
        h = h[sample]
    # Same positions across variables; at most D * N signatures total.
    bank = NormalPathBank(cfg.cov_shrinkage, cfg.eps).fit(h)
    return bank, len(h)*dimensions


@torch.no_grad()
def raw_components(series, bounds, cfg, extractor, bank, trainer, device):
    """Return [time, variables, (V,C,D,R,A)] before empirical-CDF mapping."""
    local_windows, starts = windows(series, cfg.window, cfg.eval_stride, bounds)
    extracted = []
    for offset in range(0, len(local_windows), cfg.batch_size):
        x = torch.as_tensor(local_windows[offset:offset+cfg.batch_size], device=device)
        v, c, h, _ = extractor(x)
        path = bank.components(v, c, h)
        context = trainer.masked_context_error(x)
        # Concatenated five local components (no averaging over variables).
        chunk = torch.cat((path, context[..., None]), dim=-1).cpu().numpy()
        extracted.append(chunk.astype(np.float32))
    path_context = fold_windows(np.concatenate(extracted), starts, len(series))
    history = causal_history_residual(series, bounds)
    return np.concatenate((path_context, history[..., None]), axis=-1)


def run_single_smap(cfg, prepared_path, run_folder, seed=42, device_name='cpu', threads=2):
    output = Path(run_folder)
    if (output/'metrics.json').exists():
        raise FileExistsError('Refusing to overwrite an existing completed run')
    output.mkdir(parents=True, exist_ok=True)
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    device = torch.device(device_name)
    seed_everything(seed, threads)
    start_time = time.monotonic()

    original = load_prepared_smap(prepared_path, cfg.window)
    smap, mean, sigma = standardize(original)
    cfg_path = output/'configuration.json'
    cfg_path.write_text(json.dumps({**asdict(cfg), 'seed': int(seed)}, indent=2), encoding='utf-8')

    # Diffusion coefficients depend only on the normal training sequence.
    coeff = variable_coefficients(smap.train, cfg, smap.boundaries['train'])
    schedule = VariableAwareSchedule(coeff, cfg, device)
    trainer = NormalTrainer(smap.train.shape[1], cfg, schedule, device)
    train_windows, _ = windows(smap.train, cfg.window, cfg.train_stride, smap.boundaries['train'])
    val_windows, _ = windows(smap.validation, cfg.window, cfg.window, smap.boundaries['validation'])
    best_epoch = trainer.train(train_windows, val_windows, output)
    extractor = DiffusionPathExtractor(trainer.predictor, schedule, cfg)
    bank, signature_count = build_normal_bank(
        cfg, train_windows, extractor, smap.train.shape[1], device, seed)
    bank.save(output/'normal_path_bank.npz')
    np.savez_compressed(output/'normalization.npz', mean=mean, std=sigma, coefficients=coeff)

    print('Calibrating five score components on normal SMAP calibration data', flush=True)
    raw_cal = raw_components(smap.calibration, smap.boundaries['calibration'],
                             cfg, extractor, bank, trainer, device)
    calibrator = EmpiricalUpperTail().fit(raw_cal)
    print('Selecting the path gate using validation proxies only', flush=True)
    raw_clean = raw_components(smap.validation, smap.boundaries['validation'],
                               cfg, extractor, bank, trainer, device)
    clean_calibrated = calibrator.transform(raw_clean)
    proxy_scores, proxy_labels = [], []
    for index, severity in enumerate(cfg.proxy_severities):
        proxy, proxy_y = inject_validation_proxies(
            smap.validation, smap.boundaries['validation'],
            seed+109*(index+1), severity=severity, fraction=cfg.proxy_fraction)
        proxy_raw = raw_components(proxy, smap.boundaries['validation'],
                                   cfg, extractor, bank, trainer, device)
        proxy_scores.append(calibrator.transform(proxy_raw))
        proxy_labels.append(proxy_y)
    gate, selection = select_rejectable_gate(cfg, proxy_scores, proxy_labels, clean_calibrated)
    (output/'validation_gate.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')

    # No test label access before the following evaluation call.
    print('Scoring SMAP test stream', flush=True)
    raw_test = raw_components(smap.test, smap.boundaries['test'], cfg, extractor, bank, trainer, device)
    transformed = calibrator.transform(raw_test)
    local = fuse_local_scores(transformed, cfg, gate)
    scores = local_top_q(local, cfg.top_q)
    thresholds, warmup_meta = warmup_thresholds(scores, smap.boundaries['test'], cfg)
    evaluated = evaluate_pointwise(smap.labels, scores, thresholds)
    np.savez_compressed(output/'predictions.npz', scores=scores, thresholds=thresholds,
                        predicted=(scores>thresholds).astype(np.int8),
                        variable_contributions=local, labels=smap.labels)

    report = {
        'dataset': 'SMAP', 'method': 'VADiffAD', 'seed': int(seed),
        'protocol': 'normal-only field training; normal path reference; normal CDF calibration; validation proxies; unlabeled warmup; strict point-wise (no PA)',
        'selected_epoch': int(best_epoch), 'normal_reference_signatures': int(signature_count),
        'gate': float(gate), 'top_q': int(min(cfg.top_q, smap.train.shape[1])),
        'warmup_segments': warmup_meta, 'metrics': evaluated,
        'runtime_seconds': float(time.monotonic()-start_time), 'device': str(device),
        'input_sha256': hashlib.sha256(Path(prepared_path).read_bytes()).hexdigest(),
    }
    (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'seed': seed, 'SMAP_pointwise_F1': evaluated['f1'],
                      'SMAP_AP': evaluated['average_precision'], 'gate': gate}, indent=2), flush=True)
    return report
