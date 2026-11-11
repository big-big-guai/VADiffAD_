"""SMAP-only VADiffAD training, normal-reference construction and point-wise evaluation.

Expected input: a pre-existing SMAP.npz file; this release intentionally contains
neither dataset files nor dataset download/preparation routines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from metrics import metrics
from vadiffad_paper import (
    Config, ContextReconstructor, NormalPrototype, NormalScoreField,
    PathExtractor, Schedule, aggregate_overlaps, make_windows,
    variable_coefficients,
)


def boundaries(data, split):
    key = f"{split}_boundaries"
    return [tuple(map(int, row)) for row in data[key]] if key in data.files else None


def robust_location_scale(values, ratio=.05, eps=1e-6):
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    positive = mad[mad > eps]
    floor = (float(np.median(positive)) if positive.size else 1.0) * ratio
    return median, 1.4826 * np.maximum(mad, max(floor, eps))


def empirical_surprisal(reference, target):
    """Map each coordinate to right-tail surprisal under normal reference data."""
    reference = np.asarray(reference)
    target = np.asarray(target)
    if reference.ndim != 2 or target.ndim != 2 or reference.shape[1] != target.shape[1]:
        raise ValueError("reference and target must be 2-D with matching columns")
    result = np.empty_like(target, dtype=np.float64)
    denominator = reference.shape[0] + 1.0
    for column in range(reference.shape[1]):
        ordered = np.sort(reference[:, column])
        rank = np.searchsorted(ordered, target[:, column], side="right")
        survival = np.maximum((reference.shape[0] - rank + 1.0) / denominator, 1.0 / denominator)
        result[:, column] = -np.log(survival)
    return result


def smooth(scores, width):
    if width <= 1:
        return scores
    left = width // 2
    padded = np.pad(scores, (left, width - left - 1), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    return (cumulative[width:] - cumulative[:-width]) / width


def top_q(variable_scores, q, smoothing=1):
    q = min(q, variable_scores.shape[1])
    score = np.sort(variable_scores, axis=1)[:, -q:].mean(1)
    return smooth(score, smoothing)


def causal_deviation(series, split_boundaries=None, lags=(5, 20, 50)):
    """Multi-scale causal residual with no access to future observations."""
    result = np.zeros_like(series, dtype=np.float32)
    segments = [(0, len(series))] if split_boundaries is None else split_boundaries
    for begin, end in segments:
        values = series[begin:end]
        accumulated = np.zeros_like(values, dtype=np.float64)
        cumsum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0, dtype=np.float64)])
        index = np.arange(len(values))
        for lag in lags:
            left = np.maximum(0, index - lag)
            count = np.maximum(index - left, 1)[:, None]
            prediction = (cumsum[index] - cumsum[left]) / count
            prediction[0] = values[0]
            accumulated += np.abs(values - prediction)
        result[begin:end] = accumulated / len(lags)
    return result



def inject_proxy_anomalies(
    series, seed, columns=None, segment_boundaries=None, target_fraction=None,
    severity=1.0,
):
    """Create validation-only proxy anomalies; test labels are never inspected."""
    rng = np.random.RandomState(seed + 1701)
    corrupted = series.copy()
    labels = np.zeros(len(series), dtype=np.int8)
    usable = max(16, len(series) // 30)
    segment = max(8, min(64, usable // 3))
    if target_fraction is None:
        centers = np.linspace(segment, len(series) - segment - 1, 8, dtype=int)
    else:
        ranges = (
            np.asarray(segment_boundaries, dtype=int)
            if segment_boundaries is not None else np.array([[0, len(series)]])
        )
        planned = []
        for left, right in ranges:
            entity_length = int(right - left)
            if entity_length < 16:
                continue
            entity_segment = max(8, min(64, entity_length // 20))
            expected_width = .75 * entity_segment
            count = max(1, int(round(target_fraction * entity_length / expected_width)))
            starts = np.linspace(
                left, max(left, right - entity_segment), count + 2, dtype=int
            )[1:-1]
            planned.extend((int(start), entity_segment) for start in starts)
        centers = planned
    scale = np.maximum(np.std(series, axis=0), 1.0)
    eligible = np.asarray(columns if columns is not None else np.where(np.std(series, axis=0) > .05)[0])
    if not len(eligible):
        eligible = np.arange(series.shape[1])
    # The eight transformations deliberately cover both marginal outliers and
    # contextual anomalies.  The earlier four-transform suite was dominated by
    # large value changes and therefore systematically preferred reconstruction
    # scores over temporal-dependence scores.
    for index, item in enumerate(centers):
        center, local_segment = item if isinstance(item, tuple) else (int(item), segment)
        length = int(rng.randint(max(6, local_segment // 2), local_segment + 1))
        begin, end = center, min(center + length, len(series))
        dims = rng.choice(eligible, size=min(3, len(eligible)), replace=False)
        kind = index % 8
        if kind == 0:
            corrupted[begin:end, dims] += (
                rng.choice([-1, 1], len(dims)) * severity * 4 * scale[dims]
            )
        elif kind == 1:
            corrupted[begin:end, dims] *= 1 + severity * rng.uniform(.8, 1.8, len(dims))
        elif kind == 2:
            original = corrupted[begin:end, dims].copy()
            corrupted[begin:end, dims] = (
                (1 - severity) * original + severity * original[::-1]
            )
        elif kind == 3:
            phase = np.arange(end - begin)[:, None]
            corrupted[begin:end, dims] += (
                severity * 2.5 * scale[dims] * np.sin(phase * np.pi / 2)
            )
        elif kind == 4:  # contextual segment replacement
            source = int(rng.randint(0, max(1, len(series) - (end - begin) + 1)))
            replacement = series[source:source + end - begin, dims]
            original = corrupted[begin:end, dims].copy()
            corrupted[begin:end, dims] = (
                (1 - severity) * original + severity * replacement
            )
        elif kind == 5:  # local temporal lag without changing the value range
            lag = max(2, min(end - begin - 1, local_segment // 4))
            original = corrupted[begin:end, dims].copy()
            shifted = np.roll(original, lag, axis=0)
            corrupted[begin:end, dims] = (
                (1 - severity) * original + severity * shifted
            )
        elif kind == 6:  # break cross-variable synchrony
            original = corrupted[begin:end, dims].copy()
            shuffled = original.copy()
            for column in range(shuffled.shape[1]):
                shuffled[:, column] = shuffled[rng.permutation(len(shuffled)), column]
            corrupted[begin:end, dims] = (
                (1 - severity) * original + severity * shuffled
            )
        else:  # gradual drift, difficult for pointwise magnitude detectors
            ramp = np.linspace(-1, 1, end - begin)[:, None]
            corrupted[begin:end, dims] += severity * 2 * ramp * scale[dims]
        labels[begin:end] = 1
    return corrupted, labels


def choose_stable_candidate(candidates):
    """Apply the predeclared path gate and near-best simplicity rule."""
    candidates = sorted(
        candidates,
        key=lambda row: (
            row["proxy_average_precision"], row["proxy_worst_f1"],
            row["proxy_f1"], row["proxy_recall"], -row["q"],
        ), reverse=True,
    )
    best_proxy_ap = candidates[0]["proxy_average_precision"]
    no_path_candidates = [row for row in candidates if row["path_weight"] == 0]
    best_no_path = no_path_candidates[0]
    best_no_path_ap = best_no_path["proxy_average_precision"]
    path_relative_gain = best_proxy_ap / max(best_no_path_ap, 1e-12) - 1
    path_worst_f1_delta = candidates[0]["proxy_worst_f1"] - best_no_path["proxy_worst_f1"]
    path_gate_passed = path_relative_gain >= .05 and path_worst_f1_delta >= 0
    gate_pool = (
        candidates if path_gate_passed
        else [row for row in candidates if row["path_weight"] == 0]
    )
    gate_best_ap = max(row["proxy_average_precision"] for row in gate_pool)
    near_best = [
        row for row in gate_pool
        if row["proxy_average_precision"] >= .98 * gate_best_ap
    ]
    minimum_path_weight = min(row["path_weight"] for row in near_best)
    stable_simple = [
        row for row in near_best if row["path_weight"] == minimum_path_weight
    ]
    stable_simple.sort(
        key=lambda row: (
            row["proxy_worst_f1"], row["proxy_f1"],
            row["proxy_recall"], -row["q"],
        ), reverse=True,
    )
    selected = dict(stable_simple[0])
    selected["path_gate_relative_proxy_ap_gain"] = float(path_relative_gain)
    selected["path_gate_worst_suite_f1_delta"] = float(path_worst_f1_delta)
    selected["path_gate_passed"] = bool(path_gate_passed)
    selected["selection_rule"] = (
        "require_5pct_path_ap_gain_and_nonnegative_worst_suite_f1_delta_then_simplicity"
    )
    return selected, candidates


def select_operating_point(
    clean_components, proxy_components, proxy_labels, quantiles,
    normalization="robust",
):
    """Tune on one or more proxy suites, prioritizing ranking then robust F1."""
    if isinstance(proxy_components[0], np.ndarray):
        proxy_component_suites = (proxy_components,)
        proxy_label_suites = (proxy_labels,)
    else:
        proxy_component_suites = tuple(proxy_components)
        proxy_label_suites = tuple(proxy_labels)
    tuning_step = max(1, int(np.ceil(len(proxy_label_suites[0]) / 50000)))
    if tuning_step > 1:
        clean_components = tuple(values[::tuning_step] for values in clean_components)
        proxy_component_suites = tuple(
            tuple(values[::tuning_step] for values in suite)
            for suite in proxy_component_suites
        )
        proxy_label_suites = tuple(labels[::tuning_step] for labels in proxy_label_suites)
    path_clean, recon_clean, causal_clean = clean_components
    if normalization == "ecdf":
        clean_p = empirical_surprisal(path_clean, path_clean)
        clean_r = empirical_surprisal(recon_clean, recon_clean)
        clean_c = empirical_surprisal(causal_clean, causal_clean)
        standardized_proxy_suites = tuple(
            tuple(empirical_surprisal(reference, target)
                  for reference, target in zip(clean_components, suite))
            for suite in proxy_component_suites
        )
    elif normalization == "robust":
        path_median, path_scale = robust_location_scale(path_clean)
        recon_median, recon_scale = robust_location_scale(recon_clean)
        causal_median, causal_scale = robust_location_scale(causal_clean)
        clean_p = (path_clean - path_median) / path_scale
        clean_r = (recon_clean - recon_median) / recon_scale
        clean_c = (causal_clean - causal_median) / causal_scale
        standardized_proxy_suites = tuple(
            (
                (suite[0] - path_median) / path_scale,
                (suite[1] - recon_median) / recon_scale,
                (suite[2] - causal_median) / causal_scale,
            )
            for suite in proxy_component_suites
        )
    else:
        raise ValueError("unknown score normalization: " + normalization)
    candidates = []
    for path_weight in (0.0, 0.5, 1.0):
        for recon_weight in (0.0, 0.5, 1.0):
          for causal_weight in (0.0, 0.5, 1.0, 2.0, 4.0):
            clean_variable = path_weight * clean_p + recon_weight * clean_r + causal_weight * clean_c
            proxy_variables = [
                path_weight * suite[0] + recon_weight * suite[1]
                + causal_weight * suite[2]
                for suite in standardized_proxy_suites
            ]
            clean_ordered = np.sort(clean_variable, axis=1)
            proxy_ordered = [np.sort(values, axis=1) for values in proxy_variables]
            for q in sorted(set((1, min(3, clean_p.shape[1]), min(5, clean_p.shape[1])))):
                clean_base = clean_ordered[:, -q:].mean(1)
                proxy_bases = [values[:, -q:].mean(1) for values in proxy_ordered]
                for width in (1, 5, 11, 21, 51):
                    clean_score = smooth(clean_base, width)
                    proxy_scores = [smooth(values, width) for values in proxy_bases]
                    ranking_aps = [
                        float(average_precision_score(labels, score))
                        for labels, score in zip(proxy_label_suites, proxy_scores)
                    ]
                    for quantile in quantiles:
                        threshold = float(np.quantile(clean_score, quantile))
                        suite_metrics = []
                        for labels, proxy_score in zip(proxy_label_suites, proxy_scores):
                            prediction = proxy_score > threshold
                            positive = labels.astype(bool)
                            true_positive = int(np.logical_and(prediction, positive).sum())
                            false_positive = int(np.logical_and(prediction, ~positive).sum())
                            false_negative = int(np.logical_and(~prediction, positive).sum())
                            precision = true_positive / max(true_positive + false_positive, 1)
                            recall = true_positive / max(true_positive + false_negative, 1)
                            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                            suite_metrics.append((precision, recall, f1))
                        candidates.append({
                            "path_weight": path_weight, "reconstruction_weight": recon_weight,
                            "causal_weight": causal_weight,
                            "q": q, "smoothing": width, "quantile": quantile,
                            "proxy_average_precision": float(np.mean(ranking_aps)),
                            "proxy_f1": float(np.mean([row[2] for row in suite_metrics])),
                            "proxy_worst_f1": float(np.min([row[2] for row in suite_metrics])),
                            "proxy_precision": float(np.mean([row[0] for row in suite_metrics])),
                            "proxy_recall": float(np.mean([row[1] for row in suite_metrics])),
                        })
    return choose_stable_candidate(candidates)



# The following settings specify only the SMAP main-method configuration.
# They are not exposed as separate ablation/comparison experiment switches.
SMAP_PROXY_SEVERITIES = (0.25, 0.5, 1.0)
SMAP_PROXY_FRACTION = 0.10
SMAP_THRESHOLD_QUANTILES = (0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995)


def main(args):
    input_file = Path(args.data)
    if input_file.stem.upper() != 'SMAP':
        raise ValueError('This reduced release accepts only a pre-existing SMAP.npz file')
    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    cfg = Config(
        seed=args.seed, epochs=args.epochs, nodes=10, uniform=False,
        fixed_depth=False, revision='revised', window=100, stride=25,
        eval_stride=100, hidden=64, coarse_steps=20,
        batch=args.batch, prototype_points=30000,
        reconstruction_steps=(500, 750, 900), reconstruction_repeats=2,
        geometry_version='v2', prototype_components=4,
        path_response='clean_residual',
    )
    output = Path(args.output)
    if (output / 'metrics.json').exists():
        raise FileExistsError('Existing completed run: choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    source_files = (Path(__file__), Path(__file__).with_name('vadiffad_paper.py'))
    (output / 'provenance.json').write_text(
        json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in source_files}, indent=2), encoding='utf-8'
    )
    (output / 'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    begin = time.time()

    source = np.load(input_file, allow_pickle=False)
    required = ('train', 'validation', 'calibration', 'test', 'labels')
    missing = [key for key in required if key not in source.files]
    if missing:
        raise ValueError('SMAP.npz is missing: ' + ', '.join(missing))
    fit, val, cal, test = [source[key].astype('float32') for key in required[:4]]
    labels = source['labels'].reshape(-1).astype(int)
    for split, array in zip(required[:4], (fit, val, cal, test)):
        if array.ndim != 2 or array.shape[1] != fit.shape[1] or not np.isfinite(array).all():
            raise ValueError(f'{split} must be finite [time, variables] with shared variable count')
        if len(array) < cfg.window:
            raise ValueError(f'{split} must contain at least {cfg.window} time points')
    if labels.shape != (len(test),) or not np.isin(labels, [0, 1]).all():
        raise ValueError('labels must be a binary vector aligned with the test stream')

    mean = fit.mean(0)
    raw_std = fit.std(0)
    std = np.where(raw_std < cfg.eps, 1.0, raw_std)
    fit, val, cal, test = [(values - mean) / std for values in (fit, val, cal, test)]
    coefficients = variable_coefficients(fit, cfg)
    schedule = Schedule(coefficients, cfg, device)
    model = NormalScoreField(fit.shape[1], cfg).to(device)
    context_model = ContextReconstructor(fit.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(context_model.parameters()),
        lr=cfg.lr, weight_decay=1e-4,
    )
    fw, _ = make_windows(fit, cfg.window, cfg.stride, boundaries(source, 'train'))
    vw, _ = make_windows(val, cfg.window, cfg.window, boundaries(source, 'validation'))
    fx, vx = torch.from_numpy(fw).to(device), torch.from_numpy(vw).to(device)

    def loss(batch):
        t = torch.randint(1, cfg.timesteps + 1, (len(batch),), device=device).float()
        alpha = schedule.alpha(t)[:, None]
        noise = torch.randn_like(batch)
        noisy = alpha.sqrt() * batch + (1 - alpha).sqrt() * noise
        diffusion_loss = (model(noisy, t, schedule.c) - noise).square().mean()
        mask = (torch.rand_like(batch) < cfg.mask_probability).to(batch.dtype)
        context_loss = ((context_model(batch, mask) - batch).square() * mask).sum() / mask.sum().clamp(min=1)
        return diffusion_loss + cfg.context_weight * context_loss

    history, best, patience = [], float('inf'), 0
    for epoch in range(cfg.epochs):
        model.train()
        context_model.train()
        losses = []
        for ids in torch.randperm(len(fx), device=device).split(cfg.batch):
            batch_loss = loss(fx[ids])
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(batch_loss.item())
        model.eval()
        context_model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[0] if device == 'cuda' else []):
            torch.manual_seed(9321)
            validation_loss = float(np.mean([loss(batch).item() for batch in vx.split(cfg.batch)]))
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
               'validation_loss': validation_loss}
        history.append(row)
        if validation_loss < best - 1e-5:
            best, patience = validation_loss, 0
            torch.save({
                'model': model.state_dict(), 'context_model': context_model.state_dict(),
                'config': asdict(cfg), 'mean': mean, 'std': std,
                'c': coefficients, 'epoch': epoch + 1, 'validation_loss': validation_loss,
            }, output / 'model.pt')
        else:
            patience += 1
        (output / 'training.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps(row), flush=True)
        if patience >= args.patience:
            print(f'early stopping at epoch {epoch + 1}', flush=True)
            break

    # This checkpoint was generated locally above, not loaded from an untrusted source.
    try:
        saved = torch.load(output / 'model.pt', map_location=device, weights_only=False)
    except TypeError:  # compatible with older PyTorch versions
        saved = torch.load(output / 'model.pt', map_location=device)
    model.load_state_dict(saved['model'])
    context_model.load_state_dict(saved['context_model'])
    model.eval()
    context_model.eval()
    extractor = PathExtractor(model, schedule, cfg)

    # Construct normal path reference exclusively from the normal training stream.
    rng = np.random.RandomState(cfg.seed + 991)
    reservoir_windows = min(len(fw), int(np.ceil(cfg.prototype_points / cfg.window)))
    reservoir_ids = np.sort(rng.choice(len(fw), size=reservoir_windows, replace=False))
    reservoir = fx[torch.as_tensor(reservoir_ids, device=device, dtype=torch.long)]
    selected_v, selected_c, selected_h = [], [], []
    print('constructing SMAP normal path bank', flush=True)
    for batch in reservoir.split(cfg.batch):
        vv, cc, hh, _ = extractor(batch)
        count = len(batch) * cfg.window
        selected_v.append(vv.cpu().numpy().reshape(count, -1))
        selected_c.append(cc.cpu().numpy().reshape(count, -1))
        selected_h.append(hh.cpu().numpy().reshape(count, fit.shape[1], -1))
    prototype = NormalPrototype(cfg).fit(
        np.concatenate(selected_v), np.concatenate(selected_c), np.concatenate(selected_h)
    )
    np.savez_compressed(output / 'prototype.npz', mu=prototype.mu,
                        inverse=prototype.inv, median=prototype.median, mad=prototype.mad)

    @torch.no_grad()
    def reconstruction_error(batch):
        error = torch.zeros_like(batch)
        time_index = torch.arange(batch.shape[1], device=device)[:, None]
        variable_index = torch.arange(batch.shape[2], device=device)[None, :]
        for phase in range(4):
            mask = (((time_index + variable_index) % 4) == phase).to(batch.dtype)[None].expand_as(batch)
            prediction = context_model(batch, mask)
            error += (prediction - batch).abs() * mask
        return error

    def score_components(series, name, split_boundaries=None):
        print('scoring ' + name, flush=True)
        windows, starts = make_windows(series, cfg.window, cfg.eval_stride, split_boundaries)
        path_parts, recon_parts = [], []
        for index in range(0, len(windows), cfg.batch):
            batch = torch.from_numpy(windows[index:index + cfg.batch]).to(device)
            vv, cc, hh, _ = extractor(batch)
            count = len(batch) * cfg.window
            normalized_components = prototype.standardized_components(
                vv.cpu().numpy().reshape(count, -1),
                cc.cpu().numpy().reshape(count, -1),
                hh.cpu().numpy().reshape(count, fit.shape[1], -1),
            ).reshape(len(batch), cfg.window, fit.shape[1], 3)
            path_parts.append((normalized_components @ np.asarray(cfg.weights)).astype('float32'))
            recon_parts.append(reconstruction_error(batch).cpu().numpy().astype('float32'))
        path_scores = aggregate_overlaps(np.concatenate(path_parts), starts, len(series))
        recon_scores = aggregate_overlaps(np.concatenate(recon_parts), starts, len(series))
        return path_scores, recon_scores

    validation_bounds = boundaries(source, 'validation')
    calibration_bounds = boundaries(source, 'calibration')
    test_bounds = boundaries(source, 'test')
    score_columns = np.array([0])  # SMAP target variable, as in the supplied runner.

    # Validation-only proxy examples are needed for main-method gate/score selection.
    clean_path, clean_recon = score_components(val, 'validation_normal', validation_bounds)
    clean_causal = causal_deviation(val, validation_bounds)
    clean_components = (clean_path[:, score_columns],
                        clean_recon[:, score_columns], clean_causal[:, score_columns])
    proxy_component_suites, proxy_label_suites = [], []
    for suite_index, severity in enumerate(SMAP_PROXY_SEVERITIES):
        proxy_stream, proxy_labels = inject_proxy_anomalies(
            val, cfg.seed + 101 * suite_index, score_columns,
            validation_bounds, SMAP_PROXY_FRACTION, severity,
        )
        path, recon = score_components(proxy_stream, f'validation_proxy_{suite_index}', validation_bounds)
        causal = causal_deviation(proxy_stream, validation_bounds)
        proxy_component_suites.append((path[:, score_columns],
                                       recon[:, score_columns], causal[:, score_columns]))
        proxy_label_suites.append(proxy_labels)
    selected, _ = select_operating_point(
        clean_components, tuple(proxy_component_suites), tuple(proxy_label_suites),
        SMAP_THRESHOLD_QUANTILES, normalization='robust',
    )
    (output / 'validation_selection.json').write_text(
        json.dumps(selected, indent=2), encoding='utf-8'
    )

    cal_path, cal_recon = score_components(cal, 'calibration', calibration_bounds)
    test_path, test_recon = score_components(test, 'test', test_bounds)
    cal_causal = causal_deviation(cal, calibration_bounds)
    test_causal = causal_deviation(test, test_bounds)
    cal_components = (cal_path, cal_recon, cal_causal)
    test_components = (test_path, test_recon, test_causal)
    locations_scales = tuple(
        robust_location_scale(values, cfg.score_floor_ratio, cfg.eps)
        for values in cal_components
    )
    cal_standardized = tuple((values - location) / scale
                             for values, (location, scale) in zip(cal_components, locations_scales))
    test_standardized = tuple((values - location) / scale
                              for values, (location, scale) in zip(test_components, locations_scales))
    weights = (selected['path_weight'], selected['reconstruction_weight'], selected['causal_weight'])
    cal_variable = sum(w * values for w, values in zip(weights, cal_standardized))
    test_variable = sum(w * values for w, values in zip(weights, test_standardized))
    cal_scores = top_q(cal_variable[:, score_columns], selected['q'], selected['smoothing'])
    test_scores = top_q(test_variable[:, score_columns], selected['q'], selected['smoothing'])

    # Primary protocol: a label-free threshold from the first 5% of the test stream.
    warmup_count = min(len(test_scores), max(cfg.window, int(.05 * len(test_scores))))
    threshold = float(np.quantile(test_scores[:warmup_count], selected['quantile']))
    result = metrics(labels, test_scores, threshold)  # no point adjustment, no oracle selection
    report = {
        'dataset': 'SMAP',
        'method': 'VADiffAD',
        'protocol': 'normal-only training; validation-only proxy selection; '
                    'normal-reference score calibration; first-5%-unlabeled-warmup threshold; '
                    'strict point-wise evaluation without point adjustment',
        'configuration': asdict(cfg),
        'selected_epoch': int(saved['epoch']),
        'validation_selection': selected,
        'metrics': result,
        'warmup_points': int(warmup_count),
        'runtime_seconds': float(time.time() - begin),
        'data_sha256': hashlib.sha256(input_file.read_bytes()).hexdigest(),
        'device': device,
    }
    np.savez_compressed(output / 'predictions.npz', scores=test_scores,
                        labels=labels, calibration_scores=cal_scores)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': 'SMAP', 'metrics': result,
                      'output': str(output)}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VADiffAD: SMAP main experiment only')
    parser.add_argument('--data', default='data/SMAP.npz', help='pre-existing SMAP.npz; not generated by this package')
    parser.add_argument('--output', default='runs/smap_seed42')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch < 1 or arguments.patience < 1 or arguments.threads < 1:
        parser.error('--epochs, --batch, --patience and --threads must be positive')
    main(arguments)
