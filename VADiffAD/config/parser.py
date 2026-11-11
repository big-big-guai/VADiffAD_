"""Parse the SMAP main-method YAML (no alternate dataset/ablation registry)."""
from dataclasses import dataclass, fields
from pathlib import Path
import yaml


@dataclass
class SMAPConfig:
    dataset: str = 'SMAP'
    data_file: str = 'data/SMAP.npz'
    output_dir: str = 'runs/SMAP'
    window: int = 100
    train_stride: int = 25
    eval_stride: int = 100
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    rho: float = 0.5
    cmin: float = 0.5
    cmax: float = 2.0
    hidden_dim: int = 64
    probe_nodes: int = 20
    path_intervals: int = 10
    retention_threshold: float = 0.2
    retention_delta_threshold: float = 0.02
    bank_max_signatures: int = 30000
    cov_shrinkage: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 20
    patience: int = 6
    context_loss_weight: float = 0.5
    mask_probability: float = 0.25
    path_weights: tuple = (1/3, 1/3, 1/3)
    residual_weights: tuple = (0.5, 0.5)
    top_q: int = 3
    proxy_margin: float = 0.06
    proxy_quantile: float = 0.99
    warmup_fraction: float = 0.05
    warmup_quantile: float = 0.99
    proxy_severities: tuple = (0.25, 0.5, 1.0)
    proxy_fraction: float = 0.1
    seeds: tuple = (42, 43, 44)
    eps: float = 1e-6


def load_config(path):
    obj = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError('Configuration must be a YAML mapping')
    known = {f.name for f in fields(SMAPConfig)}
    extra = set(obj) - known
    if extra:
        raise ValueError(f'Unrecognized configuration keys: {sorted(extra)}')
    for key in ('path_weights', 'residual_weights', 'proxy_severities', 'seeds'):
        if key in obj:
            obj[key] = tuple(obj[key])
    cfg = SMAPConfig(**obj)
    if cfg.dataset != 'SMAP':
        raise ValueError('This release supports SMAP only')
    if cfg.path_intervals < 2 or cfg.window < 2 or cfg.probe_nodes < 2:
        raise ValueError('Invalid window/path/probe dimensions')
    if cfg.bank_max_signatures < cfg.path_intervals * 2 * 25:
        raise ValueError('Normal path bank too small for covariance estimation')
    if len(cfg.path_weights) != 3 or len(cfg.residual_weights) != 2:
        raise ValueError('Path/residual weights must have lengths 3 and 2')
    if not (0 <= cfg.proxy_margin < 1 and 0 < cfg.warmup_fraction <= 1):
        raise ValueError('Invalid gate margin/warm-up fraction')
    return cfg
