"""Read already-prepared SMAP arrays and generate windows; no data preparation.

Boundaries are provided by the input archive and keep independent entities separate.
We deliberately do not perform train/validation/test splitting here.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class PreparedSMAP:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    boundaries: dict


def _bounds(data, name, n):
    key = f'{name}_boundaries'
    if key not in data.files:
        return [(0, n)]
    b = np.asarray(data[key], dtype=int)
    if b.ndim != 2 or b.shape[1] != 2 or not len(b):
        raise ValueError(f'{key} must be [segments,2]')
    if b[0, 0] != 0 or b[-1, 1] != n or np.any(b[:, 0] >= b[:, 1]) or np.any(b[1:, 0] != b[:-1, 1]):
        raise ValueError(f'{key} must exactly partition the time axis')
    return [tuple(map(int, row)) for row in b]


def load_prepared_smap(path, window):
    path = Path(path)
    if path.name.lower() != 'smap.npz':
        raise ValueError('The input must be a pre-prepared SMAP.npz')
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        keys = ['train', 'validation', 'calibration', 'test', 'labels']
        absent = [k for k in keys if k not in source.files]
        if absent:
            raise ValueError('Missing arrays: ' + ', '.join(absent))
        arrays = {k: np.asarray(source[k], dtype=np.float32) for k in keys[:4]}
        labels = np.asarray(source['labels']).reshape(-1)
        d = arrays['train'].shape[-1]
        if d != 25:
            raise ValueError('This release follows the paper SMAP representation with D=25 variables')
        limits = {}
        for k, arr in arrays.items():
            if arr.ndim != 2 or arr.shape[1] != d or not np.isfinite(arr).all():
                raise ValueError(f'{k} must be finite [time,{d}]')
            limits[k] = _bounds(source, k, len(arr))
            if any(end - start < window for start, end in limits[k]):
                raise ValueError(f'{k} has an entity shorter than the window')
        if len(labels) != len(arrays['test']) or not np.isin(labels, (0, 1)).all():
            raise ValueError('Test labels must be binary and aligned to test timestamps')
    return PreparedSMAP(**arrays, labels=labels.astype(np.int8), boundaries=limits)


def standardize(smap):
    """Use normal training statistics only, for all four previously separated splits."""
    mu = smap.train.mean(axis=0, dtype=np.float64)
    sigma = smap.train.std(axis=0, dtype=np.float64)
    sigma = np.where(sigma > 1e-6, sigma, 1.)
    transformed = {k: ((getattr(smap, k) - mu) / sigma).astype(np.float32)
                   for k in ('train', 'validation', 'calibration', 'test')}
    return PreparedSMAP(**transformed, labels=smap.labels, boundaries=smap.boundaries), mu, sigma


def windows(series, window, stride, boundaries):
    starts = []
    for left, right in boundaries:
        positions = np.arange(left, right - window + 1, stride)
        starts.extend(np.unique(np.r_[positions, right - window]).tolist())
    starts = np.asarray(sorted(starts), dtype=np.int64)
    return np.stack([series[k:k+window] for k in starts]).astype(np.float32), starts


def fold_windows(local, starts, length):
    """Median aggregation for overlapping estimates, including edge windows."""
    local = np.asarray(local)
    starts = np.asarray(starts, dtype=np.int64)
    if np.any(np.diff(starts) < 0):
        raise ValueError('Window starts must be sorted')
    out = np.empty((length, *local.shape[2:]), dtype=np.float32)
    coverage = np.zeros(length, dtype=np.int32)
    for s in starts:
        coverage[s:s+local.shape[1]] += 1
    if (coverage == 0).any():
        raise ValueError('Uncovered timestamp detected')
    max_cov = int(coverage.max())
    tail = int(np.prod(local.shape[2:])) if local.ndim > 2 else 1
    chunk = max(256, min(10000, (16 * 1024 * 1024) // max(4*max_cov*tail, 1)))
    for start in range(0, length, chunk):
        end = min(length, start+chunk)
        cube = np.full((end-start, max_cov, *local.shape[2:]), np.nan, np.float32)
        occupied = np.zeros(end-start, dtype=np.int32)
        first = np.searchsorted(starts, start-local.shape[1]+1, side='left')
        last = np.searchsorted(starts, end, side='left')
        for row, offset in zip(local[first:last], starts[first:last]):
            l, r = max(start, int(offset)), min(end, int(offset)+local.shape[1])
            idx = np.arange(l-start, r-start)
            cube[idx, occupied[idx]] = row[l-int(offset):r-int(offset)]
            occupied[idx] += 1
        out[start:end] = np.nanmedian(cube, axis=1)
    return out
