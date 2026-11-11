"""Normal-only score calibration, local fusion and strict point-wise aggregation."""
import numpy as np


class EmpiricalUpperTail:
    """Z_b,d(u) = -log(max(1 - F_b,d(u), epsilon)), manuscript Eq. (22).

    Stable rank estimator uses (n-rank+1)/(n+1), avoiding -log(0)
    for samples exceeding the calibration maximum.
    """
    def fit(self, values):
        sample = np.asarray(values, dtype=np.float64)
        if sample.ndim != 3:  # [time, variables, statistics]
            raise ValueError('Expected [time, variables, statistics]')
        self.sorted_reference = np.sort(sample, axis=0)
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        reference = self.sorted_reference
        if values.shape[1:] != reference.shape[1:]:
            raise ValueError('Calibration and target feature dimensions must match')
        n, d, f = reference.shape
        result = np.empty_like(values, dtype=np.float64)
        for col in range(d):
            for feature in range(f):
                rank = np.searchsorted(reference[:, col, feature], values[:, col, feature], side='right')
                survival = np.maximum((n-rank+1)/(n+1), 1/(n+1))
                result[:, col, feature] = -np.log(survival)
        return result.astype(np.float32)


def causal_history_residual(series, boundaries, lags=(5, 20, 50)):
    """Historical prediction residual A_l,d; never crosses entity boundaries."""
    result = np.zeros_like(series, dtype=np.float32)
    for begin, end in boundaries:
        x = series[begin:end]
        csum = np.concatenate((np.zeros((1, x.shape[1]), dtype=np.float64),
                               np.cumsum(x, axis=0, dtype=np.float64)), axis=0)
        t = np.arange(len(x))
        out = np.zeros_like(x, dtype=np.float64)
        for lag in lags:
            left = np.maximum(0, t-lag)
            estimate = (csum[t]-csum[left])/np.maximum(t-left, 1)[:, None]
            estimate[0] = x[0]
            out += np.abs(x-estimate)
        result[begin:end] = out / len(lags)
    return result


def fuse_local_scores(calibrated, cfg, gate):
    """Equations (23-25), tensor [time, d, 5] in order V,C,D,R,A."""
    path = np.einsum('tdk,k->td', calibrated[..., :3], cfg.path_weights)
    residual = np.einsum('tdk,k->td', calibrated[..., 3:], cfg.residual_weights)
    return gate*path + residual


def local_top_q(local, q):
    """Top-q across ALL D variables, without limiting scoring to variable 0."""
    q = min(max(int(q), 1), local.shape[1])
    return np.partition(local, local.shape[1]-q, axis=1)[:, -q:].mean(axis=1)


def warmup_thresholds(scores, boundaries, cfg):
    """First 5% of each unlabeled test entity, threshold before examining labels."""
    scores = np.asarray(scores)
    cutoff = np.empty(len(scores), dtype=np.float64)
    summary = []
    for start, end in boundaries:
        n = max(1, int(np.ceil((end-start)*cfg.warmup_fraction)))
        threshold = float(np.quantile(scores[start:start+n], cfg.warmup_quantile))
        cutoff[start:end] = threshold
        summary.append({'start': start, 'end': end, 'warmup_points': n, 'threshold': threshold})
    return cutoff, summary
