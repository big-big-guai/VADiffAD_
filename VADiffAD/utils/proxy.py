"""Synthetic validation proxy families for rejectable gate selection only.

These synthetic events are not real anomaly labels, and are not used for
training the noise predictor, fitting normal references, or thresholding test.
"""
import numpy as np
from sklearn.metrics import average_precision_score
from base.metrics import evaluate_pointwise
from vadiffad.scoring import fuse_local_scores, local_top_q


def inject_validation_proxies(clean, boundaries, seed, severity=1., fraction=.1):
    rng = np.random.default_rng(seed)
    modified = clean.copy()
    labels = np.zeros(len(clean), dtype=np.int8)
    # One family per synthetic pattern, chosen before evaluation.
    for left, right in boundaries:
        n = right-left
        width = max(6, min(32, n//12))
        if width >= n:
            continue
        count = max(1, int(round(fraction*n/width)))
        offsets = np.linspace(left+1, right-width-1, count, dtype=int)
        for index, start in enumerate(offsets):
            stop = min(start+width, right)
            cols = rng.choice(clean.shape[1], min(3, clean.shape[1]), replace=False)
            selection = np.arange(start, stop)
            kind = index % 4
            if kind == 0:  # isolated level shift
                modified[np.ix_(selection, cols)] += severity * rng.choice((-4., 4.), len(cols))
            elif kind == 1:  # periodic response
                phase = np.arange(len(selection))[:, None]
                modified[np.ix_(selection, cols)] += severity*2*np.sin(phase*np.pi/2)
            elif kind == 2:  # local temporal reversal
                slice_ = modified[np.ix_(selection, cols)].copy()
                modified[np.ix_(selection, cols)] = (1-severity)*slice_ + severity*slice_[::-1]
            else:  # trend change
                drift = np.linspace(-2, 2, len(selection))[:, None]
                modified[np.ix_(selection, cols)] += severity*drift
            labels[selection] = 1
    return modified, labels


def select_rejectable_gate(cfg, proxy_calibrated, proxy_labels, clean_calibrated):
    """An explicit realization of the manuscript's abstract G_delta gate.

    Compare every path-on candidate against g=0 with the same Top-q and a
    fixed normal-validation quantile. Admit only if proxy AP improves by
    at least delta AND F1 does not decrease for any proxy family.
    Never use real test labels. Paper does not specify exact functional G_delta.
    """
    baseline = local_top_q(fuse_local_scores(clean_calibrated, cfg, 0.), cfg.top_q)
    fixed_threshold = float(np.quantile(baseline, cfg.proxy_quantile))
    base_aps = []
    base_f1s = []
    for suite, y in zip(proxy_calibrated, proxy_labels):
        s = local_top_q(fuse_local_scores(suite, cfg, 0.), cfg.top_q)
        base_aps.append(float(average_precision_score(y, s)) if y.sum() else 0.)
        base_f1s.append(evaluate_pointwise(y, s, fixed_threshold)['f1'])
    base_ap = float(np.mean(base_aps))
    eligible = []
    report = []
    for g in (0., .25, .5, .75, 1.):
        normal_scores = local_top_q(fuse_local_scores(clean_calibrated, cfg, g), cfg.top_q)
        threshold = float(np.quantile(normal_scores, cfg.proxy_quantile))
        ap, f1 = [], []
        for suite, y in zip(proxy_calibrated, proxy_labels):
            s = local_top_q(fuse_local_scores(suite, cfg, g), cfg.top_q)
            ap.append(float(average_precision_score(y, s)) if y.sum() else 0.)
            f1.append(evaluate_pointwise(y, s, threshold)['f1'])
        mean_ap = float(np.mean(ap))
        relative_ap_gain = (mean_ap-base_ap)/max(base_ap, 1e-12)
        suite_differences = [float(a-b) for a, b in zip(f1, base_f1s)]
        accepted = bool(g>0 and relative_ap_gain >= cfg.proxy_margin and
                        all(delta >= -1e-12 for delta in suite_differences))
        report.append({'gate': g, 'mean_proxy_ap': mean_ap,
                       'relative_proxy_ap_gain': relative_ap_gain,
                       'f1_differences': suite_differences, 'admitted': accepted})
        if accepted:
            eligible.append((mean_ap, -g, g))
    gate = max(eligible)[2] if eligible else 0.
    return gate, {'margin': cfg.proxy_margin, 'base_proxy_ap': base_ap,
                  'gate': gate, 'candidates': report,
                  'selection_rule': 'proxy mean AP relative gain >= margin and no proxy-family F1 loss'}
