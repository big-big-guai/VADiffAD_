"""Strict point-wise metrics. No point adjustment or test-label threshold search."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_pointwise(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    threshold = np.asarray(threshold)
    prediction = scores > threshold
    truth = labels == 1
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    p = tp / max(tp+fp, 1)
    r = tp / max(tp+fn, 1)
    f1 = 2*p*r / max(p+r, 1e-12)
    return {'precision': p, 'recall': r, 'f1': f1,
            'average_precision': float(average_precision_score(labels, scores)) if truth.any() else None,
            'auroc': float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            'tp': tp, 'fp': fp, 'fn': fn}
