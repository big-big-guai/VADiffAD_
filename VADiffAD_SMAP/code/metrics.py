"""Evaluation metrics shared by training and controlled experiments."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def metrics(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, scores > threshold, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "threshold": float(threshold),
    }
