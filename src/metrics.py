"""Classification metrics and a threshold selected only on VALIDATION."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def _validate(y, probabilities):
    labels = np.asarray(y)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Labels and probabilities must be aligned, nonempty 1-D arrays.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return labels.astype(np.int64), scores


def select_validation_threshold(y, probabilities) -> float:
    """Maximize VALIDATION F1; an exact tie uses the highest threshold.

    The caller must provide VALIDATION labels and scores, never TEST. Predictions
    are positive when score >= threshold. All-equal scores are handled explicitly
    by the precision/recall curve, without splitting tied scores.
    """
    labels, scores = _validate(y, probabilities)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both VALIDATION classes.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    best = np.flatnonzero(f1 == f1.max())
    return float(thresholds[best[-1]])


def classification_metrics(y, probabilities, threshold: float) -> dict:
    labels, scores = _validate(y, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The fixed classification threshold must be in [0, 1].")
    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
