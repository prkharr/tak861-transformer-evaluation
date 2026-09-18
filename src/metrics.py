"""Classification metrics and a threshold selected only on VALIDATION."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def _validate(y, probabilities):
    labels = np.asarray(y)
    raw_scores = np.asarray(probabilities)
    if np.iscomplexobj(raw_scores) or (
        raw_scores.dtype == object
        and any(isinstance(value, (complex, np.complexfloating)) for value in raw_scores.flat)
    ):
        raise ValueError("Probabilities must be real values, not complex numbers.")
    scores = np.asarray(raw_scores, dtype=float)
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
    are positive when score >= threshold. Equal scores are never split, and
    integer cross-products identify exact F1 ties without rounding ambiguity.
    """
    labels, scores = _validate(y, probabilities)
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both VALIDATION classes.")
    order = np.argsort(scores, kind="stable")[::-1]
    ranked_scores = scores[order]
    true_positives = np.cumsum(labels[order], dtype=np.int64)
    group_ends = np.r_[np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), len(labels) - 1]
    total_positives = int(labels.sum())
    best_numerator, best_denominator = 0, 1
    best_threshold = float(ranked_scores[0])
    for end in group_ends:
        # F1 = 2 TP / (number selected + total positives). Python integers
        # keep cross-products exact and avoid fixed-width integer overflow.
        numerator = 2 * int(true_positives[end])
        denominator = int(end) + 1 + total_positives
        if numerator * best_denominator > best_numerator * denominator:
            best_numerator, best_denominator = numerator, denominator
            best_threshold = float(ranked_scores[end])
        # Descending thresholds retain the highest cutoff on an exact tie.
    return best_threshold


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
