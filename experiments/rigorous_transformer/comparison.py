"""Aggregate evaluation for locked finalists, never model selection on TEST.

Average precision (AP) and trapezoidal PR area are deliberately separate.
Bootstrap intervals condition on the fitted predictions and resample patients,
keeping all snapshots and both models paired. They do not include retraining
variation, resolve unknown provenance, or make a reused holdout prospective.
"""
import hashlib
import json
import math

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)


def _inputs(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or not len(y) or len(y) != len(scores):
        raise ValueError("Labels and probability scores must be equal nonempty vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary and nonmissing")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0,1]; logits require an explicit conversion")
    return y.astype(np.int8), scores


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _tie_keys(n, tie_breaker):
    if tie_breaker is None:
        return np.arange(n), "frozen_input_row_order"
    keys = np.asarray(tie_breaker).astype(str)
    if keys.ndim != 1 or len(keys) != n or len(np.unique(keys)) != n:
        raise ValueError("Tie keys must be unique, stable snapshot keys in prediction order")
    return keys, "supplied_unique_snapshot_keys"


def topk_metrics(y, scores, fraction=0.1, tie_breaker=None):
    """Exactly ceil(fraction*N) selected; stable ties are independent of labels."""
    y, scores = _inputs(y, scores)
    if not 0 < float(fraction) <= 1:
        raise ValueError("Top fraction must lie in (0,1]")
    keys, tie_rule = _tie_keys(len(y), tie_breaker)
    k = max(1, int(math.ceil(float(fraction) * len(y))))
    order = np.lexsort((keys, -scores))
    selected = order[:k]
    positives = int(y.sum())
    captured = int(y[selected].sum())
    boundary = float(scores[selected[-1]])
    tied = scores == boundary
    tied_selected = int(tied[selected].sum())
    precision = captured / k
    prevalence = positives / len(y)
    return dict(fraction=float(fraction), selected=k, selected_positives=captured,
                actual_population_fraction=k / len(y), precision=precision,
                recall=_ratio(captured, positives), lift=_ratio(precision, prevalence),
                cutoff_score=boundary, n_tied_at_cutoff=int(tied.sum()),
                n_tied_selected=tied_selected, tie_crosses_boundary=bool(tied.sum() > tied_selected),
                tie_rule=tie_rule, rule="ceil(fraction*N), descending probability, ascending stable key")


def binary_metrics(y, scores, threshold=None, top_fractions=(0.1,), calibration_bins=10,
                   tie_breaker=None):
    y, scores = _inputs(y, scores)
    if isinstance(calibration_bins, bool) or not 1 <= int(calibration_bins) <= 100:
        raise ValueError("Use 1 to 100 fixed-width calibration bins")
    calibration_bins = int(calibration_bins)
    positives, n = int(y.sum()), len(y)
    both = 0 < positives < n
    precision, recall, _ = precision_recall_curve(y, scores) if positives else (None, None, None)
    integrate = getattr(np, "trapezoid", np.trapz if hasattr(np, "trapz") else None)
    result = dict(snapshots=n, positives=positives, negatives=n - positives, prevalence=positives / n,
                  average_precision=float(average_precision_score(y, scores)) if positives else None,
                  pr_auc_trapezoidal=float(integrate(precision[::-1], recall[::-1])) if positives else None,
                  roc_auc=float(roc_auc_score(y, scores)) if both else None,
                  brier_score=float(brier_score_loss(y, scores)),
                  log_loss=float(log_loss(y, scores, labels=[0, 1])),
                  metric_definition="AP=sum(recall increment*precision); PR area uses trapezoids; these are not interchangeable",
                  single_class=not both)
    if threshold is not None:
        if isinstance(threshold, dict):
            if threshold.get("selection_split") != "validation" or threshold.get("objective") != "f1":
                raise ValueError("Threshold policy must be locked on VALIDATION")
            if threshold.get("policy_sha256") != _json_hash({k: v for k, v in threshold.items() if k != "policy_sha256"}):
                raise ValueError("Validation threshold policy hash mismatch")
            value = float(threshold["threshold"])
        else:
            value = float(threshold)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Probability threshold must lie in [0,1]")
        predicted = scores >= value
        tp, fp = int((predicted & (y == 1)).sum()), int((predicted & (y == 0)).sum())
        fn, tn = positives - tp, n - positives - fp
        result.update(threshold=value, tp=tp, fp=fp, fn=fn, tn=tn,
                      precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
                      specificity=_ratio(tn, tn + fp), f1=_ratio(2 * tp, 2 * tp + fp + fn),
                      accuracy=(tp + tn) / n,
                      confusion_matrix=[[tn, fp], [fn, tp]],
                      threshold_scope="validation_locked" if isinstance(threshold, dict) else "caller_supplied")
    bins = np.minimum((scores * calibration_bins).astype(int), calibration_bins - 1)
    rows = []
    for b in range(calibration_bins):
        mask = bins == b
        count = int(mask.sum())
        rows.append(dict(bin=b, lower=b / calibration_bins, upper=(b + 1) / calibration_bins,
                         count=count, mean_probability=float(scores[mask].mean()) if count else None,
                         observed_rate=float(y[mask].mean()) if count else None))
    result["calibration"] = rows
    result["expected_calibration_error"] = float(sum(
        row["count"] / n * abs(row["mean_probability"] - row["observed_rate"])
        for row in rows if row["count"]))
    result["calibration_note"] = "Fixed-width descriptive calibration; no calibration model fitted to evaluation labels"
    result["topk"] = [topk_metrics(y, scores, f, tie_breaker) for f in top_fractions]
    return result


def select_validation_threshold(y_validation, scores_validation, objective="f1", *, selection_split="validation"):
    """Choose once on VALIDATION; highest threshold breaks an exact F1 tie."""
    if selection_split != "validation" or objective != "f1":
        raise ValueError("Only the prespecified VALIDATION F1 policy is supported")
    y, scores = _inputs(y_validation, scores_validation)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes in VALIDATION")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    record = dict(threshold=float(thresholds[best]), objective=objective, selection_split="validation",
                  validation_f1=float(f1[best]), validation_snapshots=len(y),
                  tie_rule="highest probability threshold among exactly equal maximum F1 values",
                  decision_rule="score >= threshold")
    record["policy_sha256"] = _json_hash(record)
    return record


def lock_validation_policy(y_validation, scores_validation, top_fractions=(0.1,)):
    """Top-K locks population fractions, not a score cutoff derived from TEST."""
    fractions = list(map(float, top_fractions))
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError("Invalid operational population fractions")
    return dict(f1=select_validation_threshold(y_validation, scores_validation),
                top_fractions=fractions,
                topk_rule="ceil(fraction*N); descending score; ascending prespecified unique snapshot key",
                selection_split="validation")


def _patient_codes(patient_ids, n):
    patients = np.asarray(patient_ids)
    if patients.ndim != 1 or len(patients) != n:
        raise ValueError("Patient groups must match predictions")
    if any(v is None or str(v).strip().lower() in ("", "nan", "none", "nat") for v in patients):
        raise ValueError("Patient IDs must be nonmissing")
    _, codes = np.unique(patients.astype(str), return_inverse=True)
    return codes


def paired_patient_bootstrap(y, scores_a, scores_b, patient_ids, n_bootstrap=1000,
                             seed=20261006, confidence=0.95):
    """Delta = model A minus B. Cluster multiplicities act as row weights."""
    y, a = _inputs(y, scores_a)
    _, b = _inputs(y, scores_b)
    if len(np.unique(y)) != 2:
        raise ValueError("Paired AP/AUC inference requires both classes")
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 20:
        raise ValueError("At least 20 bootstrap draws required; use >=1000 for reports")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0,1)")
    codes = _patient_codes(patient_ids, len(y))
    groups = int(codes.max()) + 1
    if groups < 2:
        raise ValueError("At least two patients required")
    rng = np.random.default_rng(seed)
    draws, invalid = [], 0
    for _ in range(int(n_bootstrap)):
        multiplicity = np.bincount(rng.integers(0, groups, size=groups), minlength=groups)
        weights = multiplicity[codes]
        if weights[y == 1].sum() == 0 or weights[y == 0].sum() == 0:
            invalid += 1
            continue
        ap = average_precision_score(y, a, sample_weight=weights) - average_precision_score(y, b, sample_weight=weights)
        auc = roc_auc_score(y, a, sample_weight=weights) - roc_auc_score(y, b, sample_weight=weights)
        draws.append((ap, auc))
    result = dict(unit="patient", patients=groups, requested_draws=int(n_bootstrap), valid_draws=len(draws),
                  single_class_draws=invalid, seed=int(seed), confidence=float(confidence),
                  direction="model_a_minus_model_b", paired=True,
                  uncertainty_scope="conditional on fitted predictions; excludes retraining and model-selection uncertainty")
    enough = len(draws) >= max(20, int(math.ceil(0.8 * n_bootstrap)))
    result["interval_reliable"] = enough
    values = np.asarray(draws) if draws else np.empty((0, 2))
    alpha = (1 - confidence) / 2
    for index, (name, func) in enumerate((("average_precision", average_precision_score), ("roc_auc", roc_auc_score))):
        result[name] = dict(delta=float(func(y, a) - func(y, b)),
                            lower=float(np.quantile(values[:, index], alpha)) if enough else None,
                            upper=float(np.quantile(values[:, index], 1 - alpha)) if enough else None)
    return result


def latest_patient_indices(patient_ids, end_dates):
    """One latest snapshot per patient; reject ambiguous duplicate latest keys."""
    dates = np.asarray(end_dates, dtype="datetime64[D]")
    codes = _patient_codes(patient_ids, len(dates))
    if dates.ndim != 1 or np.isnat(dates).any():
        raise ValueError("Latest-patient evaluation requires exact nonmissing dates")
    result = []
    for patient in range(int(codes.max()) + 1):
        rows = np.flatnonzero(codes == patient)
        latest = rows[dates[rows] == dates[rows].max()]
        if len(latest) != 1:
            raise ValueError("Duplicate latest patient-snapshot keys")
        result.append(int(latest[0]))
    return np.asarray(result, dtype=int)


def summarize_seed_runs(results):
    """Accept training.py's candidate -> complete seed-run list; retain every seed."""
    if not isinstance(results, dict) or not results:
        raise ValueError("Expected candidate-to-seed-runs dictionary")
    all_seeds, candidates = [], []
    for candidate, runs in results.items():
        seeds = []
        for run in runs:
            summary = run["summary"]
            if not summary.get("training_complete"):
                raise ValueError("Incomplete runs cannot disappear from the final comparison")
            seed = int(summary["seed"])
            if seed in seeds:
                raise ValueError("Duplicate seed in candidate")
            seeds.append(seed)
            row = dict(candidate=candidate, seed=seed,
                       family=summary["config"]["family"],
                       parameter_count=summary.get("parameter_count"),
                       complexity_unit=summary.get("complexity_unit"),
                       best_epoch=summary.get("best_epoch"), wall_seconds=summary.get("wall_seconds"),
                       validation_evaluations=summary.get("validation_evaluations"))
            for split in ("train", "validation", "test"):
                for metric, value in summary.get(split + "_metrics", {}).items():
                    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                        row[split + "_" + metric] = float(value)
            train_ap, val_ap = row.get("train_average_precision"), row.get("validation_average_precision")
            row["train_validation_gap_ap"] = train_ap - val_ap if train_ap is not None and val_ap is not None else None
            test_ap = row.get("test_average_precision")
            row["train_test_gap_ap"] = train_ap - test_ap if train_ap is not None and test_ap is not None else None
            all_seeds.append(row)
        if not seeds:
            raise ValueError("Candidate has no completed seed repetitions")
        candidate_rows = [r for r in all_seeds if r["candidate"] == candidate]
        values = [r["validation_average_precision"] for r in candidate_rows]
        candidates.append(dict(candidate=candidate, seeds=seeds, n_seeds=len(seeds),
                               mean_validation_ap=float(np.mean(values)),
                               sd_validation_ap=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                               min_validation_ap=float(np.min(values)), max_validation_ap=float(np.max(values)),
                               seed_selection="all prespecified seeds, none selected by outcome"))
    return dict(all_seeds=all_seeds, candidates=candidates,
                note="Seed SD is training variability, not a confidence interval; bootstrap intervals are separate")


def comparison_conclusion(bootstrap, provenance):
    """Wording is gated by comparison provenance, never VALIDATION significance."""
    provenance = dict(provenance or {})
    base = dict(status="UNRESOLVED", conclusion=None, restrictions=[])
    if provenance.get("baseline_kind") == "original_v63" or not provenance.get("fit_membership_verified", False):
        base.update(status="DESCRIPTIVE_ONLY", conclusion="The existing LightGBM scores have unresolved fitting provenance; they cannot establish comparative generalization or Transformer superiority.")
        return base
    required = ("same_cohort", "same_labels", "patient_disjoint_split", "selection_locked")
    missing = [key for key in required if not provenance.get(key, False)]
    if missing or provenance.get("test_used_for_selection", True):
        base["restrictions"] = missing + (["test_used_for_selection_or_unknown"] if provenance.get("test_used_for_selection", True) else [])
        base["conclusion"] = "A matched final comparison is not established; resolve the listed protocol conditions before making a superiority claim."
        return base
    if provenance.get("comparison_split") != "test":
        base.update(status="DEVELOPMENT_ONLY", conclusion="Validation differences are development evidence after model selection; final-test superiority is not established.")
        return base
    for key in ("upstream_cutoff_verified", "holdout_never_previously_inspected"):
        if not provenance.get(key, False):
            base["restrictions"].append(key)
    if not provenance.get("same_feature_information", False):
        base["restrictions"].append("different_feature_information_architecture_effect_not_isolated")
    ap = bootstrap["average_precision"]
    if not bootstrap.get("interval_reliable") or ap["lower"] is None:
        base.update(status="UNCERTAINTY_UNRESOLVED", conclusion="Too few valid patient-bootstrap draws support a reliable interval; report the point difference without a superiority conclusion.")
        return base
    prefix = "In this retrospective matched comparison" if base["restrictions"] else "On the locked patient-disjoint test cohort"
    if ap["lower"] > 0:
        verdict = "Transformer AP is higher than matched LightGBM AP"
        status = "RETROSPECTIVE_AP_ADVANTAGE" if base["restrictions"] else "TEST_AP_ADVANTAGE"
    elif ap["upper"] < 0:
        verdict = "matched LightGBM AP is higher than Transformer AP"
        status = "LIGHTGBM_AP_ADVANTAGE"
    else:
        verdict = "the AP interval includes no difference; superiority is not established and this is not proof of equivalence"
        status = "NO_DEMONSTRATED_AP_ADVANTAGE"
    base.update(status=status, conclusion=f"{prefix}, {verdict}. Delta AP (Transformer minus LightGBM) = {ap['delta']:.6f}, patient-bootstrap interval [{ap['lower']:.6f}, {ap['upper']:.6f}]. This interval conditions on the fitted models and excludes training/search uncertainty.")
    if base["restrictions"]:
        base["conclusion"] += " Unresolved provenance or prior holdout inspection prevents a prospective generalization claim."
    return base


def compare_predictions(y, transformer_scores, lightgbm_scores, patient_ids, *,
                        threshold_policies=None, provenance=None, tie_breaker=None,
                        top_fractions=(0.1,), n_bootstrap=1000, seed=20261006):
    """Caller aligns both score vectors to identical frozen keys and persists lock first."""
    policies = threshold_policies or {}
    t = binary_metrics(y, transformer_scores, policies.get("transformer"), top_fractions=top_fractions,
                       tie_breaker=tie_breaker)
    l = binary_metrics(y, lightgbm_scores, policies.get("lightgbm"), top_fractions=top_fractions,
                       tie_breaker=tie_breaker)
    interval = paired_patient_bootstrap(y, transformer_scores, lightgbm_scores, patient_ids,
                                        n_bootstrap=n_bootstrap, seed=seed)
    return dict(transformer=t, lightgbm=l, paired_uncertainty=interval,
                conclusion=comparison_conclusion(interval, provenance), provenance=dict(provenance or {}))
