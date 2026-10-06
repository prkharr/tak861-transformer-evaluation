"""Bounded TRAIN-only signals about saved count sequences, never lineage proof.

Exact overlapping-month equality can arise from correct windows, copied values,
stable activity or no activity. It cannot establish historical orientation or an
event cutoff. E05B did not audit all count values; NaN is unknown, not zero. A
mask from the separate 49-feature representation cannot certify this tensor.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inputs(X, metadata, feature_map, valid):
    _require(isinstance(X, np.ndarray) and X.ndim == 3,
             "X must be a raw-count ndarray/memmap in (snapshot, step, feature) order.")
    _require(X.shape[1] >= 4 and X.shape[2] > 0, "Need at least four steps and one feature.")
    _require(np.issubdtype(X.dtype, np.number) and not np.iscomplexobj(X),
             "Raw counts must have a real numeric dtype.")
    meta = pd.DataFrame(metadata)
    _require(len(meta) == len(X) and {"SPLIT", "PATIENT_ID", "END_DT"} <= set(meta),
             "Metadata needs aligned SPLIT, PATIENT_ID and END_DT columns.")
    # Only the split selector is read across all rows. Non-TRAIN dates, IDs,
    # labels, masks and tensor values are never inspected by these diagnostics.
    rows = np.flatnonzero(meta.SPLIT.astype(str).str.lower().eq("train").to_numpy())
    _require(len(rows) > 0, "No TRAIN rows available.")
    train = meta.iloc[rows].copy().reset_index(drop=True)
    _require(train.PATIENT_ID.map(lambda v: isinstance(v, str) and bool(v)).all(),
             "TRAIN patient IDs must be nonempty strings.")
    dates = pd.to_datetime(train.END_DT, errors="raise")
    _require(not dates.isna().any() and dates.dt.tz is None,
             "TRAIN END_DT must contain nonmissing timezone-naive dates.")
    train["END_DT"] = dates
    _require(not train[["PATIENT_ID", "END_DT"]].duplicated().any(),
             "Duplicate TRAIN snapshot identity.")
    train["_source_row"] = rows
    train["_month"] = dates.dt.year * 12 + dates.dt.month
    fmap = pd.DataFrame(feature_map).copy().reset_index(drop=True)
    _require({"FEATURE_INDEX", "FEATURE_NAME", "FEATURE_COLUMN"} <= set(fmap),
             "Feature map needs FEATURE_INDEX, FEATURE_NAME and FEATURE_COLUMN.")
    _require(len(fmap) == X.shape[2] and
             np.array_equal(fmap.FEATURE_INDEX.to_numpy(), np.arange(X.shape[2])),
             "Feature map must already match exact zero-based tensor order.")
    for col in ("FEATURE_NAME", "FEATURE_COLUMN"):
        _require(fmap[col].map(lambda v: isinstance(v, str) and bool(v)).all()
                 and fmap[col].is_unique, "Feature names and aliases must be unique strings.")
    if valid is not None:
        _require(isinstance(valid, np.ndarray) and valid.dtype == bool
                 and valid.shape == X.shape[:2], "valid must be a boolean (N,T) array.")
    return train, fmap


def _counts(block):
    """Validate only an already permitted TRAIN slice; preserve missing values."""
    _require(not np.isinf(block).any(), "Infinite value in inspected TRAIN counts.")
    observed = block[np.isfinite(block)]
    _require(not ((observed < 0) | (observed != np.floor(observed))).any(),
             "Inspected raw TRAIN counts must be nonnegative integers or NaN.")


def _partial_month_risk(train):
    dates = train.END_DT
    remaining = dates.dt.days_in_month - dates.dt.day
    return {
        "status": "INCONCLUSIVE",
        "train_snapshots": int(len(train)),
        "anchors_before_calendar_month_end": int((remaining > 0).sum()),
        "possible_post_anchor_days_if_whole_month_min": int(remaining.min()),
        "possible_post_anchor_days_if_whole_month_max": int(remaining.max()),
        "event_cutoff_verified": False,
        "whole_month_aggregation_verified": False,
        "reason": "Calendar dates describe exposure to risk only; saved counts do not establish event cutoffs.",
        "without_step0_arm_advisable": True,
        "arm_condition": "Predeclare without_step0 sensitivity while cutoff is unresolved; it cannot prove leakage removal.",
    }


def _pairs(train, steps, limit):
    candidates = []
    for patient, group in train.groupby("PATIENT_ID", sort=True):
        ordered = group.sort_values("END_DT", kind="stable")
        records = list(ordered[["_source_row", "_month", "END_DT"]].itertuples(index=False, name=None))
        for earlier, later in zip(records, records[1:]):
            delta = int(later[1] - earlier[1])
            if not 1 <= delta <= steps - 3:
                continue  # At least one common interior month must remain.
            identity = json.dumps([patient, earlier[2].isoformat(), later[2].isoformat()],
                                  ensure_ascii=True, separators=(",", ":"))
            digest = hashlib.sha256(identity.encode()).hexdigest()
            candidates.append((digest, int(earlier[0]), int(later[0]), delta))
    selected = sorted(candidates)[:limit]
    # A combined fingerprint permits repeatability checks without exporting IDs.
    fingerprint = hashlib.sha256("\n".join(x[0] for x in selected).encode()).hexdigest()
    return selected, len(candidates), fingerprint


def overlap_month_consistency(X, metadata, feature_map, valid=None, max_pairs=200,
                              min_informative_pairs=3, min_varying_comparisons=20,
                              agreement_threshold=0.99):
    """Test both month-order hypotheses on <=200 deterministic adjacent TRAIN pairs.

    The first and last positions are excluded in *both* snapshots. Consequently
    no hypothesized current partial month or oldest window boundary participates.
    Comparisons with NaN or a supplied invalid position are excluded explicitly.
    Status describes only observed agreement on active, temporally varying count
    cells. A high agreement status never certifies orientation or historical fit.
    """
    _require(type(max_pairs) is int and 1 <= max_pairs <= 200, "max_pairs must be in 1..200.")
    _require(type(min_informative_pairs) is int and min_informative_pairs > 0,
             "min_informative_pairs must be positive.")
    _require(type(min_varying_comparisons) is int and min_varying_comparisons > 0,
             "min_varying_comparisons must be positive.")
    _require(0 < agreement_threshold <= 1, "agreement_threshold must be in (0,1].")
    train, fmap = _inputs(X, metadata, feature_map, valid)
    pairs, candidate_count, fingerprint = _pairs(train, X.shape[1], max_pairs)
    results = {}
    for orientation in ("newest_first", "oldest_first"):
        counts = dict(finite_comparisons=0, equal_comparisons=0, active_comparisons=0,
                      active_equal_comparisons=0, varying_active_comparisons=0,
                      varying_active_equal_comparisons=0, missing_comparisons=0,
                      masked_comparisons=0, pair_count=0, informative_pair_count=0)
        for _, earlier, later, delta in pairs:
            base = np.arange(1, X.shape[1] - 1 - delta)
            old_steps, new_steps = ((base, base + delta) if orientation == "newest_first"
                                    else (base + delta, base))
            a, b = X[earlier, old_steps, :], X[later, new_steps, :]
            _counts(a)
            _counts(b)
            step_ok = (np.ones(len(base), bool) if valid is None
                       else valid[earlier, old_steps] & valid[later, new_steps])
            finite = np.isfinite(a) & np.isfinite(b) & step_ok[:, None]
            equal = (a == b) & finite
            active = ((a > 0) | (b > 0)) & finite
            # Stable nonzero columns cannot distinguish copied/static snapshots.
            varying_feature = np.zeros(len(fmap), bool)
            for j in np.flatnonzero(active.any(axis=0)):
                values_a, values_b = a[finite[:, j], j], b[finite[:, j], j]
                varying_feature[j] = ((len(values_a) > 1 and np.ptp(values_a) > 0)
                                      or (len(values_b) > 1 and np.ptp(values_b) > 0))
            varying = active & varying_feature[None, :]
            counts["finite_comparisons"] += int(finite.sum())
            counts["equal_comparisons"] += int(equal.sum())
            counts["active_comparisons"] += int(active.sum())
            counts["active_equal_comparisons"] += int((active & equal).sum())
            counts["varying_active_comparisons"] += int(varying.sum())
            counts["varying_active_equal_comparisons"] += int((varying & equal).sum())
            counts["missing_comparisons"] += int(((~np.isfinite(a) | ~np.isfinite(b)) & step_ok[:, None]).sum())
            counts["masked_comparisons"] += int((~step_ok).sum()) * len(fmap)
            counts["pair_count"] += int(finite.any())
            counts["informative_pair_count"] += int(varying.any())
        for name, num, den in (("agreement_ratio", "equal_comparisons", "finite_comparisons"),
                               ("active_agreement_ratio", "active_equal_comparisons", "active_comparisons"),
                               ("varying_active_agreement_ratio", "varying_active_equal_comparisons", "varying_active_comparisons")):
            counts[name] = counts[num] / counts[den] if counts[den] else None
        informative = (counts["informative_pair_count"] >= min_informative_pairs
                       and counts["varying_active_comparisons"] >= min_varying_comparisons)
        counts["status"] = ("INCONCLUSIVE" if not informative else
                            "CORROBORATED" if counts["varying_active_agreement_ratio"] >= agreement_threshold
                            else "CONTRADICTED")
        counts["interpretation"] = ("Insufficient active temporal variation; equality is uninformative."
                                    if not informative else "Observed interior-month consistency only; not historical orientation or cutoff proof.")
        results[orientation] = counts
    return {
        "scope": "TRAIN_ONLY_BOUNDED_INTERIOR_MONTH_CONSISTENCY",
        "candidate_adjacent_pair_count": candidate_count,
        "sampled_patient_pair_count": len(pairs),
        "max_pairs": max_pairs,
        "sample_fingerprint": fingerprint,
        "sampling": "SHA256-ordered adjacent source TRAIN snapshots; independent of labels and counts",
        "excluded_positions": [0, X.shape[1] - 1],
        "thresholds": dict(min_informative_pairs=min_informative_pairs,
                           min_varying_comparisons=min_varying_comparisons,
                           agreement_threshold=agreement_threshold),
        "mask_source": "all_positions_retained_no_coverage_claim" if valid is None else "caller_supplied_not_certified_here",
        "hypotheses": results,
        "orientation_status": "INCONCLUSIVE",
        "orientation_verified": False,
        "historical_provenance_verified": False,
        "partial_month_risk": _partial_month_risk(train),
        "limitation": "Agreement can reflect stable/copy/no-activity behavior. Even discriminating hypotheses cannot certify source execution or whole-month cutoffs.",
    }


def step0_label_screen(X, metadata, feature_map, valid=None, feature_batch_size=128):
    """Per-feature physical step-0/step-1 associations on paired observed TRAIN rows.

    Returns a DataFrame sorted by descending AP difference, then positive-class
    nonzero-rate difference. Both AP values use exactly the same complete cases
    with equal total weight per represented patient. Missing counts are reported
    and excluded, not assumed absent or imputed. No p-values or leakage verdicts
    are produced; these supervised descriptive screens are not feature selection.
    """
    _require(type(feature_batch_size) is int and feature_batch_size > 0,
             "feature_batch_size must be positive.")
    train, fmap = _inputs(X, metadata, feature_map, valid)
    _require("RESP" in train and train.RESP.isin([0, 1]).all(), "TRAIN RESP must be binary and nonmissing.")
    rows, y = train._source_row.to_numpy(), train.RESP.to_numpy(dtype=np.int8)
    ids = train.PATIENT_ID.to_numpy()
    step_ok = (np.ones(len(rows), bool) if valid is None
               else valid[rows, 0] & valid[rows, 1])
    records = []
    for start in range(0, len(fmap), feature_batch_size):
        indices = np.arange(start, min(start + feature_batch_size, len(fmap)))
        block = X[np.ix_(rows, np.array([0, 1]), indices)]
        _counts(block)
        for offset, j in enumerate(indices):
            a, b = block[:, 0, offset], block[:, 1, offset]
            observed = np.isfinite(a) & np.isfinite(b) & step_ok
            yy, aa, bb, patients = y[observed], a[observed], b[observed], ids[observed]
            positive, negative = yy == 1, yy == 0
            weights = np.empty(0)
            if len(patients):
                _, inverse, multiplicity = np.unique(patients, return_inverse=True, return_counts=True)
                weights = 1.0 / multiplicity[inverse]
            estimable = bool(positive.any() and negative.any())
            ap0 = float(average_precision_score(yy, aa, sample_weight=weights)) if estimable else None
            ap1 = float(average_precision_score(yy, bb, sample_weight=weights)) if estimable else None
            def rate(values, subset):
                return float(np.average(values[subset] > 0, weights=weights[subset])) if subset.any() else None
            pos0, pos1 = rate(aa, positive), rate(bb, positive)
            row = fmap.iloc[j].to_dict()
            row.update(paired_observed_snapshots=int(observed.sum()),
                       paired_observed_patients=int(len(np.unique(patients))),
                       missing_pair_snapshots=int((~(np.isfinite(a) & np.isfinite(b)) & step_ok).sum()),
                       masked_pair_snapshots=int((~step_ok).sum()),
                       positive_snapshots=int(positive.sum()), negative_snapshots=int(negative.sum()),
                       step0_ap=ap0, step1_ap=ap1,
                       ap_difference=None if not estimable else ap0 - ap1,
                       step0_positive_nonzero_rate=pos0, step1_positive_nonzero_rate=pos1,
                       positive_nonzero_rate_difference=None if pos0 is None else pos0 - pos1,
                       step0_negative_nonzero_rate=rate(aa, negative),
                       step1_negative_nonzero_rate=rate(bb, negative),
                       status="CORROBORATED" if estimable and ap0 > ap1 and pos0 > pos1 else "INCONCLUSIVE",
                       claim="descriptive_step0_enrichment_only_not_leakage_proof")
            records.append(row)
    screen = pd.DataFrame(records).sort_values(
        ["ap_difference", "positive_nonzero_rate_difference", "FEATURE_INDEX"],
        ascending=[False, False, True], na_position="last", kind="stable").reset_index(drop=True)
    screen.insert(0, "association_rank", np.arange(1, len(screen) + 1))
    screen.attrs.update(scope="TRAIN_ONLY", physical_steps=[0, 1],
                        temporal_orientation="DECLARED_NEWEST_FIRST_NOT_VERIFIED",
                        missingness="paired_complete_cases_no_imputation",
                        ap_weighting="equal_total_weight_per_patient_among_feature_complete_cases",
                        historical_provenance_verified=False,
                        warning="Temporal association is neither a leakage test nor a feature-selection rule.")
    return screen


def audit_count_lineage(X, metadata, feature_map, valid=None, max_pairs=200):
    """Convenience API: JSON-compatible aggregate report plus full feature screen."""
    overlap = overlap_month_consistency(X, metadata, feature_map, valid, max_pairs)
    screen = step0_label_screen(X, metadata, feature_map, valid)
    # pandas converts missing scalar estimates to JSON null, never invented zero.
    overlap["step0_label_screen"] = {
        "scope": "TRAIN_ONLY_DESCRIPTIVE_ASSOCIATION",
        "feature_count": len(screen),
        "top20": json.loads(screen.head(20).to_json(orient="records")),
        "ap_weighting": screen.attrs["ap_weighting"],
        "claim": "Suspicious associations can reflect real timing, missingness or confounding; never leakage proof.",
    }
    return overlap, screen
