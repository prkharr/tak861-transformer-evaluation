"""TRAIN-only screening for the saved monthly claim-count representation.

Selection is a *proxy shortlist*, never evidence of the optimal Transformer
feature count. Original tensor indices are preserved separately from rank order.
No method here verifies upstream event cutoffs, counting lineage, or encoding.
"""
from dataclasses import dataclass
import hashlib
import inspect
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


DEFAULTS = dict(seed=42, n_splits=3, expected_features=1028, expected_steps=12,
                candidate_sizes=(16, 32, 64, 128, 256, 512), batch_rows=256,
                scaling="log1p", scale_floor=1.0, near_constant_fraction=0.995,
                low_variance_threshold=1e-6, proxy_C=0.1, proxy_max_iter=1500,
                proxy_tol=1e-4, association_bins=5)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _config(config):
    incoming = {} if config is None else dict(config)
    _require(not (set(incoming) - set(DEFAULTS)), "Unknown feature-selection configuration.")
    cfg = {**DEFAULTS, **incoming}
    for key in ("n_splits", "expected_features", "expected_steps", "batch_rows",
                "proxy_max_iter", "association_bins"):
        _require(type(cfg[key]) is int and cfg[key] > 0, key + " must be a positive integer.")
    _require(cfg["n_splits"] >= 3, "Use at least three patient-grouped folds.")
    _require(cfg["expected_steps"] % 4 == 0, "Four equally sized temporal blocks are required.")
    _require(cfg["scaling"] in {"log1p", "floored_standard"}, "Unknown count scaling.")
    _require(cfg["scale_floor"] >= 1.0, "Scale floor must be >=1 to avoid amplifying rare counts.")
    _require(0 < cfg["near_constant_fraction"] <= 1, "Invalid near-constant threshold.")
    _require(cfg["low_variance_threshold"] >= 0, "Invalid low-variance threshold.")
    _require(cfg["proxy_C"] > 0 and cfg["proxy_tol"] > 0, "Invalid logistic settings.")
    _require(all(type(k) is int and k > 0 for k in cfg["candidate_sizes"]), "Invalid candidate sizes.")
    cfg["candidate_sizes"] = tuple(sorted(set(cfg["candidate_sizes"])))
    return cfg


def _rows(rows, n):
    arr = np.asarray(rows)
    if arr.dtype == bool:
        _require(arr.shape == (n,), "Boolean row selector has wrong shape.")
        arr = np.flatnonzero(arr)
    _require(arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer), "Rows must be integer indices.")
    _require(len(arr) > 0 and np.all((arr >= 0) & (arr < n)), "Rows are empty or out of range.")
    _require(len(np.unique(arr)) == len(arr), "Duplicate fitting rows are not allowed.")
    return arr.astype(np.int64, copy=False)


def _mask(valid, shape):
    if valid is None:
        return None
    valid = np.asarray(valid)
    _require(valid.shape == shape[:2] and valid.dtype == bool, "Validity must be boolean (N,T).")
    return valid


def patient_snapshot_weights(patient_ids):
    """Equal total mass per patient, normalized to mean one per snapshot."""
    ids = np.asarray(patient_ids)
    _require(ids.ndim == 1 and len(ids) > 0 and not pd.isna(ids).any(), "Invalid patient identifiers.")
    _, inverse, counts = np.unique(ids.astype(str), return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def _feature_map(feature_map, width):
    frame = pd.DataFrame(feature_map).copy()
    required = {"FEATURE_INDEX", "FEATURE_NAME", "FEATURE_COLUMN"}
    _require(required <= set(frame.columns), "Feature map requires FEATURE_INDEX/NAME/COLUMN.")
    _require(len(frame) == width, "Feature map width differs from tensor width.")
    _require(np.array_equal(frame.FEATURE_INDEX.to_numpy(), np.arange(width)),
             "Feature map must already be in exact zero-based tensor order; do not silently sort.")
    for column in ("FEATURE_NAME", "FEATURE_COLUMN"):
        _require(frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
                 column + " contains invalid names.")
        _require(frame[column].is_unique, column + " is not unique.")
    _require(frame.FEATURE_COLUMN.tolist() == [f"F{i:04d}" for i in range(width)],
             "Feature aliases do not match the saved F0000... tensor contract.")
    forbidden = {"RESP", "PATIENT_ID", "END_DT", "TIME_STEP", "SPLIT"}
    _require(not forbidden.intersection(frame.FEATURE_NAME.str.upper()), "Identifier/target in features.")
    return frame.reset_index(drop=True)


def validate_count_inputs(X, metadata, feature_map, valid=None, expected_features=1028,
                          expected_steps=12, value_rows=None, batch_rows=256):
    """Validate identity globally, but values only on requested rows.

    The selection entrypoint passes TRAIN rows. Calling with no value_rows audits
    all input values and belongs in preparation or after the evaluation lock.
    NaN is preserved as missing; inf, negatives, fractions and nonzero padding fail.
    """
    _require(isinstance(X, np.ndarray) and X.ndim == 3, "X must be an ndarray/memmap (N,T,F).")
    _require(X.shape[1:] == (expected_steps, expected_features), "Unexpected count tensor shape.")
    _require(np.issubdtype(X.dtype, np.number) and not np.iscomplexobj(X), "Counts must be real numeric.")
    _require(batch_rows > 0, "batch_rows must be positive.")
    fm = _feature_map(feature_map, X.shape[2])
    meta = pd.DataFrame(metadata)
    _require(len(meta) == len(X), "Metadata rows do not match tensor rows.")
    _require({"PATIENT_ID", "END_DT", "RESP", "SPLIT"} <= set(meta), "Incomplete frozen metadata.")
    _require(not meta[["PATIENT_ID", "END_DT", "RESP", "SPLIT"]].isna().any().any(), "Null identity/label/split.")
    _require(meta.PATIENT_ID.map(lambda x: isinstance(x, str) and bool(x)).all(), "Patient IDs must be strings.")
    dates = pd.to_datetime(meta.END_DT, errors="raise")
    keys = pd.DataFrame({"PATIENT_ID": meta.PATIENT_ID.to_numpy(), "END_DT": dates.to_numpy()})
    _require(not keys.duplicated().any(), "Duplicate patient-snapshot keys.")
    _require(meta.RESP.isin([0, 1]).all(), "RESP must be binary.")
    _require(set(meta.SPLIT) <= {"train", "validation", "test"} and "train" in set(meta.SPLIT),
             "Expected frozen train/validation/test split labels.")
    _require(meta.groupby("PATIENT_ID").SPLIT.nunique().max() == 1, "A patient crosses frozen splits.")
    valid = _mask(valid, X.shape)
    rows = np.arange(len(X)) if value_rows is None else _rows(value_rows, len(X))
    missing = 0
    for start in range(0, len(rows), batch_rows):
        rr = rows[start:start + batch_rows]
        block = X[rr]
        vv = np.ones(block.shape[:2], bool) if valid is None else valid[rr]
        observed = block[vv]
        _require(not np.isinf(observed).any(), "Infinite count value.")
        finite = observed[np.isfinite(observed)]
        _require(np.all(finite >= 0), "Negative count value.")
        _require(np.all(finite == np.floor(finite)), "Fractional count: encoded values are not raw counts.")
        _require(np.all(np.isfinite(block[~vv])) and np.all(block[~vv] == 0),
                 "Explicit padded positions must be finite zero; zero alone does not define padding.")
        missing += int(np.isnan(observed).sum())
    digest = hashlib.sha256(fm.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    return dict(shape=list(X.shape), feature_map_sha256=digest, value_rows_audited=len(rows),
                missing_values=missing, mask_source="all_positions_retained" if valid is None else "supplied_boolean_mask",
                zero_is_padding=False, upstream_provenance="NOT_VERIFIED_BY_THIS_CHECK")


@dataclass
class CountPreprocessor:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    eligible_indices: np.ndarray
    quality: pd.DataFrame
    scaling: str
    scale_floor: float
    n_features: int
    n_steps: int
    fit_patients: int
    fit_snapshots: int
    batch_rows: int = 256

    def to_dict(self):
        return dict(schema=1, median=self.median.tolist(), mean=self.mean.tolist(), scale=self.scale.tolist(),
                    eligible_indices=self.eligible_indices.tolist(),
                    quality=self.quality.astype(object).where(pd.notna(self.quality), None).to_dict("records"),
                    scaling=self.scaling, scale_floor=float(self.scale_floor), n_features=self.n_features,
                    n_steps=self.n_steps, fit_patients=self.fit_patients, fit_snapshots=self.fit_snapshots,
                    batch_rows=self.batch_rows)

    @classmethod
    def from_dict(cls, state):
        state = dict(state)
        _require(state.pop("schema", None) == 1, "Unsupported preprocessing schema.")
        for key in ("median", "mean", "scale"):
            state[key] = np.asarray(state[key], dtype=np.float64)
        state["eligible_indices"] = np.asarray(state["eligible_indices"], dtype=np.int64)
        state["quality"] = pd.DataFrame(state["quality"])
        obj = cls(**state)
        _require(all(v.shape == (obj.n_features,) and np.isfinite(v).all()
                     for v in (obj.median, obj.mean, obj.scale)), "Invalid saved preprocessing vectors.")
        _require((obj.median >= 0).all() and (obj.scale > 0).all(), "Invalid saved median/scale.")
        _require(np.array_equal(obj.eligible_indices, np.unique(obj.eligible_indices)) and
                 np.all((obj.eligible_indices >= 0) & (obj.eligible_indices < obj.n_features)),
                 "Invalid saved eligible feature order.")
        return obj


def _weighted_quantile(x, weights, quantiles):
    order = np.argsort(x, kind="stable")
    values, weights = x[order], weights[order]
    cumulative = np.cumsum(weights)
    return np.interp(np.asarray(quantiles) * cumulative[-1], cumulative - 0.5 * weights, values)


def fit_count_preprocessor(X, train_rows, patient_ids, valid=None, config=None):
    """One feature at a time avoids a second full raw tensor in memory.

    Removes only all-missing, fully observed exact constants, and exact duplicate
    sequences. A constant observed value with variable missingness is retained.
    Near-constant, rare and low-variance features are diagnostics, not exclusions.
    """
    cfg = _config(config)
    _require(X.ndim == 3, "Expected three-dimensional count tensor.")
    rows = _rows(train_rows, len(X))
    ids = np.asarray(patient_ids)
    _require(ids.shape == (len(X),), "Patient ID shape differs from tensor rows.")
    valid = _mask(valid, X.shape)
    vv = np.ones((len(rows), X.shape[1]), bool) if valid is None else valid[rows]
    sw = patient_snapshot_weights(ids[rows])
    weights = sw[:, None] / np.maximum(vv.sum(axis=1, keepdims=True), 1)
    weights = np.broadcast_to(weights, vv.shape)[vv]
    _require(weights.sum() > 0, "No valid TRAIN timesteps.")
    _, patient_codes = np.unique(ids[rows].astype(str), return_inverse=True)
    f = X.shape[2]
    medians, means, scales = np.zeros(f), np.zeros(f), np.ones(f)
    ledger, eligible, hashes = [], [], {}
    for j in range(f):
        raw = np.asarray(X[rows, :, j], dtype=np.float64)
        values = raw[vv]
        _require(not np.isinf(values).any(), "Infinite fitting count.")
        present = np.isfinite(values)
        obs = values[present]
        _require(np.all(obs >= 0) and np.all(obs == np.floor(obs)), "Fitting values must be nonnegative integer counts.")
        nonzero_rows = np.any(vv & np.isfinite(raw) & (raw > 0), axis=1)
        observed_rows = np.any(vv & np.isfinite(raw), axis=1)
        missing_fraction = float(np.average(~present, weights=weights))
        observed_variance = 0.0
        mode_fraction = 1.0
        raw_quantiles = [None, None, None]
        raw_max = raw_mean = raw_sd = None
        if len(obs):
            ow = weights[present]
            raw_quantiles = _weighted_quantile(obs, ow, [0.5, 0.95, 0.99]).tolist()
            medians[j] = raw_quantiles[0]
            raw_max, raw_mean = float(obs.max()), float(np.average(obs, weights=ow))
            raw_sd = float(np.sqrt(np.average((obs - raw_mean) ** 2, weights=ow)))
            unique, codes = np.unique(obs, return_inverse=True)
            mode_fraction = float(np.bincount(codes, weights=ow).max() / ow.sum())
            logged = np.log1p(obs)
            observed_variance = float(np.average((logged - np.average(logged, weights=ow)) ** 2, weights=ow))
        complete = np.where(present, values, medians[j])
        logged = np.log1p(complete)
        if cfg["scaling"] == "floored_standard":
            means[j] = np.average(logged, weights=weights)
            scales[j] = max(cfg["scale_floor"], np.sqrt(np.average((logged - means[j]) ** 2, weights=weights)))
        canonical = np.where(np.isnan(values), -1.0, values).astype("<f8", copy=False)
        fingerprint = hashlib.sha256(canonical.tobytes()).hexdigest()
        duplicate = None
        for previous in hashes.get(fingerprint, []):
            if np.array_equal(X[rows, :, previous][vv], values, equal_nan=True):
                duplicate = previous
                break
        if len(obs) == 0:
            reason = "all_missing_in_fit"
        elif np.ptp(obs) == 0 and present.all():
            reason = "constant_in_fit"
        elif duplicate is not None:
            reason = "exact_duplicate_in_fit"
        else:
            reason = "eligible"
            eligible.append(j)
            hashes.setdefault(fingerprint, []).append(j)
        ledger.append(dict(FEATURE_INDEX=j, ELIGIBLE=reason == "eligible", QUALITY_REASON=reason,
                           DUPLICATE_OF=-1 if duplicate is None else duplicate,
                           MISSING_FRACTION=missing_fraction, IMPUTED_FIT_VALUES=int((~present).sum()),
                           OBSERVED_PATIENTS=int(len(np.unique(patient_codes[observed_rows]))),
                           NONZERO_PATIENTS=int(len(np.unique(patient_codes[nonzero_rows]))),
                           OBSERVED_COUNT_CELLS=int(len(obs)), RAW_Q50=raw_quantiles[0],
                           RAW_Q95=raw_quantiles[1], RAW_Q99=raw_quantiles[2],
                           RAW_MAX=raw_max, RAW_MEAN=raw_mean, RAW_SD=raw_sd,
                           OBSERVED_MODE_FRACTION=mode_fraction,
                           NEAR_CONSTANT=bool(mode_fraction >= cfg["near_constant_fraction"]),
                           LOG_COUNT_VARIANCE=observed_variance,
                           LOW_VARIANCE=bool(observed_variance <= cfg["low_variance_threshold"]),
                           MISSINGNESS_VARIES=bool(present.any() and (~present).any())))
    _require(len(eligible) > 0, "No eligible feature remains in fitting partition.")
    return CountPreprocessor(medians, means, scales, np.asarray(eligible, dtype=np.int64),
                             pd.DataFrame(ledger), cfg["scaling"], cfg["scale_floor"], f, X.shape[1],
                             len(np.unique(ids[rows])), len(rows), cfg["batch_rows"])


def _indices(state, feature_indices):
    indices = state.eligible_indices if feature_indices is None else np.asarray(feature_indices)
    _require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer), "Invalid feature indices.")
    _require(len(indices) > 0 and len(np.unique(indices)) == len(indices) and
             np.isin(indices, state.eligible_indices).all(), "Selection contains ineligible/duplicate features.")
    return indices.astype(np.int64, copy=False)


def transform_counts(X, state, valid=None, rows=None, feature_indices=None, out=None):
    """Apply the frozen TRAIN state, leaving inputs untouched and padding zero."""
    _require(X.shape[1:] == (state.n_steps, state.n_features), "Tensor/preprocessor shape mismatch.")
    valid = _mask(valid, X.shape)
    rows = np.arange(len(X)) if rows is None else _rows(rows, len(X))
    columns = _indices(state, feature_indices)
    shape = (len(rows), state.n_steps, len(columns))
    if out is None:
        out = np.empty(shape, dtype=np.float32)
    _require(out.shape == shape and out.dtype == np.float32, "Output must be float32 with selected tensor shape.")
    for start in range(0, len(rows), state.batch_rows):
        rr = rows[start:start + state.batch_rows]
        block = np.array(X[np.ix_(rr, np.arange(state.n_steps), columns)], dtype=np.float64)
        vv = np.ones(block.shape[:2], bool) if valid is None else valid[rr]
        observed = block[vv]
        finite = observed[np.isfinite(observed)]
        _require(not np.isinf(observed).any() and np.all(finite >= 0) and np.all(finite == np.floor(finite)),
                 "Invalid raw counts during transform.")
        block = np.where(np.isnan(block), state.median[columns][None, None, :], block)
        block = (np.log1p(block) - state.mean[columns]) / state.scale[columns]
        block[~vv] = 0.0
        out[start:start + len(rr)] = block.astype(np.float32)
    return out


def temporal_count_summaries(X, state, rows, valid=None, feature_indices=None):
    """log1p sums in four blocks and newest 1/3/6/12 positions (for T=12).

    Sums use raw count medians only where counts are missing. No count is copied
    across a timestep; explicit padded positions contribute zero. The screening
    summaries intentionally use log1p sums regardless of tensor scaling ablation.
    Output (rows, summaries, features) has feature identity on the last axis.
    """
    rows = _rows(rows, len(X))
    valid = _mask(valid, X.shape)
    columns = _indices(state, feature_indices)
    t = state.n_steps
    _require(t % 4 == 0, "Four equal temporal blocks required.")
    q = t // 4
    windows = [(i * q, (i + 1) * q) for i in range(4)] + [(0, n) for n in (1, q, 2 * q, t)]
    names = [f"block_{i}" for i in range(4)] + [f"newest_{n}" for n in (1, q, 2 * q, t)]
    output = np.empty((len(rows), len(windows), len(columns)), dtype=np.float32)
    for start in range(0, len(rows), state.batch_rows):
        rr = rows[start:start + state.batch_rows]
        raw = np.asarray(X[np.ix_(rr, np.arange(t), columns)], dtype=np.float64)
        raw = np.where(np.isnan(raw), state.median[columns][None, None, :], raw)
        if valid is not None:
            raw[~valid[rr]] = 0.0
        for i, (begin, end) in enumerate(windows):
            output[start:start + len(rr), i] = np.log1p(raw[:, begin:end].sum(axis=1))
    _require(np.isfinite(output).all(), "Nonfinite temporal summaries.")
    return output, names


def _weighted_corr(a, b, w):
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    denominator = np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))
    return 0.0 if denominator <= 0 else float(np.sum(w * a * b) / denominator)


def _weighted_midranks(x, w):
    unique, codes = np.unique(x, return_inverse=True)
    masses = np.bincount(codes, weights=w, minlength=len(unique))
    ranks = (np.cumsum(masses) - 0.5 * masses) / masses.sum()
    return ranks[codes]


def _binned_mi(x, y, w, bins, n_patients):
    edges = np.unique(_weighted_quantile(x, w, np.linspace(0, 1, bins + 1)[1:-1]))
    codes = np.searchsorted(edges, x, side="right")
    joint = np.bincount(codes * 2 + y, weights=w, minlength=(len(edges) + 1) * 2).reshape(-1, 2)
    joint /= joint.sum()
    expected = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    present = joint > 0
    mi = float(np.sum(joint[present] * np.log(joint[present] / expected[present])))
    occupied = int((joint.sum(axis=1) > 0).sum())
    # Conservative patient-count null-bias correction; descriptive, no p-value claim.
    corrected = max(0.0, mi - (occupied - 1) / (2.0 * n_patients))
    return mi, corrected


def rank_count_features(summaries, y, patient_ids, original_indices, summary_names, bins=5):
    """Type-aware count ranking: weighted rank association plus binned nonlinear MI.

    Counts have ties and zero inflation. Weighted midranks handle ties; bounded
    quantile bins avoid treating each count as a separate discrete category.
    Scores are descriptive, not significance/causality or reliable interaction tests.
    Recent-minus-old block contrast adds a coarse temporal-change diagnostic.
    """
    y = np.asarray(y, dtype=np.int64)
    weights = patient_snapshot_weights(patient_ids)
    _require(set(np.unique(y)) == {0, 1}, "Ranking requires both RESP classes.")
    yrank = _weighted_midranks(y, weights)
    records = []
    n_patients = len(np.unique(patient_ids))
    for j, original in enumerate(original_indices):
        x = summaries[:, :, j]
        vectors = [x[:, s] for s in range(x.shape[1])] + [x[:, 0] - x[:, 3]]
        correlations, mis, corrected = [], [], []
        for vector in vectors:
            correlations.append(_weighted_corr(_weighted_midranks(vector, weights), yrank, weights))
            mi, adjustment = _binned_mi(vector, y, weights, bins, n_patients)
            mis.append(mi)
            corrected.append(adjustment)
        association = np.maximum(np.abs(correlations), np.sqrt(2 * np.asarray(corrected)))
        best = int(np.argmax(association))
        records.append(dict(FEATURE_INDEX=int(original), RANKING_SCORE=float(association[best]),
                            MAX_ABS_SPEARMAN=float(np.max(np.abs(correlations))),
                            MAX_BINNED_MI=float(np.max(mis)), MAX_CORRECTED_MI=float(np.max(corrected)),
                            TEMPORAL_CONTRAST_SCORE=float(association[-1]),
                            STRONGEST_SUMMARY=(list(summary_names) + ["recent_minus_oldest_block"])[best],
                            ASSOCIATION_DIRECTION=float(correlations[best])))
    result = pd.DataFrame(records).sort_values(["RANKING_SCORE", "FEATURE_INDEX"], ascending=[False, True])
    result["RANK"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def _proxy(cfg):
    parameter = inspect.signature(LogisticRegression).parameters.get("penalty")
    kwargs = {"l1_ratio": 0.0} if parameter is None or parameter.default == "deprecated" else {"penalty": "l2"}
    return LogisticRegression(**kwargs, C=cfg["proxy_C"], solver="lbfgs", max_iter=cfg["proxy_max_iter"],
                              tol=cfg["proxy_tol"], random_state=cfg["seed"])


def _patient_folds(labels, ids, cfg):
    patients, inverse = np.unique(ids.astype(str), return_inverse=True)
    patient_labels = np.zeros(len(patients), dtype=int)
    np.maximum.at(patient_labels, inverse, labels)
    _require(np.bincount(patient_labels, minlength=2).min() >= cfg["n_splits"],
             "Too few positive/negative TRAIN patients for grouped folds.")
    splitter = StratifiedKFold(cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    for fit_groups, score_groups in splitter.split(patients, patient_labels):
        yield np.flatnonzero(np.isin(inverse, fit_groups)), np.flatnonzero(np.isin(inverse, score_groups))


def one_se_candidates(cv_summary):
    """Choose proxy best/one-SE/one-smaller; all-eligible stays a control."""
    table = cv_summary.copy()
    _require(len(table) > 0 and np.isfinite(table[["MEAN_AP", "SE_AP"]]).all().all(), "Invalid CV summary.")
    ordered = table.sort_values(["MEAN_AP", "FINAL_FEATURE_COUNT", "CANDIDATE"], ascending=[False, True, True])
    best = ordered.iloc[0]
    eligible = table.loc[table.MEAN_AP >= best.MEAN_AP - best.SE_AP]
    one = eligible.sort_values(["FINAL_FEATURE_COUNT", "MEAN_AP"], ascending=[True, False]).iloc[0]
    smaller = table.loc[table.FINAL_FEATURE_COUNT < one.FINAL_FEATURE_COUNT].sort_values("FINAL_FEATURE_COUNT")
    choices = {"proxy_best": str(best.CANDIDATE), "proxy_one_se": str(one.CANDIDATE)}
    if len(smaller):
        choices["one_smaller"] = str(smaller.iloc[-1].CANDIDATE)
    choices["all_eligible"] = "all_eligible"
    return choices, float(best.MEAN_AP - best.SE_AP)


@dataclass
class FeatureSelectionResult:
    selected_indices: np.ndarray
    eligible_indices: np.ndarray
    candidate_indices: dict
    candidate_tags: dict
    ranking: pd.DataFrame
    quality: pd.DataFrame
    cv_scores: pd.DataFrame
    cv_summary: pd.DataFrame
    fold_rankings: pd.DataFrame
    fold_audit: pd.DataFrame
    preprocessor: CountPreprocessor
    chosen_candidate: str
    input_audit: dict
    config: dict


def select_ranked_features(X, metadata, feature_map, valid=None, config=None):
    """No outer VALIDATION/TEST values or labels enter ranking or proxy scores.

    Metadata identity/split checks are global; actual value validation and every
    fitted statistic are TRAIN-only. The proxy is weighted regularized logistic
    regression, intentionally a transparent floor, not a nonlinear model claim.
    """
    cfg = _config(config)
    meta = pd.DataFrame(metadata).reset_index(drop=True)
    train = np.flatnonzero(meta.SPLIT.to_numpy() == "train")
    audit = validate_count_inputs(X, meta, feature_map, valid, cfg["expected_features"],
                                  cfg["expected_steps"], value_rows=train, batch_rows=cfg["batch_rows"])
    fm = _feature_map(feature_map, X.shape[2])
    ids = meta.PATIENT_ID.to_numpy()
    labels = meta.RESP.to_numpy(dtype=int)
    candidate_requests = {f"top_{k}": k for k in cfg["candidate_sizes"] if k < X.shape[2]}
    candidate_requests["all_eligible"] = X.shape[2]
    scores, ranks, audits = [], [], []
    for fold, (fit_local, score_local) in enumerate(_patient_folds(labels[train], ids[train], cfg)):
        fit_rows, score_rows = train[fit_local], train[score_local]
        state = fit_count_preprocessor(X, fit_rows, ids, valid, cfg)
        fit_values, names = temporal_count_summaries(X, state, fit_rows, valid)
        score_values, _ = temporal_count_summaries(X, state, score_rows, valid)
        ranking = rank_count_features(fit_values, labels[fit_rows], ids[fit_rows], state.eligible_indices,
                                      names, cfg["association_bins"])
        ranked = ranking.FEATURE_INDEX.to_numpy()
        lookup = {int(original): j for j, original in enumerate(state.eligible_indices)}
        ranking["FOLD"] = fold
        ranks.append(ranking)
        fit_weights, score_weights = patient_snapshot_weights(ids[fit_rows]), patient_snapshot_weights(ids[score_rows])
        cached = {}
        for tag, requested in candidate_requests.items():
            selected = np.sort(ranked[:min(requested, len(ranked))])
            key = tuple(selected)
            if key not in cached:
                positions = [lookup[int(i)] for i in selected]
                a = fit_values[:, :, positions].reshape(len(fit_rows), -1)
                b = score_values[:, :, positions].reshape(len(score_rows), -1)
                model = _proxy(cfg)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(a, labels[fit_rows], sample_weight=fit_weights)
                _require(not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                         "Proxy did not converge; do not select using an unfinished fit. Increase declared proxy budget.")
                cached[key] = (average_precision_score(labels[score_rows], model.predict_proba(b)[:, 1], sample_weight=score_weights),
                               average_precision_score(labels[fit_rows], model.predict_proba(a)[:, 1], sample_weight=fit_weights))
            ap, fit_ap = cached[key]
            scores.append(dict(FOLD=fold, CANDIDATE=tag, FEATURE_COUNT=len(selected), AP=float(ap),
                               FIT_AP=float(fit_ap), GAP=float(fit_ap - ap), METRIC="patient_weighted_proxy_AP"))
        audits.append(dict(FOLD=fold, FIT_PATIENTS=len(np.unique(ids[fit_rows])), SCORE_PATIENTS=len(np.unique(ids[score_rows])),
                           PATIENT_OVERLAP=int(len(set(ids[fit_rows]) & set(ids[score_rows]))),
                           FIT_SNAPSHOTS=len(fit_rows), SCORE_SNAPSHOTS=len(score_rows),
                           FIT_POSITIVES=int(labels[fit_rows].sum()), SCORE_POSITIVES=int(labels[score_rows].sum()),
                           ELIGIBLE_FEATURES=len(state.eligible_indices),
                           FIT_ROWS_SHA256=hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest(),
                           SCORE_ROWS_SHA256=hashlib.sha256(score_rows.astype("<i8").tobytes()).hexdigest()))
        del fit_values, score_values
    state = fit_count_preprocessor(X, train, ids, valid, cfg)
    summaries, names = temporal_count_summaries(X, state, train, valid)
    ranking = rank_count_features(summaries, labels[train], ids[train], state.eligible_indices, names, cfg["association_bins"])
    del summaries
    fold_rankings = pd.concat(ranks, ignore_index=True)
    stability = fold_rankings.groupby("FEATURE_INDEX").agg(FOLD_ELIGIBILITY_COUNT=("RANK", "count"),
                                                          MEAN_FOLD_RANK=("RANK", "mean"), SD_FOLD_RANK=("RANK", "std"),
                                                          MEAN_FOLD_SCORE=("RANKING_SCORE", "mean"))
    ranking = ranking.merge(stability, on="FEATURE_INDEX", how="left").merge(fm, on="FEATURE_INDEX", how="left")
    ranking["FOLD_ELIGIBILITY_FRACTION"] = ranking.FOLD_ELIGIBILITY_COUNT.fillna(0) / cfg["n_splits"]
    quality = fm.merge(state.quality, on="FEATURE_INDEX", validate="one_to_one")
    cv_scores = pd.DataFrame(scores)
    summary = cv_scores.groupby("CANDIDATE", sort=False).agg(MEAN_AP=("AP", "mean"), SD_AP=("AP", "std"),
                                                           N_FOLDS=("AP", "size"), MEAN_FIT_AP=("FIT_AP", "mean"),
                                                           MEAN_GAP=("GAP", "mean")).reset_index()
    summary["SE_AP"] = summary.SD_AP / np.sqrt(summary.N_FOLDS)
    ranked = ranking.FEATURE_INDEX.to_numpy(dtype=int)
    full_candidates = {tag: np.sort(ranked[:min(k, len(ranked))]) for tag, k in candidate_requests.items()}
    summary["FINAL_FEATURE_COUNT"] = summary.CANDIDATE.map(lambda tag: len(full_candidates[tag]))
    choices, boundary = one_se_candidates(summary)
    summary["ONE_SE_BOUNDARY"] = boundary
    summary["WITHIN_ONE_SE"] = summary.MEAN_AP >= boundary
    shortlist, tags, seen = {}, {}, {}
    # Preserve the literal all_eligible control name even when another tag selects the same set.
    for tag in ("all_eligible", "proxy_best", "proxy_one_se", "one_smaller"):
        if tag not in choices:
            continue
        values = full_candidates[choices[tag]]
        key = tuple(values)
        canonical = seen.get(key, tag)
        tags[tag] = canonical
        if key not in seen:
            seen[key] = canonical
            shortlist[canonical] = values
    selected = full_candidates[choices["proxy_one_se"]]
    quality["PROXY_ONE_SE_SELECTED"] = quality.FEATURE_INDEX.isin(selected)
    quality["SELECTION_REASON"] = np.where(~quality.ELIGIBLE, quality.QUALITY_REASON,
                                           np.where(quality.PROXY_ONE_SE_SELECTED, "proxy_one_se_shortlist", "eligible_control_only_or_other_candidate"))
    audit.update(selection_scope="TRAIN_ONLY", proxy="L2_logistic_on_log1p_temporal_sums",
                 proxy_weighting="inverse_patient_snapshot_count", proxy_is_transformer_optimum=False,
                 summary_names=names, outer_test_diagnostics_computed=False)
    return FeatureSelectionResult(selected, state.eligible_indices, shortlist, tags, ranking, quality, cv_scores,
                                  summary, fold_rankings, pd.DataFrame(audits), state, choices["proxy_one_se"], audit, cfg)


def feature_diagnostics(X, metadata, feature_map, state, valid=None, threshold=0.90, max_features=128):
    """TRAIN latest-patient correlations; no O(F) VIF regressions or target filtering.

    Correlation uses annual counts, so complementary temporal shape can remain.
    Effective rank/condition are bounded to the first max_features original-order
    eligible columns and are explicitly a diagnostic submatrix, not global VIF.
    """
    _require(0 < threshold <= 1 and max_features > 1, "Invalid redundancy settings.")
    meta = pd.DataFrame(metadata).reset_index(drop=True)
    train = meta.loc[meta.SPLIT.eq("train")].copy()
    train["_date"] = pd.to_datetime(train.END_DT)
    rows = train.sort_values(["PATIENT_ID", "_date"]).groupby("PATIENT_ID", sort=False).tail(1).index.to_numpy()
    summary, _ = temporal_count_summaries(X, state, rows, valid)
    values = summary[:, -1, :]
    frame = pd.DataFrame(values).rank(method="average").to_numpy(dtype=float)
    centered = frame - frame.mean(axis=0)
    norms = np.sqrt((centered ** 2).sum(axis=0))
    safe = norms > 0
    centered[:, safe] /= norms[safe]
    centered[:, ~safe] = 0
    corr = centered.T @ centered
    a, b = np.where(np.triu(np.abs(corr) >= threshold, k=1))
    fm = _feature_map(feature_map, state.n_features).set_index("FEATURE_INDEX")
    pairs = pd.DataFrame(dict(FEATURE_INDEX_A=state.eligible_indices[a], FEATURE_INDEX_B=state.eligible_indices[b],
                              SPEARMAN=corr[a, b]))
    for side in ("A", "B"):
        pairs["FEATURE_NAME_" + side] = pairs["FEATURE_INDEX_" + side].map(fm.FEATURE_NAME)
    sub = centered[:, :min(max_features, centered.shape[1])]
    eigenvalues = np.maximum(np.linalg.eigvalsh(sub.T @ sub), 0)
    positive = eigenvalues[eigenvalues > 1e-10]
    entropy_rank = 0.0
    if len(positive):
        probabilities = positive / positive.sum()
        entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return dict(correlation_pairs=pairs.sort_values("SPEARMAN", key=lambda x: x.abs(), ascending=False),
                summary=dict(scope="TRAIN_latest_snapshot_per_patient", patients=len(rows),
                             correlated_pairs=len(pairs), spectral_columns=sub.shape[1],
                             spectral_original_indices=state.eligible_indices[:sub.shape[1]].tolist(),
                             effective_rank=entropy_rank, singular=bool(len(positive) < sub.shape[1]),
                             nonzero_eigenvalue_condition=float(positive.max() / positive.min()) if len(positive) else None,
                             global_vif_computed=False, redundancy_auto_removal=False))


def shift_summary(X, metadata, feature_map, state, valid=None, comparison="validation"):
    """Descriptive TRAIN vs VALIDATION only, never a feature-selection input."""
    _require(comparison == "validation", "TEST shift diagnostics are forbidden before the evaluation lock.")
    meta = pd.DataFrame(metadata).reset_index(drop=True)
    fm = _feature_map(feature_map, state.n_features)
    values = {}
    for split in ("train", comparison):
        rows = np.flatnonzero(meta.SPLIT.to_numpy() == split)
        _require(len(rows) > 0, "Missing shift population: " + split)
        summaries, _ = temporal_count_summaries(X, state, rows, valid)
        annual = summaries[:, -1, :]
        weights = patient_snapshot_weights(meta.PATIENT_ID.to_numpy()[rows])
        mean = np.average(annual, axis=0, weights=weights)
        variance = np.average((annual - mean) ** 2, axis=0, weights=weights)
        values[split] = (mean, variance, np.average(annual > 0, axis=0, weights=weights))
    train_mean, train_var, train_presence = values["train"]
    val_mean, val_var, val_presence = values[comparison]
    denominator = np.sqrt((train_var + val_var) / 2)
    standardized = np.divide(val_mean - train_mean, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    result = fm.iloc[state.eligible_indices].copy()
    result["TRAIN_LOG_ANNUAL_MEAN"] = train_mean
    result["VALIDATION_LOG_ANNUAL_MEAN"] = val_mean
    result["STANDARDIZED_MEAN_DIFFERENCE"] = standardized
    result["TRAIN_NONZERO_FRACTION"] = train_presence
    result["VALIDATION_NONZERO_FRACTION"] = val_presence
    result["USED_FOR_SELECTION"] = False
    return result.reset_index(drop=True)
