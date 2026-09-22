"""Validation and preprocessing for the V63 selected snapshot-feature experiment."""
import io
import json
import hashlib
import re
import numpy as np
import pandas as pd


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def parse_features(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError("FEATURES must be a nonempty array of exact column names.")
    if len(set(x.upper() for x in value)) != len(value):
        raise ValueError("Duplicate feature names in MODEL_TYPE.FEATURES.")
    prohibited = {"PATIENT_ID", "START_DT", "END_DT", "RESP", "SPLIT", "RND", "SCORE", "DECILE", "CENTILE", "MILLILE"}
    if prohibited.intersection(x.upper() for x in value):
        raise ValueError("The selected list contains an identifier, target, split, random helper or prediction output.")
    return value


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def normalize_metadata(frame):
    out = frame[["PATIENT_ID", "END_DT", "RESP"]].copy()
    if out.empty or out.isna().any().any():
        raise ValueError("Missing snapshot keys or labels.")
    if not out.PATIENT_ID.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Patient IDs must remain nonempty strings.")
    dates = pd.to_datetime(out.END_DT, errors="raise")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Snapshot cutoffs must be exact dates.")
    out["END_DT"] = dates.dt.strftime("%Y-%m-%d")
    if not out.RESP.isin([0, 1]).all():
        raise ValueError("Nonbinary labels.")
    out["RESP"] = out.RESP.astype("int64")
    if out.duplicated(["PATIENT_ID", "END_DT"]).any():
        raise ValueError("Duplicate patient/date keys; no automatic deduplication is permitted.")
    return out


def align_features(snapshots, model_data, features):
    features = parse_features(features)
    expected = normalize_metadata(snapshots).sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    actual = normalize_metadata(model_data)
    missing = set(features).difference(model_data.columns)
    if missing:
        raise ValueError("Selected columns absent from MODEL_DATA: " + repr(sorted(missing)))
    source = actual.copy()
    for name in features:
        # Decimal fractions are converted to float, never through an integer cast.
        source[name] = pd.to_numeric(model_data[name], errors="raise").to_numpy(dtype=np.float64)
    aligned = expected.merge(source, on=["PATIENT_ID", "END_DT"], how="left",
                             validate="one_to_one", suffixes=("", "_SOURCE"), indicator=True)
    if not aligned._merge.eq("both").all() or not aligned.RESP.eq(aligned.RESP_SOURCE).all():
        raise ValueError("Missing source keys or conflicting labels in MODEL_DATA.")
    X = aligned[features].to_numpy(dtype=np.float64)
    if np.isinf(X).any():
        raise ValueError("Infinite source feature values.")
    return expected, X


def raw_hash(X, metadata, features):
    h = hashlib.sha256()
    h.update(canonical_json(features).encode())
    h.update(metadata.to_csv(index=False).encode())
    h.update(canonical_json(list(X.shape)).encode())
    # A separate missing mask avoids platform-dependent NaN payload hashes.
    h.update(np.isnan(X).astype("u1").tobytes())
    h.update(np.nan_to_num(X, nan=0).astype("<f8").tobytes())
    return h.hexdigest()


def npz_bytes(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def json_bytes(value):
    return canonical_json(value).encode()


def bind_split(metadata, frozen, reference):
    original = normalize_metadata(metadata).sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    normalized = normalize_metadata(frozen)
    normalized["SPLIT"] = frozen.SPLIT.to_numpy()
    normalized["SPLIT_CONFIG"] = frozen.SPLIT_CONFIG.to_numpy()
    normalized = normalized.sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    if not original.equals(normalized[["PATIENT_ID", "END_DT", "RESP"]]):
        raise ValueError("Saved split differs from the prepared snapshots/labels.")
    if normalized[["SPLIT", "SPLIT_CONFIG"]].isna().any().any():
        raise ValueError("Incomplete frozen split.")
    if set(normalized.SPLIT) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split names.")
    if normalized.groupby("PATIENT_ID").SPLIT.nunique().gt(1).any():
        raise ValueError("Patient leakage between splits.")
    if normalized.SPLIT_CONFIG.nunique() != 1:
        raise ValueError("Inconsistent split configuration.")
    records = [[r.PATIENT_ID, r.END_DT, int(r.RESP), r.SPLIT] for r in normalized.itertuples()]
    hashes = {"snapshot_manifest_sha256": digest_json(records),
              "split_config_sha256": digest_json(json.loads(normalized.SPLIT_CONFIG.iloc[0]))}
    if reference.get("run_id") != "RUN_001" or reference.get("training_complete") is not True:
        raise ValueError("Expected completed original RUN_001 reference.")
    if any(reference.get("input_hashes", {}).get(k) != v for k, v in hashes.items()):
        raise ValueError("Patient assignments differ from original RUN_001 fingerprints.")
    for _, part in normalized.groupby("SPLIT"):
        if set(part.RESP) != {0, 1}:
            raise ValueError("Each split needs both outcome classes.")
    return normalized, hashes


def fit_preprocessor(X_train):
    if X_train.ndim != 2 or not len(X_train) or np.isinf(X_train).any():
        raise ValueError("Invalid training feature matrix.")
    all_missing = np.isnan(X_train).all(axis=0)
    median = np.array([0.0 if missing else np.nanmedian(X_train[:, i])
                       for i, missing in enumerate(all_missing)])
    filled = np.where(np.isnan(X_train), median, X_train)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale[scale == 0] = 1.0
    if not np.isfinite(np.r_[median, mean, scale]).all():
        raise ValueError("Nonfinite preprocessing statistics.")
    return {"median": median.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "all_missing_train": all_missing.tolist()}


def transform_features(X, state):
    if X.ndim != 2 or X.shape[1] != len(state["median"]) or np.isinf(X).any():
        raise ValueError("Feature shape or values changed.")
    mask = np.isnan(X)
    values = ((np.where(mask, state["median"], X) - state["mean"]) / state["scale"]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite standardized values.")
    return values, mask.astype(np.float32)


def rank_tables(metadata, scores):
    labels = metadata.RESP.to_numpy(dtype=int)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid evaluation probabilities.")
    ranked = metadata[["PATIENT_ID", "END_DT", "RESP"]].copy()
    ranked["SCORE"] = scores
    ranked = ranked.sort_values(["SCORE", "PATIENT_ID", "END_DT"], ascending=[False, True, True]).reset_index(drop=True)
    n, positives = len(ranked), int(labels.sum())
    base = positives / n
    ranked["DECILE"] = 10 - np.minimum(9, np.arange(n) * 10 // n)
    deciles = []
    cumulative_n = cumulative_positive = 0
    for decile in range(10, 0, -1):
        part = ranked[ranked.DECILE == decile]
        if part.empty:
            continue
        count, positive = len(part), int(part.RESP.sum())
        cumulative_n += count
        cumulative_positive += positive
        rate = positive / count
        deciles.append({"decile": decile, "snapshots": count, "positives": positive,
                        "score_min": float(part.SCORE.min()), "score_max": float(part.SCORE.max()),
                        "response_rate": rate, "lift": rate / base if base else None,
                        "cumulative_snapshots": cumulative_n, "cumulative_positives": cumulative_positive,
                        "cumulative_lift": (cumulative_positive / cumulative_n) / base if base else None,
                        "cumulative_recall": cumulative_positive / positives if positives else None})
    top = []
    for fraction in (.05, .10, .20, .30):
        k = max(1, int(np.ceil(n * fraction)))
        tp = int(ranked.RESP.iloc[:k].sum())
        top.append({"fraction": fraction, "selected": k, "positives": tp, "precision": tp / k,
                    "recall": tp / positives if positives else None, "lift": (tp / k) / base if base else None})
    return pd.DataFrame(deciles), pd.DataFrame(top)
