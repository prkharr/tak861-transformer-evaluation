"""Embedded selection helpers; no repository dependency in delivered notebooks.

The supervised selector uses TRAIN only. Its importance is a screening heuristic,
not Transformer attribution: four three-month summaries preserve broad recency
but ignore order inside each window, can divide importance among correlated
features, and can favor features with more distinct count values.
Snapshots remain the fitting unit; repeated snapshots from one patient are not
independent observations. The prevalence filter counts distinct TRAIN patients.
The float32 screening matrix adds up to about 267 MB for 16,256 TRAIN snapshots
and all 1,028 source features; sklearn adds working memory. Bounded, two-pass
TRAIN reads avoid copying the full tensor or keeping two screening matrices.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def _fs_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _fs_positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive Python integer.")
    return value


def _fs_seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("seed must be a Python integer from 0 through 2**32 - 1.")
    return value


def _fs_source_features(data):
    features = data["features"]
    if (not isinstance(features, list) or not features
            or not all(isinstance(name, str) and name for name in features)
            or len(set(features)) != len(features)):
        raise ValueError("Source feature names must be unique, nonempty strings in a list.")
    shape = data["X"].shape
    if len(shape) != 3 or shape[1] != 12 or shape[2] != len(features):
        raise ValueError("Expected source tensor [snapshots, 12, source features].")
    source_hash = _fs_digest(features)
    recorded_hash = data.get("hashes", {}).get("feature_names_sha256")
    if recorded_hash is not None and recorded_hash != source_hash:
        raise ValueError("Source feature names differ from the validated input fingerprint.")
    return features, source_hash


def _fs_training_rows(data):
    indices = np.asarray(data["indices"]["train"])
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != len(indices)
            or indices.min() < 0 or indices.max() >= data["X"].shape[0]):
        raise ValueError("TRAIN indices must be unique, valid integer row positions.")
    metadata = data["metadata"].iloc[indices]
    if not metadata["SPLIT"].eq("train").all():
        raise ValueError("A requested selector fitting row is not in TRAIN.")
    patients = metadata["PATIENT_ID"]
    if patients.isna().any() or not patients.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("TRAIN patient identifiers must be nonempty strings.")
    patient_codes, patient_names = pd.factorize(patients, sort=True)
    return indices, patient_codes, len(patient_names)


def _fs_ranker_settings(seed):
    return {"n_estimators": 128, "max_depth": 8, "min_samples_leaf": 10,
            "max_features": "sqrt", "class_weight": "balanced",
            "n_jobs": 2, "random_state": seed}


def _fs_temporal_settings():
    return {"aggregation": "log1p_of_four_consecutive_3_month_sums",
        "temporal_windows": [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        "time_step_direction": "0_newest_11_oldest",
        "flatten_order": "window_then_source_feature",
        "importance_aggregation": "sum_four_window_importances_per_feature",
        "prevalence_unit": "unique_train_patients",
        "constant_filter": "constant_in_all_four_3_month_sums"}


def _fs_train_block(data, rows, feature_count):
    block = np.asarray(data["X"][rows], dtype=np.float32)
    if block.shape != (len(rows), 12, feature_count):
        raise ValueError("A TRAIN tensor block has an unexpected shape.")
    if not np.isfinite(block).all() or (block < 0).any():
        raise ValueError("TRAIN counts must be finite and nonnegative.")
    return block


def _fs_audit(features):
    return pd.DataFrame({"source_index": range(len(features)), "feature_name": features,
        "selected": False, "selection_status": "not_selected",
        "train_nonzero_patients": pd.array([pd.NA] * len(features), dtype="Int64"),
        "train_nonzero_patient_fraction": np.nan,
        "train_aggregate_min": np.nan, "train_aggregate_max": np.nan,
        "train_variable_windows": pd.array([pd.NA] * len(features), dtype="Int64"),
        "ranking_importance": np.nan,
        "selection_rank": pd.array([pd.NA] * len(features), dtype="Int64")})


def select_features(data, mode="reduced", max_features=150, core_features=None,
                    min_train_patients=10, seed=42):
    """Return (frozen manifest, aggregate-only audit), without reading held-out values.

    ``all`` and ``core`` are fixed selections and never inspect X or y values.
    ``core`` requires exact existing tensor feature names; legacy demographic or
    engineered features need a separate upstream, date-safe feature preparation.
    Reduced selection keeps at most max_features, with no artificial padding when
    fewer survive. All modes preserve the original source order for model inputs.
    """
    features, source_hash = _fs_source_features(data)
    if mode not in {"all", "core", "reduced"}:
        raise ValueError("Feature selection mode must be all, core or reduced.")
    audit = _fs_audit(features)
    if mode == "all":
        selected = list(range(len(features)))
        settings = {"selection_method": "all_source_features"}
        fit_split = "fixed"
    elif mode == "core":
        if (not isinstance(core_features, (list, tuple)) or not core_features
                or not all(isinstance(name, str) and name for name in core_features)):
            raise ValueError("Core experiment is pending: configure a nonempty exact feature-name list.")
        if len(set(core_features)) != len(core_features):
            raise ValueError("Core feature configuration contains duplicate names.")
        missing = sorted(set(core_features) - set(features))
        if missing:
            raise ValueError(f"Core features are absent from the tensor; no name approximation is allowed: {missing}")
        requested = set(core_features)
        selected = [i for i, name in enumerate(features) if name in requested]
        settings = {"selection_method": "configured_exact_feature_names",
                    "core_features": [features[i] for i in selected]}
        fit_split = "fixed"
    else:
        _fs_positive_int(max_features, "max_features")
        _fs_positive_int(min_train_patients, "min_train_patients")
        _fs_seed(seed)
        train, patient_codes, n_patients = _fs_training_rows(data)
        # Only these explicit TRAIN row positions may be read from X or y.
        labels = np.asarray(data["y"][train])
        if labels.shape != (len(train),) or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
            raise ValueError("Supervised selection requires binary TRAIN labels with both classes.")
        patient_present = np.zeros((n_patients, len(features)), dtype=bool)
        aggregate_min = np.full(len(features), np.inf, dtype=np.float64)
        aggregate_max = np.full(len(features), -np.inf, dtype=np.float64)
        window_min = np.full((4, len(features)), np.inf, dtype=np.float64)
        window_max = np.full((4, len(features)), -np.inf, dtype=np.float64)
        for start in range(0, len(train), 128):
            stop = min(start + 128, len(train))
            block = _fs_train_block(data, train[start:stop], len(features))
            windows = block.reshape(stop - start, 4, 3, len(features)).sum(axis=2, dtype=np.float64)
            window_min = np.minimum(window_min, windows.min(axis=0))
            window_max = np.maximum(window_max, windows.max(axis=0))
            sums = windows.sum(axis=1)
            aggregate_min = np.minimum(aggregate_min, sums.min(axis=0))
            aggregate_max = np.maximum(aggregate_max, sums.max(axis=0))
            np.logical_or.at(patient_present, patient_codes[start:stop], np.any(block > 0, axis=1))
        patient_counts = patient_present.sum(axis=0, dtype=np.int64)
        del patient_present
        variable_windows = (window_max > window_min).sum(axis=0)
        nonconstant = variable_windows > 0
        eligible = np.flatnonzero((patient_counts >= min_train_patients) & nonconstant)
        audit["train_nonzero_patients"] = pd.array(patient_counts, dtype="Int64")
        audit["train_nonzero_patient_fraction"] = patient_counts / n_patients
        audit["train_aggregate_min"] = aggregate_min
        audit["train_aggregate_max"] = aggregate_max
        audit["train_variable_windows"] = pd.array(variable_windows, dtype="Int64")
        for window in range(4):
            audit[f"train_window_{window}_sum_min"] = window_min[window]
            audit[f"train_window_{window}_sum_max"] = window_max[window]
        audit["selection_status"] = "below_importance_cutoff"
        audit.loc[patient_counts < min_train_patients, "selection_status"] = "rare_train_patients"
        audit.loc[~nonconstant, "selection_status"] = "constant_all_train_windows"
        audit.loc[patient_counts == 0, "selection_status"] = "empty_train"
        if not len(eligible):
            raise ValueError("No features survive TRAIN prevalence/constant filters; review the TRAIN feature audit/settings.")
        # Fit matrix contains eligible features only. Windows stay in actual
        # TIME_STEP order: 0-2 newest, then 3-5, 6-8, and 9-11 oldest.
        aggregates = np.empty((len(train), 4 * len(eligible)), dtype=np.float32)
        for start in range(0, len(train), 128):
            stop = min(start + 128, len(train))
            block = _fs_train_block(data, train[start:stop], len(features))
            windows = block.reshape(stop - start, 4, 3, len(features)).sum(axis=2, dtype=np.float64)
            aggregates[start:stop] = np.log1p(windows[:, :, eligible]).reshape(stop - start, -1)
        ranker_settings = _fs_ranker_settings(seed)
        ranker = ExtraTreesClassifier(**ranker_settings)
        ranker.fit(aggregates, labels.astype(np.int64))
        window_importance = np.asarray(ranker.feature_importances_, dtype=np.float64)
        if (window_importance.shape != (4 * len(eligible),)
                or not np.isfinite(window_importance).all() or (window_importance < 0).any()):
            raise ValueError("Feature ranker returned invalid importance values.")
        importance = window_importance.reshape(4, len(eligible)).sum(axis=0)
        ranking = np.lexsort((eligible, -importance))
        ranked_indices = eligible[ranking]
        audit.loc[eligible, "ranking_importance"] = importance
        audit.loc[ranked_indices, "selection_rank"] = np.arange(1, len(eligible) + 1)
        selected = sorted(int(i) for i in ranked_indices[:max_features])
        settings = {"selection_method": "train_extra_trees_impurity_importance",
            "max_features": max_features, "min_train_patients": min_train_patients, "seed": seed,
            **_fs_temporal_settings(), "ranker_params": ranker_settings,
            "n_train_snapshots": int(len(train)), "n_train_patients": int(n_patients)}
        fit_split = "train"
    audit.loc[selected, "selected"] = True
    audit.loc[selected, "selection_status"] = "selected"
    manifest = {"format_version": 1, "mode": mode,
        "source_feature_names_sha256": source_hash, "selected_indices": selected,
        "selected_features": [features[i] for i in selected],
        "settings": settings, "selection_fit_split": fit_split}
    manifest["manifest_sha256"] = _fs_digest(manifest)
    validate_feature_manifest(manifest, data)
    return manifest, audit


def validate_feature_manifest(manifest, data):
    """Validate a saved selection against source schema without refitting or reading X/y values."""
    expected = {"format_version", "mode", "source_feature_names_sha256", "selected_indices",
                "selected_features", "settings", "selection_fit_split", "manifest_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("Feature manifest fields are missing or unexpected.")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise ValueError("Unsupported feature manifest version.")
    digest_body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != _fs_digest(digest_body):
        raise ValueError("Feature manifest SHA256 verification failed.")
    features, source_hash = _fs_source_features(data)
    if manifest["source_feature_names_sha256"] != source_hash:
        raise ValueError("Feature manifest refers to a different source vocabulary/order.")
    selected = manifest["selected_indices"]
    if (not isinstance(selected, list) or not selected
            or not all(type(i) is int and 0 <= i < len(features) for i in selected)
            or selected != sorted(set(selected))):
        raise ValueError("Selected feature indices must be unique Python integers in source order.")
    if manifest["selected_features"] != [features[i] for i in selected]:
        raise ValueError("Selected feature names do not exactly match their source indices.")
    mode, settings = manifest["mode"], manifest["settings"]
    if mode not in {"all", "core", "reduced"} or not isinstance(settings, dict):
        raise ValueError("Invalid feature manifest mode/settings.")
    if mode == "all":
        if selected != list(range(len(features))) or settings != {"selection_method": "all_source_features"}:
            raise ValueError("An all-features manifest must contain every source feature.")
    elif mode == "core":
        expected_settings = {"selection_method": "configured_exact_feature_names",
                             "core_features": [features[i] for i in selected]}
        if settings != expected_settings:
            raise ValueError("Core-feature settings do not match the exact saved selection.")
    else:
        setting_keys = {"selection_method", "max_features", "min_train_patients", "seed",
            "ranker_params", "n_train_snapshots", "n_train_patients"} | set(_fs_temporal_settings())
        if set(settings) != setting_keys:
            raise ValueError("Reduced feature selection settings are incomplete or unexpected.")
        _fs_positive_int(settings["max_features"], "max_features")
        _fs_positive_int(settings["min_train_patients"], "min_train_patients")
        _fs_seed(settings["seed"])
        _fs_positive_int(settings["n_train_snapshots"], "n_train_snapshots")
        _fs_positive_int(settings["n_train_patients"], "n_train_patients")
        fixed_settings = {"selection_method": "train_extra_trees_impurity_importance",
            **_fs_temporal_settings(),
            "ranker_params": _fs_ranker_settings(settings["seed"])}
        if any(settings[key] != value for key, value in fixed_settings.items()):
            raise ValueError("Reduced feature selection method/settings differ from this implementation.")
        train, _, n_patients = _fs_training_rows(data)
        if (settings["n_train_snapshots"] != len(train) or settings["n_train_patients"] != n_patients
                or settings["min_train_patients"] > n_patients or len(selected) > settings["max_features"]):
            raise ValueError("Reduced feature selection counts/settings do not match the source TRAIN split.")
    expected_split = "train" if mode == "reduced" else "fixed"
    if manifest["selection_fit_split"] != expected_split:
        raise ValueError("Feature selection fitting split is inconsistent with its mode.")
    return manifest
