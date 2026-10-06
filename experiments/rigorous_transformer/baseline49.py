"""Secondary 49-feature LightGBM refit and descriptive original V63 scores.

These encoded snapshot predictors differ from the 1,028 raw count sequence.
Neither reading current metadata nor refitting establishes upstream fitting or
historical event-cutoff provenance. All patient-level arrays stay in runtime.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


DEFAULT_SOURCE_PREFIX = "DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_"
DEFAULT_BASELINE_CONFIG = dict(family="lightgbm49", representation="encoded_snapshot",
                              n_estimators=500, learning_rate=0.03, num_leaves=16,
                              min_child_samples=200, scale_pos_weight=10.0,
                              reg_lambda=0.0, early_stopping_rounds=30, n_jobs=4)
PARAMETER_EVIDENCE = {
    "min_child_samples": {"value": 200, "source_spelling": "min_data_in_leaf", "classification": "Recovered from existing artifact"},
    "num_leaves": {"value": 16, "classification": "Recovered from existing artifact"},
    "scale_pos_weight": {"value": 10.0, "classification": "Recovered from existing artifact"},
    "source": "S09 original AutoML OCR: active visible settings at lines 101, 105, 107; lightgbm_v63_context.md",
    "scope": "Visible requested settings, not proof of all effective historical fitted-model parameters",
    "new_refit_policy": "learning_rate=0.03; maximum 500 rounds; validation-AP stopping patience 30; default L2=0; fixed config, not tuned",
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _frame(value, selected=None):
    if isinstance(value, pd.DataFrame):
        return value[selected].copy() if selected is not None else value.copy()
    if selected is not None:
        value = value.select(*selected)
    if not hasattr(value, "toPandas"):
        raise TypeError("Reader must return a Spark or pandas DataFrame")
    return value.toPandas()


def _read_frozen_source(read_table, table, frozen, selected):
    """Filter keys in Spark before collection, including the VAL-only SCORE path."""
    value = read_table(table)
    keys = frozen[["PATIENT_ID", "END_DT"]].copy()
    if isinstance(value, pd.DataFrame):
        value = _keys(value)
        return value[[name.upper() for name in selected]].merge(keys, on=["PATIENT_ID", "END_DT"], how="inner", validate="one_to_one")
    physical = {str(name).upper(): str(name) for name in value.columns}
    if len(physical) != len(value.columns):
        raise ValueError("Ambiguous physical column casing")
    if any(name.upper() not in physical for name in selected):
        raise ValueError("Required physical source column unavailable")
    # Preserve actual identifier case and quote names before applying canonical aliases.
    selected_frame = value.select(*[value["`" + physical[name.upper()].replace("`", "``") + "`"].alias(name.upper())
                                    for name in selected])
    if not hasattr(value, "sparkSession"):
        raise TypeError("Spark reader must expose sparkSession to restrict frozen keys before collection")
    keys["END_DT"] = keys.END_DT.dt.date
    key_frame = value.sparkSession.createDataFrame(keys)
    return selected_frame.join(key_frame, on=["PATIENT_ID", "END_DT"], how="inner").toPandas()


def _normalized_columns(frame):
    names = [str(name).upper() for name in frame.columns]
    if len(set(names)) != len(names):
        raise ValueError("Case-insensitive duplicate source columns")
    frame = frame.copy()
    frame.columns = names
    return frame


def _keys(frame, require_label=True):
    frame = _normalized_columns(frame)
    required = {"PATIENT_ID", "END_DT"} | ({"RESP"} if require_label else set())
    if not required <= set(frame):
        raise ValueError("Exact PATIENT_ID, END_DT and RESP contract is required")
    if frame.PATIENT_ID.isna().any() or not frame.PATIENT_ID.map(lambda x: isinstance(x, str) and bool(x) and x == x.strip()).all():
        raise ValueError("Patient IDs must be exact nonempty strings")
    dates = pd.to_datetime(frame.END_DT, errors="raise")
    if dates.isna().any() or not (dates == dates.dt.normalize()).all():
        raise ValueError("Snapshot dates must be nonmissing midnight dates")
    if dates.dt.tz is not None:
        raise ValueError("Timezone-dependent snapshot dates are unsupported")
    frame["END_DT"] = dates
    if frame.duplicated(["PATIENT_ID", "END_DT"]).any():
        raise ValueError("Duplicate patient-snapshot source keys")
    if require_label and not frame.RESP.isin([0, 1]).all():
        raise ValueError("Labels must be nonmissing binary RESP values")
    return frame


def _feature_list(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v and v == v.strip() for v in value):
        raise ValueError("MODEL_TYPE feature field must hold an ordered nonempty exact-name array")
    if len({name.upper() for name in value}) != len(value):
        raise ValueError("Duplicate configured feature names")
    forbidden = {"PATIENT_ID", "END_DT", "START_DT", "RESP", "SPLIT", "RND", "SCORE", "DECILE", "CENTILE", "MILLILE"}
    if forbidden & {v.upper() for v in value}:
        raise ValueError("Identifier, label, split or score cannot be a baseline predictor")
    return value


def _numeric_matrix(values):
    values = np.asarray(values)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Expected nonempty snapshot by feature matrix")
    try:
        values = values.astype(np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Encoded feature values must be numeric") from error
    if np.isinf(values).any():
        raise ValueError("Infinite encoded values are not permitted")
    return values


def _input_identity(metadata, values, features):
    """Order-sensitive private-input digest; no individual records are returned."""
    frame = metadata[["PATIENT_ID", "END_DT", "RESP"]].copy()
    frame["END_DT"] = frame.END_DT.dt.strftime("%Y-%m-%d")
    frame["RESP"] = frame.RESP.astype(int)
    if "SPLIT" in metadata:
        frame["SPLIT"] = metadata.SPLIT.astype(str).str.lower()
    array = np.asarray(values, dtype="<f4").copy(order="C")
    array[np.isnan(array)] = np.nan  # One canonical NaN representation.
    digest = hashlib.sha256()
    digest.update(json.dumps({"features": features, "shape": list(array.shape), "dtype": "float32",
                              "metadata": frame.to_dict("records")}, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _align(frozen, source, columns):
    source = _keys(source)
    source_columns = ["PATIENT_ID", "END_DT", "RESP"] + list(columns)
    if not set(source_columns) <= set(source):
        raise ValueError("Source lacks required predictors or scores")
    left = frozen[["PATIENT_ID", "END_DT", "RESP"]].copy().reset_index(drop=True)
    merged = left.merge(source[source_columns], how="left", on=["PATIENT_ID", "END_DT"],
                        suffixes=("_FROZEN", "_SOURCE"), validate="one_to_one", indicator=True, sort=False)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Source does not cover every exact frozen patient-snapshot key")
    if not merged.RESP_FROZEN.eq(merged.RESP_SOURCE).all():
        raise ValueError("Source labels disagree with frozen RESP")
    return merged[list(columns)]


def load_encoded_baseline(read_table, frozen_metadata, source_prefix=DEFAULT_SOURCE_PREFIX, expected_features=49):
    """Preserve MODEL_TYPE order; order-only disagreement with FINAL_MODEL is an audit."""
    frozen = _keys(_frame(frozen_metadata))
    config = _normalized_columns(_frame(read_table(source_prefix + "MODEL_TYPE")))
    if len(config) != 1:
        raise ValueError("Expected exactly one MODEL_TYPE row")
    recognized = [name for name in ("FEATURES", "FEATURE_NAMES") if name in config]
    if not recognized:
        raise ValueError("No recognized ordered feature field in MODEL_TYPE")
    # FEATURES is the recovered contract; FEATURE_NAMES is supported only if explicitly present.
    features = _feature_list(config.iloc[0][recognized[0]])
    if any(_feature_list(config.iloc[0][name]) != features for name in recognized[1:]):
        raise ValueError("Conflicting configured feature arrays")
    if len(features) != expected_features:
        raise ValueError("Configured feature count differs; never fabricate or substitute 49 features")
    final = _normalized_columns(_frame(read_table(source_prefix + "FINAL_MODEL")))
    if not {"FEATURES", "SEQ"} <= set(final):
        raise ValueError("FINAL_MODEL must expose FEATURES and SEQ for the recorded order audit")
    if final.FEATURES.isna().any() or final.FEATURES.duplicated().any():
        raise ValueError("FINAL_MODEL has missing or duplicate names")
    rank = pd.to_numeric(final.SEQ, errors="coerce")
    valid_ranks = bool(rank.notna().all() and np.isfinite(rank).all() and rank.eq(np.floor(rank)).all() and not rank.duplicated().any())
    final_names = final.assign(_rank=rank).sort_values("_rank").FEATURES.tolist() if valid_ranks else None
    configured_only = sorted(set(features) - set(final.FEATURES))
    final_only = sorted(set(final.FEATURES) - set(features))
    if configured_only or final_only:
        raise ValueError("Configured and final feature sets differ; order-only discrepancy exception does not apply")
    audit = dict(configured_feature_count=len(features), final_feature_count=len(final),
                 configured_field=recognized[0], configured_order_sha256=_hash(features),
                 final_seq_order_sha256=_hash(final_names) if final_names else None,
                 final_seq_valid=valid_ranks, exact_order_agreement=features == final_names,
                 rule="MODEL_TYPE array order retained for this refit; SEQ is an unresolved historical summary-order contract",
                 historical_scoring_order_verified=False)
    # Select only necessary columns before bringing Spark data into approved runtime memory.
    source = _read_frozen_source(read_table, source_prefix + "MODEL_DATA", frozen,
                                 ["PATIENT_ID", "END_DT", "RESP"] + features)
    aligned = _align(frozen, source, [name.upper() for name in features])
    values = _numeric_matrix(aligned.to_numpy())
    return dict(metadata=frozen, X=values, features=features, feature_sha256=_hash(features),
                input_sha256=_input_identity(frozen, values, features), order_audit=audit,
                provenance=dict(source=source_prefix + "MODEL_DATA", baseline_kind="encoded49_refit",
                                historical_fit_membership_verified=False, upstream_encoding_fit_membership="UNKNOWN",
                                input_space="Existing encoded V63 snapshot values; fractions and AGE retained",
                                preprocessing_refitted=False, architecture_comparison_to_1028="different feature information; contextual comparison"))


def load_original_v63_scores(read_table, frozen_metadata, split="validation", *, allow_test=False,
                             selection_locked=False, source_prefix=DEFAULT_SOURCE_PREFIX):
    """Read only requested split's aligned scores; never infer fitting exclusion."""
    if split not in ("train", "validation", "test"):
        raise ValueError("Unrecognized frozen split")
    if split == "test" and not (allow_test and selection_locked):
        raise ValueError("TEST original scores require explicit final comparison and a persisted selection lock")
    frozen = _keys(_frame(frozen_metadata))
    if "SPLIT" not in frozen:
        raise ValueError("Frozen split labels required")
    chosen = frozen.loc[frozen.SPLIT.str.lower().eq(split)].reset_index(drop=True)
    if chosen.empty:
        raise ValueError("Requested split has no snapshots")
    source = _read_frozen_source(read_table, source_prefix + "UNIVERSE_W_FEATURES_SCORED", chosen,
                                 ["PATIENT_ID", "END_DT", "RESP", "SCORE"])
    aligned = _align(chosen, source, ["SCORE"])
    scores = pd.to_numeric(aligned.SCORE, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Original SCORE requires confirmed finite probability semantics")
    return dict(metadata=chosen, scores=scores, provenance=dict(baseline_kind="original_v63", split=split,
                fit_membership_verified=False, historical_fitting_scope="UNKNOWN", same_keys_and_labels=True,
                score_semantics="Candidate probability-like SCORE; semantics not established by range alone",
                interpretation="Descriptive existing scores; may include fitted patients; no generalization/superiority claim"))


def _metrics(y, scores):
    return dict(average_precision=float(average_precision_score(y, scores)),
                roc_auc=float(roc_auc_score(y, scores)), log_loss=float(log_loss(y, scores, labels=[0, 1])))


def train_encoded_baseline(trainvalues, y, groups, valvalues, yval, seed, config=None):
    """One fixed contextual baseline configuration across all prespecified seeds."""
    import lightgbm as lgb
    started = time.perf_counter()
    cfg = dict(DEFAULT_BASELINE_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError("Unexpected baseline configuration fields: " + repr(sorted(unknown)))
        cfg.update(config)
    if cfg["family"] != "lightgbm49" or cfg["representation"] != "encoded_snapshot":
        raise ValueError("49-feature context baseline has a fixed family and representation")
    train, validation = _numeric_matrix(trainvalues), _numeric_matrix(valvalues)
    y, yval = np.asarray(y), np.asarray(yval)
    if train.shape[1] != validation.shape[1] or len(y) != len(train) or len(yval) != len(validation):
        raise ValueError("Train and validation dimensions differ")
    if not np.isin(y, [0, 1]).all() or not np.isin(yval, [0, 1]).all() or len(np.unique(y)) != 2 or len(np.unique(yval)) != 2:
        raise ValueError("Both fitting and validation require valid binary labels and both classes")
    if len(groups) != len(y) or any(v is None or not str(v).strip() for v in groups):
        raise ValueError("Valid TRAIN patient groups required; split disjointness is checked by data contract")
    fit_cfg = {key: value for key, value in cfg.items() if key not in ("family", "representation", "early_stopping_rounds")}
    model = lgb.LGBMClassifier(**fit_cfg, objective="binary", metric="None", random_state=int(seed),
                               deterministic=True, force_col_wise=True, verbosity=-1)
    history = {}
    def ap_metric(target, score):
        return "average_precision", float(average_precision_score(target, score)), True
    model.fit(train, y, eval_set=[(validation, yval)], eval_metric=ap_metric,
              callbacks=[lgb.early_stopping(int(cfg["early_stopping_rounds"]), first_metric_only=True, verbose=False),
                         lgb.record_evaluation(history), lgb.log_evaluation(0)])
    best = int(model.best_iteration_ or cfg["n_estimators"])
    blob = model.booster_.model_to_string(num_iteration=best).encode()
    train_scores = model.booster_.predict(train, num_iteration=best)
    val_scores = model.booster_.predict(validation, num_iteration=best)
    trajectory = history.get("valid_0", {}).get("average_precision", [])
    rows = [dict(epoch=i + 1, validation_average_precision=float(value)) for i, value in enumerate(trajectory)]
    summary = dict(config=cfg, seed=int(seed), training_complete=True, best_epoch=best,
                   epochs_observed=len(rows), validation_evaluations=len(rows), train_metrics=_metrics(y, train_scores),
                   validation_metrics=_metrics(yval, val_scores), generalization_gap_ap=float(average_precision_score(y, train_scores) - average_precision_score(yval, val_scores)),
                   parameter_count=int(sum(tree["num_leaves"] for tree in model.booster_.dump_model()["tree_info"])),
                   complexity_unit="tree_leaves", wall_seconds=time.perf_counter() - started,
                   artifact_format="lightgbm_text", model_sha256=hashlib.sha256(blob).hexdigest(),
                   transform=dict(kind="identity_encoded_numeric", input_dim=train.shape[1], imputation="native LightGBM missing-value handling"),
                   importance=model.booster_.feature_importance(importance_type="gain").tolist(),
                   train_missing_values=int(np.isnan(train).sum()), test_inference_performed=False,
                   historical_fit_membership_verified=False, upstream_encoding_fit_membership="UNKNOWN",
                   parameter_evidence=PARAMETER_EVIDENCE,
                   configured_parameter_overrides={k: v for k, v in cfg.items() if v != DEFAULT_BASELINE_CONFIG[k]},
                   selection_scope="Fixed 49-feature contextual refit; no configuration search; validation-AP early stopping",
                   source_of_defaults="Visible historical leaf/weight settings plus explicit new refit learning/iteration policy; not recovered full historical parameters")
    summary["config_sha256"] = _hash(summary["config"])
    summary["transform_sha256"] = _hash(summary["transform"])
    summary["raw_input_shape"] = list(train.shape[1:])
    return dict(summary=summary, history=rows, model_blob=blob, train_scores=train_scores, validation_scores=val_scores)


def predict_encoded(result, values):
    import lightgbm as lgb
    values = _numeric_matrix(values)
    summary, blob = result["summary"], result["model_blob"]
    if hashlib.sha256(blob).hexdigest() != summary["model_sha256"]:
        raise ValueError("Baseline artifact checksum mismatch")
    if _hash(summary["config"]) != summary["config_sha256"] or _hash(summary["transform"]) != summary["transform_sha256"]:
        raise ValueError("Baseline configuration/transform checksum mismatch")
    if values.shape[1] != summary["transform"]["input_dim"]:
        raise ValueError("Encoded baseline feature count differs")
    model = lgb.Booster(model_str=blob.decode())
    return model.predict(values)
