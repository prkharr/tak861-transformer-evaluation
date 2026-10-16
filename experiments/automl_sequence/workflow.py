"""Run-scoped orchestration for summary and historical AutoML experiments.

Private inputs and row-level predictions stay in the configured artifact store.
Selection sees TRAIN/VALIDATION only. TEST prediction requires an explicit lock.
The notebook injects input/model/evaluation modules and immutable store functions.
"""
import hashlib
import io
import json
import re
import time

import numpy as np
import pandas as pd

inputs = modeling = comparison = None
save_artifacts = read_artifacts = table_exists = None


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_bytes(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def array_bytes(value):
    out = io.BytesIO()
    np.save(out, np.asarray(value), allow_pickle=False)
    return out.getvalue()


def array_sha256(value):
    """Hash the .npy representation without allocating a second full count tensor."""
    hasher = hashlib.sha256()
    class Sink:
        def write(self, block):
            hasher.update(block)
            return len(block)
    np.save(Sink(), np.asarray(value), allow_pickle=False)
    return hasher.hexdigest()


def bytes_array(value):
    return np.load(io.BytesIO(value), allow_pickle=False)


def npz_bytes(arrays):
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def prefix(run_id):
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,23}", run_id):
        raise ValueError("RUN_ID must use uppercase letters, digits and underscores")
    return "TAK861_AUTOML_SEQUENCE_V1_" + run_id


def assign_patient_split(metadata, seed=42):
    """Match the documented sorted-patient 70/15/15 stratification algorithm.

    Reproducibility is not an assertion that historical assignments match. If a
    SPLIT column is supplied, retain and verify it rather than resplitting.
    """
    out = metadata.copy()
    needs_split = "SPLIT" not in out
    if needs_split:
        out["SPLIT"] = "train"  # Only to validate keys/labels before assignment.
    out = inputs.normalize_snapshots(out)
    if needs_split:
        from sklearn.model_selection import train_test_split
        patients = out.groupby("PATIENT_ID", sort=True)["RESP"].max()
        a, rest = train_test_split(patients.index.to_numpy(), test_size=.30,
                                   random_state=seed, stratify=patients.to_numpy())
        b, c = train_test_split(rest, test_size=.5, random_state=seed,
                               stratify=patients.loc[rest].to_numpy())
        mapping = {p: s for s, values in (("train", a), ("validation", b), ("test", c)) for p in values}
        out["SPLIT"] = out.PATIENT_ID.map(mapping)
    if set(out.SPLIT) != {"train", "validation", "test"} or out.SPLIT.isna().any():
        raise ValueError("Require complete train/validation/test membership")
    if out.groupby("PATIENT_ID").SPLIT.nunique().max() != 1:
        raise ValueError("A patient appears in multiple splits")
    for split, frame in out.groupby("SPLIT"):
        if set(frame.RESP.unique()) != {0, 1}:
            raise ValueError("Both classes required in " + split)
    return out


INPUT_NAMES = {"arrays.npz", "metadata.csv", "input_manifest.json"}


def save_inputs(run_prefix, data, panel_manifest, config, implementation_hash):
    metadata = assign_patient_split(data["metadata"])
    inputs.verify_fitting_scope(panel_manifest["contract"], metadata)
    _validate_input_arrays(data, metadata, data["feature_names"])
    metadata["END_DT"] = pd.to_datetime(metadata.END_DT).dt.strftime("%Y-%m-%d")
    arrays = {key: np.asarray(data[key]) for key in
              ("X_summary", "X_sequence", "summary_mask", "sequence_mask", "y")}
    if "X_count" in data:
        arrays["X_count"] = data["X_count"]
        arrays["count_mask"] = data["count_mask"]
    arrays_blob = npz_bytes(arrays)
    metadata_blob = metadata.to_csv(index=False).encode()
    manifest = dict(version=1, implementation_hash=implementation_hash,
                    configuration=plain(config), configuration_hash=digest(config),
                    panel_manifest=panel_manifest, feature_names=data["feature_names"],
                    feature_specs=data.get("feature_specs", panel_manifest["feature_specs"]),
                    input_report=data.get("report", {}),
                    array_sha256=hashlib.sha256(arrays_blob).hexdigest(),
                    metadata_sha256=hashlib.sha256(metadata_blob).hexdigest(),
                    count_contract=data.get("count_contract"),
                    split_counts={s: int((metadata.SPLIT == s).sum()) for s in ("train", "validation", "test")},
                    prediction_unit="patient_snapshot", private_row_level_artifacts=True)
    if table_exists(run_prefix + "_INPUTS"):
        previous = load_inputs(run_prefix, implementation_hash)["manifest"]
        # Keep original extraction timing; rerun timing is not predictor identity.
        def semantic(value):
            value = json.loads(json_bytes(value))
            value["panel_manifest"].pop("extraction_report", None)
            return value
        if semantic(previous) != semantic(manifest):
            raise ValueError("Frozen input source/content/configuration changed; use a new RUN_ID")
        return previous
    save_artifacts(run_prefix + "_INPUTS", {"arrays.npz": arrays_blob,
                   "metadata.csv": metadata_blob, "input_manifest.json": json_bytes(manifest)})
    return manifest


def load_inputs(run_prefix, implementation_hash):
    artifacts = read_artifacts(run_prefix + "_INPUTS", INPUT_NAMES)
    manifest = json.loads(artifacts["input_manifest.json"])
    if manifest["implementation_hash"] != implementation_hash:
        raise ValueError("Inputs belong to a different notebook implementation; use a new RUN_ID")
    for name, key in (("arrays.npz", "array_sha256"), ("metadata.csv", "metadata_sha256")):
        if hashlib.sha256(artifacts[name]).hexdigest() != manifest[key]:
            raise ValueError("Input content changed: " + name)
    with np.load(io.BytesIO(artifacts["arrays.npz"]), allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    metadata = pd.read_csv(io.BytesIO(artifacts["metadata.csv"]), dtype={"PATIENT_ID": str}, keep_default_na=False)
    metadata = assign_patient_split(metadata)
    inputs.verify_fitting_scope(manifest["panel_manifest"]["contract"], metadata)
    data.update(metadata=metadata, manifest=manifest, feature_names=manifest["feature_names"],
                feature_specs=manifest["feature_specs"],
                indices={s: np.flatnonzero(metadata.SPLIT.to_numpy() == s) for s in ("train", "validation", "test")})
    _validate_input_arrays(data, metadata, manifest["feature_names"])
    return data


def _validate_input_arrays(data, metadata, feature_names):
    n, f = len(metadata), len(feature_names)
    if f == 0 or len(set(feature_names)) != f:
        raise ValueError("Invalid AutoML feature order")
    for value, mask, steps in (("X_summary", "summary_mask", 1), ("X_sequence", "sequence_mask", 12)):
        x, valid = np.asarray(data[value]), np.asarray(data[mask])
        if x.shape != (n, steps, f) or valid.shape != (n, steps) or valid.dtype != bool:
            raise ValueError("Invalid AutoML array/mask shape: " + value)
        if np.isinf(x).any() or not np.isnan(x[~valid]).all():
            raise ValueError("Invalid feature values or missing-history padding")
    if not np.array_equal(data["X_summary"], data["X_sequence"][:, :1], equal_nan=True):
        raise ValueError("Summary inputs differ from historical current snapshot")
    if not np.array_equal(data["summary_mask"], data["sequence_mask"][:, :1]):
        raise ValueError("Invalid current snapshot observation mask")
    if np.asarray(data["y"]).shape != (n,) or not np.array_equal(data["y"], metadata.RESP.to_numpy()):
        raise ValueError("Saved labels do not align with metadata")
    if "X_count" in data:
        if np.asarray(data["X_count"]).shape != (n, 12, 1028) or np.asarray(data["count_mask"]).shape != (n, 12):
            raise ValueError("Count reference shape mismatch")
        if np.asarray(data["count_mask"]).dtype != bool or np.isinf(data["X_count"]).any():
            raise ValueError("Invalid count mask or nonfinite count values")


def attach_count_reference(data, values, metadata, contract, valid=None):
    """Optional verified reuse, aligned by exact keys and independently hashed.

    A provenance declaration limits claims; it is not evidence of an unknown
    historical extraction rule. Never rename a different universe to 1,028.
    """
    if not isinstance(contract, dict) or not all(contract.get(k) for k in
            ("source_id", "source_version", "content_sha256", "feature_names", "provenance_status",
             "orientation", "date_policy", "mask_kind", "mask_sha256", "asof_grid_sha256")):
        raise ValueError("Count reuse needs source/version/content, calendar, explicit mask and provenance fingerprints")
    values = np.asarray(values)
    names = contract["feature_names"]
    if values.ndim != 3 or values.shape != (len(metadata), 12, 1028) or len(names) != 1028 or len(set(names)) != 1028:
        raise ValueError("Count reference must preserve the complete 1,028 candidate universe")
    if contract.get("n_steps") != 12 or contract["orientation"] != "0=newest" or contract["date_policy"] not in inputs.DATE_POLICIES:
        raise ValueError("Count cache calendar orientation/length must be explicit")
    valid = np.asarray(valid)
    if valid.shape != values.shape[:2] or valid.dtype != bool:
        raise ValueError("Count reuse requires an explicit boolean position mask")
    if array_sha256(valid) != contract["mask_sha256"]:
        raise ValueError("Count position mask hash mismatch")
    if array_sha256(values) != contract["content_sha256"]:
        raise ValueError("Count cache content hash mismatch")
    for block in np.array_split(values, max(1, len(values) // 512)):
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError("Invalid count values")
        finite = block[np.isfinite(block)]
        if (finite != np.floor(finite)).any():
            raise ValueError("Fractional count values")
        if np.any(finite.astype(np.float32).astype(np.float64) != finite):
            raise ValueError("Count loses exact float32 precision")
    old = inputs.normalize_snapshots(metadata)
    target = inputs.normalize_snapshots(data["metadata"])
    if contract.get("metadata_sha256") != inputs.frame_hash(old):
        raise ValueError("Count cache snapshot metadata/order fingerprint mismatch")
    _, old_grid = inputs.make_requests(old, date_policy=contract["date_policy"])
    if inputs.frame_hash(old_grid) != contract["asof_grid_sha256"]:
        raise ValueError("Count cache date grid mismatch")
    target_policy = data.get("report", {}).get("date_policy", data.get("manifest", {}).get("input_report", {}).get("date_policy"))
    if target_policy != contract["date_policy"]:
        raise ValueError("Count and AutoML historical date policies differ; do not concatenate misaligned dates")
    old["_row"] = np.arange(len(old))
    joined = target.merge(old[["PATIENT_ID", "END_DT", "RESP", "_row"]],
                          on=["PATIENT_ID", "END_DT"], how="left", validate="one_to_one", suffixes=("", "_cached"))
    if joined._row.isna().any() or not joined.RESP.eq(joined.RESP_cached).all() or len(old) != len(target):
        raise ValueError("Count cache cohort/label mismatch")
    order = joined._row.to_numpy(dtype=int)
    data["X_count"] = values[order].astype(np.float32)
    data["count_mask"] = valid[order].copy()
    data["count_contract"] = dict(contract, coverage_note="Position mask semantics are explicit; a saved row does not establish clinical observation coverage")
    return data


PREP_NAMES = {"preparation.json", "protocol.json"}


def prepare_run(data, run_prefix, config, implementation_hash):
    if data["manifest"]["panel_manifest"]["contract"]["fit_scope"] != "deterministic_raw":
        raise ValueError("This fold-local selection protocol requires raw deterministic AutoML inputs. "
                         "A transform or vocabulary selected using all TRAIN labels is not fold-local. "
                         "Recover raw extraction without RESP-based fitting, or implement upstream reconstruction inside each fold.")
    table = run_prefix + "_PREPARATION"
    expected = dict(input_manifest_hash=digest(data["manifest"]), implementation_hash=implementation_hash,
                    config_hash=digest(config))
    if table_exists(table):
        saved = read_artifacts(table, PREP_NAMES)
        prep, protocol = json.loads(saved["preparation.json"]), json.loads(saved["protocol.json"])
        if protocol["identity"] != expected:
            raise ValueError("Preparation identity changed; use a new RUN_ID")
        return prep, protocol
    ti = data["indices"]["train"]
    groups = data["metadata"].PATIENT_ID.to_numpy()[ti]
    primary_fraction = float(config.get("primary_lift_fraction", .10))
    if not np.isfinite(primary_fraction) or not 0 < primary_fraction <= 1:
        raise ValueError("primary_lift_fraction must be in (0,1]")
    selection_config, model_config = dict(config.get("selection", {})), dict(config.get("model", {}))
    for settings in (selection_config, model_config):
        if "primary_lift_fraction" in settings and settings["primary_lift_fraction"] != primary_fraction:
            raise ValueError("Conflicting primary capacity; set primary_lift_fraction once in the top-level configuration")
        settings["primary_lift_fraction"] = primary_fraction
    declared_kinds = [s["kind"] for s in data["manifest"]["feature_specs"]]
    if config.get("feature_kinds", declared_kinds) != declared_kinds:
        raise ValueError("Model preprocessing kinds differ from the recovered source feature contract")
    prep = {"automl": modeling.prepare_features(
        data["X_summary"][ti], data["X_sequence"][ti], data["y"][ti], groups,
        data["feature_names"], valid_train=data["sequence_mask"][ti],
        feature_kinds=declared_kinds, config=selection_config)}
    arms = {"SUMMARY": dict(family="transformer", input="X_summary", mask="summary_mask", preparation="automl"),
            "SEQUENCE": dict(family="transformer", input="X_sequence", mask="sequence_mask", preparation="automl")}
    if "X_count" in data:
        count = data["X_count"][ti]
        prep["count"] = modeling.prepare_features(count[:, :1], count, data["y"][ti], groups,
            data["manifest"]["count_contract"]["feature_names"], valid_train=data["count_mask"][ti],
            feature_kinds=["count"] * count.shape[-1], config=selection_config)
        arms["COUNT"] = dict(family="transformer", input="X_count", mask="count_mask", preparation="count")
        arms["STATIC_FUSION"] = dict(arms["COUNT"], static_input="X_summary", static_mask="summary_mask",
                                    static_preparation="automl", experiment="count_sequence_plus_snapshot_automl")
        arms["TEMPORAL_FUSION"] = dict(arms["COUNT"], temporal_input="X_sequence", temporal_mask="sequence_mask",
                                      temporal_preparation="automl", experiment="count_plus_historical_automl_per_timestep")
    if config.get("require_count_reference", True) and "COUNT" not in arms:
        raise ValueError("Both requested incremental-value experiments require the matched 1,028-count reference")
    if not config.get("include_automl_only_controls", False):
        arms = {name: arm for name, arm in arms.items() if name not in ("SUMMARY", "SEQUENCE")}
    if config.get("include_lightgbm", False):
        raise ValueError("LightGBM refit is deferred in this bounded two-experiment protocol")
    seeds = config.get("seeds", [42, 142, 242])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be a prespecified unique nonempty list")
    protocol = dict(identity=expected, arms=arms, seeds=seeds, preparation_hash=digest(prep),
                    model_config=model_config, primary_metric="lift_at_primary_capacity",
                    primary_lift_fraction=primary_fraction,
                    primary_capacity_provenance="configured before fit; 10% default is an operating assumption, not a recovered prior agreement",
                    lift_population="patient_snapshots; not unique-patient interventions",
                    lift_tie_policy="score descending then frozen snapshot row order; exact ceil(fraction*N)",
                    selection_split="train_grouped_cv_and_validation", test_used_for_selection=False,
                    top_fractions=sorted({.1, .2, .3, primary_fraction}),
                    test_evaluation_ordinal=None if 'client_universe' in config else 3,
                    legacy_test_evaluations=2,
                    test_scope=("Revised universe includes previously inspected TEST patients and possibly new patients; not an untouched holdout"
                                if 'client_universe' in config else "retrospective; this cohort already evaluated twice"),
                    partial_seed_rule="all completed runs shown; incomplete protocol cannot be locked",
                    comparison_scope="matched count control versus static and historical AutoML augmentation; descriptive development evidence")
    save_artifacts(table, {"preparation.json": json_bytes(prep), "protocol.json": json_bytes(protocol)})
    return prep, protocol


RUN_NAMES = {"model.bin", "run.json", "train.npy", "validation.npy"}


def pack_run(run, identity):
    summary = {k: v for k, v in run.items() if k not in ("model_blob", "train_scores", "val_scores", "validation_scores", "model")}
    blobs = {"model.bin": run["model_blob"], "train.npy": array_bytes(run["train_scores"]),
             "validation.npy": array_bytes(run["val_scores"])}
    summary.update(identity=identity, payload_hashes={k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()})
    blobs["run.json"] = json_bytes(summary)
    return blobs


def unpack_run(artifacts, identity):
    summary = json.loads(artifacts["run.json"])
    if summary["identity"] != identity:
        raise ValueError("Cached fitted model identity changed")
    for name, expected in summary["payload_hashes"].items():
        if hashlib.sha256(artifacts[name]).hexdigest() != expected:
            raise ValueError("Fitted model artifact hash mismatch")
    return dict(summary, model_blob=artifacts["model.bin"], train_scores=bytes_array(artifacts["train.npy"]),
                val_scores=bytes_array(artifacts["validation.npy"]))


def arm_arrays(data, prep, arm, split):
    idx = data["indices"][split]
    valid = data[arm["mask"]][idx]
    x = modeling.transform_inputs(data[arm["input"]][idx], prep[arm["preparation"]], valid=valid)
    static = None
    if "static_input" in arm:
        static_mask = data[arm["static_mask"]][idx]
        static = modeling.transform_inputs(data[arm["static_input"]][idx], prep[arm["static_preparation"]],
                                          valid=static_mask)[:, 0]
        static = np.concatenate([static, static_mask.astype(np.float32)], axis=-1)
    if "temporal_input" in arm:
        temporal_mask = data[arm["temporal_mask"]][idx]
        temporal = modeling.transform_inputs(data[arm["temporal_input"]][idx], prep[arm["temporal_preparation"]],
                                             valid=temporal_mask)
        if temporal.shape[:2] != x.shape[:2]:
            raise ValueError("Count and AutoML monthly shapes disagree")
        # Missing AutoML history does not mask an otherwise valid count token.
        # The indicator distinguishes missing history from a transformed zero.
        x = np.concatenate([x, temporal, temporal_mask[:, :, None].astype(np.float32)], axis=-1)
    return x, valid, static


def train_registered(data, prep, protocol, run_prefix):
    if digest(prep) != protocol["preparation_hash"] or digest(data["manifest"]) != protocol["identity"]["input_manifest_hash"]:
        raise ValueError("Training inputs or preparation changed")
    groups = data["metadata"].PATIENT_ID.to_numpy()
    ti, vi = data["indices"]["train"], data["indices"]["validation"]
    results = {}
    for name, arm in protocol["arms"].items():
        train, train_mask, static_train = arm_arrays(data, prep, arm, "train")
        validation, val_mask, static_val = arm_arrays(data, prep, arm, "validation")
        results[name] = []
        for seed in protocol["seeds"]:
            identity = dict(protocol_hash=digest(protocol), arm=name, seed=seed)
            table = run_prefix + "_" + name + "_S" + str(seed)
            if not table_exists(table):
                trainer = modeling.train_encoder if arm["family"] == "transformer" else modeling.train_lightgbm
                run = trainer(train, data["y"][ti], groups[ti], validation, data["y"][vi],
                              valid_train=train_mask, valid_val=val_mask,
                              static_train=static_train, static_val=static_val,
                              groups_val=groups[vi],
                              config=protocol["model_config"], seed=seed)
                save_artifacts(table, pack_run(run, identity))
            result = unpack_run(read_artifacts(table, RUN_NAMES), identity)
            if len(result["train_scores"]) != len(ti) or len(result["val_scores"]) != len(vi):
                raise ValueError("Prediction length does not match frozen split")
            results[name].append(result)
            print(name, "seed", seed, "completed; validation lift",
                  modeling.lift_at_fraction(data["y"][vi], result["val_scores"], protocol["primary_lift_fraction"]),
                  "secondary AP", comparison.binary_metrics(data["y"][vi], result["val_scores"])["average_precision"])
        del train, validation
    return results


def read_runs(protocol, run_prefix):
    return {name: [unpack_run(read_artifacts(run_prefix + "_" + name + "_S" + str(seed), RUN_NAMES),
                  dict(protocol_hash=digest(protocol), arm=name, seed=seed)) for seed in protocol["seeds"]]
            for name in protocol["arms"]}


def make_lock(data, prep, protocol, results, run_prefix):
    if set(results) != set(protocol["arms"]) or digest(prep) != protocol["preparation_hash"]:
        raise ValueError("Incomplete registered experiments")
    vi, ti = data["indices"]["validation"], data["indices"]["train"]
    models, metrics = {}, {}
    for name, runs in results.items():
        if len(runs) != len(protocol["seeds"]) or [r["identity"]["seed"] for r in runs] != protocol["seeds"]:
            raise ValueError("All prespecified seeds must complete before locking")
        val = np.mean([r["val_scores"] for r in runs], axis=0)
        train = np.mean([r["train_scores"] for r in runs], axis=0)
        policy = comparison.select_validation_threshold(data["y"][vi], val)
        models[name] = dict(predictor_hashes=[r["payload_hashes"] for r in runs], threshold=policy)
        metrics[name] = dict(train=comparison.binary_metrics(data["y"][ti], train, policy,
                                                             top_fractions=protocol["top_fractions"]),
                             validation=comparison.binary_metrics(data["y"][vi], val, policy,
                                                                  top_fractions=protocol["top_fractions"]))
        for split_name, yy, ss in (("train", data["y"][ti], train), ("validation", data["y"][vi], val)):
            metrics[name][split_name]["primary_lift"] = modeling.lift_at_fraction(yy, ss, protocol["primary_lift_fraction"])
            metrics[name][split_name]["primary_lift_fraction"] = protocol["primary_lift_fraction"]
    lock = dict(protocol_hash=digest(protocol), preparation_hash=digest(prep),
                input_manifest_hash=digest(data["manifest"]), models=models, metrics=metrics,
                test_prediction_performed=False, test_use_status_note="False at immutable lock creation; completed TEST use is recorded in FINAL_TEST report",
                test_scope=protocol["test_scope"],
                primary_metric=protocol["primary_metric"], primary_lift_fraction=protocol["primary_lift_fraction"],
                threshold_note="Validation F1 threshold is secondary; primary lift uses the registered capacity, not this threshold")
    save_artifacts(run_prefix + "_LOCK", {"lock.json": json_bytes(lock)})
    return lock


def paired_lift_bootstrap(y, scores_a, scores_b, patient_ids, fractions=(.1, .2, .3),
                          n_bootstrap=1000, seed=20261013, confidence=.95):
    """Paired patient resampling for snapshot lift, with exact repeated-row capacity.

    Patient multiplicities repeat every snapshot of a sampled patient. Ranking
    ties use frozen original row order in every draw. Integer multiplicities
    avoid materializing duplicate rows; a boundary row may be selected partly.
    """
    y, a, b, groups = [np.asarray(v) for v in (y, scores_a, scores_b, patient_ids)]
    fractions = sorted(set(float(f) for f in fractions))
    if (y.ndim != 1 or any(v.shape != y.shape for v in (a, b, groups))
            or set(np.unique(y)) != {0, 1} or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not fractions or any(not np.isfinite(f) or not 0 < f <= 1 for f in fractions)
            or type(n_bootstrap) is not int or n_bootstrap < 20 or not 0 < confidence < 1):
        raise ValueError("Paired lift bootstrap requires aligned two-class data, valid fractions and at least 20 draws")
    if any(v is None or str(v) in ("", "nan") for v in groups):
        raise ValueError("Patient IDs must be complete for clustered lift uncertainty")
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    orders = [np.lexsort((np.arange(len(y)), -scores.astype(float))) for scores in (a, b)]
    def lifts(order, multiplicities):
        counts = multiplicities[order]
        before = np.cumsum(counts) - counts
        n = int(multiplicities.sum())
        prevalence = float(np.dot(y, multiplicities) / n)
        values = []
        for fraction in fractions:
            k = max(1, int(np.ceil(fraction*n)))
            taken = np.minimum(counts, np.maximum(k-before, 0))
            values.append(float(np.dot(y[order], taken) / k / prevalence))
        return np.asarray(values)
    points = lifts(orders[0], np.ones(len(y), dtype=int)) - lifts(orders[1], np.ones(len(y), dtype=int))
    rng, draws, invalid = np.random.default_rng(seed), [], 0
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(unique), size=len(unique))
        multiplicities = np.bincount(sampled, minlength=len(unique))[inverse]
        positives, total = int(np.dot(y, multiplicities)), int(multiplicities.sum())
        if positives == 0 or positives == total:
            invalid += 1
            continue
        draws.append(lifts(orders[0], multiplicities)-lifts(orders[1], multiplicities))
    reliable = len(draws) >= max(20, int(np.ceil(.8*n_bootstrap)))
    values = np.asarray(draws)
    tail = (1-confidence)/2
    by_fraction = {}
    for j, fraction in enumerate(fractions):
        lower, upper = (np.quantile(values[:, j], [tail, 1-tail]).tolist() if reliable else (None, None))
        by_fraction[str(fraction)] = dict(fraction=fraction, delta=float(points[j]), lower=lower, upper=upper,
                                           excludes_zero=bool(reliable and (lower > 0 or upper < 0)))
    return dict(lift_by_fraction=by_fraction, unit="patient_cluster", population="patient_snapshots",
                patients=len(unique), requested_draws=n_bootstrap, valid_draws=len(draws), single_class_draws=invalid,
                seed=seed, confidence=confidence, direction="model_a_minus_model_b", paired=True,
                interval_reliable=reliable, tie_rule="frozen original row order; repeated rows counted with patient multiplicity",
                capacity_rule="exact ceil(fraction * resampled_snapshot_count)")


def compare_run(data, prep, protocol, results, lock, run_prefix, *, evaluate_test=False, n_bootstrap=1000):
    if lock["protocol_hash"] != digest(protocol) or lock["input_manifest_hash"] != digest(data["manifest"]) or lock["preparation_hash"] != digest(prep):
        raise ValueError("Evaluation lock mismatch")
    saved = json.loads(read_artifacts(run_prefix + "_LOCK", {"lock.json"})["lock.json"])
    if saved != plain(lock):
        raise ValueError("Persisted lock mismatch")
    split = "test" if evaluate_test else "validation"
    table = run_prefix + ("_FINAL_TEST" if evaluate_test else "_VALIDATION_COMPARISON")
    names = {"report.json"} | {n + ".npy" for n in protocol["arms"]}
    if table_exists(table):
        report = json.loads(read_artifacts(table, names)["report.json"])
        if report["lock_hash"] != digest(lock) or report["bootstrap_resamples"] != n_bootstrap:
            raise ValueError("Existing comparison differs from lock or uncertainty protocol")
        return report
    idx = data["indices"][split]
    scores, metrics = {}, {}
    for name, arm in protocol["arms"].items():
        runs = results[name]
        if [r["payload_hashes"] for r in runs] != lock["models"][name]["predictor_hashes"]:
            raise ValueError("Final fitted model changed")
        if evaluate_test:
            x, valid, static = arm_arrays(data, prep, arm, "test")
            predict = modeling.predict_encoder if arm["family"] == "transformer" else modeling.predict_lightgbm
            scores[name] = np.mean([predict(r, x, valid=valid, static=static) for r in runs], axis=0)
            del x
        else:
            scores[name] = np.mean([r["val_scores"] for r in runs], axis=0)
        metrics[name] = comparison.binary_metrics(data["y"][idx], scores[name], lock["models"][name]["threshold"],
                                                   top_fractions=protocol["top_fractions"])
        metrics[name]["primary_lift"] = modeling.lift_at_fraction(data["y"][idx], scores[name], protocol["primary_lift_fraction"])
        metrics[name]["primary_lift_fraction"] = protocol["primary_lift_fraction"]
    pairs = [("SEQUENCE", "SUMMARY")] if "SEQUENCE" in scores and "SUMMARY" in scores else []
    if "COUNT" in scores:
        pairs += [(name, "COUNT") for name in ("SUMMARY", "SEQUENCE", "STATIC_FUSION", "TEMPORAL_FUSION") if name in scores]
    if "STATIC_FUSION" in scores and "TEMPORAL_FUSION" in scores:
        pairs += [("TEMPORAL_FUSION", "STATIC_FUSION")]
    for name in ("SUMMARY", "SEQUENCE"):
        if "GB_" + name in scores:
            pairs.append((name, "GB_" + name))
    comparisons = {}
    for a, b in pairs:
        uncertainty = comparison.paired_patient_bootstrap(data["y"][idx], scores[a], scores[b],
            data["metadata"].PATIENT_ID.to_numpy()[idx], n_bootstrap=n_bootstrap, seed=20261013)
        lift_uncertainty = paired_lift_bootstrap(data["y"][idx], scores[a], scores[b],
            data["metadata"].PATIENT_ID.to_numpy()[idx], fractions=protocol["top_fractions"],
            n_bootstrap=n_bootstrap, seed=20261013)
        comparisons[a + "_MINUS_" + b] = dict(model_a=a, model_b=b, **uncertainty,
            primary_lift=lift_uncertainty["lift_by_fraction"][str(protocol["primary_lift_fraction"])],
            lift_by_fraction=lift_uncertainty["lift_by_fraction"], lift_bootstrap=lift_uncertainty)
    report = dict(lock_hash=digest(lock), split=split, metrics=metrics, comparisons=comparisons,
                  test_prediction_performed=bool(evaluate_test),
                  bootstrap_resamples=n_bootstrap, execution_status="completed_on_configured_inputs",
                  primary_metric=protocol["primary_metric"], primary_lift_fraction=protocol["primary_lift_fraction"],
                  lift_population=protocol["lift_population"], lift_tie_policy=protocol["lift_tie_policy"],
                  source_id=data["manifest"]["panel_manifest"]["contract"]["source_id"],
                  test_evaluation_ordinal=protocol["test_evaluation_ordinal"] if evaluate_test else None,
                  limitations=["Validation is development evidence, not independent confirmation" if not evaluate_test else protocol["test_scope"],
                    "Bootstrap conditions on fitted predictions; excludes training/search uncertainty",
                    "Validation intervals condition on validation-selected checkpoints and are optimistic development evidence",
                    "Revised-universe prevalence differs; old absolute lift is not a like-for-like comparator",
                    "Unadjusted intervals across fixed contrasts; no multiplicity-controlled superiority claim",
                    "Augmentation changes input width/capacity as reported; no matched zero-input capacity ablation",
                    "Primary capacity was fixed before fitting; other capacity points are secondary, not selected from TEST",
                    "Lift counts patient snapshots; repeated monthly opportunities are not unique-patient intervention counts",
                    "Input cutoff and source availability claims limited to recorded construction evidence",
                    "Augmentation comparisons are conditional on the registered training/selection procedure; no causal claim"],
                  count_reference_status="matched refit provided" if "COUNT" in scores else "not available; added-value-versus-1028 conclusion pending",
                  lightgbm_status="included" if "GB_SUMMARY" in scores else "deferred")
    save_artifacts(table, dict({"report.json": json_bytes(report)}, **{n + ".npy": array_bytes(s) for n, s in scores.items()}))
    return report


def report_table(report):
    return pd.DataFrame([dict(arm=name, split=report["split"],
        **{"lift_" + str(int(round(100*row["fraction"]))) + "pct": row["lift"] for row in values["topk"]},
        **{key: values.get(key) for key in
        ("primary_lift", "primary_lift_fraction", "snapshots", "positives", "average_precision", "roc_auc", "log_loss", "brier_score", "precision", "recall", "f1", "specificity")})
        for name, values in report["metrics"].items()])
