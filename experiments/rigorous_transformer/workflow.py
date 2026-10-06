"""Stage contracts, immutable private artifacts and declared search plan.

The delivery builder embeds these helpers and the previously tested warehouse
artifact codec. No repository imports are needed in the delivered notebooks.
"""
import base64
import copy
import hashlib
import io
import json
import platform
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def json_bytes(value):
    def default(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=default).encode()


def array_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def frame_bytes(frame):
    return frame.to_csv(index=False).encode()


def runtime_versions():
    import importlib.metadata
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "UNAVAILABLE"
    return dict(python=platform.python_version(), libraries=versions)


def load_count_inputs():
    """Stream saved rows into a driver memmap; preserve NaNs and original order.

    Structural/value assertions do not use held-out distributions or scores.
    Never print or export patient rows from this function.
    """
    import tempfile
    from pathlib import Path
    from pyspark.sql import functions as F
    mapping = [r.asDict() for r in read_sf("FEATURE_MAP").orderBy("FEATURE_INDEX").collect()]
    if len(mapping) != 1028:
        raise ValueError("Full 1,028 candidate universe required")
    features = [r["FEATURE_NAME"] for r in mapping]
    aliases = [r["FEATURE_COLUMN"] for r in mapping]
    if [r["FEATURE_INDEX"] for r in mapping] != list(range(1028)) or aliases != [f"F{i:04d}" for i in range(1028)]:
        raise ValueError("Unexpected FEATURE_MAP index/alias order")
    source = checked_metadata(read_sf("SNAPSHOTS").select("PATIENT_ID", "END_DT", "RESP").toPandas())
    frozen = checked_metadata(read_sf("PATIENT_SPLIT").select(
        "PATIENT_ID", "END_DT", "RESP", "SPLIT", "SPLIT_CONFIG").toPandas(), include_split=True)
    if not source.equals(frozen[["PATIENT_ID", "END_DT", "RESP"]]):
        raise ValueError("Frozen cohort key or label mismatch")
    expected = {"train": (16256, 8712, 941), "validation": (3481, 1867, 202), "test": (3414, 1868, 202)}
    for split, (n, p, positives) in expected.items():
        part = frozen[frozen.SPLIT == split]
        if (len(part), part.PATIENT_ID.nunique(), int(part.RESP.sum())) != (n, p, positives):
            raise ValueError("Frozen population changed")
    monthly = read_sf("TENSOR_MONTHLY")
    if set(monthly.columns) != set(["PATIENT_ID", "END_DT", "RESP", "TIME_STEP"] + aliases):
        raise ValueError("Tensor schema differs from FEATURE_MAP")
    # Cast to double first: do not round fractions or turn negatives into missing.
    grouped = (monthly.groupBy("PATIENT_ID", "END_DT", "RESP")
               .agg(F.collect_list(F.struct(F.col("TIME_STEP").alias("T"),
                    F.array(*[F.col(name).cast("double") for name in aliases]).alias("V"))).alias("SEQUENCE")))
    temporary = tempfile.TemporaryDirectory(prefix="tak861_counts_")
    X = np.memmap(Path(temporary.name) / "counts.f32", dtype="<f4", mode="w+", shape=(len(frozen), 12, 1028))
    positions = {(r.PATIENT_ID, r.END_DT): i for i, r in enumerate(frozen.itertuples())}
    seen = np.zeros(len(frozen), dtype=bool)
    try:
        for row in grouped.toLocalIterator(prefetchPartitions=False):
            key = (row["PATIENT_ID"], pd.Timestamp(row["END_DT"]).strftime("%Y-%m-%d"))
            if key not in positions:
                raise ValueError("Tensor contains an unexpected snapshot")
            i = positions[key]
            seq = sorted(row["SEQUENCE"], key=lambda r: r["T"])
            if seen[i] or row["RESP"] != frozen.RESP.iloc[i] or [r["T"] for r in seq] != list(range(12)):
                raise ValueError("Tensor duplicate, label mismatch or invalid timesteps")
            values = np.array([r["V"] for r in seq], dtype=np.float64)
            finite = np.isfinite(values)
            if values.shape != (12, 1028) or np.isinf(values).any() or (values[finite] < 0).any() or np.any(values[finite] != np.floor(values[finite])):
                raise ValueError("Invalid raw counts; missing counts may be NaN")
            # Exact count storage: reject float32 integer precision loss rather than silently round.
            converted = values.astype(np.float32)
            if np.any(converted[finite].astype(np.float64) != values[finite]):
                raise ValueError("Counts exceed exact float32 representation")
            X[i] = converted
            seen[i] = True
        if not seen.all():
            raise ValueError("Missing tensor snapshots")
        X.flush()
        X.flags.writeable = False
    except BaseException:
        X._mmap.close()
        temporary.cleanup()
        raise
    hashes = input_fingerprints(X, frozen, features)
    creation = json.loads(frozen.SPLIT_CONFIG.iloc[0])
    original_order_hash = hashlib.sha256(json.dumps(features, ensure_ascii=False).encode()).hexdigest()
    historical_order_match = creation.get("feature_order_sha256") == original_order_hash
    return dict(X=X, y=frozen.RESP.to_numpy(dtype=np.int64), metadata=frozen,
                features=features, feature_map=pd.DataFrame(mapping), hashes=hashes,
                indices={s: np.flatnonzero(frozen.SPLIT.to_numpy() == s) for s in expected},
                temporary_directory=temporary, split_creation=creation,
                historical_split_feature_hash_matches_current=historical_order_match)


def release_count_inputs(data):
    if data and isinstance(data.get("X"), np.memmap):
        data["X"]._mmap.close()
        data["temporary_directory"].cleanup()


def preparation_summary(data, implementation_hash):
    summary = data["metadata"].groupby("SPLIT").agg(snapshots=("RESP", "size"),
                         patients=("PATIENT_ID", "nunique"), positives=("RESP", "sum")).reset_index()
    return dict(format_version=1, input_hashes=data["hashes"], tensor_shape=list(data["X"].shape),
                feature_map=data["feature_map"].to_dict("records"), cohort=summary.to_dict("records"),
                implementation_hash=implementation_hash,
                historical_split_feature_hash_matches_current=data["historical_split_feature_hash_matches_current"],
                orientation="0 newest: recovered declaration; historical execution still unresolved",
                coverage="All 12 saved positions retained. Zero does not establish observation or padding.",
                provenance="Reproduced structural checks on saved inputs, not event-time validation",
                historical_test="Previously inspected; any future evaluation is retrospective",
                value_contract="nonnegative integer or missing; no infinities; exact float32 storage")


def verify_stage(data, report, implementation_hash):
    if report["input_hashes"] != data["hashes"] or report["implementation_hash"] != implementation_hash:
        raise ValueError("Inputs/code changed since prior stage; use a new run ID and rerun in order")


def build_protocol(candidate_indices, candidate_tags, input_hashes, implementation_hash):
    """Grid resources are chosen before model validation; k comes from TRAIN shortlist."""
    subsets = {k: sorted(map(int, v)) for k, v in candidate_indices.items()}
    if "all_eligible" not in subsets or not subsets["all_eligible"]:
        raise ValueError("All-eligible control is mandatory")
    for values in subsets.values():
        if not values or len(values) != len(set(values)) or not set(values) <= set(subsets["all_eligible"]):
            raise ValueError("Invalid feature shortlist")
    if any(name not in subsets for name in candidate_tags.values()):
        raise ValueError("Candidate tags reference unavailable TRAIN subsets")
    # Phase-B subset is not selected by the screening proxy. Its family-specific
    # count anchor is resolved from complete phase-A seed means under this rule.
    default_subset = "__PHASE_A_FAMILY_ANCHOR__"
    pairs = []
    for name in sorted(subsets):
        pairs.append(("count_" + name, name, {}, {}))
    # One changed factor per Transformer arm; each tabular fit is separately declared.
    pairs.extend([
        ("capacity", default_subset, dict(d_model=128, depth=2, ff=256), dict(num_leaves=31)),
        ("dropout", default_subset, dict(dropout=.4), dict(colsample_bytree=.6)),
        ("decay", default_subset, dict(weight_decay=.01), dict(reg_lambda=10.)),
        ("sqrt_weight", default_subset, dict(balance="sqrt"), dict(balance="sqrt")),
        ("full_weight", default_subset, dict(balance="full"), dict(balance="full")),
        ("projection_l1", default_subset, dict(projection_l1=1e-5), dict(reg_alpha=.1)),
        ("explicit_l2", default_subset, dict(weight_decay=0., explicit_l2=1e-4), dict(min_child_samples=100)),
        ("attention_pool", default_subset, dict(pooling="attention"), dict(learning_rate=.015)),
        ("scale", default_subset, dict(scale=True), dict(representation="windows")),
        ("quarterly", default_subset, dict(representation="quarterly"), dict(representation="annual")),
        ("patient_weight", default_subset, dict(weighting="patient"), dict(weighting="patient")),
        # All-eligible prevents the step-0 sensitivity depending on a label-ranked subset.
        ("without_step0", "all_eligible", dict(representation="without_step0"), dict(drop_step0=True)),
    ])
    candidates = {}
    for arm, subset, tf_changes, gb_changes in pairs:
        declared_columns = subsets[subset] if subset in subsets else None
        tf = dict(family="transformer", representation="monthly", subset=subset, columns=declared_columns,
                  d_model=64, depth=1, heads=4, ff=128, dropout=.2, pooling="mean", scale=False,
                  weight_decay=.001, explicit_l2=0., projection_l1=0., balance="none",
                  weighting="snapshot", learning_rate=.0003, batch_size=256)
        gb = dict(family="lightgbm", representation="flattened", subset=subset, columns=declared_columns,
                  scale=False, num_leaves=15, min_child_samples=50, colsample_bytree=.8, subsample=.8,
                  reg_lambda=1., reg_alpha=0., balance="none", weighting="snapshot", learning_rate=.03,
                  n_estimators=600)
        tf.update(tf_changes)
        gb.update(gb_changes)
        candidates["tf_"+arm], candidates["gb_"+arm] = tf, gb
    candidates = {name: candidates[name] for name in sorted(candidates)}
    phase_a = [name for name in sorted(candidates) if name.startswith(("tf_count_", "gb_count_"))]
    phase_b = [name for name in sorted(candidates) if name not in phase_a]
    return dict(protocol_version=3, search_stage="design", seeds=[42, 142, 242], candidates=candidates,
                feature_subsets=subsets, candidate_tags=dict(candidate_tags), phase_a=phase_a, phase_b=phase_b,
                count_anchor_rule="Within each family, highest mean VALIDATION AP across every prespecified seed; exact ties prefer fewer features, then lower mean complexity, then candidate name",
                phase_b_rule="Resolve each family's ablation subset from its phase-A count winner; without_step0 stays all-eligible. Persist hashed resolution before any phase-B fit",
                artifact_candidate_order=sorted(candidates),
                prediction_counts={"train":16256, "validation":3481},
                epochs=30, patience=5, cpu_threads=8, device="cpu",
                reliance_diagnostic=dict(max_individual_features=30, top_train=10, top_internal=10,
                    lowest_train=5, random_remaining=5, probe_seed=20261006, repeat_seeds=[271,811,1297],
                    split="validation", population="latest snapshot per patient", strata="END_DT calendar month",
                    correlation_threshold=.9, max_correlation_groups=10,
                    groups=["selected TRAIN ranking terciles", "DX", "PX", "RX"], selection_changes=False),
                input_hashes=input_hashes, implementation_hash=implementation_hash,
                objective="Max mean per-seed frozen VALIDATION average precision, exact ties prefer simpler",
                threshold_rule="F1 maximized on ensemble VALIDATION; top10% budget rule separately reported",
                evaluation_estimand="Snapshot-level primary; latest-patient secondary; patient-cluster uncertainty",
                final_predictor="Uniform average of all three prespecified seed predictions; never seed selection",
                historical_test="Previously inspected: retrospective final evaluation only",
                upstream_provenance="Unresolved; no leakage-free certification from these experiments",
                budget="Same number of candidate fits and seeds per family; actual validation calls/seconds reported",
                restrictions=["Proxy scores shortlist only", "No OOT inference without complete inputs and provenance",
                              "Validation selection uncertainty is descriptive, not independent confirmation",
                              "Dropping step0 is sensitivity, not a proven fix for all source leakage"])


def candidate_views(data, cfg, split):
    rows = data["indices"][split]
    # Slice rows first: callers never pass held-out rows during development.
    x = np.asarray(data["X"][rows])[:, :, cfg["columns"]].copy()
    if cfg.get("drop_step0", False):
        x[:, 0, :] = 0
    return x, data["y"][rows], data["metadata"].iloc[rows].PATIENT_ID.to_numpy()


def run_artifact_table(run_prefix, candidate_index, seed):
    if (not re.fullmatch(r"[A-Z][A-Z0-9_]*", run_prefix)
            or type(candidate_index) is not int or candidate_index < 0
            or type(seed) is not int or seed < 0):
        raise ValueError("Invalid run artifact identity")
    return f"{run_prefix}_C{candidate_index:03d}_S{seed}"


def _payload_hash(payload):
    return hashlib.sha256(payload).hexdigest()


def _validate_run(result, contract, expected_lengths=None, expected_config=None):
    """Validate both count and encoded-49 run contracts without model unpickling."""
    summary = result["summary"]
    if summary.get("training_complete") is not True or summary.get("seed") != contract.get("seed"):
        raise ValueError("Incomplete run or wrong seed")
    if (summary.get("model_sha256") != _payload_hash(result["model_blob"])
            or summary.get("config_sha256") != _payload_hash(json_bytes(summary.get("config")))
            or summary.get("transform_sha256") != _payload_hash(json_bytes(summary.get("transform")))):
        raise ValueError("Model/config/transform hash mismatch")
    if "protocol_hash" in summary and summary["protocol_hash"] != contract.get("protocol_hash"):
        raise ValueError("Run summary has another protocol identity")
    if "input_hashes" in contract and summary.get("input_hashes") != contract["input_hashes"]:
        raise ValueError("Run input identity mismatch")
    if expected_config is not None:
        config = summary["config"]
        if any(key not in config or json_bytes(config[key]) != json_bytes(value)
               for key, value in expected_config.items()):
            raise ValueError("Saved configuration differs from the declared candidate")
        if ("candidate_config_sha256" in contract and
                contract["candidate_config_sha256"] != _payload_hash(json_bytes(expected_config))):
            raise ValueError("Candidate configuration hash mismatch")
    shape = summary.get("raw_input_shape")
    if (not isinstance(shape, list) or len(shape) not in (1, 2)
            or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError("Missing or invalid raw input shape")
    if not isinstance(result["history"], list) or not result["history"]:
        raise ValueError("A completed run requires its training history")
    lengths = expected_lengths if expected_lengths is not None else contract.get("prediction_counts")
    for split in ("train", "validation"):
        values = np.asarray(result[split+"_scores"])
        if (values.ndim != 1 or len(values) == 0 or not np.issubdtype(values.dtype, np.number)
                or np.iscomplexobj(values) or not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()):
            raise ValueError("Saved predictions must be nonempty aligned finite probability vectors")
        if lengths is not None and (split not in lengths or len(values) != lengths[split]):
            raise ValueError("Saved prediction length does not match the frozen split")
        metric = summary.get(split+"_metrics", {}).get("average_precision")
        if metric is None or not np.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError("Missing or invalid completed-run average precision")


def pack_run(result, contract):
    _validate_run(result, contract)
    artifacts = {"summary.json": json_bytes(result["summary"]), "history.json": json_bytes(result["history"]),
            "model.bin": result["model_blob"], "train.npy": array_bytes(result["train_scores"]),
            "validation.npy": array_bytes(result["validation_scores"])}
    envelope = dict(format_version=2, contract=contract,
                    payload_sha256={name:_payload_hash(value) for name,value in artifacts.items()})
    artifacts["contract.json"] = json_bytes(envelope)
    return artifacts


RUN_ARTIFACT_NAMES = {"summary.json", "history.json", "model.bin", "train.npy", "validation.npy", "contract.json"}


def unpack_run(artifacts, contract, expected_lengths=None, expected_config=None):
    if set(artifacts) != RUN_ARTIFACT_NAMES:
        raise ValueError("Incomplete run artifact bundle")
    envelope = json.loads(artifacts["contract.json"])
    if envelope.get("format_version") != 2 or envelope.get("contract") != contract:
        raise ValueError("Run contract changed; never mix trials from different inputs/protocols")
    expected_payloads = RUN_ARTIFACT_NAMES - {"contract.json"}
    hashes = envelope.get("payload_sha256", {})
    if set(hashes) != expected_payloads or any(hashes[name] != _payload_hash(artifacts[name]) for name in expected_payloads):
        raise ValueError("Run artifact payload hash mismatch")
    result = dict(summary=json.loads(artifacts["summary.json"]), history=json.loads(artifacts["history.json"]),
                model_blob=artifacts["model.bin"], train_scores=np.load(io.BytesIO(artifacts["train.npy"]), allow_pickle=False),
                validation_scores=np.load(io.BytesIO(artifacts["validation.npy"]), allow_pickle=False))
    _validate_run(result, contract, expected_lengths, expected_config)
    return result


def _candidate_names(protocol):
    names = sorted(protocol["candidates"])
    seeds = protocol["seeds"]
    if not names or not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Declare nonempty candidates and unique valid seeds")
    if protocol.get("artifact_candidate_order", names) != names:
        raise ValueError("Artifact candidate order must be the deterministic sorted order")
    return names


def _run_contract(protocol, name, seed):
    design = protocol.get("design_protocol", protocol)
    contract = dict(protocol_hash=stable_hash(design), candidate=name, seed=seed,
                candidate_config_sha256=_payload_hash(json_bytes(protocol["candidates"][name])),
                input_hashes=protocol["input_hashes"], implementation_hash=protocol["implementation_hash"],
                prediction_counts=protocol["prediction_counts"])
    if protocol.get("protocol_version") == 3:
        if name in design["phase_a"]:
            contract["phase"] = "A"
        elif name in design["phase_b"] and protocol.get("search_stage") == "resolved":
            contract.update(phase="B", resolution_sha256=protocol["resolution_sha256"])
        else:
            raise ValueError("Phase-B fit requires a persisted resolved anchor")
    return contract


def _execute_candidates(data, protocol, run_prefix, requested_names):
    """Internal phase executor; candidate table indices never depend on phase order."""
    if protocol["input_hashes"] != data["hashes"]:
        raise ValueError("Protocol input hash mismatch")
    lengths = {split:len(data["indices"][split]) for split in ("train", "validation")}
    if protocol.get("prediction_counts") != lengths:
        raise ValueError("Protocol population lengths differ from frozen inputs")
    all_names = _candidate_names(protocol)
    if len(requested_names) != len(set(requested_names)) or not set(requested_names) <= set(all_names):
        raise ValueError("Requested phase candidates differ from declared candidates")
    results = {}
    for name in requested_names:
        number = all_names.index(name)
        cfg = protocol["candidates"][name]
        runs = []
        for seed in protocol["seeds"]:
            table = run_artifact_table(run_prefix, number, seed)
            contract = _run_contract(protocol, name, seed)
            if table_exists(table):
                result = unpack_run(read_artifacts(table, RUN_ARTIFACT_NAMES), contract, lengths, cfg)
            else:
                xtr, ytr, groups = candidate_views(data, cfg, "train")
                xv, yv, validation_groups = candidate_views(data, cfg, "validation")
                result = train_candidate(xtr, ytr, groups, xv, yv, cfg, seed,
                           epochs=protocol["epochs"], patience=protocol["patience"],
                           threads=protocol["cpu_threads"], device=protocol["device"],
                           groups_validation=validation_groups)
                result["summary"]["protocol_hash"] = contract["protocol_hash"]
                result["summary"]["input_hashes"] = data["hashes"]
                _validate_run(result, contract, lengths, cfg)
                save_artifacts(table, pack_run(result, contract))
                del xtr, xv
            runs.append(result)
            print(name, "seed", seed, "VAL AP", round(result["summary"]["validation_metrics"]["average_precision"], 6),
                  "seconds", round(result["summary"]["wall_seconds"], 1), flush=True)
        results[name] = runs
    return results


def execute_search(data, protocol, run_prefix):
    """Legacy single-phase executor. Version 3 must use the registered two phases."""
    if protocol.get("protocol_version") == 3:
        raise ValueError("Use execute_two_phase_search for the registered count-anchor design")
    return _execute_candidates(data, protocol, run_prefix, _candidate_names(protocol))


def _validate_design(design):
    if design.get("protocol_version") != 3 or design.get("search_stage") != "design" or "design_protocol" in design:
        raise ValueError("Expected immutable version-3 DESIGN protocol")
    names = _candidate_names(design)
    a, b = design.get("phase_a", []), design.get("phase_b", [])
    if (not a or not b or a != sorted(set(a)) or b != sorted(set(b))
            or set(a) & set(b) or set(a) | set(b) != set(names)):
        raise ValueError("Every candidate must belong to exactly one declared phase")
    subsets = design.get("feature_subsets", {})
    if not subsets.get("all_eligible"):
        raise ValueError("All-eligible feature control is required")
    by_family = {}
    for name in a:
        cfg = design["candidates"][name]
        family, subset = cfg.get("family"), cfg.get("subset")
        if family not in {"transformer", "lightgbm"} or subset not in subsets or cfg.get("columns") != subsets[subset]:
            raise ValueError("Phase A requires explicit TRAIN-derived count subsets")
        by_family.setdefault(family, []).append(subset)
    if set(by_family) != {"transformer", "lightgbm"} or any(sorted(v) != sorted(subsets) for v in by_family.values()):
        raise ValueError("Both families require exactly every prespecified phase-A count arm")
    for family in by_family:
        settings = [{k:v for k,v in design["candidates"][name].items() if k not in {"columns", "subset"}}
                    for name in a if design["candidates"][name]["family"] == family]
        if any(settings[0] != other for other in settings[1:]):
            raise ValueError("Phase-A count arms must share each family's base configuration")
    b_counts = {family: 0 for family in by_family}
    for name in b:
        cfg = design["candidates"][name]
        family = cfg.get("family")
        if family not in b_counts:
            raise ValueError("Unknown phase-B family")
        b_counts[family] += 1
        if name.endswith("without_step0"):
            if cfg.get("subset") != "all_eligible" or cfg.get("columns") != subsets["all_eligible"]:
                raise ValueError("Step-0 sensitivity must retain all eligible features")
        elif cfg.get("subset") != "__PHASE_A_FAMILY_ANCHOR__" or cfg.get("columns") is not None:
            raise ValueError("Phase-B ablations must be unresolved family-anchor templates")
    if len(set(b_counts.values())) != 1:
        raise ValueError("Both families require equal phase-B candidate and seed budgets")


def _verify_saved_design(design, run_prefix):
    _validate_design(design)
    if not table_exists(run_prefix + "_PROTOCOL"):
        raise ValueError("Persist the immutable DESIGN protocol before phase-A fitting")
    saved = read_artifacts(run_prefix + "_PROTOCOL", {"protocol.json", "runtime.json"})
    if json.loads(saved["protocol.json"]) != design:
        raise ValueError("Saved DESIGN differs; use a new run identifier")


def resolve_count_anchors(design, phase_a_results):
    """Pure resolution: complete seed means, no TEST, no CI-based equivalence rule."""
    _validate_design(design)
    if set(phase_a_results) != set(design["phase_a"]):
        raise ValueError("Cannot resolve count anchors from an incomplete phase A")
    rows, evidence = [], []
    for name in design["phase_a"]:
        cfg = design["candidates"][name]
        runs = phase_a_results[name]
        if sorted(r["summary"]["seed"] for r in runs) != sorted(design["seeds"]):
            raise ValueError("Every prespecified seed must complete exactly once before anchor resolution")
        values, complexity = [], []
        for result in sorted(runs, key=lambda r: r["summary"]["seed"]):
            summary = result["summary"]
            contract = _run_contract(design, name, summary["seed"])
            _validate_run(result, contract, expected_config=cfg)
            values.append(float(summary["validation_metrics"]["average_precision"]))
            size = summary.get("parameter_count")
            if size is None or not np.isfinite(size) or size < 0:
                raise ValueError("Finite model complexity is required for declared exact-tie handling")
            complexity.append(float(size))
            evidence.append(dict(candidate=name, seed=summary["seed"], summary_sha256=_payload_hash(json_bytes(summary)),
                                 model_sha256=summary["model_sha256"],
                                 validation_predictions_sha256=_payload_hash(array_bytes(result["validation_scores"]))))
        rows.append(dict(candidate=name, family=cfg["family"], subset=cfg["subset"],
                         columns=cfg["columns"], n_features=len(cfg["columns"]),
                         seed_validation_ap=values, mean_validation_ap=float(np.mean(values)),
                         mean_complexity=float(np.mean(complexity))))
    anchors = {}
    for family in ("transformer", "lightgbm"):
        eligible = [row for row in rows if row["family"] == family]
        winner = sorted(eligible, key=lambda r: (-r["mean_validation_ap"], r["n_features"],
                                               r["mean_complexity"], r["candidate"]))[0]
        anchors[family] = copy.deepcopy(winner)
    return dict(resolution_version=1, design_sha256=stable_hash(design),
                rule=design["count_anchor_rule"], anchors=anchors, phase_a_summary=rows,
                phase_a_evidence=evidence, selection_split="validation", test_used=False,
                interpretation="Registered adaptive phase-B anchor; VALIDATION is development evidence, not independent confirmation")


def _resolved_protocol(design, resolution):
    if resolution.get("design_sha256") != stable_hash(design):
        raise ValueError("Anchor resolution belongs to another DESIGN")
    resolved = copy.deepcopy(design)
    for name in design["phase_b"]:
        cfg = resolved["candidates"][name]
        if name.endswith("without_step0"):
            continue
        anchor = resolution["anchors"][cfg["family"]]
        cfg["subset"], cfg["columns"] = anchor["subset"], list(anchor["columns"])
    resolved.update(search_stage="resolved", design_protocol=copy.deepcopy(design),
                    design_sha256=stable_hash(design), resolution_sha256=stable_hash(resolution),
                    count_anchor_resolution=copy.deepcopy(resolution))
    return resolved


def _read_candidates(protocol, run_prefix, names):
    results = {}
    all_names = _candidate_names(protocol)
    for name in names:
        i = all_names.index(name)
        results[name] = [unpack_run(read_artifacts(run_artifact_table(run_prefix, i, seed), RUN_ARTIFACT_NAMES),
                        _run_contract(protocol, name, seed), expected_config=protocol["candidates"][name])
                        for seed in protocol["seeds"]]
    return results


def read_resolved_protocol(design, run_prefix):
    """Recompute anchors from hashed phase-A artifacts, then verify saved resolution."""
    _verify_saved_design(design, run_prefix)
    phase_a_results = _read_candidates(design, run_prefix, design["phase_a"])
    expected = resolve_count_anchors(design, phase_a_results)
    stored = json.loads(read_artifacts(run_prefix + "_ANCHORS", {"resolution.json"})["resolution.json"])
    if stored != expected:
        raise ValueError("Saved anchor resolution differs from complete phase-A evidence")
    return _resolved_protocol(design, stored)


def execute_two_phase_search(data, design, run_prefix):
    """Run count arms, freeze anchors, then run ablations on each family's winner."""
    _verify_saved_design(design, run_prefix)
    phase_a_results = _execute_candidates(data, design, run_prefix, design["phase_a"])
    resolution = resolve_count_anchors(design, phase_a_results)
    table = run_prefix + "_ANCHORS"
    if table_exists(table):
        existing = json.loads(read_artifacts(table, {"resolution.json"})["resolution.json"])
        if existing != resolution:
            raise ValueError("Existing count anchors differ; never overwrite resolution or mix phase-B trials")
    else:
        # This immutable write must finish before any phase-B model sees data.
        save_artifacts(table, {"resolution.json": json_bytes(resolution)})
    resolved = read_resolved_protocol(design, run_prefix)
    phase_b_results = _execute_candidates(data, resolved, run_prefix, design["phase_b"])
    combined = {**phase_a_results, **phase_b_results}
    return resolved, {name: combined[name] for name in _candidate_names(resolved)}


def read_search(protocol, run_prefix):
    if protocol.get("protocol_version") == 3:
        if protocol.get("search_stage") != "resolved" or "design_protocol" not in protocol:
            raise ValueError("Load read_resolved_protocol(DESIGN, prefix) before reading the two-phase search")
        recovered = read_resolved_protocol(protocol["design_protocol"], run_prefix)
        if recovered != protocol:
            raise ValueError("Resolved protocol differs from saved phase-A evidence")
    return _read_candidates(protocol, run_prefix, _candidate_names(protocol))


def ensemble_scores(runs, split):
    if not runs or split not in {"train", "validation"}:
        raise ValueError("Development ensembles require complete TRAIN/VALIDATION runs")
    seeds = [run["summary"]["seed"] for run in runs]
    if len(set(seeds)) != len(seeds) or any(run["summary"].get("training_complete") is not True for run in runs):
        raise ValueError("Ensemble contains duplicate seeds or incomplete runs")
    arrays = [r[split+"_scores"] for r in runs]
    if (len({a.shape for a in arrays}) != 1 or
            any(a.ndim != 1 or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any() for a in arrays)):
        raise ValueError("Seed score alignment mismatch")
    return np.mean(arrays, axis=0)
