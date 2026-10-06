"""Post-lock validation diagnostics: association rank is not learned reliance.

These routines do not fit, tune, select a new model, or inspect TEST. Internal
parameter/gain diagnostics remain separate from measured permutation effects.
Permutation results are conditional marginal diagnostics, not causal effects.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _map(feature_map):
    frame = pd.DataFrame(feature_map).copy()
    _require({"FEATURE_INDEX", "FEATURE_NAME"} <= set(frame), "Feature map requires index and exact name.")
    _require(np.issubdtype(frame.FEATURE_INDEX.dtype, np.integer), "Feature map indices must be integers.")
    _require(np.array_equal(frame.FEATURE_INDEX.to_numpy(), np.arange(len(frame))), "Feature map is not in original tensor order.")
    _require(frame.FEATURE_NAME.map(lambda v: isinstance(v, str) and bool(v)).all() and frame.FEATURE_NAME.is_unique,
             "Feature names must be nonempty and unique.")
    return frame[["FEATURE_INDEX", "FEATURE_NAME"]].reset_index(drop=True)


def _agreement(frame, k):
    paired = frame.loc[frame.TRAIN_RANK.notna() & frame.INTERNAL_RANK.notna()]
    if len(paired) >= 2 and paired.TRAIN_RANK.nunique() > 1 and paired.INTERNAL_RANK.nunique() > 1:
        rho = float(paired.TRAIN_RANK.corr(paired.INTERNAL_RANK, method="spearman"))
    else:
        rho = None
    actual_k = min(k, len(paired))
    train_top = set(paired.sort_values(["TRAIN_RANK", "FEATURE_INDEX"]).head(actual_k).FEATURE_INDEX)
    model_top = set(paired.sort_values(["INTERNAL_RANK", "FEATURE_INDEX"]).head(actual_k).FEATURE_INDEX)
    overlap = len(train_top & model_top)
    return dict(paired_selected_features=len(paired), spearman_rank_agreement=rho,
                top_k_within_selected=actual_k, top_k_overlap=overlap,
                top_k_overlap_fraction=overlap / actual_k if actual_k else None,
                interpretation="Descriptive agreement only; disagreement is an investigation flag, never a model failure criterion.")


def join_rank_and_internal_diagnostics(feature_map, train_ranking, runs, expected_columns,
                                       expected_seeds, top_k=10):
    """Return full-universe and every-seed tables keyed by original feature ID.

    Internal magnitude is projection-column norm for a Transformer or summed
    temporal-block gain for count LightGBM. Neither is held-out decision influence.
    Config equality prevents combining different architectures or feature axes.
    """
    fm = _map(feature_map)
    columns = np.asarray(expected_columns)
    _require(columns.ndim == 1 and np.issubdtype(columns.dtype, np.integer) and len(columns) > 0,
             "Expected columns must be nonempty original integer indices.")
    _require(len(np.unique(columns)) == len(columns) and np.all((columns >= 0) & (columns < len(fm))),
             "Expected columns are duplicate or outside the map.")
    _require(np.array_equal(columns, np.sort(columns)), "Model columns must preserve original FEATURE_MAP order.")
    seeds = list(expected_seeds)
    _require(seeds and len(set(seeds)) == len(seeds) and all(type(s) is int for s in seeds), "Declare every unique prespecified seed.")
    _require(type(top_k) is int and top_k >= 1, "top_k must be a positive integer.")
    ranking = pd.DataFrame(train_ranking).copy()
    _require({"FEATURE_INDEX", "FEATURE_NAME", "RANK"} <= set(ranking), "TRAIN ranking lacks its index/name/rank identity.")
    _require(np.issubdtype(ranking.FEATURE_INDEX.dtype, np.integer), "TRAIN feature indices must be integers.")
    _require(ranking.FEATURE_INDEX.is_unique and ranking.FEATURE_NAME.is_unique, "Duplicate TRAIN ranking keys.")
    _require(ranking.FEATURE_INDEX.isin(fm.FEATURE_INDEX).all(), "TRAIN ranking contains an unknown feature index.")
    reference = fm.set_index("FEATURE_INDEX").FEATURE_NAME
    _require(ranking.FEATURE_NAME.eq(ranking.FEATURE_INDEX.map(reference)).all(), "TRAIN feature name/index mismatch.")
    ranks = pd.to_numeric(ranking.RANK, errors="coerce")
    _require(np.isfinite(ranks).all() and (ranks > 0).all(), "TRAIN ranks must be finite and positive.")
    ranking["RANK"] = ranks
    _require(set(columns) <= set(ranking.FEATURE_INDEX), "Every selected feature must have an explicit TRAIN ranking.")
    rename = {key: "TRAIN_" + key for key in ranking.columns if key not in {"FEATURE_INDEX", "FEATURE_NAME"}}
    ranked = ranking.drop(columns="FEATURE_NAME").rename(columns=rename)
    full = fm.merge(ranked, on="FEATURE_INDEX", how="left", validate="one_to_one")
    position = {int(original): i for i, original in enumerate(columns)}
    full["SELECTED_IN_MODEL"] = full.FEATURE_INDEX.isin(columns)
    full["MODEL_POSITION"] = full.FEATURE_INDEX.map(position).astype("Int64")
    full["TRAIN_RANK_AVAILABLE"] = full.TRAIN_RANK.notna()
    full["PERMUTATION_STATUS"] = np.where(full.SELECTED_IN_MODEL, "NOT_TESTED", "NOT_IN_MODEL")
    _require(len(runs) == len(seeds), "Missing seed runs; no favorable-seed subset is allowed.")
    observed_seeds = [run["summary"]["seed"] for run in runs]
    _require(sorted(observed_seeds) == sorted(seeds), "Missing or duplicate seed runs.")
    canonical_config = runs[0]["summary"]["config"]
    family = canonical_config.get("family")
    _require(family in {"transformer", "lightgbm"}, "This diagnostic uses the ordered count-model contract.")
    rows, agreements, hashes = [], [], []
    for run in sorted(runs, key=lambda run: run["summary"]["seed"]):
        summary = run["summary"]
        config = summary["config"]
        _require(summary.get("training_complete") is True, "Incomplete model run.")
        _require(config == canonical_config, "Seed configurations differ; cannot align their internal diagnostics.")
        _require(config.get("columns") == columns.tolist(), "Saved model columns do not match the locked input order.")
        _require(summary.get("config_sha256") == _hash(config), "Saved model configuration hash mismatch.")
        model_hash = summary.get("model_sha256")
        _require(isinstance(model_hash, str) and re.fullmatch(r"[0-9a-f]{64}", model_hash) is not None,
                 "Saved model identity hash is missing or invalid.")
        if "model_blob" in run:
            _require(hashlib.sha256(run["model_blob"]).hexdigest() == model_hash, "Saved model bytes/hash mismatch.")
        values = np.asarray(summary.get("aggregated_original_importance"), dtype=float)
        _require(values.shape == (len(columns),) and np.isfinite(values).all() and (values >= 0).all(),
                 "Internal diagnostic width/values do not match original selected features.")
        representation = config.get("representation")
        expected_blocks = 1 if family == "transformer" else {"flattened": 12, "windows": 4, "annual": 1}.get(representation)
        _require(expected_blocks is not None and summary.get("importance_blocks_per_original_feature") == expected_blocks,
                 "Internal diagnostic temporal-block aggregation does not match representation.")
        raw = np.asarray(summary.get("importance"), dtype=float)
        _require(raw.shape == (expected_blocks * len(columns),) and np.isfinite(raw).all() and (raw >= 0).all(),
                 "Original internal diagnostic array has wrong width/values.")
        _require(np.allclose(raw.reshape(expected_blocks, len(columns)).sum(axis=0), values, rtol=1e-10, atol=1e-12),
                 "Aggregated importance is not aligned to the original feature axis.")
        part = full.loc[full.SELECTED_IN_MODEL].sort_values("MODEL_POSITION").copy()
        part["SEED"] = int(summary["seed"])
        part["MODEL_FAMILY"] = family
        part["INTERNAL_MAGNITUDE"] = values
        part["INTERNAL_RANK"] = pd.Series(values).rank(method="average", ascending=False).to_numpy()
        part["INTERNAL_SHARE"] = values / values.sum() if values.sum() else np.zeros_like(values)
        part["INTERNAL_DIAGNOSTIC"] = "projection_column_norm" if family == "transformer" else "temporal_block_gain_sum"
        part["NOT_HELDOUT_RELIANCE"] = True
        # A TRAIN-only transformed standard deviation adjusts the raw projection
        # norm for input scale. This remains a parameter diagnostic, not reliance.
        train_sd = summary.get("transform", {}).get("training_transformed_sd") if family == "transformer" else None
        part["SCALE_AWARE_INTERNAL_AVAILABLE"] = train_sd is not None
        part["PROJECTION_NORM_X_TRAIN_SD"] = np.nan
        part["SCALE_AWARE_INTERNAL_RANK"] = np.nan
        if train_sd is not None:
            sd = np.asarray(train_sd, dtype=float)
            _require(sd.shape == (len(columns),) and np.isfinite(sd).all() and (sd >= 0).all(),
                     "TRAIN transformed SD width/values do not match selected model features.")
            scale_aware = values * sd
            _require(np.isfinite(scale_aware).all(), "Scale-aware internal magnitude overflowed.")
            part["PROJECTION_NORM_X_TRAIN_SD"] = scale_aware
            part["SCALE_AWARE_INTERNAL_RANK"] = pd.Series(scale_aware).rank(method="average", ascending=False).to_numpy()
        rows.append(part)
        agreements.append(dict(seed=int(summary["seed"]), model_family=family, **_agreement(part, top_k)))
        hashes.append(dict(seed=int(summary["seed"]), model_sha256=summary.get("model_sha256"), config_sha256=summary["config_sha256"]))
    seed_table = pd.concat(rows, ignore_index=True)
    aggregate = seed_table.groupby("FEATURE_INDEX", as_index=False).agg(
        MEAN_INTERNAL_MAGNITUDE=("INTERNAL_MAGNITUDE", "mean"), SD_INTERNAL_MAGNITUDE=("INTERNAL_MAGNITUDE", "std"),
        MEAN_INTERNAL_RANK=("INTERNAL_RANK", "mean"), SD_INTERNAL_RANK=("INTERNAL_RANK", "std"),
        MEAN_INTERNAL_SHARE=("INTERNAL_SHARE", "mean"), SEEDS_REPORTED=("SEED", "size"),
        MEAN_PROJECTION_NORM_X_TRAIN_SD=("PROJECTION_NORM_X_TRAIN_SD", "mean"),
        SD_PROJECTION_NORM_X_TRAIN_SD=("PROJECTION_NORM_X_TRAIN_SD", "std"),
        MEAN_SCALE_AWARE_INTERNAL_RANK=("SCALE_AWARE_INTERNAL_RANK", "mean"),
        SD_SCALE_AWARE_INTERNAL_RANK=("SCALE_AWARE_INTERNAL_RANK", "std"),
        SCALE_AWARE_SEEDS_REPORTED=("PROJECTION_NORM_X_TRAIN_SD", "count"))
    full = full.merge(aggregate, on="FEATURE_INDEX", how="left", validate="one_to_one")
    aggregate_agreement = _agreement(full.rename(columns={"MEAN_INTERNAL_RANK": "INTERNAL_RANK"}), top_k)
    summary = dict(model_family=family, original_feature_count=len(fm), selected_feature_count=len(columns),
                   selected_columns=columns.tolist(), selected_names=reference.loc[columns].tolist(),
                   seeds=seeds, model_identities=hashes, aggregate_agreement=aggregate_agreement,
                   permutation_measured=False, evidence="Internal diagnostics joined to TRAIN-only screening ranks; no production result implied.",
                   scale_aware_available_seeds=sorted(seed_table.loc[seed_table.SCALE_AWARE_INTERNAL_AVAILABLE, "SEED"].unique().tolist()),
                   scale_aware_definition="Projection column norm times per-feature TRAIN transformed-input SD; unavailable when absent. Internal, noncausal, not measured reliance.",
                   warning="Association can be weak while interactions are useful. Norms/gain can be large without useful held-out influence. Never force matching ranks.")
    return dict(full_table=full, seed_table=seed_table, agreement_table=pd.DataFrame(agreements), summary=summary)


def prespecify_probes(joined, *, max_features=30, random_seed=20261006,
                     top_train=10, top_internal=10, lowest_train=5, random_remaining=5,
                     correlation_pairs=None, correlation_threshold=.9, max_correlation_groups=10):
    """Fix a bounded probe union before permutation results are inspected.

    Top internal probes use mean within-seed rank, so a seed's gain scale cannot
    dominate. Source-family groups recognize DX_/DX__, etc., preserving names.
    No random feature is selected using VALIDATION labels or permutation results.
    """
    _require(all(type(n) is int and n >= 0 for n in (max_features, top_train, top_internal, lowest_train, random_remaining))
             and max_features > 0, "Probe limits must be nonnegative integers with a positive cap.")
    _require(top_train + top_internal + lowest_train + random_remaining <= max_features,
             "Probe quota sum exceeds the declared feature cap; do not silently truncate a quota.")
    _require(type(random_seed) is int and random_seed >= 0, "Invalid label-independent probe seed.")
    _require(type(max_correlation_groups) is int and 0 <= max_correlation_groups <= 10 and
             np.isfinite(correlation_threshold) and .9 <= correlation_threshold <= 1.,
             "Declare a correlation threshold >=.9 and at most ten correlation groups.")
    full = joined["full_table"]
    selected = full.loc[full.SELECTED_IN_MODEL].sort_values("MODEL_POSITION")
    columns = selected.FEATURE_INDEX.astype(int).tolist()
    names = selected.FEATURE_NAME.tolist()
    _require(columns == joined["summary"]["selected_columns"] and names == joined["summary"]["selected_names"],
             "Joined table was reordered or relabeled after the mapping audit.")
    choices = {}
    def add(indices, reason):
        for index in indices:
            choices.setdefault(int(index), []).append(reason)
    add(selected.sort_values(["TRAIN_RANK", "FEATURE_INDEX"]).head(top_train).FEATURE_INDEX, "top_train_rank")
    add(selected.sort_values(["MEAN_INTERNAL_RANK", "FEATURE_INDEX"]).head(top_internal).FEATURE_INDEX, "top_internal_rank")
    add(selected.sort_values(["TRAIN_RANK", "FEATURE_INDEX"], ascending=[False, True]).head(lowest_train).FEATURE_INDEX, "lowest_train_rank")
    remaining = sorted(set(columns) - set(choices))
    rng = np.random.default_rng(random_seed)
    random = rng.choice(remaining, size=min(random_remaining, len(remaining)), replace=False) if remaining else []
    add(sorted(random), "label_independent_random_remaining")
    _require(len(choices) <= max_features, "Probe cap exceeded.")
    index_to_position = {index: i for i, index in enumerate(columns)}
    probes = []
    for index in sorted(choices):
        pos = index_to_position[index]
        probes.append(dict(probe_id=f"feature:{index}", kind="feature", feature_indices=[index],
                           model_positions=[pos], feature_names=[names[pos]], reasons=choices[index]))
    family = {}
    for pos, name in enumerate(names):
        match = re.match(r"^(DX|PX|RX)_+", name, flags=re.IGNORECASE)
        if match:
            family.setdefault(match.group(1).upper(), []).append(pos)
    for group, positions in sorted(family.items()):
        probes.append(dict(probe_id="family:" + group, kind="family", model_positions=positions,
                           feature_indices=[columns[p] for p in positions], feature_names=[names[p] for p in positions],
                           reasons=["joint_source_family_probe; not a validated clinical interaction group"]))
    train_order = selected.sort_values(["TRAIN_RANK", "FEATURE_INDEX"]).FEATURE_INDEX.astype(int).tolist()
    for label, indices in zip(("top", "middle", "bottom"), np.array_split(train_order, 3)):
        indices = sorted(int(i) for i in indices)
        if indices:
            positions = [index_to_position[i] for i in indices]
            probes.append(dict(probe_id="train_tercile:" + label, kind="train_tercile", model_positions=positions,
                               feature_indices=indices, feature_names=[names[p] for p in positions],
                               reasons=["Selected-feature TRAIN composite-rank tercile; fixed-model joint reliance, not added predictive information."]))
    # Only TRAIN-derived pairs may be passed. Restrict the graph to selected nodes;
    # an unselected feature must not bridge two otherwise disconnected groups.
    components = []
    if correlation_pairs is not None:
        pairs = pd.DataFrame(correlation_pairs).copy()
        required = {"FEATURE_INDEX_A", "FEATURE_INDEX_B", "FEATURE_NAME_A", "FEATURE_NAME_B", "SPEARMAN"}
        _require(required <= set(pairs), "TRAIN correlation pairs lack exact feature identities or Spearman values.")
        if "SPLIT" in pairs:
            _require(pairs.SPLIT.eq("train").all(), "Correlation groups must be defined on TRAIN only.")
        reference = full.set_index("FEATURE_INDEX").FEATURE_NAME
        for side in ("A", "B"):
            index, name = "FEATURE_INDEX_" + side, "FEATURE_NAME_" + side
            _require(pairs[index].isin(reference.index).all() and pairs[name].eq(pairs[index].map(reference)).all(),
                     "TRAIN correlation name/index mismatch.")
        rho = pd.to_numeric(pairs.SPEARMAN, errors="coerce")
        _require(np.isfinite(rho).all() and rho.abs().le(1. + 1e-12).all(), "Invalid TRAIN Spearman values.")
        parents = {i:i for i in columns}
        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        edges = pairs.loc[rho.abs().ge(correlation_threshold) & pairs.FEATURE_INDEX_A.isin(columns) & pairs.FEATURE_INDEX_B.isin(columns)]
        for a, b in sorted(zip(edges.FEATURE_INDEX_A.astype(int), edges.FEATURE_INDEX_B.astype(int))):
            aa, bb = find(a), find(b)
            parents[max(aa, bb)] = min(aa, bb)
        connected = {}
        for index in columns:
            connected.setdefault(find(index), []).append(index)
        components = sorted((v for v in connected.values() if len(v) >= 2), key=lambda v:(-len(v), tuple(v)))
    component_coverage = []
    for number, indices in enumerate(components):
        positions = [index_to_position[i] for i in indices]
        included = number < max_correlation_groups
        probe_id = "train_correlation:" + str(number)
        component_coverage.append(dict(component_id=probe_id, feature_indices=indices,
                                       status="INCLUDED" if included else "OMITTED_BY_PRESPECIFIED_GROUP_CAP"))
        if included:
            probes.append(dict(probe_id=probe_id, kind="train_correlation", model_positions=positions,
                               feature_indices=indices, feature_names=[names[p] for p in positions],
                               reasons=["TRAIN-only absolute-Spearman connected component; largest groups first, original-index tie break."]))
    _require(bool(probes), "Declare at least one feature or source-family probe.")
    plan = dict(schema_version=2, selected_columns=columns, selected_names=names,
                model_family=joined["summary"]["model_family"], model_identities=joined["summary"]["model_identities"],
                random_seed=random_seed, max_individual_features=max_features,
                quotas=dict(top_train=top_train, top_internal=top_internal, lowest_train=lowest_train, random_remaining=random_remaining),
                probes=probes, individual_probe_count=len(choices), family_probe_count=len(family),
                tercile_probe_count=sum(p["kind"] == "train_tercile" for p in probes),
                correlation_probe_count=sum(p["kind"] == "train_correlation" for p in probes),
                correlation_threshold=float(correlation_threshold), max_correlation_groups=max_correlation_groups,
                correlation_pairs_supplied=correlation_pairs is not None, correlation_component_coverage=component_coverage,
                omitted_correlation_components=max(0, len(components)-max_correlation_groups),
                grouping_scope="TRAIN rankings/pairs only; pair provenance is a caller contract, not inferred from values.",
                selection_rule="Union of fixed TRAIN/internal extremes and label-independent random remainder; no permutation-result selection.",
                intended_use="Post-lock VALIDATION diagnosis only; no automatic feature or configuration changes.")
    plan["plan_sha256"] = _hash(plan)
    return plan


def _validate_plan(plan, feature_names):
    _require(plan.get("schema_version") == 2 and plan.get("plan_sha256") == _hash({k:v for k,v in plan.items() if k != "plan_sha256"}),
             "Permutation probe plan hash mismatch.")
    _require(list(feature_names) == plan["selected_names"], "Raw tensor feature names/order differ from the locked probe plan.")
    _require(len(set(feature_names)) == len(feature_names) and len(feature_names) == len(plan["selected_columns"]),
             "Invalid selected feature identity.")
    probe_ids = set()
    _require(bool(plan["probes"]), "The permutation probe plan is empty.")
    for probe in plan["probes"]:
        positions = probe["model_positions"]
        _require(probe["probe_id"] not in probe_ids and positions and len(set(positions)) == len(positions), "Duplicate/empty probe.")
        _require(all(type(p) is int and 0 <= p < len(feature_names) for p in positions), "Probe position outside model input.")
        _require(probe["feature_names"] == [feature_names[p] for p in positions] and
                 probe["feature_indices"] == [plan["selected_columns"][p] for p in positions], "Probe feature name/index/position mismatch.")
        _require(probe["kind"] in {"feature", "family", "train_tercile", "train_correlation"}, "Unrecognized prespecified probe kind.")
        probe_ids.add(probe["probe_id"])


def _probabilities(predictor, values):
    output = np.asarray(predictor(values), dtype=float)
    _require(output.shape == (len(values),) and np.isfinite(output).all() and ((output >= 0) & (output <= 1)).all(),
             "Frozen predictor must return one finite probability per input patient.")
    return output


def _lift_at_ten_percent(labels, probabilities):
    # Same ceil(top-fraction*N) rule and frozen-input-order score-tie policy as
    # comparison.py. Diagnostic unit here is latest validation patient only.
    count = max(1, int(np.ceil(.1 * len(labels))))
    top = np.argsort(-probabilities, kind="stable")[:count]
    return float(np.mean(labels[top]) / np.mean(labels))


def validation_permutation_diagnostic(X, metadata, feature_names, predictor, plan, lock_hash, *,
                                     split="validation", repeat_seeds=(271, 811, 1297),
                                     all_validation_metadata=None):
    """Whole 12-step trajectories within END_DT month; groups share permutations.

    X and metadata must contain one latest validation snapshot per patient. If
    full validation metadata is supplied, latest dates and labels are verified.
    Otherwise latest-row choice remains a documented caller precondition. The
    predictor must be the frozen saved ensemble, with no refitting/calibration.
    """
    _require(split == "validation", "TEST and TRAIN permutation diagnostics are not permitted by this API.")
    _require(isinstance(lock_hash, str) and bool(lock_hash.strip()), "A persisted nonempty decision-lock hash is required.")
    _validate_plan(plan, list(feature_names))
    seeds = tuple(repeat_seeds)
    _require(len(seeds) >= 2 and len(set(seeds)) == len(seeds) and all(type(s) is int and s >= 0 for s in seeds),
             "Declare at least two unique label-independent repeat seeds.")
    values = np.asarray(X)
    _require(values.ndim == 3 and values.shape[1:] == (12, len(feature_names)) and len(values) >= 2,
             "Permutation input must be raw (patients,12,selected_features).")
    _require(np.issubdtype(values.dtype, np.number) and not np.iscomplexobj(values), "Raw counts must be real numeric values.")
    for start in range(0, len(values), 128):
        block = values[start:start+128]
        observed = block[np.isfinite(block)]
        _require(not np.isinf(block).any() and np.all(observed >= 0) and np.all(observed == np.floor(observed)),
                 "Permutation input must preserve nonnegative integer raw counts or NaN.")
    meta = pd.DataFrame(metadata).reset_index(drop=True)
    _require(len(meta) == len(values) and {"PATIENT_ID", "END_DT", "RESP", "SPLIT"} <= set(meta), "Missing/alignment-error validation metadata.")
    _require(meta.SPLIT.eq("validation").all(), "Only frozen VALIDATION rows may be diagnosed.")
    _require(meta.PATIENT_ID.map(lambda v: isinstance(v, str) and bool(v)).all() and meta.PATIENT_ID.is_unique,
             "Supply exactly one latest validation snapshot per distinct patient.")
    dates = pd.to_datetime(meta.END_DT, errors="raise")
    _require(dates.notna().all() and dates.eq(dates.dt.normalize()).all() and dates.dt.tz is None, "Invalid snapshot dates.")
    y = meta.RESP.to_numpy()
    _require(set(np.unique(y)) == {0, 1}, "Diagnostic labels must contain both binary RESP classes.")
    latest_verified = False
    if all_validation_metadata is not None:
        all_meta = pd.DataFrame(all_validation_metadata).copy()
        _require({"PATIENT_ID", "END_DT", "RESP", "SPLIT"} <= set(all_meta) and all_meta.SPLIT.eq("validation").all(),
                 "Latest-row reference must contain only validation metadata.")
        all_meta["END_DT"] = pd.to_datetime(all_meta.END_DT, errors="raise")
        _require(all_meta.PATIENT_ID.map(lambda v: isinstance(v, str) and bool(v)).all() and
                 all_meta.END_DT.notna().all() and all_meta.END_DT.eq(all_meta.END_DT.dt.normalize()).all() and
                 all_meta.END_DT.dt.tz is None and all_meta.RESP.isin([0, 1]).all(), "Invalid latest-row reference metadata.")
        _require(not all_meta.duplicated(["PATIENT_ID", "END_DT"]).any(), "Duplicate validation reference keys.")
        latest = all_meta.sort_values(["PATIENT_ID", "END_DT"]).groupby("PATIENT_ID", sort=False).tail(1).set_index("PATIENT_ID")
        _require(set(meta.PATIENT_ID) == set(latest.index), "Latest-patient diagnostic omits validation patients.")
        reference = latest.loc[meta.PATIENT_ID]
        _require(np.array_equal(reference.END_DT.to_numpy(), dates.to_numpy()) and np.array_equal(reference.RESP.to_numpy(), y),
                 "Selected rows are not the latest frozen validation dates/labels.")
        latest_verified = True
    base = np.array(values, copy=True)
    base.flags.writeable = False
    baseline = _probabilities(predictor, base)
    baseline_ap = float(average_precision_score(y, baseline))
    baseline_lift = _lift_at_ten_percent(y, baseline)
    month = dates.dt.to_period("M").astype(str).to_numpy()
    strata = [np.flatnonzero(month == value) for value in sorted(set(month))]
    permutations = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = np.arange(len(base))
        for indices in strata:
            order[indices] = rng.permutation(indices)
        permutations[seed] = order
    repeats = []
    for probe in plan["probes"]:
        positions = probe["model_positions"]
        for seed in seeds:
            order = permutations[seed]
            changed = base.copy()
            changed[:, :, positions] = base[np.ix_(order, np.arange(12), positions)]
            before_values, after_values = base[:, :, positions], changed[:, :, positions]
            equal = (before_values == after_values) | (np.isnan(before_values) & np.isnan(after_values))
            changed_rows = int((~equal.all(axis=(1, 2))).sum())
            changed.flags.writeable = False
            predicted = _probabilities(predictor, changed)
            shuffled_ap = float(average_precision_score(y, predicted))
            repeats.append(dict(PROBE_ID=probe["probe_id"], KIND=probe["kind"], REPEAT_SEED=seed,
                                FEATURES_IN_PROBE=len(positions), BASELINE_AP=baseline_ap, PERMUTED_AP=shuffled_ap,
                                AP_DROP=baseline_ap-shuffled_ap, MEAN_ABS_PREDICTION_CHANGE=float(np.mean(np.abs(predicted-baseline))),
                                MEAN_SIGNED_PREDICTION_CHANGE=float(np.mean(predicted-baseline)),
                                BASELINE_LIFT_AT10=baseline_lift, PERMUTED_LIFT_AT10=_lift_at_ten_percent(y, predicted),
                                LIFT_AT10_DROP=baseline_lift-_lift_at_ten_percent(y, predicted),
                                PATIENTS_ACTUALLY_SHUFFLED=int((order != np.arange(len(base))).sum()),
                                MOVED_PATIENT_FRACTION=float(np.mean(order != np.arange(len(base)))),
                                RAW_VALUE_CHANGED_PATIENTS=changed_rows, RAW_VALUE_CHANGED_FRACTION=changed_rows/len(base)))
    repeat_table = pd.DataFrame(repeats)
    summary = repeat_table.groupby(["PROBE_ID", "KIND"], sort=False, as_index=False).agg(
        FEATURES_IN_PROBE=("FEATURES_IN_PROBE", "first"), BASELINE_AP=("BASELINE_AP", "first"),
        MEAN_AP_DROP=("AP_DROP", "mean"), REPEAT_SD_AP_DROP=("AP_DROP", "std"),
        BASELINE_LIFT_AT10=("BASELINE_LIFT_AT10", "first"), MEAN_LIFT_AT10_DROP=("LIFT_AT10_DROP", "mean"),
        REPEAT_SD_LIFT_AT10_DROP=("LIFT_AT10_DROP", "std"),
        MEAN_MOVED_PATIENT_FRACTION=("MOVED_PATIENT_FRACTION", "mean"),
        MEAN_RAW_VALUE_CHANGED_FRACTION=("RAW_VALUE_CHANGED_FRACTION", "mean"),
        MEAN_ABS_PREDICTION_CHANGE=("MEAN_ABS_PREDICTION_CHANGE", "mean"),
        REPEAT_SD_PREDICTION_CHANGE=("MEAN_ABS_PREDICTION_CHANGE", "std"), REPEATS=("REPEAT_SEED", "size"))
    individual = {p["feature_indices"][0]:p["probe_id"] for p in plan["probes"] if p["kind"] == "feature"}
    family_members = {index:[] for index in plan["selected_columns"]}
    for probe in plan["probes"]:
        if probe["kind"] != "feature":
            for index in probe["feature_indices"]:
                family_members[index].append(probe["probe_id"])
    coverage = pd.DataFrame(dict(FEATURE_INDEX=plan["selected_columns"], FEATURE_NAME=feature_names,
                                 MODEL_POSITION=np.arange(len(feature_names))))
    coverage["INDIVIDUALLY_TESTED"] = coverage.FEATURE_INDEX.isin(individual)
    coverage["PERMUTATION_STATUS"] = np.where(coverage.INDIVIDUALLY_TESTED, "MEASURED_VALIDATION_MARGINAL", "NOT_TESTED_UNKNOWN")
    coverage["PROBE_ID"] = coverage.FEATURE_INDEX.map(individual)
    coverage["JOINT_GROUP_PROBES"] = coverage.FEATURE_INDEX.map(lambda index:";".join(family_members[index]))
    coverage = coverage.merge(summary.loc[summary.KIND.eq("feature"), ["PROBE_ID", "MEAN_AP_DROP", "REPEAT_SD_AP_DROP", "MEAN_ABS_PREDICTION_CHANGE",
                              "MEAN_LIFT_AT10_DROP", "REPEAT_SD_LIFT_AT10_DROP", "MEAN_RAW_VALUE_CHANGED_FRACTION"]],
                              on="PROBE_ID", how="left", validate="many_to_one")
    protocol = dict(lock_hash=lock_hash, plan_sha256=plan["plan_sha256"], split="validation",
                    evaluation_unit="one_latest_snapshot_per_patient", latest_selection_verified=latest_verified,
                    comparison_scope="Secondary latest-patient diagnostic; not the primary repeated-snapshot metric population.",
                    patients=len(base), positive_patients=int(y.sum()), baseline_ap=baseline_ap, baseline_lift_at10=baseline_lift,
                    permutation_strata="END_DT_calendar_month", calendar_strata=len(strata),
                    singleton_strata=sum(len(indices) == 1 for indices in strata),
                    patients_in_singleton_strata=sum(len(indices) for indices in strata if len(indices) == 1),
                    repeat_seeds=list(seeds), predictor_calls=1+len(plan["probes"])*len(seeds),
                    full_population_reliance_measured=False, all_features_individually_tested=bool(coverage.INDIVIDUALLY_TESTED.all()),
                    fitting_performed=False, configuration_changes_allowed=False, test_rows_used=False,
                    uncertainty="Repeat SD measures permutation randomness, not a confidence interval or patient sampling uncertainty.",
                    limitation="Marginal shuffling can break correlations and create unrealistic combinations; joint groups preserve only within-group trajectories/dependence. AP drop is conditional predictive reliance, not causation or optimal feature selection.",
                    baseline_reuse="One frozen ensemble prediction reused across probes; identical label-independent patient permutations for each repeat seed.")
    return dict(probe_summary=summary, repeats=repeat_table, coverage=coverage, protocol=protocol)
