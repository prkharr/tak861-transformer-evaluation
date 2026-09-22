"""Build the two input-audit notebooks for the temporal feature-selection experiment.

The saved V63 tensor and patient split are the inputs. This module deliberately
does not reconstruct missing raw-claims SQL or invent new split assignments.
"""

from textwrap import dedent


EARLY_HELPERS_SOURCE = dedent(r'''
REFERENCE_TRAINING_ARTIFACT_NAMES = {
    "checkpoint.pt", "training_summary.json", "training_history.csv", "training_history.png"
}


def bind_report_to_reference(report, artifacts, reference_run_id, stage):
    """Bind a source audit to the original completed run without loading PyTorch."""
    if (not isinstance(reference_run_id, str)
            or not re.fullmatch(r"[A-Z][A-Z0-9_]*", reference_run_id)):
        raise ValueError("REFERENCE_RUN_ID must be the original saved uppercase run identifier.")
    if not isinstance(artifacts, dict) or set(artifacts) != REFERENCE_TRAINING_ARTIFACT_NAMES:
        raise ValueError("The original reference run must contain its exact four training artifacts.")
    try:
        summary = json.loads(artifacts["training_summary.json"])
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("The original reference training summary is not valid JSON.") from error
    if (not isinstance(summary, dict) or summary.get("training_complete") is not True
            or summary.get("run_id") != reference_run_id):
        raise ValueError("The reference training run is incomplete or belongs to another RUN_ID.")
    hashes = summary.get("input_hashes")
    expected_hashes = {"model_input_sha256", "snapshot_manifest_sha256",
                       "feature_names_sha256", "split_config_sha256"}
    if (not isinstance(hashes, dict) or set(hashes) != expected_hashes
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in hashes.values())):
        raise ValueError("The original reference summary lacks valid full input fingerprints.")
    checkpoint = artifacts["checkpoint.pt"]
    if not isinstance(checkpoint, bytes) or not checkpoint:
        raise ValueError("The original reference checkpoint is missing or empty.")
    stages = {
        "preparation": ("reuse_and_validate_saved_v63_tensor",
                        ("model_input_sha256", "feature_names_sha256")),
        "split": ("reuse_original_patient_split",
                  ("snapshot_manifest_sha256", "split_config_sha256", "feature_names_sha256")),
    }
    if stage not in stages or not isinstance(report, dict) or report.get("mode") != stages[stage][0]:
        raise ValueError("Unexpected source audit stage for reference verification.")
    for field in stages[stage][1]:
        if report.get(field) != hashes[field]:
            raise ValueError(
                f"Current saved inputs differ from original {reference_run_id} on {field}. "
                "Restore the original saved inputs; matching population counts alone are insufficient.")
    bound = dict(report)
    bound["reference_run_id"] = reference_run_id
    bound["reference_checkpoint_sha256"] = hashlib.sha256(checkpoint).hexdigest()
    bound["reference_input_hashes"] = dict(hashes)
    bound["checks"] = dict(report.get("checks", {}), matches_original_reference_run=True)
    return bound


def validate_feature_mapping(mapping):
    if not mapping or [r["FEATURE_INDEX"] for r in mapping] != list(range(len(mapping))):
        raise ValueError("Invalid feature indices in the saved FEATURE_MAP.")
    features = [r["FEATURE_NAME"] for r in mapping]
    aliases = [r["FEATURE_COLUMN"] for r in mapping]
    if (aliases != [f"F{i:04d}" for i in range(len(features))]
            or not all(isinstance(f, str) and f for f in features)
            or len(set(features)) != len(features)):
        raise ValueError("Invalid saved feature names, aliases or order.")
    return features, aliases


def original_snapshot_hash(metadata):
    records = [[str(r.PATIENT_ID), str(r.END_DT), int(r.RESP)]
               for r in metadata.itertuples()]
    return digest_json(records)


def validate_original_population(metadata, features):
    observed = (len(metadata), int(metadata.PATIENT_ID.nunique()),
                int(metadata.RESP.sum()), len(features))
    if observed != (23151, 12447, 1345, 1028):
        raise ValueError(
            "This handoff reuses the completed V63 cohort; "
            f"snapshots/patients/positives/features changed to {observed}.")


def make_preparation_report(X, metadata, features, source_prefix):
    # fill_tensor has already validated all rows, labels, timesteps and counts.
    validate_original_population(metadata, features)
    if tuple(X.shape) != (len(metadata), 12, len(features)):
        raise ValueError("The saved model-input shape must be snapshots x 12 x features.")
    tensor_hash = hashlib.sha256()
    tensor_hash.update(canonical_json(list(X.shape)).encode("utf-8"))
    zero_months = 0
    zero_snapshots = 0
    for start in range(0, len(X), 128):
        block = np.asarray(X[start:start + 128], dtype="<f4")
        tensor_hash.update(block.tobytes(order="C"))
        empty_months = np.all(block == 0, axis=2)
        zero_months += int(empty_months.sum())
        zero_snapshots += int(np.all(empty_months, axis=1).sum())
    groups = {}
    for feature in features:
        group = feature.split("__", 1)[0] if "__" in feature else "OTHER"
        groups[group] = groups.get(group, 0) + 1
    return {
        "report_version": 1,
        "mode": "reuse_and_validate_saved_v63_tensor",
        "source_prefix": source_prefix,
        "source_tables": [source_prefix + "_" + suffix
                          for suffix in ("FEATURE_MAP", "SNAPSHOTS", "TENSOR_MONTHLY")],
        "n_snapshots": len(metadata),
        "n_patients": int(metadata.PATIENT_ID.nunique()),
        "n_positive_snapshots": int(metadata.RESP.sum()),
        "n_negative_snapshots": int(len(metadata) - metadata.RESP.sum()),
        "n_features": len(features),
        "n_timesteps": 12,
        "n_monthly_rows": int(X.shape[0] * X.shape[1]),
        "tensor_shape": list(X.shape),
        "model_input_dtype": "little_endian_float32",
        "minimum_snapshot_end_date": str(metadata.END_DT.min()),
        "maximum_snapshot_end_date": str(metadata.END_DT.max()),
        "zero_activity_months": zero_months,
        "zero_activity_snapshots": zero_snapshots,
        "feature_group_counts": groups,
        "model_input_sha256": tensor_hash.hexdigest(),
        "source_snapshots_sha256": original_snapshot_hash(metadata),
        "feature_names_sha256": digest_json(features),
        "checks": {
            "exact_saved_population": True,
            "unique_patient_date_keys": True,
            "exact_binary_labels": True,
            "exact_12_unique_timesteps": True,
            "complete_snapshot_coverage": True,
            "finite_nonnegative_counts": True,
            "zero_activity_sequences_preserved": True,
        },
        "scope": "Saved tensor validation only; raw claims and mappings were not regenerated.",
        "feature_selection": "Deferred to TRAIN-only processing in notebook 03.",
    }


def make_split_audit_report(metadata, creation, features, source_prefix):
    # validate_metadata_pair has already checked source coverage and all split rules.
    validate_original_population(metadata, features)
    expected = {
        "train": (16256, 8712, 941),
        "validation": (3481, 1867, 202),
        "test": (3414, 1868, 202),
    }
    summary = []
    patient_sets = {}
    for split_name in ("train", "validation", "test"):
        part = metadata.loc[metadata.SPLIT == split_name]
        observed = (len(part), int(part.PATIENT_ID.nunique()), int(part.RESP.sum()))
        if observed != expected[split_name]:
            raise ValueError(
                f"Saved {split_name} split differs from RUN_001: {observed}. "
                "Restore the original manifest; this notebook does not reshuffle patients.")
        patient_sets[split_name] = set(part.PATIENT_ID)
        summary.append({
            "split": split_name,
            "snapshots": observed[0],
            "patients": observed[1],
            "positive_snapshots": observed[2],
            "negative_snapshots": observed[0] - observed[2],
            "positive_fraction": observed[2] / observed[0],
            "minimum_snapshot_end_date": str(part.END_DT.min()),
            "maximum_snapshot_end_date": str(part.END_DT.max()),
        })
    overlaps = {
        "train_validation": len(patient_sets["train"] & patient_sets["validation"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "validation_test": len(patient_sets["validation"] & patient_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError("A patient appears in more than one split.")
    records = [[str(r.PATIENT_ID), str(r.END_DT), int(r.RESP), str(r.SPLIT)]
               for r in metadata.itertuples()]
    return {
        "report_version": 1,
        "mode": "reuse_original_patient_split",
        "source_prefix": source_prefix,
        "source_table": source_prefix + "_PATIENT_SPLIT",
        "split_summary": summary,
        "patient_overlap_counts": overlaps,
        "source_snapshots_sha256": original_snapshot_hash(metadata),
        "snapshot_manifest_sha256": digest_json(records),
        "feature_names_sha256": digest_json(features),
        "split_config_sha256": digest_json(creation),
        "saved_split_creation": creation,
        "new_assignments_created": False,
        "checks": {
            "source_keys_and_labels_match": True,
            "both_classes_in_each_split": True,
            "patient_overlap_zero": True,
            "original_split_counts_match": True,
            "feature_order_and_timesteps_match": True,
        },
    }


def compare_preparation_to_split(preparation_report, split_report):
    for report in (preparation_report, split_report):
        if (not report.get("reference_run_id") or not report.get("reference_checkpoint_sha256")
                or not isinstance(report.get("reference_input_hashes"), dict)):
            raise ValueError("Both source audits must be bound to the original reference training run.")
    for field in ("source_prefix", "source_snapshots_sha256", "feature_names_sha256",
                  "reference_run_id", "reference_checkpoint_sha256", "reference_input_hashes"):
        if preparation_report.get(field) != split_report.get(field):
            raise ValueError(f"Stage 01 preparation and frozen split disagree on {field}.")
    if preparation_report.get("mode") != "reuse_and_validate_saved_v63_tensor":
        raise ValueError("Unexpected preparation report; run notebook 01 from this delivery.")
    return True
''').strip()


def _cell(number, title, source):
    return f"# CELL {number} — {title}\n" + dedent(source).strip() + "\n"


def early_notebook_cells(connection_source, input_helpers_source):
    """Return eight self-contained code cells for each of the saved-input stages."""
    connection = connection_source.strip()
    common_helpers = input_helpers_source.strip() + "\n\n" + EARLY_HELPERS_SOURCE
    first = [
        _cell(1, "Connection and saved-input preparation scope", connection + '\n\n' + dedent('''
            PREPARATION_TABLE = EXPERIMENT_PREFIX + "_PREPARATION"
            REFERENCE_MODEL_TABLE = PREFIX + "_MODEL_" + REFERENCE_RUN_ID
            print("Reuse and validate the completed V63 tensor; no raw claims rebuild.")
            print("Original FEATURE_MAP, SNAPSHOTS and TENSOR_MONTHLY tables stay unchanged.")
            print("Feature reduction is fitted on TRAIN only in notebook 03.")
        ''')),
        _cell(2, "Input contracts and aggregate-report helpers", common_helpers),
        _cell(3, "Load the saved feature map and source snapshots", '''
            for suffix in ("FEATURE_MAP", "SNAPSHOTS", "TENSOR_MONTHLY"):
                if not table_exists(PREFIX + "_" + suffix):
                    raise FileNotFoundError(
                        f"Missing completed V63 input: {PREFIX}_{suffix}. "
                        "Restore the saved input; this notebook does not reconstruct raw claims.")
            mapping = read_sf("FEATURE_MAP").orderBy("FEATURE_INDEX").collect()
            features, aliases = validate_feature_mapping(mapping)
            metadata = checked_metadata(read_sf("SNAPSHOTS").select(
                "PATIENT_ID", "END_DT", "RESP").toPandas())
            validate_original_population(metadata, features)
            print(f"Saved inputs: {len(metadata):,} snapshots; "
                  f"{metadata.PATIENT_ID.nunique():,} patients; {len(features):,} features.")
        '''),
        _cell(4, "Validate all saved monthly sequences without needing a split", '''
            from pyspark.sql import functions as F
            if "X" in globals() and hasattr(X, "_mmap") and not X._mmap.closed:
                X._mmap.close()
            if "temporary" in globals():
                temporary.cleanup()
            monthly = read_sf("TENSOR_MONTHLY")
            if set(monthly.columns) != set(["PATIENT_ID", "END_DT", "RESP", "TIME_STEP"] + aliases):
                raise ValueError("Monthly table columns differ from the saved feature map.")
            grouped = (monthly.groupBy("PATIENT_ID", "END_DT", "RESP")
                .agg(F.collect_list(F.struct(
                    F.col("TIME_STEP").alias("T"),
                    F.array(*[F.when(F.col(name) >= 0, F.col(name).cast("float"))
                              .otherwise(F.lit(None).cast("float")) for name in aliases]).alias("V")
                )).alias("SEQUENCE"))
                .repartition(128))
            print("Validating the full tensor in a temporary driver file (about 1.06 GiB).", flush=True)
            temporary, X = new_tensor_buffer(metadata, len(features))
            try:
                fill_tensor(X, metadata, grouped.toLocalIterator(prefetchPartitions=False))
                X.flags.writeable = False
            except BaseException:
                X._mmap.close()
                temporary.cleanup()
                raise
        '''),
        _cell(5, "Fingerprint the full tensor and build a deterministic audit report", '''
            try:
                preparation_report = make_preparation_report(X, metadata, features, PREFIX)
                if not table_exists(REFERENCE_MODEL_TABLE):
                    raise FileNotFoundError(
                        f"Missing original saved model {REFERENCE_MODEL_TABLE}. "
                        "Restore the completed RUN_001 training artifacts before preparing new experiments.")
                reference_artifacts = read_artifacts(REFERENCE_MODEL_TABLE, REFERENCE_TRAINING_ARTIFACT_NAMES)
                preparation_report = bind_report_to_reference(
                    preparation_report, reference_artifacts, REFERENCE_RUN_ID, "preparation")
            except BaseException:
                X._mmap.close()
                temporary.cleanup()
                raise
            preparation_artifacts = {
                "preparation_report.json": canonical_json(preparation_report).encode("utf-8")
            }
            print("Complete tensor coverage, labels, finite counts and 12 timesteps verified.")
            print("Tensor values and feature order match the original saved", REFERENCE_RUN_ID)
            print("Snapshot dates are prediction cutoffs, not patient enrolment dates.")
        '''),
        _cell(6, "Inspect aggregate data coverage", '''
            display(pd.DataFrame([{
                "snapshots": preparation_report["n_snapshots"],
                "patients": preparation_report["n_patients"],
                "positive_snapshots": preparation_report["n_positive_snapshots"],
                "features": preparation_report["n_features"],
                "timesteps": preparation_report["n_timesteps"],
                "minimum_END_DT": preparation_report["minimum_snapshot_end_date"],
                "maximum_END_DT": preparation_report["maximum_snapshot_end_date"],
                "zero_activity_snapshots": preparation_report["zero_activity_snapshots"],
                "zero_activity_months": preparation_report["zero_activity_months"],
            }]))
            display(pd.DataFrame([
                {"feature_group": name, "feature_count": count}
                for name, count in sorted(preparation_report["feature_group_counts"].items())
            ]))
            print("No patient IDs or individual feature values are saved in this audit report.")
        '''),
        _cell(7, "Save the preparation report in the new experiment namespace", '''
            save_artifacts(PREPARATION_TABLE, preparation_artifacts)
            # An identical existing report is verified; differing artifacts are never overwritten.
        '''),
        _cell(8, "Verify saved preparation and release the temporary tensor", '''
            try:
                if read_artifacts(PREPARATION_TABLE, preparation_artifacts) != preparation_artifacts:
                    raise ValueError("Preparation report read-back mismatch.")
                print("Preparation saved and verified. Continue with notebook 02.")
            finally:
                X._mmap.close()
                temporary.cleanup()
                del X
        '''),
    ]
    second = [
        _cell(1, "Connection and frozen patient split", connection + '\n\n' + dedent('''
            PREPARATION_TABLE = EXPERIMENT_PREFIX + "_PREPARATION"
            SPLIT_AUDIT_TABLE = EXPERIMENT_PREFIX + "_SPLIT_AUDIT"
            REFERENCE_MODEL_TABLE = PREFIX + "_MODEL_" + REFERENCE_RUN_ID
            print("Reuse the original patient assignments from the completed RUN_001.")
            print("This notebook validates and audits the split; it never reshuffles patients.")
        ''')),
        _cell(2, "Input contracts and split-audit helpers", common_helpers),
        _cell(3, "Load source metadata and the original saved split", '''
            for suffix in ("FEATURE_MAP", "SNAPSHOTS", "PATIENT_SPLIT"):
                if not table_exists(PREFIX + "_" + suffix):
                    raise FileNotFoundError(
                        f"Missing original {PREFIX}_{suffix}. Restore the completed V63 inputs; "
                        "new split assignments are not created by this handoff.")
            mapping = read_sf("FEATURE_MAP").orderBy("FEATURE_INDEX").collect()
            features, aliases = validate_feature_mapping(mapping)
            source = read_sf("SNAPSHOTS").select("PATIENT_ID", "END_DT", "RESP").toPandas()
            frozen = read_sf("PATIENT_SPLIT").select(
                "PATIENT_ID", "END_DT", "RESP", "SPLIT", "SPLIT_CONFIG").toPandas()
        '''),
        _cell(4, "Verify the exact frozen manifest and aggregate counts", '''
            metadata, creation = validate_metadata_pair(source, frozen, features)
            split_report = make_split_audit_report(metadata, creation, features, PREFIX)
            print("Snapshot keys, labels, patient separation and saved split settings verified.")
        '''),
        _cell(5, "Check that the split matches the stage 01 preparation", '''
            if not table_exists(PREPARATION_TABLE):
                raise FileNotFoundError("Run and save notebook 01 preparation before this split audit.")
            preparation_report = json.loads(read_artifacts(
                PREPARATION_TABLE, ["preparation_report.json"])["preparation_report.json"])
            if not table_exists(REFERENCE_MODEL_TABLE):
                raise FileNotFoundError(
                    f"Missing original saved model {REFERENCE_MODEL_TABLE}. "
                    "Restore the completed RUN_001 training artifacts before auditing its frozen split.")
            reference_artifacts = read_artifacts(REFERENCE_MODEL_TABLE, REFERENCE_TRAINING_ARTIFACT_NAMES)
            split_report = bind_report_to_reference(split_report, reference_artifacts, REFERENCE_RUN_ID, "split")
            compare_preparation_to_split(preparation_report, split_report)
            split_artifacts = {"split_audit.json": canonical_json(split_report).encode("utf-8")}
            print("The split uses the exact source snapshot population and feature order audited in stage 01.")
            print("Patient assignments and split settings match the original saved", REFERENCE_RUN_ID)
        '''),
        _cell(6, "Inspect frozen split totals", '''
            display(pd.DataFrame(split_report["split_summary"]))
            display(pd.DataFrame([split_report["patient_overlap_counts"]]))
            print("All snapshots of a patient stay in the same split. Patient overlap = 0.")
            print("This is the original patient split; it is not a new temporal holdout.")
            print("Select features and model settings with TRAIN/VALIDATION only in notebook 03.")
        '''),
        _cell(7, "Save the split audit in the new experiment namespace", '''
            save_artifacts(SPLIT_AUDIT_TABLE, split_artifacts)
        '''),
        _cell(8, "Verify saved audit before training experiments", '''
            if read_artifacts(SPLIT_AUDIT_TABLE, split_artifacts) != split_artifacts:
                raise ValueError("Split audit read-back mismatch.")
            print("Frozen patient split audit saved and verified. Continue with notebook 03.")
            print("Keep the original TEST out of feature selection and experiment selection.")
        '''),
    ]
    return {
        "01_tensor_initialization_DL_POC.ipynb": first,
        "02_patient_level_split_DL_POC.ipynb": second,
    }
