"""Contract checks for saved-tensor reuse and the unchanged patient split."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from early_stages import EARLY_HELPERS_SOURCE, early_notebook_cells


INPUT_SOURCE = (Path(__file__).parent / "input_helpers.py").read_text(encoding="utf-8")


@pytest.fixture
def helpers():
    namespace = {}
    exec(INPUT_SOURCE, namespace)
    exec(EARLY_HELPERS_SOURCE, namespace)
    return namespace


@pytest.fixture
def original_cohort():
    records = []
    for split, snapshots, patients, positives in [
        ("train", 16256, 8712, 941),
        ("validation", 3481, 1867, 202),
        ("test", 3414, 1868, 202),
    ]:
        for index in range(snapshots):
            patient = f"{split}_{index % patients:05d}"
            cutoff = "2023-06-21" if index < patients else "2025-05-21"
            records.append([patient, cutoff, int(index < positives), split])
    features = sorted(
        [f"RX__R{i}" for i in range(42)]
        + [f"DX__D{i}" for i in range(490)]
        + [f"PX__P{i}" for i in range(496)]
    )
    creation = {
        "feature_order_sha256": hashlib.sha256(
            json.dumps(features, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "n_timesteps": 12,
        "random_state": 42,
    }
    manifest = pd.DataFrame(records, columns=["PATIENT_ID", "END_DT", "RESP", "SPLIT"])
    manifest["SPLIT_CONFIG"] = json.dumps(creation)
    source = manifest[["PATIENT_ID", "END_DT", "RESP"]].copy()
    return source, manifest, creation, features


def test_all_generated_cells_compile_and_do_not_embed_credentials():
    cells_by_name = early_notebook_cells(
        'sf_options_dl_poc = {}\nPREFIX = "TAK861_TX_READY_V63_DL_POC"\n'
        'EXPERIMENT_PREFIX = PREFIX + "_TEMPORAL_SELECTION_V1"\nREFERENCE_RUN_ID = "RUN_001"',
        INPUT_SOURCE,
    )
    assert len(cells_by_name) == 2
    for filename, cells in cells_by_name.items():
        assert len(cells) == 8
        for number, cell in enumerate(cells, 1):
            assert cell.startswith(f"# CELL {number} ")
            compile(cell, f"{filename}:cell{number}", "exec")
    assert "train_test_split" not in "\n".join(cells_by_name["02_patient_level_split_DL_POC.ipynb"])


def test_mapping_rejects_reordering_and_duplicate_names(helpers):
    valid = [
        {"FEATURE_INDEX": 0, "FEATURE_NAME": "RX__a", "FEATURE_COLUMN": "F0000"},
        {"FEATURE_INDEX": 1, "FEATURE_NAME": "DX__b", "FEATURE_COLUMN": "F0001"},
    ]
    assert helpers["validate_feature_mapping"](valid) == (["RX__a", "DX__b"], ["F0000", "F0001"])
    with pytest.raises(ValueError, match="indices"):
        helpers["validate_feature_mapping"](list(reversed(valid)))
    duplicate = [valid[0], dict(valid[1], FEATURE_NAME="RX__a")]
    with pytest.raises(ValueError, match="names"):
        helpers["validate_feature_mapping"](duplicate)
    incorrect_alias = [valid[0], dict(valid[1], FEATURE_COLUMN="F0002")]
    with pytest.raises(ValueError, match="aliases"):
        helpers["validate_feature_mapping"](incorrect_alias)


def test_split_audit_preserves_exact_assignments_and_matches_saved_hash_contract(helpers, original_cohort):
    source, manifest, creation, features = original_cohort
    original = manifest.copy(deep=True)
    metadata, saved_creation = helpers["validate_metadata_pair"](source, manifest, features)
    report = helpers["make_split_audit_report"](metadata, saved_creation, features, "ORIGINAL")
    pd.testing.assert_frame_equal(manifest, original)
    assert report["new_assignments_created"] is False
    assert report["patient_overlap_counts"] == {
        "train_validation": 0, "train_test": 0, "validation_test": 0
    }
    assert [r["positive_snapshots"] for r in report["split_summary"]] == [941, 202, 202]
    reference = helpers["input_fingerprints"](np.zeros((len(metadata), 1, 1)), metadata, features)
    for name in ("snapshot_manifest_sha256", "feature_names_sha256", "split_config_sha256"):
        assert report[name] == reference[name]
    assert report["saved_split_creation"] == creation
    # The artifact contains totals and hashes, never individual patient IDs.
    serialized = json.dumps(report)
    assert "train_00000" not in serialized
    assert "PATIENT_ID" not in serialized


def test_source_label_change_rejected_before_split_audit(helpers, original_cohort):
    source, manifest, _, features = original_cohort
    source.loc[0, "RESP"] = 0
    with pytest.raises(ValueError, match="keys or labels"):
        helpers["validate_metadata_pair"](source, manifest, features)


def test_changed_split_counts_rejected(helpers, original_cohort):
    source, manifest, _, features = original_cohort
    patient = manifest.loc[0, "PATIENT_ID"]
    manifest.loc[manifest.PATIENT_ID == patient, "SPLIT"] = "validation"
    metadata, creation = helpers["validate_metadata_pair"](source, manifest, features)
    with pytest.raises(ValueError, match="differs from RUN_001"):
        helpers["make_split_audit_report"](metadata, creation, features, "ORIGINAL")


def test_patient_overlap_rejected(helpers, original_cohort):
    source, manifest, _, features = original_cohort
    manifest.loc[0, "SPLIT"] = "validation"
    with pytest.raises(ValueError, match="Patient overlap"):
        helpers["validate_metadata_pair"](source, manifest, features)


def test_preparation_hash_covers_all_counts_and_keeps_zero_activity(helpers):
    # Small shapes exercise the hashing/report logic independently of the fixed V63 count gate.
    helpers["validate_original_population"] = lambda metadata, features: None
    metadata = pd.DataFrame({
        "PATIENT_ID": ["a", "b"],
        "END_DT": ["2023-06-21", "2025-05-21"],
        "RESP": [0, 1],
        "SPLIT": ["train", "validation"],
        "SPLIT_CONFIG": ["{}", "{}"],
    })
    features = ["RX__a", "DX__b", "PX__c"]
    X = np.zeros((2, 12, 3), dtype=np.float32)
    X[1, 11, 2] = 2
    report = helpers["make_preparation_report"](X, metadata, features, "ORIGINAL")
    reference = helpers["input_fingerprints"](X, metadata, features)
    assert report["model_input_sha256"] == reference["model_input_sha256"]
    assert report["zero_activity_snapshots"] == 1
    assert report["zero_activity_months"] == 23
    assert report["feature_group_counts"] == {"RX": 1, "DX": 1, "PX": 1}
    assert report["minimum_snapshot_end_date"] == "2023-06-21"
    assert report["maximum_snapshot_end_date"] == "2025-05-21"
    assert report == helpers["make_preparation_report"](X, metadata, features, "ORIGINAL")
    X[1, 11, 2] = 3
    changed = helpers["make_preparation_report"](X, metadata, features, "ORIGINAL")
    assert changed["model_input_sha256"] != report["model_input_sha256"]


def test_original_population_gate_is_enforced(helpers, original_cohort):
    source, _, _, features = original_cohort
    metadata = helpers["checked_metadata"](source)
    helpers["validate_original_population"](metadata, features)
    with pytest.raises(ValueError, match="changed"):
        helpers["validate_original_population"](metadata.iloc[:-1], features)
    with pytest.raises(ValueError, match="changed"):
        helpers["validate_original_population"](metadata, features[:-1])


def test_preparation_and_split_must_match(helpers):
    prep = {
        "mode": "reuse_and_validate_saved_v63_tensor", "source_prefix": "SAVED",
        "source_snapshots_sha256": "abc", "feature_names_sha256": "def",
        "reference_run_id": "RUN_001", "reference_checkpoint_sha256": "checkpoint-hash",
        "reference_input_hashes": {"model_input_sha256": "tensor-hash"},
    }
    split = dict(prep)
    assert helpers["compare_preparation_to_split"](prep, split)
    for field in ("source_prefix", "source_snapshots_sha256", "feature_names_sha256",
                  "reference_run_id", "reference_checkpoint_sha256"):
        changed = dict(split, **{field: "changed"})
        with pytest.raises(ValueError, match=field):
            helpers["compare_preparation_to_split"](prep, changed)


def reference_bundle(helpers):
    hashes = {
        "model_input_sha256": "1" * 64, "snapshot_manifest_sha256": "2" * 64,
        "feature_names_sha256": "3" * 64, "split_config_sha256": "4" * 64,
    }
    summary = {"run_id": "RUN_001", "training_complete": True, "input_hashes": hashes}
    artifacts = {
        "checkpoint.pt": b"original numeric checkpoint bytes",
        "training_summary.json": helpers["canonical_json"](summary).encode("utf-8"),
        "training_history.csv": b"epoch,validation_ap\n6,0.109\n",
        "training_history.png": b"original png bytes",
    }
    return hashes, summary, artifacts


def test_source_audits_are_bound_to_the_same_original_run(helpers):
    hashes, _, artifacts = reference_bundle(helpers)
    preparation = {
        "mode": "reuse_and_validate_saved_v63_tensor", "source_prefix": "SAVED",
        "source_snapshots_sha256": "5" * 64, **hashes,
    }
    split = dict(preparation, mode="reuse_original_patient_split")
    first = helpers["bind_report_to_reference"](preparation, artifacts, "RUN_001", "preparation")
    second = helpers["bind_report_to_reference"](split, artifacts, "RUN_001", "split")
    assert first["reference_checkpoint_sha256"] == hashlib.sha256(artifacts["checkpoint.pt"]).hexdigest()
    assert first["reference_input_hashes"] == hashes
    assert first["checks"]["matches_original_reference_run"] is True
    assert helpers["compare_preparation_to_split"](first, second)
    assert "reference_run_id" not in preparation  # Pure helper does not mutate its input.
    changed_artifacts = dict(artifacts, **{"checkpoint.pt": b"different saved checkpoint"})
    changed = helpers["bind_report_to_reference"](split, changed_artifacts, "RUN_001", "split")
    with pytest.raises(ValueError, match="reference_checkpoint_sha256"):
        helpers["compare_preparation_to_split"](first, changed)


@pytest.mark.parametrize("stage,mode,field", [
    ("preparation", "reuse_and_validate_saved_v63_tensor", "model_input_sha256"),
    ("preparation", "reuse_and_validate_saved_v63_tensor", "feature_names_sha256"),
    ("split", "reuse_original_patient_split", "snapshot_manifest_sha256"),
    ("split", "reuse_original_patient_split", "split_config_sha256"),
    ("split", "reuse_original_patient_split", "feature_names_sha256"),
])
def test_reference_hash_mismatch_rejected_even_when_counts_are_unchanged(helpers, stage, mode, field):
    hashes, _, artifacts = reference_bundle(helpers)
    report = {"mode": mode, "n_snapshots": 23151, "n_patients": 12447, **hashes}
    report[field] = "a" * 64
    with pytest.raises(ValueError, match=field):
        helpers["bind_report_to_reference"](report, artifacts, "RUN_001", stage)


@pytest.mark.parametrize("change", ["incomplete", "wrong_run", "missing_hash", "malformed_hash"])
def test_reference_summary_must_be_complete_and_identifiable(helpers, change):
    hashes, summary, artifacts = reference_bundle(helpers)
    report = {"mode": "reuse_and_validate_saved_v63_tensor", **hashes}
    if change == "incomplete":
        summary["training_complete"] = False
    elif change == "wrong_run":
        summary["run_id"] = "RUN_002"
    elif change == "missing_hash":
        del summary["input_hashes"]["snapshot_manifest_sha256"]
    else:
        summary["input_hashes"]["feature_names_sha256"] = "not-a-sha256"
    artifacts["training_summary.json"] = helpers["canonical_json"](summary).encode("utf-8")
    with pytest.raises(ValueError):
        helpers["bind_report_to_reference"](report, artifacts, "RUN_001", "preparation")


def test_reference_artifact_names_and_checkpoint_are_required(helpers):
    hashes, _, artifacts = reference_bundle(helpers)
    report = {"mode": "reuse_and_validate_saved_v63_tensor", **hashes}
    missing = dict(artifacts)
    del missing["training_history.png"]
    with pytest.raises(ValueError, match="exact four"):
        helpers["bind_report_to_reference"](report, missing, "RUN_001", "preparation")
    empty = dict(artifacts, **{"checkpoint.pt": b""})
    with pytest.raises(ValueError, match="empty"):
        helpers["bind_report_to_reference"](report, empty, "RUN_001", "preparation")
