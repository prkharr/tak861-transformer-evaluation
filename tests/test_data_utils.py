"""Independent synthetic checks of tensor construction and frozen patient splits."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_utils  # noqa: E402


KEYS = ["PATIENT_ID", "END_DT"]
FEATURES = ["RX__SYNTH_A", "DX__SYNTH_B", "PX__SYNTH_C"]
SOURCE_REVIEW = {
    "population_version": "V63",
    "claims_vintage": "20260825",
    "value_representation": "raw_counts",
    "feature_order_verified": True,
    "snapshot_semantics_verified": True,
    "predictor_cutoff_verified": True,
    "outcome_events_excluded": True,
    "review_reference": "Synthetic test fixture, not clinical validation",
}


def make_bundle(n_patients=80):
    """Two snapshots each; alternate patients have one positive snapshot."""
    keys, labels = [], []
    for patient in range(n_patients):
        for snapshot, date in enumerate(["2025-01-31", "2025-02-28"]):
            keys.append((f"SYNTHETIC_{patient:04d}", date))
            labels.append(int(patient % 2 == 0 and snapshot == 1))
    n = len(keys)
    counts = (np.arange(n * 12 * 3).reshape(n, 12, 3) % 7).astype(np.float32)
    counts[:, 3, :] = 0  # A true zero-activity month must remain in the tensor.
    return data_utils.SequenceBundle(
        X=counts,
        y=np.array(labels, dtype=np.int64),
        snapshots=pd.DataFrame(keys, columns=KEYS),
        feature_names=FEATURES.copy(),
        time_steps=tuple(range(12)),
    )


def make_monthly(bundle):
    records = []
    for i, key in bundle.snapshots.iterrows():
        for step in bundle.time_steps:
            record = {"PATIENT_ID": key.PATIENT_ID, "END_DT": key.END_DT,
                      "RESP": int(bundle.y[i]), "TIME_STEP": step}
            record.update(dict(zip(bundle.feature_names, bundle.X[i, step].tolist())))
            records.append(record)
    return pd.DataFrame.from_records(records)


def reorder(bundle, order):
    return replace(bundle, X=bundle.X[order].copy(), y=bundle.y[order].copy(),
                   snapshots=bundle.snapshots.iloc[order].reset_index(drop=True))


def assert_same_bundle(actual, expected):
    np.testing.assert_array_equal(actual.X, expected.X)
    np.testing.assert_array_equal(actual.y, expected.y)
    assert list(actual.feature_names) == list(expected.feature_names)
    assert tuple(actual.time_steps) == tuple(expected.time_steps)
    pd.testing.assert_frame_equal(actual.snapshots[KEYS].reset_index(drop=True),
                                  expected.snapshots[KEYS].reset_index(drop=True),
                                  check_dtype=False)


def test_valid_bundle_preserves_counts_order_labels_and_zero_months():
    original = make_bundle()
    checked = data_utils.validate_bundle(original)
    assert_same_bundle(checked, original)
    assert checked.X.shape == (160, 12, 3)
    assert np.all(checked.X[:, 3, :] == 0)


def test_canonical_keys_with_pandas_string_dtype_are_valid():
    bundle = make_bundle()
    string_keys = bundle.snapshots.astype({"PATIENT_ID": "string", "END_DT": "string"})
    checked = data_utils.validate_bundle(replace(bundle, snapshots=string_keys))
    assert_same_bundle(checked, bundle)


@pytest.mark.parametrize("bad_count", [-1, np.nan, np.inf, -np.inf])
def test_negative_or_nonfinite_raw_counts_rejected(bad_count):
    bundle = make_bundle()
    bundle.X[0, 0, 0] = bad_count
    with pytest.raises(ValueError):
        data_utils.validate_bundle(bundle)


@pytest.mark.parametrize("bad_label", [-1, 2, np.nan, "yes"])
def test_nonbinary_or_missing_labels_rejected(bad_label):
    bundle = make_bundle()
    labels = bundle.y.astype(object)
    labels[0] = bad_label
    with pytest.raises(ValueError):
        data_utils.validate_bundle(replace(bundle, y=labels))


def test_label_and_identity_lengths_must_match_tensor_rows():
    bundle = make_bundle()
    for changed in [replace(bundle, y=bundle.y[:-1]),
                    replace(bundle, snapshots=bundle.snapshots.iloc[:-1].copy())]:
        with pytest.raises(ValueError):
            data_utils.validate_bundle(changed)


def test_duplicate_snapshot_keys_rejected_but_repeated_patients_are_valid():
    bundle = make_bundle()
    assert bundle.snapshots.PATIENT_ID.nunique() == 80
    data_utils.validate_bundle(bundle)
    keys = bundle.snapshots.copy()
    keys.loc[1, KEYS] = keys.loc[0, KEYS].to_numpy()
    with pytest.raises(ValueError):
        data_utils.validate_bundle(replace(bundle, snapshots=keys))


@pytest.mark.parametrize("bad_identity", [None, 12, "", " leading_space"])
def test_missing_nonstring_or_invalid_patient_keys_rejected(bad_identity):
    bundle = make_bundle()
    keys = bundle.snapshots.copy()
    keys.loc[0, "PATIENT_ID"] = bad_identity
    with pytest.raises(ValueError):
        data_utils.validate_bundle(replace(bundle, snapshots=keys))


@pytest.mark.parametrize("features", [
    ["RX__SYNTH_A", "RX__SYNTH_A", "PX__SYNTH_C"],
    ["AGE", "DX__SYNTH_B", "PX__SYNTH_C"],
    ["RX__SYNTH_A", "DX__SYNTH_B"],
])
def test_feature_schema_must_be_unique_prefixed_and_match_tensor_width(features):
    with pytest.raises(ValueError):
        data_utils.validate_bundle(replace(make_bundle(), feature_names=features))


@pytest.mark.parametrize("steps", [tuple(range(11)), tuple(reversed(range(12))),
                                  tuple(range(1, 13)), (0,) * 12])
def test_timestep_metadata_must_match_contiguous_zero_based_tensor_axis(steps):
    with pytest.raises(ValueError):
        data_utils.validate_bundle(replace(make_bundle(), time_steps=steps))


def test_tensor_must_have_three_dimensions():
    bundle = make_bundle()
    with pytest.raises(ValueError):
        data_utils.validate_bundle(replace(bundle, X=bundle.X.reshape(160, -1)))


def test_expected_population_shape_is_enforced():
    bundle = make_bundle()
    data_utils.validate_bundle(bundle, expected_shape=(160, 12, 3))
    with pytest.raises(ValueError):
        data_utils.validate_bundle(bundle, expected_shape=(159, 12, 3))


def test_monthly_reshape_uses_explicit_feature_order_and_sorted_keys():
    expected = make_bundle()
    monthly = make_monthly(expected).sample(frac=1, random_state=9)
    actual = data_utils.bundle_from_monthly(monthly, feature_names=FEATURES, expected_steps=12)
    assert_same_bundle(actual, expected)
    assert np.all(actual.X[:, 3, :] == 0)


def test_monthly_default_features_are_sorted_and_values_follow_that_order():
    expected = make_bundle()
    monthly = make_monthly(expected)
    monthly["IGNORED_METADATA"] = "synthetic-only"
    actual = data_utils.bundle_from_monthly(monthly.sample(frac=1, random_state=10))
    names = sorted(FEATURES)
    assert list(actual.feature_names) == names
    feature_positions = [FEATURES.index(name) for name in names]
    np.testing.assert_array_equal(actual.X, expected.X[:, :, feature_positions])
    np.testing.assert_array_equal(actual.y, expected.y)


def test_monthly_categorical_patient_keys_do_not_create_phantom_snapshots():
    expected = reorder(make_bundle(n_patients=2), np.array([0, 3]))
    monthly = make_monthly(expected)
    monthly["PATIENT_ID"] = pd.Categorical(
        monthly.PATIENT_ID,
        categories=["SYNTHETIC_0001", "SYNTHETIC_0000", "UNUSED_PATIENT"],
    )
    actual = data_utils.bundle_from_monthly(monthly.sample(frac=1, random_state=5), feature_names=FEATURES)
    np.testing.assert_array_equal(actual.X, expected.X)
    np.testing.assert_array_equal(actual.y, expected.y)
    pd.testing.assert_frame_equal(actual.snapshots.astype({"PATIENT_ID": object}), expected.snapshots)


@pytest.mark.parametrize("complex_value,object_column", [
    (1 + 2j, False), (1 + 2j, True), (np.complex64(1 + 2j), True),
])
def test_monthly_complex_counts_rejected_before_lossy_float_conversion(complex_value, object_column):
    monthly = make_monthly(make_bundle(n_patients=2))
    monthly[FEATURES[0]] = monthly[FEATURES[0]].astype(object if object_column else complex)
    monthly.loc[0, FEATURES[0]] = complex_value
    with pytest.raises(ValueError, match="real numeric raw counts"):
        data_utils.bundle_from_monthly(monthly)


def test_monthly_missing_zero_activity_month_rejected_instead_of_silent_padding():
    monthly = make_monthly(make_bundle())
    monthly = monthly.drop(index=3)  # Counts are all zero, but the key/month is still required.
    with pytest.raises(ValueError):
        data_utils.bundle_from_monthly(monthly)


def test_monthly_duplicate_snapshot_timestep_rejected():
    monthly = make_monthly(make_bundle())
    monthly = pd.concat([monthly, monthly.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        data_utils.bundle_from_monthly(monthly)


@pytest.mark.parametrize("bad_step", [-1, 12, 0.5, np.nan])
def test_monthly_invalid_timestep_rejected(bad_step):
    monthly = make_monthly(make_bundle())
    monthly["TIME_STEP"] = monthly.TIME_STEP.astype(float)
    monthly.loc[0, "TIME_STEP"] = bad_step
    with pytest.raises(ValueError):
        data_utils.bundle_from_monthly(monthly)


def test_monthly_inconsistent_responses_within_snapshot_rejected():
    monthly = make_monthly(make_bundle())
    monthly.loc[0, "RESP"] = 1 - monthly.loc[0, "RESP"]
    with pytest.raises(ValueError):
        data_utils.bundle_from_monthly(monthly)


@pytest.mark.parametrize("column", ["PATIENT_ID", "END_DT", "RESP", "TIME_STEP"])
def test_monthly_missing_required_columns_rejected(column):
    monthly = make_monthly(make_bundle()).drop(columns=column)
    with pytest.raises(ValueError):
        data_utils.bundle_from_monthly(monthly)


def test_frozen_split_is_patient_disjoint_stratified_and_complete(tmp_path):
    bundle = make_bundle()
    manifest = data_utils.freeze_patient_split(bundle, tmp_path / "split.csv", seed=42)
    assert len(manifest) == 160
    assert set(manifest.columns) == {"PATIENT_ID", "END_DT", "RESP", "SPLIT"}
    assert not manifest.duplicated(KEYS).any()
    assert manifest.groupby("PATIENT_ID").SPLIT.nunique().eq(1).all()
    assert manifest.groupby("SPLIT").PATIENT_ID.nunique().to_dict() == {
        "TRAIN": 56, "VALIDATION": 12, "TEST": 12}
    ever_positive = manifest.groupby(["SPLIT", "PATIENT_ID"]).RESP.max()
    assert ever_positive.groupby(level=0).sum().to_dict() == {
        "TRAIN": 28, "VALIDATION": 6, "TEST": 6}
    expected_labels = bundle.snapshots.assign(RESP=bundle.y).sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(manifest[KEYS + ["RESP"]].sort_values(KEYS).reset_index(drop=True),
                                  expected_labels, check_dtype=False)


def test_new_split_is_invariant_to_aligned_input_row_order(tmp_path):
    bundle = make_bundle()
    order = np.random.default_rng(13).permutation(len(bundle.y))
    first = data_utils.freeze_patient_split(bundle, tmp_path / "first.csv", seed=42)
    second = data_utils.freeze_patient_split(reorder(bundle, order), tmp_path / "second.csv", seed=42)
    pd.testing.assert_frame_equal(first.sort_values(KEYS).reset_index(drop=True),
                                  second.sort_values(KEYS).reset_index(drop=True))


def test_existing_split_reused_despite_changed_seed_and_requested_fractions(tmp_path):
    bundle = make_bundle()
    path = tmp_path / "frozen.csv"
    first = data_utils.freeze_patient_split(bundle, path, seed=42)
    original_bytes = path.read_bytes()
    second = data_utils.freeze_patient_split(bundle, path, seed=999, train_fraction=0.6,
                                             validation_fraction=0.2, test_fraction=0.2)
    pd.testing.assert_frame_equal(first.sort_values(KEYS).reset_index(drop=True),
                                  second.sort_values(KEYS).reset_index(drop=True), check_dtype=False)
    assert path.read_bytes() == original_bytes


def test_frozen_split_reload_preserves_patient_ids_that_look_like_csv_missing_values(tmp_path):
    bundle = make_bundle()
    replacements = dict(zip(bundle.snapshots.PATIENT_ID.unique()[:4], ["NA", "NULL", "N/A", "NaN"]))
    keys = bundle.snapshots.copy()
    keys["PATIENT_ID"] = keys.PATIENT_ID.replace(replacements)
    bundle = replace(bundle, snapshots=keys)
    path = tmp_path / "split.csv"
    original = data_utils.freeze_patient_split(bundle, path)
    reloaded = data_utils.freeze_patient_split(bundle, path)
    pd.testing.assert_frame_equal(original, reloaded, check_dtype=False)
    assert set(replacements.values()).issubset(set(reloaded.PATIENT_ID))


def test_split_and_summary_ignore_unused_categorical_patients(tmp_path):
    bundle = make_bundle()
    keys = bundle.snapshots.copy()
    keys["PATIENT_ID"] = pd.Categorical(
        keys.PATIENT_ID, categories=[*reversed(keys.PATIENT_ID.unique()), "UNUSED_PATIENT"],
    )
    manifest = data_utils.freeze_patient_split(replace(bundle, snapshots=keys), tmp_path / "split.csv")
    string_manifest = data_utils.freeze_patient_split(bundle, tmp_path / "string_split.csv")
    pd.testing.assert_frame_equal(manifest, string_manifest)
    summary = data_utils.split_summary(manifest)
    assert summary.n_patients.sum() == 80
    assert summary.n_snapshots.sum() == 160
    assert summary.n_resp1.sum() == 40
    assert summary.n_positive_patients.sum() == 40
    assert "UNUSED_PATIENT" not in set(manifest.PATIENT_ID)


def test_existing_split_reused_with_aligned_cohort_in_different_row_order(tmp_path):
    bundle = make_bundle()
    path = tmp_path / "frozen.csv"
    first = data_utils.freeze_patient_split(bundle, path, seed=42)
    reordered = reorder(bundle, np.arange(len(bundle.y) - 1, -1, -1))
    second = data_utils.freeze_patient_split(reordered, path, seed=7)
    pd.testing.assert_frame_equal(first.sort_values(KEYS).reset_index(drop=True),
                                  second.sort_values(KEYS).reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize("change", ["missing_snapshot", "changed_key", "changed_label"])
def test_existing_split_rejects_cohort_or_outcome_changes(tmp_path, change):
    bundle = make_bundle()
    path = tmp_path / "frozen.csv"
    data_utils.freeze_patient_split(bundle, path)
    original_bytes = path.read_bytes()
    if change == "missing_snapshot":
        altered = reorder(bundle, np.arange(len(bundle.y) - 1))
    elif change == "changed_key":
        keys = bundle.snapshots.copy()
        keys.loc[0, "PATIENT_ID"] = "SYNTHETIC_NEW_PATIENT"
        altered = replace(bundle, snapshots=keys)
    else:
        labels = bundle.y.copy()
        labels[0] = 1 - labels[0]
        altered = replace(bundle, y=labels)
    with pytest.raises(ValueError):
        data_utils.freeze_patient_split(altered, path)
    assert path.read_bytes() == original_bytes


def test_insufficient_positive_patients_for_stratification_gives_clear_value_error(tmp_path):
    bundle = make_bundle()
    labels = np.zeros_like(bundle.y)
    labels[1] = 1
    with pytest.raises(ValueError, match="(?i)(stratif|class|positive|patient)"):
        data_utils.freeze_patient_split(replace(bundle, y=labels), tmp_path / "split.csv")


def test_split_indices_map_manifest_keys_to_original_tensor_rows(tmp_path):
    bundle = make_bundle()
    manifest = data_utils.freeze_patient_split(bundle, tmp_path / "split.csv")
    bundle = reorder(bundle, np.random.default_rng(76).permutation(len(bundle.y)))
    shuffled_manifest = manifest.sample(frac=1, random_state=89).reset_index(drop=True)
    indices = data_utils.split_indices(bundle, shuffled_manifest)
    assert set(indices) == {"TRAIN", "VALIDATION", "TEST"}
    rows = bundle.snapshots.merge(manifest, on=KEYS, how="left", validate="one_to_one")
    for name in indices:
        expected_positions = np.flatnonzero(rows.SPLIT.eq(name).to_numpy())
        np.testing.assert_array_equal(indices[name], expected_positions)
        np.testing.assert_array_equal(bundle.y[indices[name]], rows.RESP.to_numpy()[expected_positions])
    np.testing.assert_array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(160))


@pytest.mark.parametrize("change", ["leakage", "wrong_label", "missing", "duplicate", "unknown_split"])
def test_split_indices_reject_invalid_or_misaligned_manifest(tmp_path, change):
    bundle = make_bundle()
    manifest = data_utils.freeze_patient_split(bundle, tmp_path / "split.csv").copy()
    if change == "leakage":
        manifest.loc[0, "SPLIT"] = "TEST" if manifest.loc[0, "SPLIT"] != "TEST" else "TRAIN"
    elif change == "wrong_label":
        manifest.loc[0, "RESP"] = 1 - manifest.loc[0, "RESP"]
    elif change == "missing":
        manifest = manifest.iloc[1:].copy()
    elif change == "duplicate":
        manifest = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    else:
        manifest.loc[0, "SPLIT"] = "UNASSIGNED"
    with pytest.raises(ValueError):
        data_utils.split_indices(bundle, manifest)


def test_bundle_fingerprint_covers_counts_labels_keys_and_feature_order():
    bundle = make_bundle()
    original = data_utils.bundle_fingerprint(bundle)
    assert len(original) == 64
    int(original, 16)
    assert data_utils.bundle_fingerprint(make_bundle()) == original
    counts = bundle.X.copy()
    counts[0, 0, 0] += 1
    labels = bundle.y.copy()
    labels[0] = 1 - labels[0]
    keys = bundle.snapshots.copy()
    keys.loc[0, "PATIENT_ID"] = "SYNTHETIC_DIFFERENT"
    variants = [replace(bundle, X=counts), replace(bundle, y=labels),
                replace(bundle, snapshots=keys),
                replace(bundle, feature_names=list(reversed(FEATURES)))]
    for altered in variants:
        assert data_utils.bundle_fingerprint(altered) != original


def test_save_load_round_trip_preserves_exact_unsorted_tensor_order(tmp_path):
    bundle = make_bundle()
    bundle = reorder(bundle, np.random.default_rng(111).permutation(len(bundle.y)))
    directory = tmp_path / "bundle"
    saved_path = data_utils.save_bundle(bundle, directory, source_review=SOURCE_REVIEW.copy())
    assert Path(saved_path).exists()
    loaded = data_utils.load_bundle(directory)
    assert_same_bundle(loaded, bundle)
    assert data_utils.bundle_fingerprint(loaded) == data_utils.bundle_fingerprint(bundle)
    assert {p.name for p in directory.iterdir()} == {
        "X.npy", "y.npy", "snapshots.parquet", "features.json", "metadata.json"}


def test_saving_existing_bundle_refuses_to_overwrite(tmp_path):
    bundle = make_bundle()
    directory = tmp_path / "bundle"
    data_utils.save_bundle(bundle, directory, source_review=SOURCE_REVIEW.copy())
    original_bytes = (directory / "X.npy").read_bytes()
    bundle.X[0, 0, 0] += 1
    with pytest.raises(ValueError):
        data_utils.save_bundle(bundle, directory, source_review=SOURCE_REVIEW.copy())
    assert (directory / "X.npy").read_bytes() == original_bytes


def test_loading_detects_post_export_count_changes(tmp_path):
    bundle = make_bundle()
    directory = tmp_path / "bundle"
    data_utils.save_bundle(bundle, directory, source_review=SOURCE_REVIEW.copy())
    changed = np.load(directory / "X.npy", allow_pickle=False)
    changed[0, 0, 0] += 1
    np.save(directory / "X.npy", changed, allow_pickle=False)
    with pytest.raises(ValueError, match="(?i)fingerprint"):
        data_utils.load_bundle(directory)
