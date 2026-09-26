import copy
import json

import numpy as np
import pandas as pd
import pytest

import feature_selection as fs


def make_data():
    rng = np.random.default_rng(7)
    # Repeated patient snapshots deliberately test prevalence at patient level.
    n_train, n_total, n_features = 80, 92, 7
    y = np.tile([0, 1], n_total // 2).astype(np.float32)
    X = rng.poisson(0.3, (n_total, 12, n_features)).astype(np.float32)
    X[:n_train, :, 0] = 0  # Held-out-only signal must never enter selection.
    X[n_train:, :, 0] = 100
    X[:n_train, :, 1] = 2  # Constant training aggregate.
    X[:n_train, :, 2] = 0
    X[:8, 0, 2] = 2       # Eight snapshots, but only one patient.
    X[:n_train, :, 3] = y[:n_train, None] * 5
    X[:n_train, :, 4] = (1 - y[:n_train, None]) * 4
    patients = [f"P{i // 8}" for i in range(n_train)] + [f"H{i}" for i in range(12)]
    split = ["train"] * n_train + ["validation"] * 6 + ["test"] * 6
    features = [f"RX__F{i}" for i in range(n_features)]
    return {"X": X, "y": y, "features": features,
        "metadata": pd.DataFrame({"PATIENT_ID": patients, "SPLIT": split}),
        "indices": {"train": np.arange(n_train), "validation": np.arange(80, 86), "test": np.arange(86, 92)},
        "hashes": {"feature_names_sha256": fs._fs_digest(features)}}


class TrainOnlyArray:
    def __init__(self, array, allowed):
        self.array, self.shape = array, array.shape
        self.allowed, self.reads = set(allowed), []

    def __array__(self, *args, **kwargs):
        raise AssertionError("Whole-array conversion is forbidden.")

    def __getitem__(self, rows):
        assert not isinstance(rows, slice), "Do not slice unrestricted source rows."
        rows = np.asarray(rows)
        assert set(rows.tolist()) <= self.allowed, "Held-out row was accessed."
        self.reads.append(rows.copy())
        return self.array[rows]


def rehash(manifest):
    manifest["manifest_sha256"] = fs._fs_digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return manifest


def test_reduced_uses_train_only_and_patient_prevalence():
    data = make_data()
    data["X"] = TrainOnlyArray(data["X"], data["indices"]["train"])
    data["y"] = TrainOnlyArray(data["y"], data["indices"]["train"])
    manifest, audit = fs.select_features(data, max_features=2, min_train_patients=2)
    assert manifest["selected_indices"] == [3, 4]
    assert manifest["selection_fit_split"] == "train"
    assert audit.loc[0, "selection_status"] == "empty_train"
    assert audit.loc[1, "selection_status"] == "constant_all_train_windows"
    assert audit.loc[2, "train_nonzero_patients"] == 1
    assert audit.loc[2, "selection_status"] == "rare_train_patients"
    assert "PATIENT_ID" not in audit.columns
    assert all(len(rows) <= 128 for rows in data["X"].reads)
    assert fs.validate_feature_manifest(manifest, data) is manifest


def test_held_out_changes_do_not_change_selection_or_audit():
    first = make_data()
    second = copy.deepcopy(first)
    second["X"][80:] = np.nan
    second["y"][80:] = np.nan
    m1, a1 = fs.select_features(first, max_features=3, min_train_patients=2)
    m2, a2 = fs.select_features(second, max_features=3, min_train_patients=2)
    assert m1 == m2
    pd.testing.assert_frame_equal(a1, a2)


def test_large_noncontiguous_train_rows_are_read_in_bounded_blocks():
    data = make_data()
    # Interleave protected rows with TRAIN and exceed the 128-row read block.
    count = 401
    train = np.arange(0, 400, 2)
    raw = np.zeros((count, 12, 7), dtype=np.float32)
    labels = np.zeros(count, dtype=np.float32)
    labels[train] = np.arange(len(train)) % 2
    raw[train, 0, 3] = labels[train] * 3
    raw[train, 2, 4] = (1 - labels[train]) * 2
    splits = np.full(count, "test", dtype=object)
    splits[train] = "train"
    data.update(X=TrainOnlyArray(raw, train), y=TrainOnlyArray(labels, train),
        metadata=pd.DataFrame({"PATIENT_ID": [f"P{i}" for i in range(count)], "SPLIT": splits}),
        indices={"train": train, "test": np.flatnonzero(splits == "test")})
    manifest, _ = fs.select_features(data, max_features=2, min_train_patients=2)
    assert manifest["selected_indices"] == [3, 4]
    assert [len(rows) for rows in data["X"].reads] == [128, 72, 128, 72]
    np.testing.assert_array_equal(np.concatenate(data["X"].reads), np.tile(train, 2))


def test_timing_only_signal_with_constant_total_is_retained():
    data = make_data()
    data["X"][:80] = 0
    train_labels = data["y"][:80]
    data["X"][np.flatnonzero(train_labels == 1), 0, 5] = 3
    data["X"][np.flatnonzero(train_labels == 0), 6, 5] = 3
    assert np.unique(data["X"][:80, :, 5].sum(axis=1)).tolist() == [3]
    manifest, audit = fs.select_features(data, max_features=1, min_train_patients=2)
    assert manifest["selected_indices"] == [5]
    assert audit.loc[5, "train_aggregate_min"] == audit.loc[5, "train_aggregate_max"] == 3
    assert audit.loc[5, "train_variable_windows"] == 2
    assert audit.loc[5, "ranking_importance"] == pytest.approx(1)
    assert manifest["settings"]["temporal_windows"] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


def test_window_order_and_importance_aggregation(monkeypatch):
    captured = {}
    class WindowRanker:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            captured["X"] = X.copy()
            # Four eligible source features (3,4,5,6); feature6 receives
            # importance only in the oldest window and must win globally.
            assert X.shape == (80, 16)
            self.feature_importances_ = np.zeros(16)
            self.feature_importances_[15] = 1
            return self
    monkeypatch.setattr(fs, "ExtraTreesClassifier", WindowRanker)
    data = make_data()
    manifest, audit = fs.select_features(data, max_features=1, min_train_patients=2)
    expected = np.log1p(data["X"][:80].reshape(80, 4, 3, 7).sum(axis=2)[:, :, [3, 4, 5, 6]])
    np.testing.assert_allclose(captured["X"], expected.reshape(80, 16), rtol=1e-6)
    assert manifest["selected_indices"] == [6]
    assert audit.loc[6, "ranking_importance"] == 1


@pytest.mark.parametrize("mode,core,expected", [("all", None, list(range(7))), ("core", ["RX__F5", "RX__F1"], [1, 5])])
def test_fixed_modes_never_inspect_tensor_or_labels(mode, core, expected):
    data = make_data()
    data["X"] = TrainOnlyArray(data["X"], [])
    data["y"] = TrainOnlyArray(data["y"], [])
    manifest, audit = fs.select_features(data, mode=mode, core_features=core)
    assert manifest["selected_indices"] == expected
    assert manifest["selection_fit_split"] == "fixed"
    assert audit["train_nonzero_patients"].isna().all()
    assert not data["X"].reads and not data["y"].reads


@pytest.mark.parametrize("core", [None, [], ["RX__F1", "RX__F1"], ["rx__f1"], ["AGE"], "RX__F1"])
def test_invalid_or_pending_core_is_rejected(core):
    with pytest.raises(ValueError):
        fs.select_features(make_data(), mode="core", core_features=core)


def test_fewer_than_requested_are_kept_without_padding():
    manifest, audit = fs.select_features(make_data(), max_features=150, min_train_patients=2)
    assert manifest["selected_indices"] == [3, 4, 5, 6]
    assert int(audit.selected.sum()) == 4
    assert manifest["settings"]["max_features"] == 150


def test_tied_importances_use_original_order(monkeypatch):
    class TiedRanker:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            self.feature_importances_ = np.zeros(X.shape[1])
            return self
    monkeypatch.setattr(fs, "ExtraTreesClassifier", TiedRanker)
    manifest, audit = fs.select_features(make_data(), max_features=2, min_train_patients=2)
    assert manifest["selected_indices"] == [3, 4]
    assert audit.loc[[3, 4, 5, 6], "selection_rank"].tolist() == [1, 2, 3, 4]


def test_manifest_round_trip_and_validation_do_not_read_values():
    data = make_data()
    manifest, _ = fs.select_features(data, max_features=3, min_train_patients=2)
    manifest = json.loads(json.dumps(manifest))
    data["X"] = TrainOnlyArray(data["X"], [])
    data["y"] = TrainOnlyArray(data["y"], [])
    fs.validate_feature_manifest(manifest, data)
    assert not data["X"].reads and not data["y"].reads


def test_hash_tampering_rejected():
    data = make_data()
    manifest, _ = fs.select_features(data, mode="all")
    manifest["selected_indices"].pop()
    with pytest.raises(ValueError, match="SHA256"):
        fs.validate_feature_manifest(manifest, data)


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(selected_indices=[True, 4]),
    lambda m: m.update(selected_indices=[4, 3]),
    lambda m: m.update(selected_indices=[3, 3]),
    lambda m: m.update(selected_features=["wrong", "RX__F4"]),
    lambda m: m.update(selection_fit_split="test"),
    lambda m: m["settings"].update(max_features=1),
    lambda m: m["settings"].update(seed=True),
    lambda m: m["settings"].update(n_train_patients=80),
    lambda m: m["settings"]["ranker_params"].update(n_jobs=-1),
    lambda m: m["settings"].update(temporal_windows=[[9, 10, 11], [6, 7, 8], [3, 4, 5], [0, 1, 2]]),
    lambda m: m.update(source_feature_names_sha256="0" * 64),
])
def test_rehashed_invalid_manifest_rejected(mutation):
    data = make_data()
    manifest, _ = fs.select_features(data, max_features=2, min_train_patients=2)
    mutation(manifest)
    rehash(manifest)
    with pytest.raises(ValueError):
        fs.validate_feature_manifest(manifest, data)


def test_source_feature_reordering_rejected():
    data = make_data()
    manifest, _ = fs.select_features(data, mode="all")
    data["features"] = list(reversed(data["features"]))
    data["hashes"]["feature_names_sha256"] = fs._fs_digest(data["features"])
    with pytest.raises(ValueError, match="vocabulary/order"):
        fs.validate_feature_manifest(manifest, data)


@pytest.mark.parametrize("parameter,value", [("max_features", 0), ("max_features", True), ("min_train_patients", 0), ("seed", -1)])
def test_invalid_reduced_settings_rejected(parameter, value):
    kwargs = {"min_train_patients": 2, parameter: value}
    with pytest.raises(ValueError):
        fs.select_features(make_data(), **kwargs)


def test_no_eligible_features_is_explicit_error():
    with pytest.raises(ValueError, match="No features survive"):
        fs.select_features(make_data(), min_train_patients=11)


def test_bad_train_indices_or_count_values_rejected():
    data = make_data()
    data["indices"]["train"] = np.array([0, 81])
    with pytest.raises(ValueError, match="not in TRAIN"):
        fs.select_features(data)
    data = make_data()
    data["X"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fs.select_features(data, min_train_patients=2)


def test_reduced_requires_both_train_classes():
    data = make_data()
    data["y"][:80] = 0
    with pytest.raises(ValueError, match="both classes"):
        fs.select_features(data)
