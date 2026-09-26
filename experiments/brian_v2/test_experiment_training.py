"""Synthetic CPU regression tests; no source claims or Snowflake connection."""
import copy
import hashlib
import io
import json
from pathlib import Path
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ((ROOT / "tak861-transformer-evaluation" / "src")
          if HERE.name == "dl_poc_experiment_work" else ROOT / "src")


def load_namespace():
    module = types.ModuleType("experiment_test_runtime")
    for path in [SOURCE / "model.py", SOURCE / "metrics.py", SOURCE / "reproducibility.py",
                 HERE / "feature_selection.py", HERE / "experiment_training.py"]:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


rt = load_namespace()
torch.set_num_threads(1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()


def make_data():
    rng = np.random.default_rng(42)
    n, f = 72, 6
    y = np.tile([0, 0, 1], n // 3).astype(np.float32)
    X = rng.poisson(0.3, (n, 12, f)).astype(np.float32)
    X[:, 0, 2] += y * 4
    features = [f"RX__F{i}" for i in range(f)]
    split = ["train"] * 48 + ["validation"] * 12 + ["test"] * 12
    metadata = pd.DataFrame({"PATIENT_ID": [f"P{i:03d}" for i in range(n)],
                             "END_DT": ["2024-05-21"] * n, "RESP": y.astype(int), "SPLIT": split})
    return {"X": X, "y": y, "features": features, "metadata": metadata,
            "indices": {"train": np.arange(48), "validation": np.arange(48, 60), "test": np.arange(60, 72)},
            "hashes": {"feature_names_sha256": digest(features),
                       "model_input_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
                       "snapshot_manifest_sha256": digest(metadata.astype(str).values.tolist()),
                       "split_config_sha256": digest({"seed": 42})}}


def settings(**changes):
    return {"seed": 42, "device": "cpu", "batch_size": 16, "epochs": 3,
            "patience": 2, "learning_rate": 0.001, "weight_decay": 0.001,
            "grad_clip": 1.0, "min_delta": 0.0001, **changes}


def config(input_dim=3):
    return rt.ModelConfig(input_dim=input_dim, seq_len=12, d_model=8, n_heads=2,
                          encoder_layers=1, feedforward_dim=16, dropout=0.2)


def core_manifest(data):
    return rt.select_features(data, mode="core", core_features=["RX__F0", "RX__F2", "RX__F5"])[0]


class GuardedRows:
    def __init__(self, values, forbidden):
        self.values, self.shape = values, values.shape
        self.forbidden = set(forbidden)
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("Full array conversion must never be used.")

    def __getitem__(self, item):
        assert not isinstance(item, tuple), "Select a source row before feature columns."
        if isinstance(item, slice):
            rows = np.arange(len(self.values))[item]
        else:
            rows = np.asarray(item).reshape(-1)
        assert not (set(rows.tolist()) & self.forbidden), "TEST row accessed during training."
        self.reads.extend(rows.tolist())
        return self.values[item]


def checkpoint_update(blob, update):
    payload = torch.load(io.BytesIO(blob), weights_only=True)
    update(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def trained():
    data = make_data()
    manifest = core_manifest(data)
    blob, summary, history = rt.run_experiment(data, manifest, config(), settings(), "TEST_TRANSFORMER")
    return data, manifest, blob, summary, history


def test_selected_feature_axis_and_raw_values():
    data = make_data()
    values, label = rt.ExperimentSequenceDataset(data, "train", [0, 2, 5])[2]
    assert values.shape == (12, 3)
    np.testing.assert_array_equal(values.numpy(), data["X"][2][:, [0, 2, 5]])
    assert label.item() == 1


@pytest.mark.parametrize("kind", ["transformer", "logistic"])
def test_training_never_reads_test_features_or_labels(kind):
    data = make_data()
    manifest = core_manifest(data)
    data["X"] = GuardedRows(data["X"], data["indices"]["test"])
    data["y"] = GuardedRows(data["y"], data["indices"]["test"])
    blob, summary, _ = rt.run_experiment(data, manifest, config(), settings(epochs=1), "NO_TEST", kind)
    assert blob and summary["test_inference_performed"] is False
    assert summary["class_counts"]["train"]["positives"] == 16
    assert summary["train_pos_weight"] == 2
    assert data["X"].reads and data["y"].reads


def test_transformer_restore_prediction_and_best_metrics(trained):
    data, manifest, blob, summary, history = trained
    first, payload, device = rt.restore_experiment(blob, data, "TEST_TRANSFORMER", "cpu")
    second, _, _ = rt.restore_experiment(blob, data, "TEST_TRANSFORMER", "cpu")
    val_y, val_scores = rt.predict_experiment(first, payload, data, "validation", device)
    _, again = rt.predict_experiment(second, payload, data, "validation", device)
    np.testing.assert_array_equal(val_scores, again)
    assert payload["format_version"] == 2
    assert payload["feature_names"] == data["features"]
    assert payload["feature_manifest"] == manifest
    assert summary["feature_manifest_sha256"] == manifest["manifest_sha256"]
    assert summary["best_validation_average_precision"] == rt.average_precision_score(val_y, val_scores)
    assert summary["best_validation_average_precision"] == history.validation_average_precision.max()
    assert summary["best_epoch"] == int(history.loc[history.validation_average_precision.idxmax(), "epoch"])
    assert summary["validation_threshold"] == rt.select_validation_threshold(val_y, val_scores)
    train_y, train_scores = rt.predict_experiment(first, payload, data, "train", device)
    assert summary["training_metrics"]["average_precision"] == rt.average_precision_score(train_y, train_scores)
    assert history["training_loss"].notna().all()
    assert history["optimization_loss"].notna().all()
    assert set(history.epoch) <= {1, 2, 3}
    test_y, test_scores = rt.predict_experiment(first, payload, data, "test", device)
    assert len(test_y) == len(test_scores) == 12
    # Default JSON serialization must accept summaries without NumPy scalars/NaN.
    json.dumps(summary, allow_nan=False)


def test_diagnostics_use_same_weighted_loss_and_eval_mode(trained):
    data, _, blob, summary, _ = trained
    model, payload, device = rt.restore_experiment(blob, data, "TEST_TRANSFORMER", "cpu")
    for split in ("train", "validation"):
        labels, scores = rt.predict_experiment(model, payload, data, split, device)
        # Recover BCE directly from the reported scores (finite float rounding).
        expected = -(2 * labels * np.log(scores) + (1 - labels) * np.log1p(-scores)).mean()
        key = "training_metrics" if split == "train" else "validation_metrics"
        assert summary[key]["loss"] == pytest.approx(float(expected), abs=2e-6)
    assert model.training is False


@pytest.mark.parametrize("change", [
    lambda d: d["hashes"].update(model_input_sha256="changed"),
    lambda d: d["features"].reverse(),
    lambda d: d.update(X=d["X"][:, :11, :]),
])
def test_changed_full_inputs_rejected(trained, change):
    data, _, blob, _, _ = trained
    data = copy.deepcopy(data)
    change(data)
    with pytest.raises(ValueError):
        rt.restore_experiment(blob, data, "TEST_TRANSFORMER", "cpu")


@pytest.mark.parametrize("change", [
    lambda p: p["feature_manifest"]["selected_indices"].reverse(),
    lambda p: p["feature_manifest"]["selected_features"].reverse(),
    lambda p: p["model_config"].update(input_dim=2),
    lambda p: p.update(transform="incorrect"),
    lambda p: p.update(validation_threshold=float("nan")),
    lambda p: p.update(training_complete=False),
    lambda p: p.update(format_version=1),
])
def test_checkpoint_contract_tampering_rejected(trained, change):
    data, _, blob, _, _ = trained
    with pytest.raises(ValueError):
        rt.restore_experiment(checkpoint_update(blob, change), data, "TEST_TRANSFORMER", "cpu")


def test_wrong_run_rejected(trained):
    data, _, blob, _, _ = trained
    with pytest.raises(ValueError):
        rt.restore_experiment(blob, data, "ANOTHER_RUN", "cpu")


def test_topk_matches_evaluation_tie_order():
    data = make_data()
    labels = data["y"][data["indices"]["validation"]]
    scores = np.full(len(labels), 0.5)
    report = rt._experiment_topk(data, "validation", labels, scores)
    namespace = {}
    helper = HERE / "evaluation_helpers.py"
    exec(compile(helper.read_text(encoding="utf-8"), str(helper), "exec"), namespace)
    scored = data["metadata"].iloc[data["indices"]["validation"]].copy()
    scored["P_RESP1"] = scores
    expected = namespace["evaluate_scores"](scored, .5, make_plots=False)["topk"].drop(columns="model")
    pd.testing.assert_frame_equal(pd.DataFrame(report)[expected.columns], expected)


def test_logistic_scaler_train_only_and_roundtrip():
    data = make_data()
    data["X"][48:] += 1000  # Large held-out distribution must not influence scaler.
    manifest = core_manifest(data)
    blob, summary, history = rt.run_experiment(data, manifest, config(), settings(), "LOGISTIC", "logistic")
    model, payload, device = rt.restore_experiment(blob, data, "LOGISTIC", "cpu")
    cols = manifest["selected_indices"]
    train = np.stack([data["X"][row][:, cols] for row in data["indices"]["train"]])
    expected_mean = np.log1p(train.sum(axis=1, dtype=np.float64)).mean(axis=0)
    np.testing.assert_allclose(payload["model_state_dict"]["scaler_mean"].numpy(), expected_mean, rtol=0, atol=1e-12)
    y, scores = rt.predict_experiment(model, payload, data, "train", device)
    other, _, _ = rt.restore_experiment(blob, data, "LOGISTIC", "cpu")
    _, again = rt.predict_experiment(other, payload, data, "train", device)
    np.testing.assert_array_equal(scores, again)
    values = np.log1p(train.sum(axis=1, dtype=np.float64))
    values = (values - model.scaler_mean.numpy()) / model.scaler_scale.numpy()
    expected = 1 / (1 + np.exp(-(values @ model.coefficient.numpy() + model.intercept.numpy()[0])))
    np.testing.assert_allclose(scores, expected, atol=1e-12)
    assert summary["epochs_completed"] == 0 and summary["best_epoch"] is None
    assert summary["optimizer_iterations"] > 0 and len(history) == 1
    assert summary["training_metrics"]["average_precision"] == rt.average_precision_score(y, scores)
    assert all(torch.is_tensor(value) for value in payload["model_state_dict"].values())
    json.dumps(summary, allow_nan=False)


def test_logistic_convergence_warning_prevents_checkpoint(monkeypatch):
    data = make_data()
    original = rt.LogisticRegression
    class NonConverged(original):
        def fit(self, X, y, **kwargs):
            result = super().fit(X, y, **kwargs)
            warnings.warn("Synthetic convergence failure", rt.ConvergenceWarning)
            return result
    monkeypatch.setattr(rt, "LogisticRegression", NonConverged)
    with pytest.raises(RuntimeError, match="did not converge"):
        rt.run_experiment(data, core_manifest(data), config(), settings(), "FAILED", "logistic")


def test_logistic_invalid_numeric_checkpoint_rejected():
    data = make_data()
    blob, _, _ = rt.run_experiment(data, core_manifest(data), config(), settings(), "LOGISTIC", "logistic")
    def invalidate(payload):
        payload["model_state_dict"]["scaler_scale"][0] = 0
    with pytest.raises(ValueError, match="Invalid numeric"):
        rt.restore_experiment(checkpoint_update(blob, invalidate), data, "LOGISTIC", "cpu")


@pytest.mark.parametrize("changes", [{"batch_size": 0}, {"epochs": 0}, {"patience": True},
    {"learning_rate": float("nan")}, {"weight_decay": -1}, {"min_delta": -1}, {"grad_clip": 0}])
def test_invalid_training_settings_rejected(changes):
    data = make_data()
    with pytest.raises(ValueError):
        rt.run_experiment(data, core_manifest(data), config(), settings(**changes), "BAD")
