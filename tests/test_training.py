"""Synthetic-only tests of training isolation, checkpoint identity, and thresholds."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import training
from src.data_utils import SequenceBundle
from src.metrics import classification_metrics, select_validation_threshold
from src.model import ClaimsTransformer, ModelConfig
from src.reproducibility import seed_everything
from src.training import TrainConfig, load_trained_model, score_split, train_transformer


def synthetic_bundle():
    generator = np.random.default_rng(71)
    x = generator.poisson(1.0, (36, 12, 4)).astype(np.float32)
    y = np.tile(np.array([0, 0, 0, 1]), 9)
    snapshots = pd.DataFrame({
        "PATIENT_ID": [f"synthetic_{i:03d}" for i in range(36)],
        "END_DT": ["2025-01-01"] * 36, "RESP": y,
    })
    manifest = snapshots.copy()
    manifest["SPLIT"] = np.repeat(["TRAIN", "VALIDATION", "TEST"], 12)
    bundle = SequenceBundle(
        X=x, y=y, snapshots=snapshots,
        feature_names=("RX__a", "DX__b", "PX__c", "PX__d"),
        time_steps=tuple(range(12)),
    )
    return bundle, manifest


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    bundle, manifest = synthetic_bundle()
    model_config = ModelConfig(input_dim=4, d_model=8, n_heads=2, encoder_layers=1,
                               feedforward_dim=16, dropout=0.0)
    train_config = TrainConfig(epochs=2, patience=2, batch_size=6, device="cpu")
    run = train_transformer(bundle, manifest, tmp_path_factory.mktemp("training") / "run",
                            model_config=model_config, train_config=train_config)
    yield bundle, manifest, model_config, train_config, run
    torch.set_num_threads(prior_threads)


def test_forward_and_independent_encoder_layers():
    model = ClaimsTransformer(ModelConfig(input_dim=4, seq_len=12, d_model=8,
                              n_heads=2, encoder_layers=2, feedforward_dim=16))
    assert model(torch.zeros(2, 12, 4)).shape == (2,)
    assert not torch.equal(model.encoder.layers[0].linear1.weight,
                           model.encoder.layers[1].linear1.weight)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(2, 4, 12))


def test_validation_threshold_boundaries_and_ties():
    assert select_validation_threshold([0, 1], [0.0, 1.0]) == 1.0
    assert select_validation_threshold([0, 1], [0.0, 0.0]) == 0.0
    # F1=2/3 at 0.1 (all selected) and 0.9 (one TP); the higher cutoff wins.
    assert select_validation_threshold([1, 0, 0, 1], [0.9, 0.7, 0.5, 0.1]) == 0.9
    with pytest.raises(ValueError, match="both VALIDATION"):
        select_validation_threshold([0, 0], [0.1, 0.9])


@pytest.mark.parametrize("scores", [
    np.arange(8, 0, -1, dtype=float) / 9,
    np.array([0.9, 0.9, 0.8, 0.8, 0.6, 0.5, 0.5, 0.2]),
])
def test_validation_threshold_exact_f1_ties_survive_rounding_and_score_groups(scores):
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    # At scores[4]: TP=3, FP=2, FN=1. At scores[7]: TP=4, FP=4,
    # FN=0. Both have exact F1=2/3, though a floating precision/recall
    # harmonic mean can make the lower cutoff appear slightly better.
    assert select_validation_threshold(labels, scores) == scores[4]
    permutation = np.array([7, 0, 4, 2, 5, 1, 6, 3])
    assert select_validation_threshold(labels[permutation], scores[permutation]) == scores[4]


@pytest.mark.parametrize("operation", ["threshold", "classification"])
@pytest.mark.parametrize("scores", [
    np.array([0.2 + 0.1j, 0.8]),
    np.array([0.2 + 0j, 0.8 + 0j]),
    np.array([np.complex128(0.2 + 0.1j), 0.8], dtype=object),
    np.array([complex(0.2, 0), 0.8], dtype=object),
])
def test_classification_helpers_reject_complex_scores_before_conversion(operation, scores):
    with pytest.raises(ValueError, match="real values"):
        if operation == "threshold":
            select_validation_threshold([0, 1], scores)
        else:
            classification_metrics([0, 1], scores, threshold=0.5)


def test_fixed_threshold_metrics():
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.8, 0.7, 0.2], threshold=0.7)
    assert {key: metrics[key] for key in ["tn", "fp", "fn", "tp"]} == {
        "tn": 1, "fp": 1, "fn": 1, "tp": 1,
    }
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert classification_metrics([0, 0], [0.1, 0.2], 0.5)["roc_auc"] is None


def test_train_only_weight_and_no_test_metrics(trained):
    _, _, _, _, run = trained
    summary = json.loads(run["metadata_path"].read_text())
    assert summary["train_pos_weight"] == 3.0
    assert set(summary["class_counts"]) == {"TRAIN", "VALIDATION"}
    assert summary["test_inference_performed"] is False
    assert "test_metrics" not in summary
    history = pd.read_csv(run["history_path"])
    assert set(history) == {"epoch", "training_loss", "validation_loss",
                            "validation_average_precision", "validation_roc_auc"}
    assert summary["best_validation_average_precision"] == pytest.approx(history.validation_average_precision.max())


def test_checkpoint_reload_scores_identity_and_fixed_threshold(trained):
    bundle, manifest, _, _, run = trained
    predictions = score_split(bundle, manifest, run["checkpoint_path"], device="cpu")
    reloaded = score_split(bundle, manifest, run["checkpoint_path"], device="cpu")
    pd.testing.assert_frame_equal(predictions, reloaded, check_exact=True)
    assert predictions.PATIENT_ID.tolist() == manifest.loc[manifest.SPLIT.eq("TEST"), "PATIENT_ID"].tolist()
    assert predictions.P_RESP1.between(0, 1).all()
    assert set(predictions.SPLIT) == {"TEST"}
    _, payload = load_trained_model(run["checkpoint_path"], device="cpu")
    validation = score_split(bundle, manifest, run["checkpoint_path"], split="VALIDATION", device="cpu")
    assert payload["validation_threshold"] == select_validation_threshold(validation.RESP, validation.P_RESP1)


def test_training_updates_every_trainable_parameter_group(trained):
    _, _, model_config, train_config, run = trained
    seed_everything(train_config.seed)
    initial = ClaimsTransformer(model_config).state_dict()
    model, _ = load_trained_model(run["checkpoint_path"], device="cpu")
    learned = model.state_dict()
    groups = {
        "projection": ("projection.",),
        "positions": ("position",),
        "attention": ("encoder.layers.0.self_attn.",),
        "feedforward": ("encoder.layers.0.linear1.", "encoder.layers.0.linear2."),
        "normalization": ("encoder.layers.0.norm1.", "encoder.layers.0.norm2.", "norm."),
        "classifier": ("head.",),
    }
    for group, prefixes in groups.items():
        names = [name for name in initial if name.startswith(prefixes)]
        assert names, group
        assert any(not torch.equal(initial[name], learned[name]) for name in names), group


def test_validation_and_scoring_leave_model_weights_unchanged(trained, monkeypatch):
    bundle, manifest, _, config, run = trained
    model, payload = load_trained_model(run["checkpoint_path"], device="cpu")
    original = {name: value.clone() for name, value in model.state_dict().items()}
    indices = training.split_indices(bundle, manifest)
    loader = training._loader(bundle, indices["VALIDATION"], config)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(payload["train_pos_weight"]))
    training._evaluate(model, loader, criterion, torch.device("cpu"), config.transform)
    assert all(torch.equal(original[name], value) for name, value in model.state_dict().items())
    # Observe the very model used for scoring, rather than just reloading the
    # saved file afterward, which would miss an in-memory mutation.
    monkeypatch.setattr(training, "load_trained_model", lambda *args, **kwargs: (model, payload))
    score_split(bundle, manifest, run["checkpoint_path"], split="TEST", device="cpu")
    assert all(torch.equal(original[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize("validation_aps,best_epoch", [
    ([0.70, 0.71, 0.705], 2),  # New best below min_delta still earns a checkpoint.
    ([0.70, 0.70, 0.69], 1),  # An equal-AP checkpoint keeps the earliest epoch.
])
def test_early_stopping_restores_best_checkpoint_independent_of_min_delta(
    trained, tmp_path, monkeypatch, validation_aps, best_epoch,
):
    bundle, manifest, model_config, config, _ = trained
    observed_states = []
    original_evaluate = training._evaluate

    def observe_validation(model, *args, **kwargs):
        observed_states.append({name: value.detach().cpu().clone()
                                for name, value in model.state_dict().items()})
        return original_evaluate(model, *args, **kwargs)

    aps = iter(validation_aps)
    monkeypatch.setattr(training, "_evaluate", observe_validation)
    monkeypatch.setattr(training, "average_precision_score", lambda *args, **kwargs: next(aps))
    run = train_transformer(
        bundle, manifest, tmp_path / "controlled_validation",
        model_config=model_config,
        train_config=replace(config, epochs=10, patience=2, min_delta=0.05, print_progress=False),
    )
    assert run["summary"]["epochs_completed"] == 3
    assert run["summary"]["best_epoch"] == best_epoch
    assert run["summary"]["best_validation_average_precision"] == max(validation_aps)
    model, payload = load_trained_model(run["checkpoint_path"], device="cpu")
    expected = observed_states[best_epoch - 1]
    assert all(torch.equal(expected[name], value) for name, value in model.state_dict().items())
    assert all(torch.equal(expected[name], value) for name, value in observed_states[-1].items())
    validation = score_split(bundle, manifest, run["checkpoint_path"], split="VALIDATION", device="cpu")
    assert payload["validation_threshold"] == select_validation_threshold(validation.RESP, validation.P_RESP1)


def test_scoring_rejects_changed_features_and_manifest(trained):
    bundle, manifest, _, _, run = trained
    changed_x = bundle.X.copy()
    changed_x[0, 0, 0] += 1
    with pytest.raises(ValueError, match="differ"):
        score_split(replace(bundle, X=changed_x), manifest, run["checkpoint_path"], device="cpu")
    changed_manifest = manifest.copy()
    changed_manifest.loc[0, "SPLIT"] = "VALIDATION"
    with pytest.raises(ValueError, match="differs"):
        score_split(bundle, changed_manifest, run["checkpoint_path"], device="cpu")


def test_test_labels_cannot_affect_training(trained, tmp_path):
    bundle, manifest, model_config, train_config, run = trained
    changed_y = bundle.y.copy()
    changed_y[24:] = 1 - changed_y[24:]
    snapshots = bundle.snapshots.copy()
    snapshots["RESP"] = changed_y
    altered_manifest = manifest.copy()
    altered_manifest["RESP"] = changed_y
    altered = replace(bundle, y=changed_y, snapshots=snapshots)
    second = train_transformer(altered, altered_manifest, tmp_path / "alternate_test_labels",
                               model_config=model_config, train_config=train_config)
    first_model, first_payload = load_trained_model(run["checkpoint_path"], device="cpu")
    second_model, second_payload = load_trained_model(second["checkpoint_path"], device="cpu")
    for key, value in first_model.state_dict().items():
        assert torch.equal(value, second_model.state_dict()[key])
    assert first_payload["validation_threshold"] == second_payload["validation_threshold"]
    assert first_payload["bundle_sha256"] != second_payload["bundle_sha256"]


def test_prevents_overwrite_and_incomplete_checkpoint(trained, tmp_path):
    bundle, manifest, model_config, train_config, run = trained
    with pytest.raises(FileExistsError, match="new or empty"):
        train_transformer(bundle, manifest, run["checkpoint_path"].parent,
                          model_config=model_config, train_config=train_config)
    payload = torch.load(run["checkpoint_path"], weights_only=True)
    payload["training_complete"] = False
    incomplete = tmp_path / "incomplete.pt"
    torch.save(payload, incomplete)
    with pytest.raises(ValueError, match="did not finish"):
        load_trained_model(incomplete, device="cpu")
