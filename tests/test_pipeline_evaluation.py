"""Tiny, synthetic end-to-end TRAIN -> checkpoint -> held-out TEST reporting."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from targeting_evaluation import manifest_fingerprint  # noqa: E402
from src.data_utils import SequenceBundle, bundle_fingerprint, freeze_patient_split  # noqa: E402
from src import evaluation  # noqa: E402
from src.model import ModelConfig  # noqa: E402
from src.training import TrainConfig, score_split, train_transformer  # noqa: E402


KEYS = ["PATIENT_ID", "END_DT"]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic_pipeline_evaluation")
    rows = [(f"SYNTHETIC_PIPELINE_{patient:04d}", date)
            for patient in range(80) for date in ("2025-01-31", "2025-02-28")]
    labels = np.array([int(patient % 2 == 0 and snapshot == 1)
                       for patient in range(80) for snapshot in range(2)], dtype=np.int64)
    counts = np.random.default_rng(83).poisson(1.2, size=(160, 12, 4)).astype(np.float32)
    counts[:, 4, :] = 0
    bundle = SequenceBundle(counts, labels, pd.DataFrame(rows, columns=KEYS),
                            ("RX__SYNTH_A", "RX__SYNTH_B", "DX__SYNTH_C", "PX__SYNTH_D"))
    manifest = freeze_patient_split(bundle, directory / "frozen_split.csv", seed=42)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trained = train_transformer(
            bundle, manifest, directory / "training",
            model_config=ModelConfig(input_dim=4, seq_len=12, d_model=8, n_heads=2,
                                     encoder_layers=1, feedforward_dim=16, dropout=0),
            train_config=TrainConfig(seed=42, epochs=1, patience=1, batch_size=32,
                                     device="cpu", num_workers=0, print_progress=False),
        )
        checkpoint = trained["checkpoint_path"]
        checkpoint_bytes = checkpoint.read_bytes()
        predictions = score_split(bundle, manifest, checkpoint, split="TEST", device="cpu")
        report_directory = directory / "test_report"
        evaluated = evaluation.evaluate_checkpoint(
            bundle, manifest.sample(frac=1, random_state=3), checkpoint, report_directory, device="cpu")
        yield {"directory": directory, "bundle": bundle, "manifest": manifest,
               "trained": trained, "checkpoint": checkpoint, "checkpoint_bytes": checkpoint_bytes,
               "predictions": predictions, "report_directory": report_directory, "evaluated": evaluated}
    finally:
        torch.set_num_threads(previous_threads)


def test_full_pipeline_scores_exact_held_out_snapshot_identities(pipeline):
    manifest, predictions = pipeline["manifest"], pipeline["predictions"]
    expected = manifest.loc[manifest.SPLIT.eq("TEST"), KEYS + ["RESP"]].sort_values(KEYS).reset_index(drop=True)
    actual = predictions[KEYS + ["RESP"]].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert len(predictions) == 24
    assert predictions.PATIENT_ID.nunique() == 12
    assert predictions.SPLIT.eq("TEST").all()
    assert predictions.P_RESP1.between(0, 1).all()
    assert np.isfinite(predictions.P_RESP1).all()
    assert pipeline["checkpoint"].read_bytes() == pipeline["checkpoint_bytes"]
    assert pipeline["trained"]["summary"]["test_inference_performed"] is False
    history = pd.read_csv(pipeline["trained"]["history_path"])
    assert len(history) == 1
    assert not any("test" in column.lower() for column in history)


def test_operating_threshold_comes_from_validation_and_is_unchanged_on_test(pipeline):
    validation = score_split(pipeline["bundle"], pipeline["manifest"], pipeline["checkpoint"],
                             split="VALIDATION", device="cpu")
    labels, probabilities = validation.RESP.to_numpy(), validation.P_RESP1.to_numpy()
    candidate_scores = []
    for threshold in np.unique(probabilities):
        predicted = probabilities >= threshold
        tp = np.sum(predicted & (labels == 1))
        fp = np.sum(predicted & (labels == 0))
        fn = np.sum(~predicted & (labels == 1))
        candidate_scores.append((2 * tp / (2 * tp + fp + fn), float(threshold)))
    expected_threshold = max(candidate_scores)[1]  # Highest threshold breaks an exact F1 tie.
    fixed = pipeline["evaluated"]["classification_metrics"].iloc[0]
    assert fixed.threshold == pytest.approx(expected_threshold)
    assert fixed.threshold == pipeline["trained"]["summary"]["validation_threshold"]
    test = pipeline["predictions"]
    predicted = test.P_RESP1.to_numpy() >= fixed.threshold
    y = test.RESP.to_numpy()
    assert int(fixed.tp) == np.sum(predicted & (y == 1))
    assert int(fixed.fp) == np.sum(predicted & (y == 0))
    assert int(fixed.fn) == np.sum(~predicted & (y == 1))
    assert int(fixed.tn) == np.sum(~predicted & (y == 0))
    assert int(fixed.tp + fixed.fp + fixed.fn + fixed.tn) == len(test)


def test_checkpoint_evaluation_audit_binds_exact_manifest_bundle_and_checkpoint(pipeline):
    evaluated = pipeline["evaluated"]
    audit = evaluated["audit"]
    assert audit["snapshot_manifest_sha256"] == manifest_fingerprint(pipeline["manifest"])
    assert audit["bundle_sha256"] == bundle_fingerprint(pipeline["bundle"])
    assert audit["checkpoint_sha256"] == hashlib.sha256(pipeline["checkpoint_bytes"]).hexdigest()
    assert audit["patient_disjointness_checked"] is True
    assert audit["exact_test_snapshot_match_checked"] is True
    assert audit["feature_order_and_input_values_match_checkpoint"] is True
    assert audit["test_used_for_training_or_threshold_selection"] is False
    assert audit["threshold"] == pipeline["trained"]["summary"]["validation_threshold"]
    stored_audit = json.loads(evaluated["paths"]["audit"].read_text(encoding="utf-8"))
    assert stored_audit == audit


def test_pipeline_deciles_and_top_k_cover_only_test_population(pipeline):
    results = pipeline["evaluated"]["results"]
    deciles = results["deciles"].sort_values("decile")
    assert deciles.decile.tolist() == list(range(1, 11))
    assert int(deciles.n_snapshots.sum()) == 24
    assert int(deciles.n_resp1.sum()) == 6
    assert deciles.iloc[-1].cumulative_recall == pytest.approx(1)
    assert deciles.iloc[-1].cumulative_lift == pytest.approx(1)
    assert deciles.iloc[-1].cumulative_precision == pytest.approx(6 / 24)
    topk = results["topk"].sort_values("top_k_pct")
    assert topk.top_k_pct.tolist() == [10, 20, 30]
    assert topk.n_selected.tolist() == [3, 5, 8]
    np.testing.assert_allclose(topk.recall, topk.n_resp1 / 6)
    np.testing.assert_allclose(topk.precision, topk.n_resp1 / topk.n_selected)
    np.testing.assert_allclose(topk.lift, topk.precision / 0.25)
    assert results["global_metrics"].iloc[0].n_test_patients == 12


def test_pipeline_exports_aggregate_tables_six_figures_and_no_patient_rows(pipeline):
    evaluated = pipeline["evaluated"]
    paths = evaluated["paths"]
    for name in ["gains", "lift", "topk_performance", "response_rate", "discrimination", "confusion_matrix"]:
        content = paths[name].read_bytes()
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(content) > 1000
    report = paths["report"].read_text(encoding="utf-8")
    assert "VALIDATION" in report
    assert "confusion_matrix.png" in report
    assert "cumulative_gains.png" in report
    patient_ids = pipeline["bundle"].snapshots.PATIENT_ID.unique()
    for path in pipeline["report_directory"].rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert not any(patient.encode("utf-8") in content for patient in patient_ids)
            if path.suffix == ".csv":
                table = pd.read_csv(path)
                assert not {"PATIENT_ID", "END_DT", "P_RESP1"} & set(table.columns)
                assert len(table) <= 10
    assert json.loads(paths["classification_metrics_json"].read_text(encoding="utf-8"))["threshold"] == \
        pipeline["trained"]["summary"]["validation_threshold"]


def test_existing_report_refused_before_loading_or_scoring_and_without_changes(pipeline, monkeypatch):
    destination = pipeline["report_directory"]
    before = {str(path.relative_to(destination)): path.read_bytes()
              for path in destination.rglob("*") if path.is_file()}

    def unexpected_load(*args, **kwargs):
        pytest.fail("An existing report must be rejected before the checkpoint is loaded.")

    monkeypatch.setattr(evaluation, "load_trained_model", unexpected_load)
    with pytest.raises(FileExistsError, match="(?i)(not empty|saved report|rerun)"):
        evaluation.evaluate_checkpoint(pipeline["bundle"], pipeline["manifest"],
                                        pipeline["checkpoint"], destination, device="cpu")
    after = {str(path.relative_to(destination)): path.read_bytes()
             for path in destination.rglob("*") if path.is_file()}
    assert after == before
    assert pipeline["checkpoint"].read_bytes() == pipeline["checkpoint_bytes"]


def test_evaluation_refuses_changed_frozen_patient_assignments(pipeline, tmp_path):
    manifest = pipeline["manifest"].copy()
    test_patient = manifest.loc[manifest.SPLIT.eq("TEST"), "PATIENT_ID"].iloc[0]
    validation_patient = manifest.loc[manifest.SPLIT.eq("VALIDATION"), "PATIENT_ID"].iloc[0]
    manifest.loc[manifest.PATIENT_ID.eq(test_patient), "SPLIT"] = "VALIDATION"
    manifest.loc[manifest.PATIENT_ID.eq(validation_patient), "SPLIT"] = "TEST"
    output = tmp_path / "invalid_manifest_report"
    with pytest.raises(ValueError, match="(?i)manifest"):
        evaluation.evaluate_checkpoint(pipeline["bundle"], manifest,
                                        pipeline["checkpoint"], output, device="cpu")
    assert not output.exists()


def test_evaluation_refuses_changed_feature_order_before_export(pipeline, tmp_path):
    bundle = replace(pipeline["bundle"], feature_names=tuple(reversed(pipeline["bundle"].feature_names)))
    output = tmp_path / "invalid_features_report"
    with pytest.raises(ValueError, match="(?i)(tensor|feature|bundle)"):
        evaluation.evaluate_checkpoint(bundle, pipeline["manifest"],
                                        pipeline["checkpoint"], output, device="cpu")
    assert not output.exists()
