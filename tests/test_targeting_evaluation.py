"""Synthetic, independently calculated checks of held-out targeting evaluation.

No clinical or project patient data is used by these tests.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import targeting_evaluation as evaluation  # noqa: E402


LABELS = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


def make_inputs(n=20, labels=None, train_labels=(1, 1, 1), repeated_patients=False):
    """Make uniquely keyed synthetic snapshots with scores decreasing by row."""
    if labels is None:
        labels = LABELS if n == 20 else [int(i % 3 == 0) for i in range(n)]
    rows = []
    for i in range(n):
        patient = f"SYNTHETIC_TEST_{i // 2 if repeated_patients else i:04d}"
        date = "2025-02-28" if repeated_patients and i % 2 else "2025-01-31"
        rows.append((patient, date, labels[i], "TEST"))
    rows.extend((f"SYNTHETIC_TRAIN_{i:04d}", "2025-01-31", y, "TRAIN")
                for i, y in enumerate(train_labels))
    rows.append(("SYNTHETIC_VALIDATION_0000", "2025-01-31", 1, "VALIDATION"))
    manifest = pd.DataFrame(rows, columns=["PATIENT_ID", "END_DT", "RESP", "SPLIT"])
    test = manifest.loc[manifest.SPLIT == "TEST", ["PATIENT_ID", "END_DT"]].copy()
    transformer = test.assign(P_RESP1=np.linspace(0.99, 0.01, n))
    return manifest, transformer


def aligned_data(**kwargs):
    return evaluation.validate_inputs(*make_inputs(**kwargs))


def ranked_data(**kwargs):
    return evaluation.rank_snapshots(aligned_data(**kwargs), "p_transformer")


def test_validation_aligns_predictions_by_snapshot_key_not_file_row_order():
    manifest, transformer = make_inputs()
    aligned = evaluation.validate_inputs(
        manifest.sample(frac=1, random_state=1),
        transformer.sample(frac=1, random_state=2),
    ).sort_values("PATIENT_ID")
    assert len(aligned) == 20
    assert aligned.RESP.tolist() == LABELS
    np.testing.assert_allclose(aligned.p_transformer, transformer.P_RESP1)
    assert set(aligned.columns) == {"PATIENT_ID", "END_DT", "RESP", "p_transformer"}


def test_exact_hand_calculated_deciles_and_top_k():
    table = evaluation.decile_table(ranked_data(), "Transformer").sort_values("decile")
    assert table.decile.tolist() == list(range(1, 11))
    assert table.n_snapshots.tolist() == [2] * 10
    assert table.n_resp1.tolist() == [2, 1, 1, 0, 1, 0, 1, 0, 0, 0]
    first, second, third, last = [table.iloc[i] for i in (0, 1, 2, 9)]
    assert first.response_rate == pytest.approx(1)
    assert first.share_of_all_resp1 == pytest.approx(2 / 6)
    assert first.cumulative_resp1 == 2
    assert first.cumulative_recall == pytest.approx(2 / 6)
    assert first.decile_lift == pytest.approx(1 / 0.3)
    assert first.cumulative_lift == pytest.approx(1 / 0.3)
    assert second.response_rate == pytest.approx(0.5)
    assert second.share_of_all_resp1 == pytest.approx(1 / 6)
    assert second.cumulative_resp1 == 3
    assert second.cumulative_recall == pytest.approx(0.5)
    assert second.decile_lift == pytest.approx(0.5 / 0.3)
    assert second.cumulative_precision == pytest.approx(3 / 4)
    assert second.cumulative_lift == pytest.approx((3 / 4) / 0.3)
    assert third.cumulative_recall == pytest.approx(4 / 6)
    assert third.cumulative_precision == pytest.approx(4 / 6)
    assert last.cumulative_resp1 == 6
    assert last.cumulative_n_snapshots == 20
    assert last.cumulative_population_fraction == pytest.approx(1)
    assert last.cumulative_recall == pytest.approx(1)
    assert last.cumulative_precision == pytest.approx(0.3)
    assert last.cumulative_lift == pytest.approx(1)
    assert table.share_of_all_resp1.sum() == pytest.approx(1)

    topk = evaluation.topk_table(table).sort_values("top_k_pct")
    assert topk.top_k_pct.tolist() == [10, 20, 30]
    assert topk.n_selected.tolist() == [2, 4, 6]
    assert topk.n_resp1.tolist() == [2, 3, 4]
    np.testing.assert_allclose(topk.recall, [2 / 6, 3 / 6, 4 / 6])
    np.testing.assert_allclose(topk.precision, [1, 3 / 4, 4 / 6])
    np.testing.assert_allclose(topk.lift, np.array([1, 3 / 4, 4 / 6]) / 0.3)
    np.testing.assert_allclose(topk.actual_population_fraction, [0.1, 0.2, 0.3])


def test_odd_sample_size_uses_declared_cumulative_ceiling_cuts():
    ranked = ranked_data(n=23)
    table = evaluation.decile_table(ranked, "Transformer").sort_values("decile")
    cuts = np.array([3, 5, 7, 10, 12, 14, 17, 19, 21, 23])
    np.testing.assert_array_equal(table.cumulative_n_snapshots, cuts)
    np.testing.assert_array_equal(table.n_snapshots, np.diff(np.r_[0, cuts]))
    assert table.n_snapshots.max() - table.n_snapshots.min() == 1
    expected = np.searchsorted(cuts, np.arange(1, 24), side="left") + 1
    np.testing.assert_array_equal(ranked.sort_values("rank").decile, expected)
    topk = evaluation.topk_table(table).sort_values("top_k_pct")
    assert topk.n_selected.tolist() == [3, 5, 7]
    np.testing.assert_allclose(topk.actual_population_fraction, [3 / 23, 5 / 23, 7 / 23])
    np.testing.assert_allclose(table.cumulative_lift,
                               table.cumulative_recall / table.cumulative_population_fraction)


@pytest.mark.parametrize("n", [10, 11, 19, 20, 21, 101])
def test_every_snapshot_appears_once_in_ten_nonempty_deciles(n):
    ranked = ranked_data(n=n)
    assert len(ranked) == n
    assert sorted(ranked["rank"].tolist()) == list(range(1, n + 1))
    assert set(ranked.decile) == set(range(1, 11))
    assert not ranked.duplicated(["PATIENT_ID", "END_DT"]).any()
    assert ranked.sort_values("rank").p_transformer.is_monotonic_decreasing


@pytest.mark.parametrize("scores", [np.full(20, 0.5), np.repeat([0.9, 0.5, 0.1, 0.0], 5)])
def test_ties_are_deterministic_and_rank_is_independent_of_labels_and_input_order(scores):
    aligned = aligned_data()
    aligned["p_transformer"] = scores
    original = evaluation.rank_snapshots(aligned.copy(), "p_transformer")
    changed = aligned.sample(frac=1, random_state=42).copy()
    changed["RESP"] = 1 - changed.RESP
    reranked = evaluation.rank_snapshots(changed, "p_transformer")
    columns = ["PATIENT_ID", "END_DT", "rank", "decile"]
    pd.testing.assert_frame_equal(
        original[columns].sort_values("rank").reset_index(drop=True),
        reranked[columns].sort_values("rank").reset_index(drop=True),
    )


def test_lift_baseline_uses_only_held_out_test_population():
    # Training prevalence moves from 0% to 100%; test prevalence remains 30%.
    tables = []
    for train_labels in ([0] * 100, [1] * 100):
        table = evaluation.decile_table(ranked_data(train_labels=train_labels), "Transformer")
        tables.append(table)
    pd.testing.assert_frame_equal(tables[0], tables[1])
    assert tables[0].iloc[0].decile_lift == pytest.approx(1 / 0.3)


def test_duplicate_prediction_keys_rejected():
    inputs = list(make_inputs())
    inputs[1] = pd.concat([inputs[1], inputs[1].iloc[[0]]])
    with pytest.raises(ValueError):
        evaluation.validate_inputs(*inputs)


def test_missing_prediction_snapshots_rejected():
    inputs = list(make_inputs())
    inputs[1] = inputs[1].iloc[1:].copy()
    with pytest.raises(ValueError):
        evaluation.validate_inputs(*inputs)


def test_equal_length_but_mismatched_prediction_snapshot_keys_rejected():
    inputs = list(make_inputs())
    inputs[1].loc[0, "PATIENT_ID"] = "SYNTHETIC_UNKNOWN"
    with pytest.raises(ValueError):
        evaluation.validate_inputs(*inputs)


def test_duplicate_manifest_snapshot_keys_rejected():
    manifest, transformer = make_inputs()
    manifest = pd.concat([manifest, manifest.iloc[[0]]])
    with pytest.raises(ValueError):
        evaluation.validate_inputs(manifest, transformer)


def test_patient_leakage_rejected_even_when_snapshot_dates_differ():
    manifest, transformer = make_inputs()
    leaked = manifest.iloc[[0]].copy()
    leaked["END_DT"] = "2024-01-31"
    leaked["SPLIT"] = "TRAIN"
    manifest = pd.concat([manifest, leaked])
    with pytest.raises(ValueError):
        evaluation.validate_inputs(manifest, transformer)


def test_multiple_snapshots_for_one_patient_are_valid_within_one_split():
    aligned = aligned_data(repeated_patients=True)
    assert len(aligned) == 20
    assert aligned.PATIENT_ID.nunique() == 10


def test_unused_categorical_patient_ids_do_not_create_phantom_leakage():
    manifest, transformer = make_inputs()
    manifest["PATIENT_ID"] = pd.Categorical(
        manifest.PATIENT_ID, categories=[*manifest.PATIENT_ID, "UNUSED_PATIENT"])
    aligned = evaluation.validate_inputs(manifest, transformer)
    assert len(aligned) == 20
    assert evaluation.validate_manifest(manifest).PATIENT_ID.nunique() == len(manifest)


def test_manifest_fingerprint_ignores_categorical_key_order_and_csv_dtypes(tmp_path):
    manifest, _ = make_inputs()
    categorical = manifest.copy()
    categorical["PATIENT_ID"] = pd.Categorical(
        categorical.PATIENT_ID, categories=[*reversed(manifest.PATIENT_ID), "UNUSED_PATIENT"])
    path = tmp_path / "manifest.csv"
    categorical.to_csv(path, index=False)
    expected = evaluation.manifest_fingerprint(manifest)
    assert evaluation.manifest_fingerprint(categorical.sample(frac=1, random_state=8)) == expected
    assert evaluation.manifest_fingerprint(evaluation.read_input(path)) == expected


def test_csv_reader_preserves_literal_patient_keys(tmp_path):
    manifest, transformer = make_inputs()
    literal_keys = ["NA", "NULL", "N/A", "nan", "0000123"]
    for index, key in enumerate(literal_keys):
        manifest.loc[index, "PATIENT_ID"] = key
        transformer.loc[index, "PATIENT_ID"] = key
    manifest_path, score_path = tmp_path / "manifest.csv", tmp_path / "scores.csv"
    manifest.to_csv(manifest_path, index=False)
    transformer.to_csv(score_path, index=False)
    aligned = evaluation.validate_inputs(evaluation.read_input(manifest_path),
                                         evaluation.read_input(score_path))
    assert set(literal_keys).issubset(aligned.PATIENT_ID)
    assert len(aligned) == 20


@pytest.mark.parametrize("column", ["PATIENT_ID", "END_DT", "RESP", "SPLIT"])
def test_csv_reader_still_rejects_missing_required_manifest_values(tmp_path, column):
    manifest, transformer = make_inputs()
    manifest[column] = manifest[column].astype(object)
    manifest.loc[0, column] = None
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError):
        evaluation.validate_inputs(evaluation.read_input(path), transformer)


def test_csv_reader_still_rejects_missing_probability(tmp_path):
    manifest, transformer = make_inputs()
    transformer.loc[0, "P_RESP1"] = np.nan
    path = tmp_path / "scores.csv"
    transformer.to_csv(path, index=False)
    with pytest.raises(ValueError):
        evaluation.validate_inputs(manifest, evaluation.read_input(path))


@pytest.mark.parametrize("invalid_score", [-0.01, 1.01, np.nan, np.inf, -np.inf, "not-a-score", 0.5 + 0.1j])
def test_invalid_probability_rejected(invalid_score):
    inputs = list(make_inputs())
    inputs[1]["P_RESP1"] = inputs[1].P_RESP1.astype(object)
    inputs[1].loc[0, "P_RESP1"] = invalid_score
    with pytest.raises(ValueError):
        evaluation.validate_inputs(*inputs)


@pytest.mark.parametrize("invalid_label", [-1, 2, np.nan, "positive"])
def test_invalid_test_response_rejected(invalid_label):
    manifest, transformer = make_inputs()
    manifest["RESP"] = manifest.RESP.astype(object)
    manifest.loc[0, "RESP"] = invalid_label
    with pytest.raises(ValueError):
        evaluation.validate_inputs(manifest, transformer)


def test_optional_prediction_response_must_match_manifest():
    inputs = list(make_inputs())
    inputs[1]["RESP"] = LABELS
    evaluation.validate_inputs(*inputs)
    inputs[1].loc[0, "RESP"] = 1 - LABELS[0]
    with pytest.raises(ValueError):
        evaluation.validate_inputs(*inputs)


def test_fewer_than_ten_test_snapshots_rejected():
    with pytest.raises(ValueError):
        evaluation.validate_inputs(*make_inputs(n=9))


def test_zero_positives_preserves_zero_counts_but_undefined_recall_and_lift():
    table = evaluation.decile_table(ranked_data(labels=[0] * 20), "Transformer")
    assert (table.n_resp1 == 0).all()
    assert (table.cumulative_resp1 == 0).all()
    assert (table.response_rate == 0).all()
    assert (table.cumulative_precision == 0).all()
    for column in ["share_of_all_resp1", "cumulative_recall", "decile_lift", "cumulative_lift"]:
        assert table[column].isna().all(), column
    topk = evaluation.topk_table(table)
    assert topk.recall.isna().all()
    assert topk.lift.isna().all()
    assert (topk.precision == 0).all()


def test_all_positives_have_unit_lift_and_capture_equals_population_share():
    table = evaluation.decile_table(ranked_data(labels=[1] * 20), "Transformer")
    assert (table.response_rate == 1).all()
    assert (table.decile_lift == 1).all()
    assert (table.cumulative_precision == 1).all()
    assert (table.cumulative_lift == 1).all()
    np.testing.assert_allclose(table.cumulative_recall, table.cumulative_population_fraction)


def test_model_evaluation_includes_transformer_and_requested_top_k_rows():
    result = evaluation.evaluate_model(aligned_data())
    assert set(result) == {"deciles", "topk", "global_metrics", "ties"}
    assert len(result["deciles"]) == 10
    assert len(result["topk"]) == 3
    assert set(result["topk"].model) == {"Transformer"}
    table = result["topk"].sort_values("top_k_pct")
    assert table.top_k_pct.tolist() == [10, 20, 30]
    assert table.n_selected.tolist() == [2, 4, 6]
    assert table.n_resp1.tolist() == [2, 3, 4]
    assert len(result["global_metrics"]) == 1


def test_evaluation_and_report_export_only_aggregate_rows(tmp_path):
    aligned = aligned_data(repeated_patients=True)
    patient_ids = aligned.PATIENT_ID.unique().tolist()
    result = evaluation.evaluate_model(aligned)
    for table in result.values():
        assert "PATIENT_ID" not in table.columns
        assert "END_DT" not in table.columns
        assert not any(patient in table.to_csv(index=False) for patient in patient_ids)
    paths = evaluation.write_report(result, tmp_path, audit={"model": "Transformer"})
    assert paths
    csv_files = list(tmp_path.rglob("*.csv"))
    assert csv_files
    for path in csv_files:
        table = pd.read_csv(path)
        assert "PATIENT_ID" not in table.columns
        assert "END_DT" not in table.columns
        assert len(table) <= 10, f"Unexpected nonaggregate table: {path.name}"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert not any(patient.encode("utf-8") in content for patient in patient_ids)


def test_provenance_requires_only_transformer_metadata():
    manifest, _ = make_inputs()
    metadata = {"Transformer": {
        "snapshot_manifest_sha256": evaluation.manifest_fingerprint(manifest),
        "population_version": "V63",
        "claims_vintage": "20260825",
        "model_frozen_before_test": True,
        "training_split": "TRAIN",
        "tuning_split": "VALIDATION",
        "scoring_split": "TEST",
    }}
    evaluation.validate_provenance(metadata, manifest)
    assert set(evaluation.MODELS) == {"Transformer"}
    with pytest.raises(ValueError):
        evaluation.validate_provenance({}, manifest)
