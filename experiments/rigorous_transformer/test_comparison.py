"""Synthetic mathematical/protocol checks; no client data or clinical results."""
import copy

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.rigorous_transformer.comparison import (
    binary_metrics, compare_predictions, comparison_conclusion, latest_patient_indices,
    paired_patient_bootstrap, select_validation_threshold, summarize_seed_runs, topk_metrics,
)


def test_ap_is_not_trapezoidal_pr_area_and_confusion_is_exact():
    metrics = binary_metrics([1, 0, 1], [.9, .8, .7], threshold=.75, top_fractions=())
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["pr_auc_trapezoidal"] == pytest.approx(19 / 24)
    assert metrics["roc_auc"] == pytest.approx(.5)
    assert metrics["brier_score"] == pytest.approx((.01 + .64 + .09) / 3)
    assert metrics["confusion_matrix"] == [[0, 1], [1, 1]]
    assert metrics["precision"] == .5
    assert metrics["recall"] == .5
    assert metrics["specificity"] == 0
    assert metrics["f1"] == .5


def test_perfect_model_and_fixed_calibration_bins():
    metrics = binary_metrics([0, 1, 0, 1], [0., 1., 0., 1.], threshold=.5, calibration_bins=2)
    assert metrics["average_precision"] == metrics["roc_auc"] == 1
    assert metrics["brier_score"] == metrics["expected_calibration_error"] == 0
    assert [r["count"] for r in metrics["calibration"]] == [2, 2]
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_threshold_is_validation_locked_and_rejected_if_modified():
    policy = select_validation_threshold([0, 1, 1, 0], [.1, .7, .9, .8])
    assert policy["threshold"] == .7
    assert policy["validation_f1"] == pytest.approx(.8)
    assert binary_metrics([0, 1], [.6, .65], policy)["confusion_matrix"] == [[1, 0], [1, 0]]
    bad = copy.deepcopy(policy)
    bad["threshold"] = .5
    with pytest.raises(ValueError, match="hash mismatch"):
        binary_metrics([0, 1], [.6, .65], bad)
    with pytest.raises(ValueError, match="VALIDATION"):
        select_validation_threshold([0, 1], [.1, .9], selection_split="test")


def test_topk_ties_use_stable_keys_and_not_labels():
    scores, labels, keys = np.array([.8, .8, .8, .1]), np.array([1, 0, 1, 0]), np.array(["c", "a", "b", "d"])
    metrics = topk_metrics(labels, scores, .5, keys)
    assert metrics["selected"] == 2
    assert metrics["selected_positives"] == 1
    assert metrics["tie_crosses_boundary"] is True
    assert metrics["n_tied_at_cutoff"] == 3
    order = np.array([3, 2, 0, 1])
    assert topk_metrics(labels[order], scores[order], .5, keys[order]) == metrics
    with pytest.raises(ValueError, match="unique"):
        topk_metrics(labels, scores, .5, ["a"] * 4)


def test_patient_cluster_bootstrap_matches_explicit_whole_patient_resampling():
    y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    a = np.array([.1, .6, .2, .3, .7, .8, .4, .9])
    b = np.array([.5, .6, .2, .7, .3, .8, .4, .1])
    patient = np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"])
    result = paired_patient_bootstrap(y, a, b, patient, n_bootstrap=100, seed=18)
    rng, draws = np.random.default_rng(18), []
    for _ in range(100):
        ids = rng.integers(0, 4, size=4)
        rows = np.concatenate([np.flatnonzero(patient == f"p{i + 1}") for i in ids])
        if len(np.unique(y[rows])) == 2:
            draws.append([average_precision_score(y[rows], a[rows]) - average_precision_score(y[rows], b[rows]),
                          roc_auc_score(y[rows], a[rows]) - roc_auc_score(y[rows], b[rows])])
    assert result["valid_draws"] == len(draws)
    assert result["average_precision"]["lower"] == pytest.approx(np.quantile(np.array(draws)[:, 0], .025))
    assert result["roc_auc"]["upper"] == pytest.approx(np.quantile(np.array(draws)[:, 1], .975))
    assert result == paired_patient_bootstrap(y, a, b, patient, n_bootstrap=100, seed=18)


def test_identical_predictions_have_zero_paired_interval():
    result = paired_patient_bootstrap([0, 1] * 10, [.2, .8] * 10, [.2, .8] * 10,
                                      np.repeat(np.arange(10), 2), n_bootstrap=50)
    assert result["average_precision"] == {"delta": 0, "lower": 0, "upper": 0}
    assert result["roc_auc"] == {"delta": 0, "lower": 0, "upper": 0}


def test_latest_patient_summary_has_one_row_and_rejects_duplicate_latest_keys():
    indices = latest_patient_indices(["a", "a", "b"], ["2025-01-01", "2025-05-01", "2025-02-01"])
    assert indices.tolist() == [1, 2]
    with pytest.raises(ValueError, match="Duplicate latest"):
        latest_patient_indices(["a", "a"], ["2025-01-01", "2025-01-01"])


def _provenance():
    return dict(baseline_kind="matched_refit", fit_membership_verified=True, same_cohort=True,
                same_labels=True, patient_disjoint_split=True, selection_locked=True,
                test_used_for_selection=False, comparison_split="test",
                upstream_cutoff_verified=True, holdout_never_previously_inspected=True,
                same_feature_information=True)


def test_original_baseline_never_receives_superiority_claim():
    interval = {"interval_reliable": True, "average_precision": {"delta": .1, "lower": .05, "upper": .15}}
    proof = _provenance()
    assert comparison_conclusion(interval, proof)["status"] == "TEST_AP_ADVANTAGE"
    proof["baseline_kind"] = "original_v63"
    assert comparison_conclusion(interval, proof)["status"] == "DESCRIPTIVE_ONLY"
    proof = _provenance()
    proof["upstream_cutoff_verified"] = False
    outcome = comparison_conclusion(interval, proof)
    assert outcome["status"] == "RETROSPECTIVE_AP_ADVANTAGE"
    assert "prevents a prospective" in outcome["conclusion"]
    proof["comparison_split"] = "validation"
    assert comparison_conclusion(interval, proof)["status"] == "DEVELOPMENT_ONLY"


def test_nonsignificance_is_not_equivalence_and_test_selection_blocks_inference():
    interval = {"interval_reliable": True, "average_precision": {"delta": .01, "lower": -.01, "upper": .02}}
    assert "not proof of equivalence" in comparison_conclusion(interval, _provenance())["conclusion"]
    proof = _provenance()
    proof["test_used_for_selection"] = True
    assert comparison_conclusion(interval, proof)["status"] == "UNRESOLVED"


def test_seed_reporting_preserves_all_runs_and_generalization_gap():
    runs = [{"summary": {"training_complete": True, "seed": seed, "config": {"family": "transformer"},
                         "train_metrics": {"average_precision": train}, "validation_metrics": {"average_precision": val}}}
            for seed, train, val in [(1, .7, .2), (2, .6, .3), (3, .8, .1)]]
    report = summarize_seed_runs({"candidate": runs})
    assert len(report["all_seeds"]) == 3
    assert report["candidates"][0]["mean_validation_ap"] == pytest.approx(.2)
    assert report["all_seeds"][0]["train_validation_gap_ap"] == pytest.approx(.5)
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize_seed_runs({"candidate": runs + [runs[0]]})


@pytest.mark.parametrize("y, scores", [([0, 2], [.1, .2]), ([0, 1], [float("nan"), .4]),
                                      ([0, 1], [-.1, 1.1]), ([0], [.1, .3]), ([], [])])
def test_invalid_labels_probabilities_and_lengths_fail(y, scores):
    with pytest.raises(ValueError):
        binary_metrics(y, scores)


def test_single_class_metrics_do_not_invent_auc_or_recall():
    metrics = binary_metrics([0, 0], [.1, .2], .5)
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["recall"] is None
    assert metrics["specificity"] == 1


def test_comparison_combines_only_matching_prediction_vectors():
    result = compare_predictions([0, 1] * 10, [.1, .9] * 10, [.3, .7] * 10,
                                 np.repeat(np.arange(10), 2), provenance=_provenance(), n_bootstrap=50)
    assert result["conclusion"]["status"] == "NO_DEMONSTRATED_AP_ADVANTAGE"
    with pytest.raises(ValueError):
        compare_predictions([0, 1], [.1, .9], [.3], [1, 2], provenance=_provenance(), n_bootstrap=50)
