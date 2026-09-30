"""Synthetic tests; no private patient data or warehouse connection required."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


_spec = importlib.util.spec_from_file_location("feature_analysis_selection", Path(__file__).with_name("selection.py"))
selection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(selection)


def small_config():
    return {"n_splits": 3, "min_nonzero_patients": 3, "min_observed_patients": 10,
        "filter_max_features": 3, "wrapper_shortlist": 5, "wrapper_max_features": 2,
        "embedded_max_features": 3, "tree_estimators": 16, "tree_min_samples_leaf": 4,
        "linear_max_iter": 1500, "n_jobs": 1}


def signal_data():
    rng = np.random.default_rng(83)
    y = rng.binomial(1, 0.2, 360)
    signal = rng.poisson(0.5 + 5 * y)
    X = pd.DataFrame({"RX__signal": signal, "DX__noise": rng.poisson(1.1, len(y)),
        "PX__noise": rng.poisson(0.7, len(y)), "numeric": rng.normal(size=len(y)),
        "constant": 0.0, "duplicate_signal": signal, "missing": np.nan,
        "rare": np.r_[1, np.zeros(len(y)-1)]})
    return X, y


def test_quality_and_signal():
    X, y = signal_data()
    original = X.copy(deep=True)
    result = selection.select_all(X, y, small_config())
    pd.testing.assert_frame_equal(X, original)
    for method in ("filter", "wrapper", "embedded"):
        table = result[method].set_index("FEATURE_NAME")
        assert len(table) == X.shape[1]
        assert bool(table.loc["RX__signal", "SELECTED"])
        assert table.loc["duplicate_signal", "QUALITY_REASON"] == "exact_duplicate"
        assert table.loc["duplicate_signal", "DUPLICATE_OF"] == "RX__signal"
        assert table.loc["constant", "QUALITY_REASON"] == "constant_observed_values"
        assert table.loc["missing", "QUALITY_REASON"] == "all_missing"
        assert table.loc["rare", "QUALITY_REASON"] == "insufficient_nonzero_support"
        assert not table.loc[["duplicate_signal", "constant", "missing", "rare"], "SELECTED"].any()
        assert table.FOLD_SELECTION_FREQUENCY.between(0, 1).all()
    diag = result["diagnostics"]
    assert len(diag) == 12
    assert set(diag.STRATEGY) == {"filter", "wrapper", "embedded", "all_eligible"}
    assert (diag.AVERAGE_PRECISION > 0.7).all()
    assert (diag.TOP10_LIFT > 2).all()
    assert diag.SELECTOR_CONVERGED.all() and diag.SCORER_CONVERGED.all()


def test_fold_refits_and_target_isolation():
    X, y = signal_data()
    calls = []
    original = selection._fit_strategies
    def record(values, labels, names, cfg):
        calls.append((len(values), len(labels)))
        return original(values, labels, names, cfg)
    selection._fit_strategies = record
    try:
        selection.select_all(X, y, small_config())
    finally:
        selection._fit_strategies = original
    assert calls == [(240, 240), (240, 240), (240, 240), (360, 360)]
    try:
        selection.select_all(X.assign(RESP=y), y, small_config())
    except ValueError as exc:
        assert "target" in str(exc)
    else:
        raise AssertionError("The target was accepted as a predictor.")


def test_no_signal_and_ties_use_baseline():
    y = np.tile([0, 0, 0, 1], 30)
    result = selection.select_all(pd.DataFrame({"constant": np.ones(len(y))}), y, small_config())
    assert not result["filter"].SELECTED.any()
    assert not result["wrapper"].SELECTED.any()
    assert not result["embedded"].SELECTED.any()
    assert np.allclose(result["diagnostics"].TOP10_LIFT, 1)
    assert np.allclose(result["diagnostics"].AVERAGE_PRECISION, 0.25)


def test_imputation_and_scaling_fit_partition_only():
    cfg = selection._selection_config(small_config())
    raw = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]] * 4)
    audit, indices, state = selection._fit_quality(raw, ["count"], cfg)
    mean_before = state["scaler"].mean_.copy()
    transformed = selection._quality_transform(np.array([[np.nan], [1e8]]), indices, state)
    assert np.isfinite(transformed).all()
    assert np.allclose(state["medians"], 2)
    assert np.array_equal(mean_before, state["scaler"].mean_)
    assert np.allclose(transformed[0], state["scaler"].transform([[np.log1p(2)]])[0])


def test_fold_quality_does_not_see_heldout_only_variation():
    X, y = signal_data()
    cfg = small_config()
    cv = selection.StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=42)
    fit_rows, score_rows = next(cv.split(X, y))
    X["heldout_only_variation"] = 0.0
    X.loc[score_rows, "heldout_only_variation"] = np.arange(len(score_rows))
    captured = []
    original = selection._fit_strategies
    def record(values, labels, names, config):
        output = original(values, labels, names, config)
        captured.append(output[0]["filter"].set_index("FEATURE_NAME"))
        return output
    selection._fit_strategies = record
    try:
        selection.select_all(X, y, cfg)
    finally:
        selection._fit_strategies = original
    assert captured[0].loc["heldout_only_variation", "QUALITY_REASON"] == "constant_observed_values"
    assert bool(captured[-1].loc["heldout_only_variation", "ELIGIBLE"])


if __name__ == "__main__":
    for test in (test_quality_and_signal, test_fold_refits_and_target_isolation,
                 test_no_signal_and_ties_use_baseline, test_imputation_and_scaling_fit_partition_only,
                 test_fold_quality_does_not_see_heldout_only_variation):
        test()
        print(test.__name__, "PASS")
