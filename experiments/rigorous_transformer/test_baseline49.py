"""Synthetic source-contract and refit tests; no clinical data."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments.rigorous_transformer.baseline49 import (
    DEFAULT_SOURCE_PREFIX, load_encoded_baseline, load_original_v63_scores,
    predict_encoded, train_encoded_baseline,
)


def fixture_sources():
    features = ["F" + str(i) for i in range(49)]
    meta = pd.DataFrame({"PATIENT_ID": ["c", "a", "b"], "END_DT": pd.to_datetime(["2025-01-01"] * 3),
                         "RESP": [1, 0, 1], "SPLIT": ["train", "validation", "test"]})
    data = meta.drop(columns="SPLIT").copy()
    for i, name in enumerate(features):
        data[name] = [.125 + i, .5 + i, .75 + i]
    scores = meta.drop(columns="SPLIT").assign(SCORE=[.8, .3, .7])
    sources = {DEFAULT_SOURCE_PREFIX + "MODEL_TYPE": pd.DataFrame({"MODEL_TYPE": ["lightgbm"], "FEATURES": [json.dumps(features)]}),
               DEFAULT_SOURCE_PREFIX + "FINAL_MODEL": pd.DataFrame({"FEATURES": features, "SEQ": np.arange(49)[::-1]}),
               DEFAULT_SOURCE_PREFIX + "MODEL_DATA": data.iloc[::-1],
               DEFAULT_SOURCE_PREFIX + "UNIVERSE_W_FEATURES_SCORED": scores.iloc[::-1]}
    return meta, features, sources


def test_order_mismatch_is_documented_and_configured_order_is_preserved():
    meta, features, sources = fixture_sources()
    result = load_encoded_baseline(sources.__getitem__, meta)
    assert result["features"] == features
    assert result["X"].shape == (3, 49)
    assert result["X"][:, 0].tolist() == [.125, .5, .75]
    assert result["order_audit"]["exact_order_agreement"] is False
    assert result["order_audit"]["historical_scoring_order_verified"] is False
    assert result["provenance"]["upstream_encoding_fit_membership"] == "UNKNOWN"


def test_input_identity_detects_values_labels_and_split_changes():
    meta, _, sources = fixture_sources()
    original = load_encoded_baseline(sources.__getitem__, meta)
    changed_sources = {k: v.copy(deep=True) for k, v in sources.items()}
    changed_sources[DEFAULT_SOURCE_PREFIX + "MODEL_DATA"].loc[0, "F0"] = .126
    changed = load_encoded_baseline(changed_sources.__getitem__, meta)
    assert original["feature_sha256"] == changed["feature_sha256"]
    assert original["input_sha256"] != changed["input_sha256"]
    changed_meta = meta.copy()
    changed_meta.loc[0, "SPLIT"] = "test"
    assert original["input_sha256"] != load_encoded_baseline(sources.__getitem__, changed_meta)["input_sha256"]


def test_mixed_case_physical_columns_remain_readable():
    meta, _, sources = fixture_sources()
    source = sources[DEFAULT_SOURCE_PREFIX + "MODEL_DATA"]
    sources[DEFAULT_SOURCE_PREFIX + "MODEL_DATA"] = source.rename(columns={name: name.lower() for name in source.columns})
    assert load_encoded_baseline(sources.__getitem__, meta)["X"][:, 0].tolist() == [.125, .5, .75]


def test_missing_feature_and_conflicting_lists_never_fabricate_fallback():
    meta, features, sources = fixture_sources()
    sources[DEFAULT_SOURCE_PREFIX + "MODEL_TYPE"].loc[0, "FEATURES"] = json.dumps(features[:-1])
    with pytest.raises(ValueError, match="count differs"):
        load_encoded_baseline(sources.__getitem__, meta)
    meta, features, sources = fixture_sources()
    sources[DEFAULT_SOURCE_PREFIX + "MODEL_TYPE"]["FEATURE_NAMES"] = [json.dumps(features[::-1])]
    with pytest.raises(ValueError, match="Conflicting"):
        load_encoded_baseline(sources.__getitem__, meta)


def test_explicit_feature_names_array_is_supported_without_guessing():
    meta, features, sources = fixture_sources()
    config = sources[DEFAULT_SOURCE_PREFIX + "MODEL_TYPE"]
    sources[DEFAULT_SOURCE_PREFIX + "MODEL_TYPE"] = config.rename(columns={"FEATURES": "FEATURE_NAMES"})
    assert load_encoded_baseline(sources.__getitem__, meta)["features"] == features


@pytest.mark.parametrize("mutation, message", [("missing", "cover every"), ("labels", "labels disagree"), ("duplicate", "Duplicate")])
def test_exact_source_keys_and_resp_are_required(mutation, message):
    meta, features, sources = fixture_sources()
    name = DEFAULT_SOURCE_PREFIX + "MODEL_DATA"
    if mutation == "missing":
        sources[name] = sources[name].iloc[1:]
    elif mutation == "labels":
        sources[name] = sources[name].copy()
        sources[name].loc[sources[name].PATIENT_ID.eq("a"), "RESP"] = 1
    else:
        sources[name] = pd.concat([sources[name], sources[name].iloc[:1]])
    with pytest.raises(ValueError, match=message):
        load_encoded_baseline(sources.__getitem__, meta)


def test_original_scores_are_validation_only_and_never_certify_fit_scope():
    meta, _, sources = fixture_sources()
    result = load_original_v63_scores(sources.__getitem__, meta)
    assert result["metadata"].PATIENT_ID.tolist() == ["a"]
    assert result["scores"].tolist() == [.3]
    assert result["provenance"]["fit_membership_verified"] is False
    assert result["provenance"]["historical_fitting_scope"] == "UNKNOWN"
    with pytest.raises(ValueError, match="selection lock"):
        load_original_v63_scores(sources.__getitem__, meta, split="test")
    allowed = load_original_v63_scores(sources.__getitem__, meta, split="test", allow_test=True, selection_locked=True)
    assert allowed["scores"].tolist() == [.7]
    assert not allowed["provenance"]["fit_membership_verified"]


def test_encoded_fractional_training_roundtrip():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(8)
    x = rng.random((100, 49), dtype=np.float32)
    # AGE-like encoded values >1 and fractions are valid; missing stays missing.
    x[:, 2] *= 8
    x[0, 1] = np.nan
    y = (x[:, 0] > .65).astype(int)
    result = train_encoded_baseline(x[:70], y[:70], [f"p{i}" for i in range(70)], x[70:], y[70:], 42,
                                    config={"n_estimators": 12, "min_child_samples": 5, "early_stopping_rounds": 3, "n_jobs": 1})
    assert result["summary"]["training_complete"]
    assert result["summary"]["train_missing_values"] == 1
    assert result["summary"]["transform"]["kind"] == "identity_encoded_numeric"
    assert not result["summary"]["historical_fit_membership_verified"]
    np.testing.assert_allclose(predict_encoded(result, x[70:]), result["validation_scores"])
    result["model_blob"] += b"changed"
    with pytest.raises(ValueError, match="checksum"):
        predict_encoded(result, x[70:])
