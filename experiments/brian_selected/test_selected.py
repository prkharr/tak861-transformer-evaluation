import importlib.util
import io
import json
from decimal import Decimal
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core
import models
import build_notebooks as builder

torch.set_num_threads(1)


def metadata(n=80):
    return pd.DataFrame({"PATIENT_ID": [f"P{i:04d}" for i in range(n)],
                         "END_DT": ["2025-05-21"]*n, "RESP": np.arange(n) % 2})


def test_exact_order_and_quoted_names():
    names = ["_A_B_", 'Long name "quoted"', "PL_22"]
    assert core.parse_features(json.dumps(names)) == names
    assert core.quote_identifier(names[1]) == '"Long name ""quoted"""'
    for bad in (["AGE", "age"], ["RESP"], ["RND"], [], "AGE", [1]):
        with pytest.raises((ValueError, TypeError)):
            core.parse_features(bad)


def test_alignment_retains_fraction_dates_and_missingness():
    meta = metadata(4)
    source = meta.copy()
    source["VALUE"] = [Decimal("0.125"), None, Decimal("1.875"), Decimal("0.5")]
    actual, X = core.align_features(meta.iloc[::-1], source.sample(frac=1, random_state=1), ["VALUE"])
    pd.testing.assert_frame_equal(actual, meta)
    assert X[0, 0] == .125 and np.isnan(X[1, 0]) and X[2, 0] == 1.875
    for changed in (source.iloc[:-1], pd.concat([source, source.iloc[:1]])):
        with pytest.raises(ValueError):
            core.align_features(meta, changed, ["VALUE"])
    source.loc[0, "RESP"] = 1
    with pytest.raises(ValueError):
        core.align_features(meta, source, ["VALUE"])


def test_train_only_preprocessing_and_hash_integrity():
    X = np.array([[1., np.nan, 2.], [3., np.nan, 2.], [np.nan, np.nan, 2.]])
    state = core.fit_preprocessor(X)
    assert state["median"] == [2., 0., 2.]
    assert state["all_missing_train"] == [False, True, False]
    validation = np.array([[100., 100., np.nan]])
    values, mask = core.transform_features(validation, state)
    assert state == core.fit_preprocessor(X)
    assert mask.tolist() == [[0, 0, 1]] and np.isfinite(values).all()
    meta = metadata(3)
    hash1 = core.raw_hash(X, meta, ["A", "B", "C"])
    X[0, 0] += 1
    assert core.raw_hash(X, meta, ["A", "B", "C"]) != hash1


def test_split_bound_to_reference_and_no_patient_overlap():
    meta = metadata(12)
    frozen = meta.copy()
    frozen["SPLIT"] = ["train"]*4 + ["validation"]*4 + ["test"]*4
    frozen["SPLIT_CONFIG"] = json.dumps({"seed": 42})
    records = [[r.PATIENT_ID, r.END_DT, r.RESP, r.SPLIT] for r in frozen.itertuples()]
    reference = {"run_id": "RUN_001", "training_complete": True, "input_hashes": {
        "snapshot_manifest_sha256": core.digest_json(records),
        "split_config_sha256": core.digest_json({"seed": 42})}}
    actual, _ = core.bind_split(meta, frozen.iloc[::-1], reference)
    assert actual.SPLIT.tolist() == frozen.SPLIT.tolist()
    frozen.loc[0, "SPLIT"] = "test"
    with pytest.raises(ValueError):
        core.bind_split(meta, frozen, reference)


def test_lift_and_ties_are_outcome_independent():
    meta = metadata(101)
    scores = np.ones(101)*.5
    deciles, top = core.rank_tables(meta, scores)
    assert deciles.snapshots.sum() == 101
    assert top.loc[top.fraction == .10, "selected"].iloc[0] == 11
    expected = meta.RESP.iloc[:11].mean()/meta.RESP.mean()
    assert top.loc[top.fraction == .10, "lift"].iloc[0] == pytest.approx(expected)
    assert deciles.cumulative_lift.iloc[-1] == pytest.approx(1)
    assert deciles.cumulative_recall.iloc[-1] == 1
    shuffled = meta.sample(frac=1, random_state=8)
    a, b = core.rank_tables(shuffled, scores)
    pd.testing.assert_frame_equal(deciles, a)
    pd.testing.assert_frame_equal(top, b)
    assert models.selection_key(meta.RESP.to_numpy(), scores)[0] == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["transformer"])
def test_fit_predict_roundtrip(kind):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    missing = np.zeros_like(X)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    recipe = next(r for r in models.default_recipes() if r["kind"] == kind)
    if kind == "transformer":
        recipe["settings"].update(width=8, heads=2, layers=1, feedforward=16, batch_size=16)
    result = models.fit_candidate(recipe, X[:60], missing[:60], y[:60], X[60:], missing[60:], y[60:], max_epochs=2, patience=1)
    scores = models.predict_candidate(result, X[60:], missing[60:])
    loaded = models.load_candidate(models.dump_candidate(result))
    np.testing.assert_allclose(scores, models.predict_candidate(loaded, X[60:], missing[60:]))
    assert "train_top10_lift" in result["history"][0]
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    with pytest.raises(ValueError):
        models.predict_candidate(loaded, X[:, :-1], missing[:, :-1])


def test_notebooks_self_contained_compilable_and_no_test_training():
    for cells in (builder.preparation, builder.splitting, builder.training, builder.evaluation):
        assert len(cells) == 8
        for source in cells:
            compile(source, "cell", "exec")
        assert "def align_features" in cells[1] and "def save_artifacts" in cells[1]
        assert "sf_options" in cells[0]
    training = '\n'.join(builder.training)
    assert 'experiment_partition(data, "test")' not in training
    assert "CORE_FEATURES" not in training and "reduced150" not in training
    assert 'select_validation_threshold(yv, pv)' in training
    assert 'candidate_reports[recipe["name"]]' in training


def test_chunk_storage_roundtrip_and_corruption():
    scope = {"hashlib": __import__("hashlib"), "re": __import__("re")}
    exec(builder.storage_source(), scope)
    blobs = {"sample": b"abc"*100}
    rows = scope["pack_artifacts"](blobs, chunk_size=17)
    assert scope["unpack_artifacts"](rows[::-1], blobs.keys()) == blobs
    with pytest.raises(ValueError):
        scope["unpack_artifacts"](rows[:-1], blobs.keys())


def test_all_four_notebooks_in_fresh_sessions_with_synthetic_warehouse(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    # Full expected population sizes, invented patients/dates/values only.
    records, counter = [], 0
    for split, count, patients, positive in (("train", 16256, 8712, 941),
                                           ("validation", 3481, 1867, 202),
                                           ("test", 3414, 1868, 202)):
        for i in range(count):
            records.append([f"SYNTHETIC_{counter + i % patients:06d}",
                            "2025-04-21" if i < patients else "2025-05-21",
                            int(i < positive), split, json.dumps({"seed": 42})])
        counter += patients
    frozen = pd.DataFrame(records, columns=["PATIENT_ID", "END_DT", "RESP", "SPLIT", "SPLIT_CONFIG"])
    frozen = frozen.sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    snapshots = frozen[["PATIENT_ID", "END_DT", "RESP"]].copy()
    feature_names = ["AGE", "_LEADING_UNDERSCORE_", "RATIO", "ALL_MISSING", "LONG_SELECTED_CATEGORY"]
    rng = np.random.default_rng(24)
    source = snapshots.copy()
    for name in feature_names:
        source[name] = rng.normal(size=len(source)) + source.RESP.to_numpy()*2
    source.loc[np.arange(len(source)) % 11 == 0, "RATIO"] = np.nan
    source["ALL_MISSING"] = np.nan
    refs = [[r.PATIENT_ID, r.END_DT, int(r.RESP), r.SPLIT] for r in frozen.itertuples()]
    reference = {"run_id": "RUN_001", "training_complete": True,
                 "input_hashes": {"snapshot_manifest_sha256": core.digest_json(refs),
                                  "split_config_sha256": core.digest_json({"seed": 42})}}
    tables = {"TAK861_TX_READY_V63_MODEL_TYPE": pd.DataFrame({"MODEL_TYPE": ["lightgbm"], "FEATURES": [json.dumps(feature_names)]}),
              "TAK861_TX_READY_V63_FINAL_MODEL": pd.DataFrame({"FEATURES": feature_names[:-1]}),
              "TAK861_TX_READY_V63_MODEL_DATA": source,
              "TAK861_TX_READY_V63_DL_POC_SNAPSHOTS": snapshots,
              "TAK861_TX_READY_V63_DL_POC_PATIENT_SPLIT": frozen}
    artifacts = {"TAK861_TX_READY_V63_DL_POC_MODEL_RUN_001": {
        "checkpoint.pt": b"synthetic-reference-not-loaded", "training_summary.json": core.json_bytes(reference),
        "training_history.csv": b"synthetic", "training_history.png": b"synthetic"}}

    class Frame:
        def __init__(self, df):
            self.df = df
        @property
        def columns(self):
            return self.df.columns.tolist()
        def select(self, *names):
            return Frame(self.df[list(names)])
        def collect(self):
            return self.df.to_dict("records")
        def toPandas(self):
            return self.df.copy()

    def read_artifact(table, expected):
        assert set(artifacts[table]) == set(expected)
        return artifacts[table]

    def save_artifact(table, blobs):
        if table in artifacts:
            assert artifacts[table] == blobs
        else:
            artifacts[table] = blobs

    def tiny_recipes():
        recipes = models.default_recipes()
        for recipe in recipes:
            if recipe["kind"] == "transformer":
                recipe["settings"].update(width=8, heads=2, layers=1, feedforward=16, batch_size=1024)
        return recipes

    for cells in (builder.preparation, builder.splitting, builder.training, builder.evaluation):
        scope = {"sf_options": {}, "spark": object(), "display": lambda x: None}
        for index, code in enumerate(cells):
            exec(compile(code, f"synthetic_cell_{index+1}", "exec"), scope)
            if index == 0:
                scope.update(MAX_EPOCHS=2, PATIENCE=1)
            if index == 1:
                scope.update(read_table=lambda name: Frame(tables[name]), read_query=lambda query: Frame(source),
                             read_artifacts=read_artifact, save_artifacts=save_artifact,
                             table_exists=lambda table: table in artifacts, default_recipes=tiny_recipes)
    report = json.loads(artifacts[scope["EVALUATION_TABLE"]]["evaluation.json"])
    assert report["features"] == feature_names
    assert set(report["metrics"]) == {"train", "validation", "test"}
    assert report["metrics"]["test"]["snapshots"] == 3414
    assert "train_deciles.csv" in artifacts[scope["EVALUATION_TABLE"]]
    selection = json.loads(artifacts[scope["SELECTION_TABLE"]]["selection.json"])
    assert selection["test_used_for_selection"] is False
    candidates = [name for name in artifacts if name.startswith(scope["RUN_PREFIX"] + "_MODEL_")]
    assert len(candidates) == 1


def test_single_transformer_configuration():
    recipes = models.default_recipes()
    assert len(recipes) == 1
    assert recipes[0]["kind"] == "transformer"
    assert recipes[0]["settings"]["dropout"] == .35
    assert 'for recipe in recipes:' not in '\n'.join(builder.training)
