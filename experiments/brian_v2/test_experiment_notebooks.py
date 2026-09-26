"""Execute the delivered notebook cells against synthetic data and memory storage.

No private tables, identifiers, credentials, or Snowflake connections are used.
Only transport and input construction are stubbed; notebook orchestration, feature
selection, fitting, checkpoint restoration, reporting, and artifact creation run.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch


HERE = Path(__file__).resolve().parent
# Tests are distributed both beside the output builder and in the repository.
DELIVERY = (HERE.parent / "Brian_Experiments" if HERE.name == "dl_poc_experiment_work"
            else HERE.parents[1] / "notebooks" / "Brian Experiments")
PREFIX = "TAK861_TX_READY_V63_DL_POC"
EXP_PREFIX = PREFIX + "_BRIAN_V2"
SOURCES = {stage: json.loads((DELIVERY / filename).read_text(encoding="utf-8"))["cells"]
           for stage, filename in {
               "train": "03_transformer_training_DL_POC.ipynb",
               "eval": "04_transformer_evaluation_DL_POC.ipynb"}.items()}
EXPERIMENTS = ["all_baseline", "reduced150", "reduced150_dropout", "reduced150_decay",
               "reduced150_small", "reduced150_logistic"]
torch.set_num_threads(1)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class RestrictedRows:
    """Reject every training-time TEST access, including implicit conversion."""
    def __init__(self, values, forbidden):
        self.values, self.shape = values, values.shape
        self.forbidden = set(forbidden)
        self.reads = []
        if hasattr(values, "_mmap"):
            self._mmap = values._mmap

    def __len__(self):
        return len(self.values)

    def __array__(self, *args, **kwargs):
        raise AssertionError("Notebook must not convert the full source array.")

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows = item[0]
        else:
            rows = item
        rows = np.arange(len(self.values))[rows] if isinstance(rows, slice) else np.asarray(rows).reshape(-1)
        assert not (set(np.asarray(rows).reshape(-1).tolist()) & self.forbidden), "TEST read during training/selection."
        self.reads.extend(np.asarray(rows).reshape(-1).tolist())
        return self.values[item]


def base_data():
    rng = np.random.default_rng(817)
    n, f = 84, 7
    y = np.tile([0, 0, 1], n // 3).astype(np.float32)
    X = rng.poisson(.25, (n, 12, f)).astype(np.float32)
    # Positive/negative examples have equal yearly totals for one temporal signal.
    X[:, :, 0] = 0
    for i in range(n):
        X[i, 0 if y[i] else 11, 0] = 6
    X[:, 3, 1] += y * 3
    X[:, :, 6] = 0  # Empty TRAIN column, excluded by reduced selection.
    features = [f"RX__SYNTHETIC_{i}" for i in range(f)]
    split = ["train"] * 60 + ["validation"] * 12 + ["test"] * 12
    metadata = pd.DataFrame({"PATIENT_ID": [f"SYNTHETIC_PERSON_{i:03d}" for i in range(n)],
                             "END_DT": ["2024-05-21"] * n, "RESP": y.astype(int), "SPLIT": split})
    hashes = {"feature_names_sha256": digest(features),
              "model_input_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
              "snapshot_manifest_sha256": digest(metadata.astype(str).values.tolist()),
              "split_config_sha256": digest({"seed": 42})}
    return {"X": X, "y": y, "features": features, "metadata": metadata,
            "indices": {"train": np.arange(60), "validation": np.arange(60, 72), "test": np.arange(72, 84)},
            "hashes": hashes, "split_creation": {"seed": 42}}


class NotebookHarness:
    def __init__(self, source=None):
        self.base = base_data() if source is None else copy.deepcopy(source)
        self.store, self.writes, self.reads, self.loads, self.displays = {}, [], [], [], []
        source_hash = digest([[str(r.PATIENT_ID), str(r.END_DT), int(r.RESP)]
                              for r in self.base["metadata"].itertuples()])
        preparation = {"mode": "reuse_and_validate_saved_v63_tensor", "source_prefix": PREFIX,
                       "tensor_shape": list(self.base["X"].shape), "source_snapshots_sha256": source_hash,
                       **{key: self.base["hashes"][key] for key in ("model_input_sha256", "feature_names_sha256")}}
        split_audit = {"mode": "reuse_original_patient_split", "source_prefix": PREFIX,
                       "new_assignments_created": False, "source_snapshots_sha256": source_hash,
                       **{key: self.base["hashes"][key] for key in (
                           "snapshot_manifest_sha256", "split_config_sha256", "feature_names_sha256")}}
        self.store[EXP_PREFIX + "_PREPARATION"] = {"preparation_report.json": canonical(preparation).encode()}
        self.store[EXP_PREFIX + "_SPLIT_AUDIT"] = {"split_audit.json": canonical(split_audit).encode()}

    def table_exists(self, table):
        return table in self.store

    def read_artifacts(self, table, names):
        assert table in self.store, f"Missing memory artifact table {table}"
        assert set(self.store[table]) == set(names), "Saved artifact names differ."
        self.reads.append(table)
        return dict(self.store[table])

    def save_artifacts(self, table, artifacts):
        assert table.startswith(EXP_PREFIX + "_"), "Original V63 namespace must never be written."
        assert all(isinstance(blob, bytes) for blob in artifacts.values())
        if table in self.store:
            assert self.store[table] == artifacts, "Changed saved artifacts require a new suite."
        else:
            self.store[table] = dict(artifacts)
        # Simulate the save helper's immediate exact read-back check as well.
        assert self.read_artifacts(table, artifacts) == artifacts
        self.writes.append(table)

    def load_inputs(self, guard_test):
        data = copy.deepcopy({key: value for key, value in self.base.items() if key != "X"})
        temporary = tempfile.TemporaryDirectory(prefix="dl_poc_notebook_test_")
        path = Path(temporary.name) / "tensor.f32"
        X = np.memmap(path, dtype="<f4", mode="w+", shape=self.base["X"].shape)
        X[:] = self.base["X"]
        X.flush()
        X.flags.writeable = False
        forbidden = data["indices"]["test"] if guard_test else []
        data["X"] = RestrictedRows(X, forbidden)
        data["y"] = RestrictedRows(data["y"], forbidden)
        data["temporary_directory"] = temporary
        self.loads.append((data, path))
        return data

    def start(self, stage):
        namespace = {"__name__": "synthetic_notebook", "sf_options": {}, "display": self.displays.append}
        self.cell(namespace, stage, 1)
        if stage == "train":
            namespace.update(EXPERIMENT_NAMES=list(EXPERIMENTS), MAX_SELECTED_FEATURES=3, MIN_TRAIN_PATIENTS=1)
        self.cell(namespace, stage, 2)
        namespace.update(table_exists=self.table_exists, read_artifacts=self.read_artifacts,
                         save_artifacts=self.save_artifacts, load_inputs=lambda: self.load_inputs(stage == "train"))
        return namespace

    @staticmethod
    def cell(namespace, stage, number):
        source = SOURCES[stage][number - 1]["source"]
        if isinstance(source, list):
            source = "".join(source)
        exec(compile(source, f"delivered_{stage}:cell{number}", "exec"), namespace)

    @staticmethod
    def fast_recipes(namespace):
        for recipe in namespace["recipes"]:
            recipe["model_settings"].update(d_model=8, n_heads=2, encoder_layers=1, feedforward_dim=16)
            recipe["training_settings"].update(epochs=2, patience=1, batch_size=8, device="cpu")

    def train(self, forbid_fit=False):
        namespace = self.start("train")
        # Apply synthetic-test sizes before cell 4 checks an already-finalized
        # suite's recipe signature; production recipes are separately tested.
        original_recipes = namespace["experiment_recipes"]
        def fast_recipes(*args, **kwargs):
            recipes = original_recipes(*args, **kwargs)
            self.fast_recipes({"recipes": recipes})
            return recipes
        namespace["experiment_recipes"] = fast_recipes
        if forbid_fit:
            def no_fit(*args, **kwargs):
                raise AssertionError("Saved completed experiments must be reused without fitting.")
            namespace["run_experiment"] = no_fit
        for number in range(3, 9):
            self.cell(namespace, "train", number)
        plt.close("all")
        return namespace

    def evaluate(self):
        namespace = self.start("eval")
        def no_fit(*args, **kwargs):
            raise AssertionError("Evaluation cannot fit or select features.")
        namespace.update(run_experiment=no_fit, select_features=no_fit)
        for number in range(3, 9):
            self.cell(namespace, "eval", number)
        plt.close("all")
        return namespace

    def cleanup(self):
        for data, _ in self.loads:
            if not data["X"]._mmap.closed:
                data["X"]._mmap.close()
            data["temporary_directory"].cleanup()
        plt.close("all")


@pytest.fixture(scope="module")
def suite():
    harness = NotebookHarness()
    try:
        training = harness.train()
        yield harness, training
    finally:
        harness.cleanup()


def test_all_six_delivered_training_experiments_save_and_choose_validation_winner(suite):
    harness, ns = suite
    assert len(SOURCES["train"]) == len(SOURCES["eval"]) == 8
    assert all(cell["cell_type"] == "code" for group in SOURCES.values() for cell in group)
    assert len(ns["run_summaries"]) == len(ns["run_records"]) == 6
    assert ns["CORE_FEATURES"] == [] and ns["selection"]["core_status"] == "pending_approved_list"
    ordered = sorted(ns["run_summaries"], key=lambda s: (-s["validation_metrics"]["average_precision"], s["run_id"]))
    assert ns["selection"]["selected_run_id"] == ordered[0]["run_id"]
    assert ns["selection"]["test_used_for_selection"] is False
    assert "previously" in ns["selection"]["test_status"].lower()
    for summary, record in zip(ns["run_summaries"], ns["run_records"]):
        artifacts = harness.store[record["model_table"]]
        assert set(artifacts) == ns["TRAINING_ARTIFACT_NAMES"] and len(artifacts) == 8
        manifest = json.loads(artifacts["feature_manifest.json"])
        checkpoint = torch.load(io.BytesIO(artifacts["checkpoint.pt"]), weights_only=True)
        assert manifest == checkpoint["feature_manifest"] == summary["feature_manifest"]
        assert summary["test_inference_performed"] is False
        assert len(manifest["selected_indices"]) == (7 if summary["experiment_name"] == "all_baseline" else 3)
        assert artifacts["training_history.png"].startswith(b"\x89PNG")
        assert len(pd.read_csv(io.BytesIO(artifacts["training_topk.csv"]))) == 3
        assert len(pd.read_csv(io.BytesIO(artifacts["validation_topk.csv"]))) == 3
    assert set(harness.store[ns["SELECTION_TABLE"]]) == ns["SELECTION_ARTIFACT_NAMES"]
    assert ns["SELECTION_TABLE"] in harness.reads
    assert ns["data"]["X"]._mmap.closed
    assert not harness.loads[0][1].exists()
    assert all(table.startswith(EXP_PREFIX + "_") for table in harness.writes)


def test_unchanged_suite_resumes_without_retraining(suite):
    harness, first = suite
    original = copy.deepcopy(harness.store)
    resumed = harness.train(forbid_fit=True)
    assert resumed["selection"] == first["selection"]
    assert resumed["run_summaries"] == first["run_summaries"]
    assert harness.store == original
    assert resumed["data"]["X"]._mmap.closed


def test_delivered_evaluation_scores_only_selected_run_and_saves_11_aggregate_artifacts(suite):
    original, trained = suite
    harness = NotebookHarness(original.base)
    harness.store = copy.deepcopy(original.store)
    try:
        ns = harness.evaluate()
        assert ns["RUN_ID"] == trained["selection"]["selected_run_id"]
        assert ns["checkpoint"]["validation_threshold"] == trained["selection"]["validation_threshold"]
        assert ns["saved_manifest"] == ns["checkpoint"]["feature_manifest"]
        assert len(ns["test_y"]) == len(ns["test_scores"]) == 12
        assert set(ns["results"]["figures"]) == {
            "gains", "lift", "topk_performance", "response_rate", "discrimination", "confusion_matrix"}
        artifacts = harness.store[ns["EVALUATION_TABLE"]]
        assert len(artifacts) == 11
        assert set(artifacts) == {"evaluation_metadata.json", "global_metrics.csv", "deciles.csv", "topk.csv", "ties.csv",
                                  "gains.png", "lift.png", "topk_performance.png", "response_rate.png", "discrimination.png", "confusion_matrix.png"}
        metadata = json.loads(artifacts["evaluation_metadata.json"])
        assert metadata["test_used_for_selection"] is False
        assert "previously" in metadata["test_status"].lower()
        assert metadata["evaluation_unit"] == "patient/date snapshot"
        assert ns["data"]["X"]._mmap.closed and not harness.loads[-1][1].exists()
        assert harness.writes == [ns["EVALUATION_TABLE"]]
        for name, blob in artifacts.items():
            if name.endswith(".csv"):
                frame = pd.read_csv(io.BytesIO(blob))
                assert not {"PATIENT_ID", "END_DT", "P_RESP1"} & set(frame.columns)
            if name.endswith((".csv", ".json")):
                assert b"SYNTHETIC_PERSON_" not in blob
        # A second TEST execution is explicitly refused before loading fresh inputs.
        second = harness.start("eval")
        with pytest.raises(FileExistsError, match="saved TEST report"):
            harness.cell(second, "eval", 3)
        assert len(harness.loads) == 1
    finally:
        harness.cleanup()


@pytest.mark.parametrize("tamper, expected", [
    ("selection_uses_test", "incompatible frozen"),
    ("checkpoint_digest", "checkpoint changed"),
    ("manifest_file", "different settings|checkpoint disagree"),
    ("preparation_hash", "Preparation audit differs"),
    ("selection_input_hash", "Inputs changed"),
])
def test_evaluation_stops_on_inconsistent_saved_provenance(suite, tamper, expected):
    original, trained = suite
    harness = NotebookHarness(original.base)
    harness.store = copy.deepcopy(original.store)
    selection_table = trained["SELECTION_TABLE"]
    selection = json.loads(harness.store[selection_table]["selection.json"])
    if tamper == "selection_uses_test":
        selection["test_used_for_selection"] = True
    elif tamper == "checkpoint_digest":
        selection["selected_checkpoint_sha256"] = "changed"
    elif tamper == "selection_input_hash":
        selection["input_hashes"]["model_input_sha256"] = "changed"
    elif tamper == "manifest_file":
        table = selection["selected_model_table"]
        manifest = json.loads(harness.store[table]["feature_manifest.json"])
        manifest["selected_features"].reverse()
        harness.store[table]["feature_manifest.json"] = canonical(manifest).encode()
    elif tamper == "preparation_hash":
        table = EXP_PREFIX + "_PREPARATION"
        audit = json.loads(harness.store[table]["preparation_report.json"])
        audit["model_input_sha256"] = "changed"
        harness.store[table]["preparation_report.json"] = canonical(audit).encode()
    harness.store[selection_table]["selection.json"] = canonical(selection).encode()
    try:
        ns = harness.start("eval")
        with pytest.raises(ValueError, match=expected):
            for number in (3, 4):
                harness.cell(ns, "eval", number)
        assert not harness.writes
        assert "test_scores" not in ns
        for data, path in harness.loads:
            assert data["X"]._mmap.closed, "Failed evaluation verification must release its mapping."
            assert not path.exists()
    finally:
        harness.cleanup()


@pytest.mark.parametrize("tamper", [
    "training_complete", "model_kind", "model_config", "training_settings",
    "best_ap", "metric_ap", "best_epoch", "feature_manifest", "manifest_sha256",
])
def test_saved_summary_inconsistency_is_rejected_and_evaluation_mapping_closed(suite, tamper):
    original, trained = suite
    harness = NotebookHarness(original.base)
    harness.store = copy.deepcopy(original.store)
    selection = trained["selection"]
    table = selection["selected_model_table"]
    summary = json.loads(harness.store[table]["training_summary.json"])
    if tamper == "training_complete":
        summary["training_complete"] = False
    elif tamper == "model_kind":
        summary["model_kind"] = "logistic" if summary["model_kind"] == "transformer" else "transformer"
    elif tamper == "model_config":
        summary["model_config"]["dropout"] += .05
    elif tamper == "training_settings":
        summary["training_settings"]["learning_rate"] *= 2
    elif tamper == "best_ap":
        summary["best_validation_average_precision"] += .01
    elif tamper == "metric_ap":
        summary["validation_metrics"]["average_precision"] += .01
    elif tamper == "best_epoch":
        summary["best_epoch"] = 999
    elif tamper == "feature_manifest":
        summary["feature_manifest"]["selected_features"].reverse()
    elif tamper == "manifest_sha256":
        summary["feature_manifest_sha256"] = "changed"
    harness.store[table]["training_summary.json"] = canonical(summary).encode()
    try:
        ns = harness.start("eval")
        harness.cell(ns, "eval", 3)
        with pytest.raises(ValueError, match="Training record|summary and checkpoint disagree"):
            harness.cell(ns, "eval", 4)
        assert len(harness.loads) == 1
        assert ns["data"]["X"]._mmap.closed
        assert not harness.loads[0][1].exists()
        assert not harness.writes and "test_scores" not in ns
    finally:
        harness.cleanup()


@pytest.mark.parametrize("tamper", ["model_kind", "model_config", "training_settings", "saved_manifest"])
def test_checked_saved_training_rejects_checkpoint_recipe_or_manifest_disagreement(suite, tamper):
    harness, trained = suite
    recipe = copy.deepcopy(trained["recipes"][0])
    manifest = copy.deepcopy(trained["selections"][recipe["feature_mode"]][0])
    record = trained["run_records"][0]
    artifacts = copy.deepcopy(harness.store[record["model_table"]])
    signature = record["experiment_signature"]
    if tamper == "model_kind":
        recipe["model_kind"] = "logistic"
    elif tamper == "model_config":
        recipe["model_settings"]["dropout"] += .05
    elif tamper == "training_settings":
        recipe["training_settings"]["learning_rate"] *= 2
    elif tamper == "saved_manifest":
        saved = json.loads(artifacts["feature_manifest.json"])
        saved["selected_features"].reverse()
        artifacts["feature_manifest.json"] = canonical(saved).encode()
    with pytest.raises(ValueError, match="summary and checkpoint disagree|different settings"):
        trained["checked_saved_training"](artifacts, trained["data"], recipe, manifest, signature)


def test_delivered_recipe_differences_are_controlled_before_speed_overrides():
    harness = NotebookHarness()
    ns = harness.start("train")
    recipes = {recipe["name"]: recipe for recipe in ns["experiment_recipes"]("CHECK", EXPERIMENTS, 42)}
    base, reduced = recipes["all_baseline"], recipes["reduced150"]
    assert base["model_settings"] == reduced["model_settings"]
    assert base["training_settings"] == reduced["training_settings"]
    assert (base["feature_mode"], reduced["feature_mode"]) == ("all", "reduced")
    for name, section, deltas in [
        ("reduced150_dropout", "model_settings", {"dropout": .3}),
        ("reduced150_decay", "training_settings", {"weight_decay": .001}),
        ("reduced150_small", "model_settings", {"d_model": 64, "encoder_layers": 1, "feedforward_dim": 128}),
    ]:
        expected = copy.deepcopy(reduced[section])
        expected.update(deltas)
        assert recipes[name][section] == expected
        unchanged = "training_settings" if section == "model_settings" else "model_settings"
        assert recipes[name][unchanged] == reduced[unchanged]
        assert recipes[name]["reference"] == "reduced150"
    assert recipes["reduced150_logistic"]["model_kind"] == "logistic"
    assert recipes["reduced150_logistic"]["feature_mode"] == "reduced"
    assert reduced["training_settings"]["epochs"] == 20
    assert reduced["training_settings"]["patience"] == 5
