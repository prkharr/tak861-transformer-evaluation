"""Execute every generated code cell against synthetic inputs and an artifact store.

The real selector, training, serializers, models, comparison, and notebook cell
code execute. Only private warehouse access, production dimensions and compute
budgets are replaced. This is software integration evidence, not a clinical run.
The builder writes exclusively into a temporary directory during this test.
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


@contextlib.contextmanager
def notebook_frontend_stub():
    """Restore only these keys, never the entire cache of imported torch modules."""
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: None
    ipython.version_info = (9, 0)
    frontend = types.ModuleType("IPython.display")
    frontend.display = lambda *a, **k: None
    replacements = {"IPython": ipython, "IPython.display": frontend}
    originals = {key: sys.modules.get(key) for key in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for key, original in originals.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


def synthetic_inputs():
    rng = np.random.default_rng(882)
    patients, snapshots, steps, width = 48, 2, 12, 8
    y = np.repeat(np.arange(patients) % 2, snapshots)
    X = rng.poisson(.5, (patients * snapshots, steps, width)).astype(np.float32)
    X[:, :3, 0] += (2 * y[:, None]).astype(np.float32)
    X[:, :, 5] = X[:, :, 4]  # Exercise quality removal and saved membership.
    X[:, :, 6] = 0
    X[0, 3, 7] = np.nan
    metadata = pd.DataFrame(dict(
        PATIENT_ID=np.repeat([f"synthetic_{i:03d}" for i in range(patients)], snapshots),
        END_DT=np.tile(["2024-01-20", "2024-04-20"], patients), RESP=y,
        SPLIT=np.repeat(["train"] * 36 + ["validation"] * 6 + ["test"] * 6, snapshots)))
    metadata["SPLIT_CONFIG"] = '{"synthetic_fixture":true}'
    fmap = pd.DataFrame(dict(FEATURE_INDEX=np.arange(width),
                             FEATURE_NAME=[f"DX_SYNTHETIC_{i}" for i in range(width)],
                             FEATURE_COLUMN=[f"F{i:04d}" for i in range(width)]))
    indices = {s: np.flatnonzero(metadata.SPLIT.eq(s)) for s in ("train", "validation", "test")}
    data = dict(X=X, y=y, metadata=metadata, feature_map=fmap, features=fmap.FEATURE_NAME.tolist(),
                indices=indices, historical_split_feature_hash_matches_current=False)
    baseline_names = [f"ENCODED_{i}" for i in range(49)]
    encoded = metadata[["PATIENT_ID", "END_DT", "RESP"]].copy()
    for i, name in enumerate(baseline_names):
        encoded[name] = rng.random(len(metadata)) + y * (i == 0)
    prefix = "DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_"
    tables = {
        prefix + "MODEL_TYPE": pd.DataFrame({"MODEL_TYPE": ["lightgbm"], "FEATURES": [json.dumps(baseline_names)]}),
        prefix + "FINAL_MODEL": pd.DataFrame({"FEATURES": baseline_names, "SEQ": np.arange(49)[::-1]}),
        prefix + "MODEL_DATA": encoded.iloc[::-1],
        prefix + "UNIVERSE_W_FEATURES_SCORED": metadata[["PATIENT_ID", "END_DT", "RESP"]].assign(SCORE=.2 + y * .5),
    }
    return data, tables


class NotebookFlow(unittest.TestCase):
    def _build_isolated(self, directory):
        spec = importlib.util.spec_from_file_location("test_notebook_builder", HERE / "build_notebooks.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        builder.REPO = directory
        builder.DEST = directory / "notebooks" / "delivery"
        builder.DEST.mkdir(parents=True)
        (directory / "notebooks" / "Historical 49 Snapshot Padding (20260927)").mkdir()
        warehouse = directory / "experiments" / "temporal_selection" / "input_helpers.py"
        warehouse.parent.mkdir(parents=True)
        shutil.copyfile(REPO / "experiments" / "temporal_selection" / "input_helpers.py", warehouse)
        builder.build()
        return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(builder.DEST.glob("*.ipynb"))]

    def _patch_runtime(self, namespace, store, data, source_tables, observations, stage):
        fs, tr, wf, cmp, b49, wh = [namespace[n] for n in ("fs", "tr", "wf", "cmp", "b49", "wh")]
        if "hashes" not in data:
            data["hashes"] = wh.input_fingerprints(data["X"], data["metadata"], data["features"])
        wf.load_count_inputs = lambda: copy.deepcopy(data)
        namespace["read_table"] = lambda table: source_tables[table].copy()
        namespace["display"] = lambda *a, **k: None
        namespace["plt"].show = lambda: namespace["plt"].close("all")

        def read(table, expected):
            return wh.unpack_artifacts(store[table], expected)

        def save(table, artifacts):
            if table in store:
                if read(table, artifacts) != artifacts:
                    raise FileExistsError("Synthetic immutable artifact conflict: " + table)
            else:
                # Exercise the real chunk codec, including checksums and lengths.
                store[table] = wh.pack_artifacts(artifacts, chunk_size=1700)
            self.assertEqual(read(table, artifacts), artifacts)

        for module in (wh, wf):
            module.table_exists = lambda table: table in store
            module.read_artifacts = read
            module.save_artifacts = save

        validate = fs.validate_count_inputs
        def small_validate(*args, **kwargs):
            if len(args) > 4:
                args = list(args)
                args[4] = data["X"].shape[-1]
            else:
                kwargs["expected_features"] = data["X"].shape[-1]
            return validate(*args, **kwargs)
        fs.validate_count_inputs = small_validate
        select = fs.select_ranked_features
        def small_selection(*args, **kwargs):
            config = dict(kwargs.get("config", {}))
            config.update(expected_features=data["X"].shape[-1], candidate_sizes=(2, 4), proxy_max_iter=200)
            kwargs["config"] = config
            return select(*args, **kwargs)
        fs.select_ranked_features = small_selection

        build_protocol = wf.build_protocol
        def tiny_protocol(*args, **kwargs):
            result = build_protocol(*args, **kwargs)
            chosen = {}
            # Deliberately retain insertion order different from alphabetical JSON
            # order to expose table-index collisions after saved protocol reload.
            for family in ("transformer", "lightgbm"):
                name, cfg = next((n, c) for n, c in result["candidates"].items()
                                 if c["family"] == family and c["subset"] == "all_eligible"
                                 and n.endswith("count_all_eligible"))
                cfg = copy.deepcopy(cfg)
                if family == "transformer":
                    cfg.update(d_model=16, depth=1, heads=2, ff=32, batch_size=16)
                else:
                    cfg.update(n_estimators=10, min_child_samples=3, num_leaves=5)
                chosen[name] = cfg
                ablation_name = "tf_capacity" if family == "transformer" else "gb_capacity"
                ablation = copy.deepcopy(result["candidates"][ablation_name])
                if family == "transformer":
                    ablation.update(d_model=16, depth=1, heads=2, ff=48, batch_size=16)
                else:
                    ablation.update(n_estimators=10, min_child_samples=3, num_leaves=7)
                chosen[ablation_name] = ablation
            result.update(candidates=chosen, seeds=[11, 12, 13], epochs=2, patience=1, cpu_threads=1)
            result["phase_a"] = sorted(n for n in chosen if "_count_" in n)
            result["phase_b"] = sorted(n for n in chosen if "_count_" not in n)
            result["feature_subsets"] = {"all_eligible": result["feature_subsets"]["all_eligible"]}
            result["candidate_tags"] = {"all_eligible": "all_eligible", "proxy_one_se": "all_eligible"}
            if "prediction_counts" in result:
                result["prediction_counts"] = {s: len(data["indices"][s]) for s in ("train", "validation")}
            if "artifact_candidate_order" in result:
                result["artifact_candidate_order"] = sorted(chosen)
            return result
        wf.build_protocol = tiny_protocol
        b49.DEFAULT_BASELINE_CONFIG.update(n_estimators=10, min_child_samples=3,
                                          early_stopping_rounds=2, n_jobs=1)
        fit_baseline = b49.train_encoded_baseline
        def small_baseline(*args, **kwargs):
            kwargs["config"] = dict(n_estimators=10, min_child_samples=3, early_stopping_rounds=2, n_jobs=1)
            return fit_baseline(*args, **kwargs)
        b49.train_encoded_baseline = small_baseline
        compare = cmp.compare_predictions
        def small_comparison(*args, **kwargs):
            kwargs["n_bootstrap"] = 30
            return compare(*args, **kwargs)
        cmp.compare_predictions = small_comparison

        select_threshold = cmp.select_validation_threshold
        def development_threshold(*args, **kwargs):
            self.assertEqual(stage, 4, "Threshold selection must finish in stage 4, before all held-out inference")
            self.assertEqual(observations["inference_calls"], 0)
            self.assertEqual(observations["inference49_calls"], 0)
            observations["threshold_calls"] += 1
            return select_threshold(*args, **kwargs)
        cmp.select_validation_threshold = development_threshold

        def assert_complete_lock():
            self.assertTrue(any(t.endswith("_LOCK") for t in store), "Inference occurred before persisted lock")
            lock = namespace["lock"]
            self.assertEqual(set(lock["threshold_policies"]), {"transformer", "lightgbm"})
            self.assertEqual(lock["baseline49"]["threshold_policy"]["selection_split"], "validation")
            self.assertEqual(len(lock["baseline49"]["model_hashes"]), 3)
            self.assertTrue(lock["baseline49"]["input_sha256"])
            for identities in [*lock["predictor_identities"].values(), lock["baseline49"]["predictor_identities"]]:
                self.assertEqual(len(identities), 3)
                self.assertEqual([item["seed"] for item in identities], [11, 12, 13])
                self.assertTrue(all(set(item) == {"seed", "model_sha256", "config_sha256", "transform_sha256"}
                                    for item in identities))

        predict, predict49 = tr.predict_saved, b49.predict_encoded
        def checked_prediction(*args, **kwargs):
            self.assertEqual(stage, 5, "Held-out inference entered a development notebook")
            assert_complete_lock()
            observations["inference_calls"] += 1
            return predict(*args, **kwargs)
        def checked_prediction49(*args, **kwargs):
            self.assertEqual(stage, 5)
            assert_complete_lock()
            observations["inference49_calls"] += 1
            return predict49(*args, **kwargs)
        tr.predict_saved, b49.predict_encoded = checked_prediction, checked_prediction49

        permutation = namespace["rel"].validation_permutation_diagnostic
        def checked_permutation(*args, **kwargs):
            self.assertEqual(stage, 4)
            assert_complete_lock()
            lock_table = namespace["RUN_PREFIX"] + "_LOCK"
            lock_names = {"lock.json", "projection_parameter_magnitude.csv"}
            before = wh.unpack_artifacts(store[lock_table], lock_names)["lock.json"]
            self.assertTrue(args[1].SPLIT.eq("validation").all())
            self.assertTrue(args[1].PATIENT_ID.is_unique)
            diagnostic = permutation(*args, **kwargs)
            self.assertEqual(before, wh.unpack_artifacts(store[lock_table], lock_names)["lock.json"])
            observations["reliance_calls"] += 1
            return diagnostic
        namespace["rel"].validation_permutation_diagnostic = checked_permutation

    def _execute(self, notebook, stage, store, data, sources, observations):
        namespace = {"spark": object(), "sf_options": {"synthetic_connection": "not_a_credential"}}
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(cells), 4)
        exec(compile(cells[0], f"stage{stage}:config", "exec"), namespace)
        exec(compile(cells[1], f"stage{stage}:helpers", "exec"), namespace)
        self._patch_runtime(namespace, store, data, sources, observations, stage)
        for index, source in enumerate(cells[2:], 2):
            try:
                exec(compile(source, f"stage{stage}:cell{index}", "exec"), namespace)
            except Exception as error:
                raise AssertionError(f"Generated stage {stage}, code cell {index} failed: {error}") from error
        observations["code_cells_executed"] += len(cells)
        return namespace

    def test_all_five_notebooks_execute_and_final_rerun_reuses_inference(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        data, sources = synthetic_inputs()
        store = {}
        observations = dict(inference_calls=0, inference49_calls=0, code_cells_executed=0, threshold_calls=0, reliance_calls=0)
        # Display is a notebook frontend facility, not part of computation.
        with tempfile.TemporaryDirectory(prefix="tak861_notebook_test_") as temp, contextlib.redirect_stdout(io.StringIO()), \
                notebook_frontend_stub():
            notebooks = self._build_isolated(Path(temp))
            self.assertEqual(len(notebooks), 5)
            hashes = {n["metadata"]["tak861"]["implementation_sha256"] for n in notebooks}
            self.assertEqual(len(hashes), 1)
            for stage, notebook in enumerate(notebooks, 1):
                namespace = self._execute(notebook, stage, store, data, sources, observations)
                if stage < 5:
                    self.assertEqual(observations["inference_calls"], 0)
                    self.assertEqual(observations["inference49_calls"], 0)
                if stage == 4:
                    self.assertEqual(observations["reliance_calls"], 2)
                    # The audit cell alone must reuse its complete artifact on rerun.
                    audit_cell = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"][-1]
                    exec(compile(audit_cell, "stage4:cached_reliance", "exec"), namespace)
                    self.assertEqual(observations["reliance_calls"], 2)
            self.assertEqual(observations["inference_calls"], 6)
            self.assertEqual(observations["inference49_calls"], 3)
            self.assertEqual(observations["threshold_calls"], 3)
            prefix = namespace["RUN_PREFIX"]
            self.assertIn(prefix + "_FINAL", store)
            self.assertIn(prefix + "_ANCHORS", store)
            self.assertIn(prefix + "_RELIANCE", store)
            final = namespace["final_report"]
            self.assertIn("retrospective", final["heldout_status"])
            self.assertEqual(set(final["metrics"]), {"transformer", "lightgbm"})
            self.assertEqual(set(final["baseline49_metrics"]), {"train", "validation", "test"})
            self.assertFalse(namespace["lock"]["test_used_for_new_selection"])
            self.assertEqual(namespace["original_context"]["evidence_class"], "Reproduced")
            self.assertEqual(set(namespace["original_context"]["original_v63_validation"]),
                             {"snapshots", "positives", "average_precision", "roc_auc"})
            before = dict(observations)
            self._execute(notebooks[-1], 5, store, data, sources, observations)
            self.assertEqual(observations["inference_calls"], before["inference_calls"])
            self.assertEqual(observations["inference49_calls"], before["inference49_calls"])
            self.assertGreater(observations["code_cells_executed"], 25)
            self.assertEqual(final["validation_reliance"], namespace["audit_report"])
            reference = namespace["audit_report"]["matched_count_lgb_reference_contract"]
            self.assertFalse(reference["architecture_controlled_reliance_comparison"])
            self.assertEqual(reference["transformer_input"]["columns"], reference["reference_count_lgb_input"]["columns"])
            self.assertIn("Same feature IDs", reference["label"])

            # Temporal information identity is stronger than matching names.
            same = namespace["same_count_feature_information"]
            tf = dict(columns=[0, 2], representation="monthly")
            gb = dict(columns=[0, 2], representation="flattened")
            with self.subTest(contract="reference temporal information"):
                self.assertTrue(same(tf, gb))
                self.assertFalse(same({**tf, "representation":"quarterly"}, gb))
                self.assertFalse(same({**tf, "representation":"without_step0"}, gb))
                self.assertTrue(same({**tf, "representation":"without_step0"}, {**gb, "drop_step0":True}))
                self.assertFalse(same(tf, {**gb, "columns":[0, 3]}))

            # Recomputed self-hashes and a valid inner/outer codec do not permit
            # changing preprocessing after lock. Test all three model families.
            wh, wf = namespace["wh"], namespace["wf"]
            for family in ("transformer", "lightgbm", "baseline49"):
                with self.subTest(lock_transform=family):
                    if family == "baseline49":
                        original_runs = namespace["baseline_runs"]
                        table = prefix + "_BASE49_S11"
                        expected = namespace["lock"]["baseline49"]["predictor_identities"]
                    else:
                        winner = namespace["winners"][family]
                        original_runs = namespace["results"][winner]
                        index = sorted(namespace["protocol"]["candidates"]).index(winner)
                        table = wf.run_artifact_table(prefix, index, 11)
                        expected = namespace["lock"]["predictor_identities"][family]
                    payload = wh.unpack_artifacts(store[table], wf.RUN_ARTIFACT_NAMES)
                    contract = json.loads(payload["contract.json"])["contract"]
                    altered_run = wf.unpack_run(payload, contract)
                    transform = altered_run["summary"]["transform"]
                    if family == "baseline49":
                        transform["imputation"] = "synthetic altered identity preprocessing"
                    else:
                        transform["median"][0] += .25
                    altered_run["summary"]["transform_sha256"] = namespace["tr"].stable_hash(transform)
                    repacked = wf.pack_run(altered_run, contract)
                    outer = wh.pack_artifacts(repacked, chunk_size=1700)
                    decoded = wf.unpack_run(wh.unpack_artifacts(outer, wf.RUN_ARTIFACT_NAMES), contract)
                    altered_runs = [decoded, *original_runs[1:]]
                    self.assertEqual(decoded["summary"]["model_sha256"], original_runs[0]["summary"]["model_sha256"])
                    with self.assertRaisesRegex(ValueError, "Locked predictor model/config/preprocessing identity changed"):
                        namespace["verify_frozen_predictors"](altered_runs, expected, family)
                    if family == "baseline49":
                        original_rows = store[table]
                        store[table] = outer
                        with self.assertRaisesRegex(AssertionError, "Locked predictor model/config/preprocessing identity changed"):
                            self._execute(notebooks[-1], 5, store, data, sources, observations)
                        store[table] = original_rows

            # A valid cached FINAL codec cannot carry a different reliance report
            # while retaining the same lock hash and model predictions.
            final_table = prefix + "_FINAL"
            original_rows = store[final_table]
            final_payload = wh.unpack_artifacts(original_rows, namespace["final_names"])
            altered_report = json.loads(final_payload["comparison.json"])
            altered_report["validation_reliance"]["interpretation"] = "synthetic stale reliance report"
            final_payload["comparison.json"] = wf.json_bytes(altered_report)
            store[final_table] = wh.pack_artifacts(final_payload, chunk_size=1700)
            with self.assertRaisesRegex(AssertionError, "Final report reliance mismatch"):
                self._execute(notebooks[-1], 5, store, data, sources, observations)
            store[final_table] = original_rows

            # Cached FINAL must not bypass current baseline identity checks.
            encoded_table = "DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_MODEL_DATA"
            original_values = sources[encoded_table].copy()
            sources[encoded_table].loc[:, "ENCODED_0"] += .25
            with self.assertRaisesRegex(AssertionError, "Baseline input/order changed"):
                self._execute(notebooks[-1], 5, store, data, sources, observations)
            sources[encoded_table] = original_values

            # Repack altered blobs through the valid outer codec. Inner model
            # contracts still must reject changes even with cached final scores.
            wh = namespace["wh"]
            for table in (prefix + "_BASE49_S11", prefix + "_C000_S11"):
                original_rows = store[table]
                payload = wh.unpack_artifacts(original_rows, namespace["wf"].RUN_ARTIFACT_NAMES)
                payload["model.bin"] += b"synthetic_tamper"
                store[table] = wh.pack_artifacts(payload, chunk_size=1700)
                with self.assertRaisesRegex(AssertionError, "hash mismatch"):
                    self._execute(notebooks[-1], 5, store, data, sources, observations)
                store[table] = original_rows

            # Any changed locked policy invalidates the persisted final report.
            lock_table = prefix + "_LOCK"
            old_lock = store[lock_table]
            payload = wh.unpack_artifacts(old_lock, {"lock.json", "projection_parameter_magnitude.csv"})
            altered_lock = json.loads(payload["lock.json"])
            altered_lock["baseline49"]["threshold_policy"]["threshold"] = .123456789
            payload["lock.json"] = namespace["wf"].json_bytes(altered_lock)
            store[lock_table] = wh.pack_artifacts(payload)
            with self.assertRaisesRegex(AssertionError, "mismatch"):
                self._execute(notebooks[-1], 5, store, data, sources, observations)
            store[lock_table] = old_lock
            self.assertEqual(observations["inference_calls"], before["inference_calls"])
            self.assertEqual(observations["inference49_calls"], before["inference49_calls"])
            self.assertEqual(observations["threshold_calls"], 3)


if __name__ == "__main__":
    unittest.main()
