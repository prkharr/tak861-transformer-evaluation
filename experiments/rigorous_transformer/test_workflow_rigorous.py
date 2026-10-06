"""Offline artifact/stage/search integrity checks using an in-memory warehouse."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("rigorous_workflow", Path(__file__).with_name("workflow.py"))
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)
w.stable_hash = lambda value: hashlib.sha256(w.json_bytes(value)).hexdigest()


def fake_run(config=None, seed=42, ntrain=8, nval=4, encoded=False):
    cfg = copy.deepcopy(config or dict(family="transformer", representation="monthly", columns=[0, 1]))
    transform = (dict(kind="identity_encoded_numeric", input_dim=49, imputation="native LightGBM missing-value handling")
                 if encoded else dict(median=[0., 0.], center=[0., 0.], divisor=[1., 1.], log=True))
    blob = ("synthetic-model-" + str(seed)).encode()
    summary = dict(config=cfg, transform=transform, seed=seed, training_complete=True,
                   model_sha256=hashlib.sha256(blob).hexdigest(), config_sha256=w.stable_hash(cfg),
                   transform_sha256=w.stable_hash(transform), raw_input_shape=[49] if encoded else [12, len(cfg["columns"])],
                   train_metrics=dict(average_precision=.6), validation_metrics=dict(average_precision=.5),
                   wall_seconds=.01, parameter_count=10, validation_evaluations=2)
    return dict(summary=summary, history=[dict(epoch=1, validation_ap=.5)], model_blob=blob,
                train_scores=np.linspace(.1, .9, ntrain), validation_scores=np.linspace(.1, .9, nval))


def retag_payload(artifacts, name, value):
    """Rehashing a payload still cannot evade semantic/schema validation."""
    result = copy.deepcopy(artifacts)
    result[name] = value
    envelope = json.loads(result["contract.json"])
    envelope["payload_sha256"][name] = hashlib.sha256(value).hexdigest()
    result["contract.json"] = w.json_bytes(envelope)
    return result


class ArtifactContracts(unittest.TestCase):
    def setUp(self):
        self.contract = dict(protocol_hash="protocol", candidate="tf_a", seed=42,
                             prediction_counts=dict(train=8, validation=4))
        self.run = fake_run()
        self.artifacts = w.pack_run(self.run, self.contract)

    def test_roundtrip_safe_arrays_and_versioned_envelope(self):
        loaded = w.unpack_run(self.artifacts, self.contract, expected_config=self.run["summary"]["config"])
        self.assertEqual(loaded["summary"], self.run["summary"])
        self.assertEqual(loaded["history"], self.run["history"])
        np.testing.assert_array_equal(loaded["train_scores"], self.run["train_scores"])
        self.assertEqual(json.loads(self.artifacts["contract.json"])["format_version"], 2)

    def test_each_payload_and_contract_tampering_is_rejected(self):
        for name in w.RUN_ARTIFACT_NAMES - {"contract.json"}:
            with self.subTest(name=name):
                altered = copy.deepcopy(self.artifacts)
                altered[name] += b"corrupt"
                with self.assertRaisesRegex(ValueError, "hash"):
                    w.unpack_run(altered, self.contract)
        with self.assertRaisesRegex(ValueError, "contract"):
            w.unpack_run(self.artifacts, {**self.contract, "seed": 99})
        missing = dict(self.artifacts)
        del missing["history.json"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            w.unpack_run(missing, self.contract)

    def test_internal_hash_checks_even_if_payload_hash_is_updated(self):
        for key in ("config", "transform"):
            summary = copy.deepcopy(self.run["summary"])
            summary[key]["changed"] = True
            altered = retag_payload(self.artifacts, "summary.json", w.json_bytes(summary))
            with self.assertRaisesRegex(ValueError, "hash"):
                w.unpack_run(altered, self.contract)
        altered = retag_payload(self.artifacts, "model.bin", b"another-model")
        with self.assertRaisesRegex(ValueError, "hash"):
            w.unpack_run(altered, self.contract)

    def test_prediction_shapes_finiteness_range_and_lengths(self):
        for values in (np.array([.2, np.nan, .5, .7]), np.array([.2, .5, .6, 1.1]),
                       np.zeros((4, 1)), np.zeros(3)):
            with self.subTest(shape=values.shape):
                altered = retag_payload(self.artifacts, "validation.npy", w.array_bytes(values))
                with self.assertRaises(ValueError):
                    w.unpack_run(altered, self.contract)
        with self.assertRaisesRegex(ValueError, "length"):
            w.unpack_run(self.artifacts, self.contract, expected_lengths={"train":9, "validation":4})

    def test_unsafe_object_npy_is_not_loaded(self):
        buffer = io.BytesIO()
        np.save(buffer, np.array([{"object": "not permitted"}], dtype=object), allow_pickle=True)
        altered = retag_payload(self.artifacts, "validation.npy", buffer.getvalue())
        with self.assertRaises(ValueError):
            w.unpack_run(altered, self.contract)

    def test_complete_seed_and_candidate_config_are_enforced(self):
        for update in (dict(training_complete=False), dict(seed=3), dict(raw_input_shape=[])):
            summary = {**self.run["summary"], **update}
            altered = retag_payload(self.artifacts, "summary.json", w.json_bytes(summary))
            with self.assertRaises(ValueError):
                w.unpack_run(altered, self.contract)
        with self.assertRaisesRegex(ValueError, "declared candidate"):
            w.unpack_run(self.artifacts, self.contract, expected_config={"family":"lightgbm"})

    def test_encoded49_contract_does_not_require_count_transform(self):
        cfg = dict(family="lightgbm49", representation="encoded_snapshot", n_estimators=500)
        result = fake_run(cfg, encoded=True)
        contract = dict(protocol_hash="fixed", candidate="lightgbm49_refit", seed=42,
                        input_sha256="encoded-source", feature_sha256="feature-order")
        loaded = w.unpack_run(w.pack_run(result, contract), contract,
                              expected_lengths={"train":8, "validation":4}, expected_config=cfg)
        self.assertEqual(loaded["summary"]["raw_input_shape"], [49])
        self.assertEqual(loaded["summary"]["transform"]["kind"], "identity_encoded_numeric")

    def test_stage_hashes_and_object_arrays(self):
        data = dict(hashes=dict(input="a", map="b"))
        report = dict(input_hashes=data["hashes"], implementation_hash="code1")
        w.verify_stage(data, report, "code1")
        with self.assertRaisesRegex(ValueError, "Inputs/code changed"):
            w.verify_stage(data, report, "code2")
        with self.assertRaises(ValueError):
            w.array_bytes(np.array([object()], dtype=object))


class SearchResume(unittest.TestCase):
    def setUp(self):
        self.store = {}
        self.calls = []
        self.data = dict(X=np.ones((16, 12, 3), dtype=np.float32), y=np.arange(16) % 2,
                         metadata=pd.DataFrame(dict(PATIENT_ID=[f"p{i}" for i in range(16)])),
                         hashes=dict(tensor="source-sha"),
                         indices={"train":np.arange(8), "validation":np.arange(8,12), "test":np.arange(12,16)})
        self.data["X"][12:] = -999  # No TEST values may reach the fitting callback.
        # Deliberately reverse insertion order: persistence sorts dictionary keys.
        self.protocol = dict(candidates={"z_last":dict(family="transformer", representation="monthly", columns=[0,2]),
                                         "a_first":dict(family="lightgbm", representation="windows", columns=[0,1,2])},
                             seeds=[42,142,242], input_hashes=self.data["hashes"], implementation_hash="code-sha",
                             epochs=2, patience=2, cpu_threads=1, device="cpu", prediction_counts={"train":8,"validation":4})
        w.table_exists = lambda table: table in self.store
        w.read_artifacts = lambda table, names: copy.deepcopy(self.store[table])
        def save(table, artifacts):
            if table in self.store:
                raise ValueError("Cannot overwrite a saved fit")
            self.store[table] = copy.deepcopy(artifacts)
        w.save_artifacts = save
        def train(x, y, groups, xv, yv, cfg, seed, **kwargs):
            self.assertTrue((x >= 0).all() and (xv >= 0).all())
            self.assertEqual(len(set(groups) & set(kwargs["groups_validation"])), 0)
            self.calls.append((cfg["family"],seed))
            return fake_run(cfg, seed, len(y), len(yv))
        w.train_candidate = train

    def test_sorted_order_resume_after_canonical_protocol_roundtrip(self):
        with contextlib.redirect_stdout(io.StringIO()):
            initial = w.execute_search(self.data, self.protocol, "SYNTHETIC_RUN")
        self.assertEqual(list(initial), ["a_first", "z_last"])
        self.assertEqual(len(self.calls), 6)
        reloaded = json.loads(w.json_bytes(self.protocol))
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = w.execute_search(self.data, reloaded, "SYNTHETIC_RUN")
        self.assertEqual(len(self.calls), 6)
        recovered = w.read_search(reloaded, "SYNTHETIC_RUN")
        self.assertEqual(list(recovered), list(resumed))
        for name in recovered:
            self.assertEqual([r["summary"]["seed"] for r in recovered[name]], [42,142,242])
        first = json.loads(self.store["SYNTHETIC_RUN_C000_S42"]["contract.json"])["contract"]
        self.assertEqual(first["candidate"], "a_first")

    def test_failed_trial_does_not_silently_complete_or_shrink_grid(self):
        original = w.train_candidate
        def failing(*args, **kwargs):
            if args[6] == 142:
                raise RuntimeError("synthetic interrupted fit")
            return original(*args, **kwargs)
        w.train_candidate = failing
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "interrupted"):
            w.execute_search(self.data, self.protocol, "SYNTHETIC_FAIL")
        self.assertEqual(len(self.store), 1)
        with self.assertRaises(KeyError):
            w.read_search(self.protocol, "SYNTHETIC_FAIL")
        w.train_candidate = original
        with contextlib.redirect_stdout(io.StringIO()):
            completed = w.execute_search(self.data, self.protocol, "SYNTHETIC_FAIL")
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(sum(len(runs) for runs in completed.values()), 6)

    def test_changed_candidate_input_code_or_order_invalidates_reuse(self):
        with contextlib.redirect_stdout(io.StringIO()):
            w.execute_search(self.data, self.protocol, "SYNTHETIC_CHANGE")
        variants = []
        changed = copy.deepcopy(self.protocol); changed["candidates"]["a_first"]["columns"] = [0,2]; variants.append(changed)
        changed = copy.deepcopy(self.protocol); changed["implementation_hash"] = "new-code"; variants.append(changed)
        changed = copy.deepcopy(self.protocol); changed["input_hashes"] = {"tensor":"new-source"}; variants.append(changed)
        changed = copy.deepcopy(self.protocol); changed["artifact_candidate_order"] = ["z_last","a_first"]; variants.append(changed)
        for variant in variants:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                w.execute_search(self.data, variant, "SYNTHETIC_CHANGE")

    def test_ensemble_uses_all_runs_and_rejects_duplicate_seeds(self):
        runs = [fake_run(seed=seed) for seed in (42,142,242)]
        runs[0]["validation_scores"] = np.ones(4) * .1
        runs[1]["validation_scores"] = np.ones(4) * .4
        runs[2]["validation_scores"] = np.ones(4) * .7
        np.testing.assert_allclose(w.ensemble_scores(runs,"validation"), .4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            w.ensemble_scores([runs[0],runs[0]],"validation")
        with self.assertRaises(ValueError):
            w.ensemble_scores(runs,"test")


class TwoPhaseSearch(SearchResume):
    """Complete phase-A evidence controls immutable, independently resolved anchors."""
    def setUp(self):
        super().setUp()
        self.design = w.build_protocol({"all_eligible":[0,1,2], "small":[0,2]},
                                      {"proxy_one_se":"small", "proxy_best":"all_eligible"},
                                      self.data["hashes"], "code-sha")
        # Keep one ablation plus the mandatory all-feature sensitivity per family
        # in this synthetic protocol; production build still declares every arm.
        keep = self.design["phase_a"] + ["tf_capacity", "gb_capacity", "tf_without_step0", "gb_without_step0"]
        self.design["candidates"] = {name:cfg for name,cfg in self.design["candidates"].items() if name in keep}
        self.design["phase_b"] = sorted(set(keep) - set(self.design["phase_a"]))
        self.design["artifact_candidate_order"] = sorted(keep)
        self.design["prediction_counts"] = {"train":8,"validation":4}
        self.design["cpu_threads"] = 1
        self.prefix = "TWO_PHASE"
        self.store[self.prefix+"_PROTOCOL"] = {"protocol.json":w.json_bytes(self.design),"runtime.json":b"{}"}
        self.fit_order = []
        self.phase_b_failure = False
        def train(x, y, groups, xv, yv, cfg, seed, **kwargs):
            self.assertTrue((x >= 0).all() and (xv >= 0).all(), "TEST sentinel reached fitting")
            is_base = ((cfg["family"] == "transformer" and cfg["d_model"] == 64 and cfg["representation"] == "monthly")
                       or (cfg["family"] == "lightgbm" and cfg["num_leaves"] == 15 and not cfg.get("drop_step0", False)))
            if not is_base:
                self.assertIn(self.prefix+"_ANCHORS", self.store, "Phase B started before immutable anchor artifact")
                if self.phase_b_failure:
                    raise RuntimeError("phase-B synthetic interruption")
            self.fit_order.append((cfg["family"], cfg["subset"], len(cfg["columns"]), seed, is_base))
            self.calls.append((cfg["family"], seed))
            result = fake_run(cfg, seed, len(y), len(yv))
            # Families intentionally choose different anchors: TF all; GB small.
            score = (.8 if cfg["subset"] == "all_eligible" else .6) if cfg["family"] == "transformer" else (.7 if cfg["subset"] == "small" else .5)
            result["summary"]["validation_metrics"]["average_precision"] = score
            result["summary"]["parameter_count"] = len(cfg["columns"]) * 10
            return result
        w.train_candidate = train

    # Parent legacy tests use a separate protocol; retain them in SearchResume only.
    test_sorted_order_resume_after_canonical_protocol_roundtrip = None
    test_failed_trial_does_not_silently_complete_or_shrink_grid = None
    test_changed_candidate_input_code_or_order_invalidates_reuse = None
    test_ensemble_uses_all_runs_and_rejects_duplicate_seeds = None

    def _run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return w.execute_two_phase_search(self.data, self.design, self.prefix)

    def test_phase_b_uses_each_family_count_winner_and_without_step0_stays_all(self):
        before = w.json_bytes(self.design)
        resolved, results = self._run()
        self.assertEqual(w.json_bytes(self.design), before)
        self.assertEqual(resolved["candidates"]["tf_capacity"]["columns"], [0,1,2])
        self.assertEqual(resolved["candidates"]["gb_capacity"]["columns"], [0,2])
        self.assertEqual(resolved["candidates"]["gb_without_step0"]["columns"], [0,1,2])
        self.assertEqual(resolved["candidates"]["tf_without_step0"]["columns"], [0,1,2])
        self.assertEqual(len(self.calls), len(self.design["candidates"]) * 3)
        self.assertEqual(list(results), sorted(self.design["candidates"]))
        self.assertEqual(w.read_resolved_protocol(self.design,self.prefix), resolved)
        self.assertEqual(list(w.read_search(resolved,self.prefix)), list(results))

    def test_resume_reuses_phase_a_and_phase_b_with_stable_global_indices(self):
        resolved, _ = self._run()
        calls = len(self.calls)
        rerun, _ = self._run()
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(resolved, rerun)
        for index,name in enumerate(sorted(self.design["candidates"])):
            envelope = json.loads(self.store[w.run_artifact_table(self.prefix,index,42)]["contract.json"])
            contract = envelope["contract"]
            self.assertEqual(contract["candidate"], name)
            self.assertEqual(contract["protocol_hash"], w.stable_hash(self.design))
            self.assertEqual(contract["phase"], "A" if name in self.design["phase_a"] else "B")
            self.assertEqual("resolution_sha256" in contract, name in self.design["phase_b"])

    def test_phase_b_interruption_does_not_refit_phase_a(self):
        self.phase_b_failure = True
        with self.assertRaisesRegex(RuntimeError,"phase-B"):
            self._run()
        self.assertEqual(len(self.calls), len(self.design["phase_a"])*3)
        self.assertIn(self.prefix+"_ANCHORS",self.store)
        self.phase_b_failure = False
        self._run()
        self.assertEqual(len(self.calls),len(self.design["candidates"])*3)

    def test_missing_or_duplicate_seed_cannot_resolve_anchor(self):
        _, results = self._run()
        phase_a = {name:results[name] for name in self.design["phase_a"]}
        name = self.design["phase_a"][0]
        for bad in (phase_a[name][:2], [phase_a[name][0]]*3):
            altered = {**phase_a,name:bad}
            with self.assertRaisesRegex(ValueError,"seed"):
                w.resolve_count_anchors(self.design,altered)
        with self.assertRaisesRegex(ValueError,"incomplete"):
            w.resolve_count_anchors(self.design,{})

    def test_exact_score_tie_prefers_smaller_count_not_confidence_band(self):
        _, results = self._run()
        phase_a = copy.deepcopy({name:results[name] for name in self.design["phase_a"]})
        for runs in phase_a.values():
            for run in runs:
                run["summary"]["validation_metrics"]["average_precision"] = .6
        resolution = w.resolve_count_anchors(self.design,phase_a)
        self.assertEqual(resolution["anchors"]["transformer"]["subset"],"small")
        self.assertEqual(resolution["anchors"]["lightgbm"]["subset"],"small")

    def test_resolution_or_phase_a_drift_rejected_before_reuse(self):
        resolved, _ = self._run()
        table = self.prefix+"_ANCHORS"
        saved = copy.deepcopy(self.store[table])
        resolution = json.loads(self.store[table]["resolution.json"])
        resolution["anchors"]["transformer"]["columns"] = [0,2]
        self.store[table]["resolution.json"] = w.json_bytes(resolution)
        with self.assertRaisesRegex(ValueError,"resolution differs"):
            w.read_resolved_protocol(self.design,self.prefix)
        self.store[table] = saved
        name = self.design["phase_a"][0]
        run_table = w.run_artifact_table(self.prefix,sorted(self.design["candidates"]).index(name),42)
        self.store[run_table] = retag_payload(self.store[run_table],"validation.npy",w.array_bytes(np.array([.2,.3,.4,.5])))
        with self.assertRaisesRegex(ValueError,"resolution differs"):
            w.read_search(resolved,self.prefix)

    def test_unpersisted_changed_or_unresolved_design_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"execute_two_phase_search"):
            w.execute_search(self.data,self.design,self.prefix)
        with self.assertRaisesRegex(ValueError,"read_resolved_protocol"):
            w.read_search(self.design,self.prefix)
        changed = copy.deepcopy(self.design)
        changed["epochs"] += 1
        with self.assertRaisesRegex(ValueError,"Saved DESIGN differs"):
            w.execute_two_phase_search(self.data,changed,self.prefix)
        del self.store[self.prefix+"_PROTOCOL"]
        with self.assertRaisesRegex(ValueError,"Persist"):
            self._run()


class EncodedRealRoundtrip(unittest.TestCase):
    def test_actual_lightgbm49_result_satisfies_saved_run_contract(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("Local LightGBM dependency unavailable")
        from experiments.rigorous_transformer.baseline49 import train_encoded_baseline, predict_encoded
        rng = np.random.default_rng(781)
        x = rng.random((90,49),dtype=np.float32)
        y = (x[:,0] > .65).astype(int)
        result = train_encoded_baseline(x[:60],y[:60],[f"p{i}" for i in range(60)],x[60:],y[60:],42,
                                        {"n_estimators":8,"min_child_samples":5,"early_stopping_rounds":3,"n_jobs":1})
        contract = dict(protocol_hash="synthetic",candidate="lightgbm49_refit",seed=42,
                        prediction_counts={"train":60,"validation":30},input_sha256="private-input-hash")
        loaded = w.unpack_run(w.pack_run(result,contract),contract,expected_config=result["summary"]["config"])
        np.testing.assert_allclose(predict_encoded(loaded,x[60:]),result["validation_scores"])


if __name__ == "__main__":
    unittest.main()
