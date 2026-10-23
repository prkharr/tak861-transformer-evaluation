"""Synthetic orchestration proof; never evidence of clinical model performance."""
import copy
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.automl_sequence import inputs, modeling, workflow as w
from experiments.rigorous_transformer import comparison


class LiftBootstrapTests(unittest.TestCase):
    def test_direction_capacities_and_perfect_separation_interval(self):
        y = np.tile([0, 1], 12)
        groups = np.repeat(np.arange(12).astype(str), 2)
        result = w.paired_lift_bootstrap(y, y.astype(float), 1-y.astype(float), groups, n_bootstrap=100)
        self.assertEqual(result["direction"], "model_a_minus_model_b")
        self.assertTrue(result["interval_reliable"])
        self.assertEqual(result["valid_draws"], 100)
        for row in result["lift_by_fraction"].values():
            self.assertEqual(row["delta"], 2.)
            self.assertEqual(row["lower"], 2.)
            self.assertEqual(row["upper"], 2.)

    def test_cluster_multiplicity_matches_explicit_repeated_snapshots(self):
        y = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0])
        a = np.array([.9, .9, .8, .7, .7, .6, .5, .4, .3, .2])
        b = np.array([.2, .3, .4, .5, .6, .6, .7, .8, .9, .9])
        groups = np.array(["a", "a", "a", "b", "b", "c", "d", "d", "d", "d"])
        result = w.paired_lift_bootstrap(y, a, b, groups, fractions=[.3], n_bootstrap=100, seed=7)
        unique, inverse = np.unique(groups, return_inverse=True)
        rng, deltas = np.random.default_rng(7), []
        for _ in range(100):
            sampled = rng.integers(0, len(unique), size=len(unique))
            counts = np.bincount(sampled, minlength=len(unique))[inverse]
            repeated = np.repeat(np.arange(len(y)), counts)
            if len(np.unique(y[repeated])) != 2:
                continue
            deltas.append(modeling.lift_at_fraction(y[repeated], a[repeated], .3)-modeling.lift_at_fraction(y[repeated], b[repeated], .3))
        self.assertEqual(result["valid_draws"], len(deltas))
        row = result["lift_by_fraction"]["0.3"]
        np.testing.assert_allclose([row["lower"], row["upper"]], np.quantile(deltas, [.025, .975]))
        self.assertAlmostEqual(row["delta"], modeling.lift_at_fraction(y, a, .3)-modeling.lift_at_fraction(y, b, .3))


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.store = {}
        def save(table, artifacts):
            artifacts = {k: v.read_bytes() if isinstance(v, Path) else v for k, v in artifacts.items()}
            if table in self.store and self.store[table] != artifacts:
                raise ValueError("Immutable artifact table changed")
            self.store[table] = copy.deepcopy(artifacts)
        def read(table, names, *, disk_names=(), output_dir=None):
            if table not in self.store or set(self.store[table]) != set(names):
                raise ValueError("Missing artifact or manifest names mismatch")
            result = copy.deepcopy(self.store[table])
            for name in disk_names:
                target = Path(output_dir) / name
                target.write_bytes(result[name]); result[name] = target
            return result
        w.inputs, w.modeling, w.comparison = inputs, modeling, comparison
        w.save_artifacts, w.read_artifacts, w.table_exists = save, read, self.store.__contains__
        self.prefix = w.prefix("SYNTHETIC20261013")
        self.impl = hashlib.sha256(b"synthetic-test-code").hexdigest()
        self.config = {
            "require_count_reference": True, "seeds": [42],
            "selection": {"candidate_sizes": [2, 4], "proxy_max_iter": 200},
            "model": {"epochs": 2, "patience": 2, "d_model": 8, "heads": 2,
                      "ff": 16, "threads": 1, "batch_size": 24},
        }
        meta = pd.DataFrame({"PATIENT_ID": [f"p{i:03d}" for i in range(96)],
                             "END_DT": ["2026-06-30"] * 96,
                             "RESP": np.tile([0, 1], 48),
                             "SPLIT": ["train"]*60 + ["validation"]*18 + ["test"]*18})
        self.specs = [dict(name=f"automl{i}", kind="continuous", definition=f"Synthetic signed measurement {i}")
                      for i in range(5)]
        contract = dict(source_id="synthetic_only", source_version="v1",
                        extraction_sha256=hashlib.sha256(b"synthetic extractor").hexdigest(),
                        cutoff_rule="event and available <= AS_OF_DATE",
                        fit_scope="deterministic_raw", feature_semantics_evidence="synthetic fixture")
        requests, grid = inputs.make_requests(meta)
        def extractor(keys):
            frame = keys.copy()
            p = frame.PATIENT_ID.str[1:].astype(int).to_numpy()
            month = frame.AS_OF_DATE.dt.month.to_numpy()
            for j in range(5):
                frame[f"automl{j}"] = np.sin(p * (j+1) + month / (j+1)) + (p % 2) * .2
            frame["IS_OBSERVED"] = True
            frame["MAX_EVENT_DATE"] = frame.AS_OF_DATE
            frame["MAX_AVAILABLE_DATE"] = frame.AS_OF_DATE
            return frame
        panel, self.panel_manifest = inputs.obtain_panel(requests, self.specs, contract, extractor=extractor)
        data = inputs.assemble_inputs(meta, panel, self.specs)
        rng = np.random.default_rng(23)
        count = np.zeros((96, 12, 1028), dtype=np.float32)
        count[:, :, :4] = rng.poisson(2, size=(96, 12, 4))
        mask = np.ones((96, 12), bool)
        cc = dict(source_id="synthetic_count", source_version="v1",
                  content_sha256=hashlib.sha256(w.array_bytes(count)).hexdigest(),
                  feature_names=[f"count{i}" for i in range(1028)], provenance_status="synthetic_test",
                  orientation="0=newest", n_steps=12, date_policy="same_day_months",
                  mask_kind="explicit_synthetic_all_observed", mask_sha256=hashlib.sha256(w.array_bytes(mask)).hexdigest(),
                  metadata_sha256=inputs.frame_hash(inputs.normalize_snapshots(meta)),
                  asof_grid_sha256=inputs.frame_hash(grid))
        self.data = w.attach_count_reference(data, count, meta, cc, mask)

    def test_three_arm_end_to_end_freeze_restore_and_test_gate(self):
        manifest = w.save_inputs(self.prefix, self.data, self.panel_manifest, self.config, self.impl)
        data = w.load_inputs(self.prefix, self.impl)
        np.testing.assert_equal(data["X_sequence"], self.data["X_sequence"])
        self.assertEqual(manifest["split_counts"], {"train": 60, "validation": 18, "test": 18})
        prep, protocol = w.prepare_run(data, self.prefix, self.config, self.impl)
        self.assertEqual(set(protocol["arms"]), {"COUNT", "STATIC_FUSION", "TEMPORAL_FUSION"})
        self.assertEqual(len(prep["count"]["quality"]), 1028)
        self.assertFalse(prep["automl"]["selection_used_test"])
        self.assertEqual(protocol["primary_lift_fraction"], .1)
        self.assertEqual(protocol["model_config"]["primary_lift_fraction"], .1)
        self.assertEqual(prep["automl"]["config"]["primary_lift_fraction"], .1)
        for name in protocol["arms"]:
            values, valid, static = w.arm_arrays(data, prep, protocol["arms"][name], "train")
            self.assertEqual(values.shape[:2], (60, 12))
            self.assertTrue(valid.all())
            self.assertEqual(static is not None, name == "STATIC_FUSION")
        results = w.train_registered(data, prep, protocol, self.prefix)
        self.assertEqual(len(results), 3)
        self.assertEqual(results["STATIC_FUSION"][0]["summary"]["static_dim"], len(prep["automl"]["selected_indices"])+1)
        with patch.object(modeling, "train_encoder", side_effect=AssertionError("Cached fit reran")):
            cached = w.train_registered(data, prep, protocol, self.prefix)
        restored = w.read_runs(protocol, self.prefix)
        for name in results:
            np.testing.assert_array_equal(restored[name][0]["val_scores"], results[name][0]["val_scores"])
            np.testing.assert_array_equal(cached[name][0]["val_scores"], results[name][0]["val_scores"])
        lock = w.make_lock(data, prep, protocol, restored, self.prefix)
        self.assertFalse(lock["test_prediction_performed"])
        with patch.object(modeling, "predict_encoder", side_effect=AssertionError("Validation called TEST prediction")):
            val_report = w.compare_run(data, prep, protocol, restored, lock, self.prefix, evaluate_test=False, n_bootstrap=20)
        self.assertEqual(val_report["split"], "validation")
        test_report = w.compare_run(data, prep, protocol, restored, lock, self.prefix, evaluate_test=True, n_bootstrap=20)
        self.assertEqual(test_report["split"], "test")
        self.assertEqual(test_report["test_evaluation_ordinal"], 3)
        self.assertIn("TEMPORAL_FUSION_MINUS_COUNT", test_report["comparisons"])
        delta = test_report["comparisons"]["TEMPORAL_FUSION_MINUS_COUNT"]
        self.assertEqual(delta["direction"], "model_a_minus_model_b")
        self.assertAlmostEqual(delta["average_precision"]["delta"],
                               test_report["metrics"]["TEMPORAL_FUSION"]["average_precision"] -
                               test_report["metrics"]["COUNT"]["average_precision"])
        self.assertAlmostEqual(delta["primary_lift"]["delta"],
                               test_report["metrics"]["TEMPORAL_FUSION"]["primary_lift"] -
                               test_report["metrics"]["COUNT"]["primary_lift"])
        self.assertEqual(set(delta["lift_by_fraction"]), {"0.1", "0.2", "0.3"})
        self.assertIn("lift_10pct", w.report_table(test_report).columns)
        with patch.object(modeling, "predict_encoder", side_effect=AssertionError("Cached TEST reevaluated")):
            self.assertEqual(w.compare_run(data, prep, protocol, restored, lock, self.prefix,
                                           evaluate_test=True, n_bootstrap=20), test_report)
        modified = copy.deepcopy(prep)
        modified["automl"]["selected_names"][0] += "_changed"
        with self.assertRaisesRegex(ValueError, "changed"):
            w.train_registered(data, modified, protocol, self.prefix)
        with self.assertRaisesRegex(ValueError, "different notebook implementation"):
            w.load_inputs(self.prefix, "new-implementation")
        altered = copy.deepcopy(lock); altered["test_prediction_performed"] = True
        with self.assertRaisesRegex(ValueError, "Persisted lock mismatch"):
            w.compare_run(data, prep, protocol, restored, altered, self.prefix, n_bootstrap=20)

    def test_train_fitted_upstream_features_cannot_bypass_fold_local_selection(self):
        manifest = copy.deepcopy(self.panel_manifest)
        manifest["contract"].update(fit_scope="train_only_fitted",
                                    fitting_patients_sha256=inputs.train_patient_hash(self.data["metadata"]))
        # TRAIN scope is valid for storage, but not proof of fold-local fitting.
        w.save_inputs(self.prefix, self.data, manifest, self.config, self.impl)
        restored = w.load_inputs(self.prefix, self.impl)
        with self.assertRaisesRegex(ValueError, "raw deterministic AutoML inputs"):
            w.prepare_run(restored, self.prefix, self.config, self.impl)

    def test_unavailable_automl_keeps_count_cohort_and_zero_availability_indicators(self):
        data = copy.deepcopy(self.data)
        data["X_sequence"][0] = np.nan
        data["X_summary"][0] = np.nan
        data["sequence_mask"][0] = False
        data["summary_mask"][0] = False
        w.save_inputs(self.prefix, data, self.panel_manifest, self.config, self.impl)
        loaded = w.load_inputs(self.prefix, self.impl)
        prep, protocol = w.prepare_run(loaded, self.prefix, self.config, self.impl)
        reference, reference_mask, _ = w.arm_arrays(loaded, prep, protocol["arms"]["COUNT"], "train")
        static_seq, static_mask, static = w.arm_arrays(loaded, prep, protocol["arms"]["STATIC_FUSION"], "train")
        temporal, temporal_mask, _ = w.arm_arrays(loaded, prep, protocol["arms"]["TEMPORAL_FUSION"], "train")
        self.assertEqual(len(static_seq), 60)
        self.assertEqual(len(temporal), 60)
        np.testing.assert_array_equal(static_seq, reference)
        np.testing.assert_array_equal(static_mask, reference_mask)
        np.testing.assert_array_equal(temporal_mask, reference_mask)
        np.testing.assert_array_equal(temporal[:, :, :reference.shape[-1]], reference)
        np.testing.assert_array_equal(static[0], 0.)
        np.testing.assert_array_equal(temporal[0, :, reference.shape[-1]:], 0.)
        self.assertTrue(reference_mask[0].all())
        self.assertEqual(static[1, -1], 1.)
        np.testing.assert_array_equal(temporal[1, :, -1], 1.)

    def test_conflicting_primary_capacities_fail_before_fitting(self):
        config = copy.deepcopy(self.config)
        config.update(primary_lift_fraction=.2)
        config["model"]["primary_lift_fraction"] = .1
        w.save_inputs(self.prefix, self.data, self.panel_manifest, config, self.impl)
        restored = w.load_inputs(self.prefix, self.impl)
        with self.assertRaisesRegex(ValueError, "Conflicting primary capacity"):
            w.prepare_run(restored, self.prefix, config, self.impl)

    def test_identical_input_replay_keeps_original_timing_but_rejects_changed_content(self):
        original = w.save_inputs(self.prefix, self.data, self.panel_manifest, self.config, self.impl)
        repeated = copy.deepcopy(self.panel_manifest)
        repeated["extraction_report"]["elapsed_seconds"] += 100
        self.assertEqual(original, w.save_inputs(self.prefix, self.data, repeated, self.config, self.impl))
        mutated = copy.deepcopy(self.data)
        mutated["X_sequence"][0, 2, 1] += 1
        with self.assertRaisesRegex(ValueError, "Frozen input source/content/configuration changed"):
            w.save_inputs(self.prefix, mutated, repeated, self.config, self.impl)


if __name__ == "__main__":
    unittest.main()
