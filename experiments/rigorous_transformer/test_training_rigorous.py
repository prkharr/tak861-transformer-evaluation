"""Bounded synthetic training tests; no project data or real model metrics."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

import numpy as np
import torch

_SPEC = importlib.util.spec_from_file_location("rigorous_training", Path(__file__).with_name("training.py"))
t = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(t)


def synthetic():
    rng = np.random.default_rng(889)
    train = rng.poisson(.7, (32, 12, 4)).astype(np.float32)
    val = rng.poisson(.7, (16, 12, 4)).astype(np.float32)
    y_train, y_val = np.arange(32) % 2, np.arange(16) % 2
    train[:, 0, 0] += 3 * y_train
    val[:, 0, 0] += 3 * y_val
    train[0, 2, 1] = np.nan
    val[0, 3, 1] = np.nan
    groups = np.repeat([f"train_{i}" for i in range(16)], 2)
    val_groups = np.repeat([f"validation_{i}" for i in range(8)], 2)
    cfg = dict(family="transformer", representation="monthly", columns=[3, 10, 27, 88],
               d_model=8, heads=2, ff=12, depth=1, dropout=.1, batch_size=16,
               learning_rate=.005, projection_l1=.001, explicit_l2=.002, weight_decay=0.,
               scheduler="none")
    return train, y_train, groups, val, y_val, val_groups, cfg


class InputAndMathTests(unittest.TestCase):
    def test_count_integrality_before_precision_cast_and_missing_preserved(self):
        x = np.zeros((2, 12, 2), dtype=np.float64)
        x[0, 0, 0] = 1.000000001
        with self.assertRaisesRegex(ValueError, "Fractional"):
            t.raw_representation(x, "monthly")
        for value in (-1, np.inf):
            x[0, 0, 0] = value
            with self.assertRaises(ValueError):
                t.raw_representation(x, "monthly")
        x[0, 0, 0] = np.nan
        result = t.raw_representation(x, "quarterly")
        self.assertTrue(np.isnan(result[0, 0, 0]))
        self.assertEqual(result.shape, (2, 4, 2))
        self.assertFalse(np.shares_memory(t.raw_representation(x, "monthly"), x))

    def test_aggregate_and_flatten_transform_contracts(self):
        x = np.arange(48, dtype=np.float32).reshape(2, 12, 2)
        windows = t.raw_representation(x, "windows")
        np.testing.assert_equal(windows[:, :2], x[:, 0])
        np.testing.assert_equal(windows[:, -2:], x.sum(axis=1))
        np.testing.assert_equal(t.raw_representation(x, "quarterly"), x.reshape(2, 4, 3, 2).sum(axis=2))
        for representation in ("monthly", "quarterly", "flattened", "annual", "windows"):
            raw = t.raw_representation(x, representation)
            state = t.fit_numeric_transform(raw, scale=True)
            restored = json.loads(json.dumps(state, allow_nan=False))
            actual = t.numeric_transform(raw, restored)
            self.assertEqual(actual.shape, raw.shape)
            self.assertTrue(np.isfinite(actual).all())
            self.assertEqual(len(state["median"]), raw.shape[-1])
            self.assertTrue((np.array(state["divisor"]) >= 1).all())

    def test_imputation_only_train_and_nonexistent_field_reported(self):
        raw = np.array([[0, np.nan, 1], [2, np.nan, np.nan]], dtype=float)
        state = t.fit_numeric_transform(raw)
        self.assertEqual(state["all_missing_fields"], 1)
        self.assertEqual(state["missing_cells"], 3)
        val = np.array([[1e8, np.nan, np.nan]], dtype=float)
        result = t.numeric_transform(val, state)
        self.assertAlmostEqual(result[0, 2], np.log(2), places=6)
        self.assertEqual(result[0, 1], 0)
        self.assertEqual(state["median"][2], np.log(2))

    def test_weighting_and_class_weight_arms(self):
        groups = np.array(["a", "a", "a", "b"])
        np.testing.assert_equal(t.sample_weights(groups), np.ones(4))
        patient = t.sample_weights(groups, "patient")
        self.assertAlmostEqual(patient[:3].sum(), patient[3])
        y = np.array([0, 0, 0, 1])
        self.assertEqual(t.class_weight(y, np.ones(4), "none"), 1)
        self.assertEqual(t.class_weight(y, np.ones(4), "full"), 3)
        self.assertAlmostEqual(t.class_weight(y, np.ones(4), "sqrt"), np.sqrt(3))
        self.assertAlmostEqual(t.class_weight(y, patient, "full"), 1)
        with self.assertRaises(ValueError):
            t.class_weight(np.ones(4), np.ones(4), "full")

    def test_l1_projection_only_and_explicit_l2_gradients(self):
        torch.manual_seed(73)
        config = dict(d_model=8, heads=2, ff=12, depth=1, projection_l1=.2, explicit_l2=0, weight_decay=0)
        model = t.make_encoder(4, config)
        l1, _ = t.penalties(model, config)
        l1.backward()
        expected = .2 * model.projection.weight.detach().sign() / model.projection.weight.numel()
        torch.testing.assert_close(model.projection.weight.grad, expected)
        for name, parameter in model.named_parameters():
            if name != "projection.weight":
                self.assertIsNone(parameter.grad, name)
        model.zero_grad(set_to_none=True)
        config.update(projection_l1=0, explicit_l2=.3)
        _, l2 = t.penalties(model, config)
        l2.backward()
        count = sum(p.numel() for p in model.parameters() if p.ndim > 1)
        for name, parameter in model.named_parameters():
            if parameter.ndim > 1:
                torch.testing.assert_close(parameter.grad, .6 * parameter.detach() / count)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertFalse(model.position.requires_grad)
        with self.assertRaisesRegex(ValueError, "never both"):
            t.penalties(model, {**config, "weight_decay": .1})

    def test_invalid_parameter_configs_rejected_before_training(self):
        *_, cfg = synthetic()
        for change in (dict(heads=0), dict(heads=3), dict(depth=0), dict(dropout=1),
                       dict(pooling="typo"), dict(projection_l1=-1), dict(scheduler="typo"),
                       dict(optimizer_betas=[1, .9]), dict(learning_rate=float("nan")),
                       dict(columns=[1, 1, 2, 3]), dict(weight_decay=.1), dict(droput=.2)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                t._resolved_config({**cfg, **change}, 4)
        omitted = {k: v for k, v in cfg.items() if k != "weight_decay"}
        with self.assertRaisesRegex(ValueError, "set weight_decay=0"):
            t._resolved_config(omitted, 4)
        self.assertEqual(t._resolved_config({**cfg, "subset": "all_eligible", "drop_step0": True}, 4)["subset"], "all_eligible")

    def test_declared_workflow_configurations_are_all_supported(self):
        spec = importlib.util.spec_from_file_location("rigorous_workflow_contract", Path(__file__).with_name("workflow.py"))
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        protocol = workflow.build_protocol({"all_eligible": [0, 1, 2, 3], "proxy_one_se": [0, 2]},
                                            {"proxy_one_se": "proxy_one_se"}, {"synthetic": "only"}, "code_hash")
        for name, config in protocol["candidates"].items():
            with self.subTest(candidate=name):
                if config["columns"] is None:
                    self.assertIn(name, protocol["phase_b"])
                    config = {**config, "columns": protocol["feature_subsets"]["all_eligible"], "subset": "all_eligible"}
                resolved = t._resolved_config(config, len(config["columns"]))
                self.assertEqual(resolved["columns"], config["columns"])


class TinyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y, cls.groups, cls.val, cls.y_val, cls.val_groups, cls.cfg = synthetic()
        cls.result = t.train_candidate(cls.x, cls.y, cls.groups, cls.val, cls.y_val,
                                       cls.cfg, 13, epochs=2, patience=2, threads=1,
                                       groups_validation=cls.val_groups)

    def test_deterministic_training_and_best_epoch_restore(self):
        run = t.train_candidate(self.x, self.y, self.groups, self.val, self.y_val,
                                 self.cfg, 13, epochs=2, patience=2, threads=1,
                                 groups_validation=self.val_groups)
        np.testing.assert_array_equal(run["validation_scores"], self.result["validation_scores"])
        np.testing.assert_array_equal(run["train_scores"], self.result["train_scores"])
        self.assertEqual(run["summary"]["model_sha256"], self.result["summary"]["model_sha256"])
        self.assertEqual(run["history"], self.result["history"])
        np.testing.assert_array_equal(t.predict_saved(self.result, self.val), self.result["validation_scores"])
        history = self.result["history"]
        best = self.result["summary"]["best_epoch"]
        self.assertEqual(history[best-1]["validation_ap"], max(h["validation_ap"] for h in history))
        self.assertEqual(self.result["summary"]["validation_metrics"]["average_precision"], history[best-1]["validation_ap"])

    def test_regularization_active_every_epoch_and_config_not_mutated(self):
        self.assertNotIn("optimizer_betas", self.cfg)
        self.assertEqual(self.result["summary"]["config"]["optimizer_betas"], [.9, .999])
        for row in self.result["history"]:
            self.assertGreater(row["projection_l1_loss"], 0)
            self.assertGreater(row["explicit_l2_loss"], 0)
            self.assertAlmostEqual(row["regularization_loss"], row["projection_l1_loss"] + row["explicit_l2_loss"], places=7)
        self.assertTrue(self.result["summary"]["patient_disjointness_verified"])
        self.assertFalse(self.result["summary"]["test_inference_performed"])
        json.dumps(self.result["summary"], allow_nan=False)

    def test_reusable_predictor_restores_once_and_freezes_numeric_contract(self):
        result = copy.deepcopy(self.result)
        predict = t.make_saved_predictor(result)
        np.testing.assert_array_equal(predict(self.val), self.result["validation_scores"])
        result["summary"]["transform"]["median"][0] = 9999
        result["summary"]["config"]["representation"] = "quarterly"
        np.testing.assert_array_equal(predict(self.val), self.result["validation_scores"])
        with self.assertRaisesRegex(ValueError, "shape"):
            predict(self.val[:, :, :-1])

    def test_checkpoint_and_preprocessing_hashes_verified(self):
        bad = copy.deepcopy(self.result)
        bad["model_blob"] += b"changed"
        with self.assertRaisesRegex(ValueError, "hash"):
            t.predict_saved(bad, self.val)
        for key in ("config", "transform"):
            bad = copy.deepcopy(self.result)
            if key == "config":
                bad["summary"][key]["dropout"] = .25
            else:
                bad["summary"][key]["median"][0] += 1
            with self.assertRaisesRegex(ValueError, "hash"):
                t.predict_saved(bad, self.val)
        with self.assertRaisesRegex(ValueError, "shape"):
            t.predict_saved(self.result, self.val[:, :, :3])

    def test_no_test_argument_and_frozen_patient_alignment(self):
        self.assertFalse(any("test" in name for name in inspect.signature(t.train_candidate).parameters))
        with self.assertRaises(TypeError):
            t.train_candidate(self.x, self.y, self.groups, self.val, self.y_val, self.cfg, 0, x_test=self.val)
        with self.assertRaisesRegex(ValueError, "overlap"):
            t.train_candidate(self.x, self.y, self.groups, self.val, self.y_val, self.cfg, 0,
                              groups_validation=self.groups[:len(self.val)])
        with self.assertRaisesRegex(ValueError, "alignment"):
            t.train_candidate(self.x, self.y, self.groups, self.val, self.y_val[:-1], self.cfg, 0)


class FinalistTests(unittest.TestCase):
    @staticmethod
    def run_record(ap, seed, candidate=0):
        return dict(summary=dict(seed=seed, training_complete=True,
                                  config=dict(family="transformer", columns=[0, 1], candidate=candidate),
                                  validation_metrics=dict(average_precision=ap), parameter_count=100,
                                  validation_evaluations=2))

    def test_all_seeds_count_no_lucky_seed_selection(self):
        results = {"lucky": [self.run_record(a, s, 0) for a, s in zip([.9, .2, .2], [1, 2, 3])],
                   "stable": [self.run_record(.5, s, 1) for s in [1, 2, 3]]}
        winners, rows = t.choose_finalists(results, ["lucky", "stable"], [1, 2, 3])
        self.assertEqual(winners["transformer"], "stable")
        self.assertTrue(all(row["fits"] == 3 for row in rows))
        self.assertTrue(all(row["validation_evaluations"] == 6 for row in rows))

    def test_incomplete_grid_seed_mismatch_and_bad_metrics_rejected(self):
        valid = {"candidate": [self.run_record(.5, s) for s in [1, 2, 3]]}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            t.choose_finalists(valid, ["candidate", "missing"], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "unique"):
            t.choose_finalists(valid, ["candidate"], [1, 1, 2])
        with self.assertRaisesRegex(ValueError, "seeds"):
            t.choose_finalists(valid, ["candidate"], [1, 2])
        bad = copy.deepcopy(valid)
        bad["candidate"][1]["summary"]["validation_metrics"]["average_precision"] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid validation"):
            t.choose_finalists(bad, ["candidate"], [1, 2, 3])
        bad = copy.deepcopy(valid)
        bad["candidate"][1]["summary"]["config"]["columns"] = [0]
        with self.assertRaisesRegex(ValueError, "different configurations"):
            t.choose_finalists(bad, ["candidate"], [1, 2, 3])


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM isolated dependency is not installed")
class TinyLightGBMTests(unittest.TestCase):
    def test_text_artifact_restore_windows_missing_values(self):
        x, y, groups, val, y_val, val_groups, _ = synthetic()
        config = dict(family="lightgbm", representation="windows", n_estimators=12,
                       num_leaves=4, min_child_samples=2, early_stopping_rounds=3)
        result = t.train_candidate(x, y, groups, val, y_val, config, 9, threads=1,
                                   groups_validation=val_groups)
        self.assertEqual(result["summary"]["artifact_format"], "lightgbm_text")
        self.assertEqual(len(result["summary"]["transform"]["median"]), 16)
        self.assertEqual(len(result["summary"]["aggregated_original_importance"]), 4)
        self.assertEqual(result["summary"]["importance_blocks_per_original_feature"], 4)
        np.testing.assert_allclose(t.predict_saved(result, val), result["validation_scores"], rtol=0, atol=0)

    def test_step0_flag_is_applied_in_direct_restore(self):
        x, y, groups, val, y_val, val_groups, _ = synthetic()
        config = dict(family="lightgbm", representation="flattened", n_estimators=8,
                       num_leaves=4, min_child_samples=2, early_stopping_rounds=3,
                       drop_step0=True, subset="all_eligible")
        result = t.train_candidate(x, y, groups, val, y_val, config, 9, threads=1,
                                   groups_validation=val_groups)
        changed = val.copy()
        changed[:, 0] = 99999
        np.testing.assert_array_equal(t.predict_saved(result, changed), result["validation_scores"])


if __name__ == "__main__":
    unittest.main()
