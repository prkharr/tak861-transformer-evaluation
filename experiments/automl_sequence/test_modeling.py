"""Synthetic-only regression checks; no clinical performance is inferred."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from experiments.automl_sequence import modeling as m


class LiftTests(unittest.TestCase):
    def test_exact_ceil_capacity_and_label_blind_ties(self):
        self.assertAlmostEqual(m.lift_at_fraction([1, 0, 1, 0, 0], [.9, .8, .7, .6, .5], .3), 1.25)
        self.assertEqual(m.lift_at_fraction([0, 1, 1, 0], [1., 1., 1., 1.], .25), 0.)
        self.assertEqual(m.lift_at_fraction([0, 1, 1, 0], [1., 1., 1., 1.], 1.), 1.)
        with self.assertRaisesRegex(ValueError, "Lift requires"):
            m.lift_at_fraction([1, 0], [.4, .5], 0.)

    def test_ranking_prefers_primary_lift_when_ap_prefers_other_feature(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        first = np.array([1., .9, .1, .8, .7, .6, .5, .4, .3, .2])
        second = np.array([1., .8, .7, .9, .6, .5, .4, .3, .2, .1])
        proxy = np.repeat(np.stack([first, second], axis=1)[:, :, None], 4, axis=2)
        ranking = m._rank(proxy, y, [0, 1], ["higher_lift", "higher_ap"], np.ones(len(y)), .2)
        self.assertEqual(ranking[0]["feature"], "higher_lift")
        self.assertLess(ranking[0]["descriptor_ap"]["summary"], ranking[1]["descriptor_ap"]["summary"])


class PreparationTests(unittest.TestCase):
    def test_blocked_preprocessing_and_proxy_match_dense_float64_formulas(self):
        rng = np.random.default_rng(723)
        x = rng.normal(size=(37, 12, 137)).astype(np.float32)
        x[:, :, 0] = rng.poisson(2, size=(37, 12))
        x[:, :, 1] = rng.integers(0, 2, size=(37, 12))
        x[0, 2, 5] = np.nan
        valid = rng.random((37, 12)) > .2
        valid[:, 0] = True
        x[~valid] = np.nan
        summary = x[:, :1].copy()
        kinds = ["count", "binary"] + ["continuous"]*135
        state = m._fit_transform(summary, x, valid, [f"f{i}" for i in range(137)], kinds)
        dense = x.astype(np.float64).copy()
        dense[~valid] = 0.
        dense[:, :, 0] = np.log1p(dense[:, :, 0])
        dense = np.where(np.isnan(dense), np.asarray(state["median"]), dense)
        dense = (dense-np.asarray(state["center"]))/np.asarray(state["scale"])
        dense[~valid] = 0.
        transformed = m._transform_all(x, state, valid)
        np.testing.assert_array_equal(transformed, dense.astype(np.float32))
        st = m._transform_all(summary, state)
        xx = transformed.astype(np.float64)
        mask, count = valid[:, :, None], valid.sum(1)[:, None]
        mean = np.where(mask, xx, 0.).sum(1)/count
        variance = np.where(mask, (xx-mean[:, None, :])**2, 0.).sum(1)/count
        t = np.arange(12, dtype=float)[None, :, None]
        tmean = np.where(mask, t, 0.).sum(1)/count
        ct = t-tmean[:, None, :]
        denominator = np.where(mask, ct**2, 0.).sum(1)
        slope = np.where(mask, ct*(xx-mean[:, None, :]), 0.).sum(1)/np.maximum(denominator, 1.)
        expected = np.stack([st[:, 0], mean, np.sqrt(np.maximum(variance, 0)), slope], axis=-1)
        np.testing.assert_array_equal(m._proxy(st, transformed, valid), expected)

    def test_rare_binary_survives_and_temporal_only_feature_is_eligible(self):
        rng = np.random.default_rng(19)
        n = 120
        y = np.tile([0, 1], n // 2)
        names = ["rare_binary", "temporal_only", "continuous", "duplicate", "empty", "constant"]
        summary = np.zeros((n, 1, 6), dtype=np.float32)
        sequence = np.zeros((n, 12, 6), dtype=np.float32)
        summary[0, 0, 0] = 1
        sequence[0, 0, 0] = 1
        sequence[:, :, 1] = y[:, None] * np.arange(12)[None, :]
        summary[:, 0, 2] = rng.normal(size=n)
        sequence[:, :, 2] = rng.normal(size=(n, 12))
        summary[:, :, 3] = summary[:, :, 2]
        sequence[:, :, 3] = sequence[:, :, 2]
        summary[:, :, 4] = np.nan
        sequence[:, :, 4] = np.nan
        prep = m.prepare_features(summary, sequence, y, np.arange(n).astype(str), names,
                                  feature_kinds={"temporal_only": "count"},
                                  config={"candidate_sizes": [2, 4, 6], "proxy_max_iter": 150})
        quality = {r["feature"]: r for r in prep["quality"]}
        self.assertEqual(quality["rare_binary"]["reason"], "retained")
        self.assertEqual(quality["temporal_only"]["reason"], "retained")
        self.assertEqual(quality["duplicate"]["reason"], "exact_duplicate")
        self.assertEqual(quality["empty"]["reason"], "all_missing")
        self.assertEqual(quality["constant"]["reason"], "constant")
        self.assertEqual(prep["ranking"][0]["feature"], "temporal_only")
        self.assertEqual(prep["transform"]["scale"][0], 1.)
        self.assertEqual(prep["transform"]["clipping"], "none")
        changed = summary.copy()
        changed[1, 0, 2] = 1e5
        before = copy.deepcopy(prep)
        out = m.transform_inputs(changed, prep)
        self.assertEqual(before, prep)
        self.assertTrue(np.isfinite(out).all())
        self.assertIn(len(prep["transform"]["eligible_indices"]), [r["k"] for r in prep["cv_summary"]])
        self.assertEqual(prep["selected_names"], [names[j] for j in prep["selected_indices"]])
        for r in prep["cv_rows"]:
            self.assertEqual(r["warnings"], [])
        best = max(prep["cv_summary"], key=lambda r: (r["mean_lift"], -r["k"]))
        expected_k = min(r["k"] for r in prep["cv_summary"] if r["mean_lift"] >= best["mean_lift"]-best["se_lift"])
        self.assertEqual(len(prep["selected_names"]), expected_k)

    def test_mask_excludes_payload_and_zero_is_available(self):
        summary = np.asarray([[[0., 1.]], [[1., 2.]]])
        sequence = np.tile(summary, (1, 12, 1))
        valid = np.ones((2, 12), bool)
        valid[:, 5:] = False
        sequence[:, 5:] = np.nan
        state = m._fit_transform(summary, sequence, valid, ["binary", "numeric"], None)
        prep = {"transform": state, "selected_indices": [0, 1]}
        result = m.transform_inputs(sequence, prep, valid)
        self.assertTrue((result[~valid] == 0).all())
        self.assertEqual(result[0, 0, 0], 0.)
        np.testing.assert_array_equal(m.transform_inputs(sequence, prep, np.zeros_like(valid)), 0.)
        # Preprocessing may retain an unavailable auxiliary branch, but an
        # encoder cannot attend to an entirely unavailable sequence by itself.
        with self.assertRaisesRegex(ValueError, "no available timestep"):
            m._array(sequence, np.zeros_like(valid), finite=True)

    def test_fully_unavailable_automl_history_can_be_prepared_without_dropping_snapshot(self):
        rng = np.random.default_rng(223)
        n = 90
        sequence = rng.normal(size=(n, 12, 4)).astype(np.float32)
        valid = np.ones((n, 12), dtype=bool)
        valid[0] = False
        sequence[0] = np.nan
        summary = sequence[:, :1].copy()
        prep = m.prepare_features(summary, sequence, np.tile([0, 1], n//2), np.arange(n).astype(str),
                                  [f"f{j}" for j in range(4)], valid_train=valid,
                                  config={"candidate_sizes": [2, 4], "proxy_max_iter": 200})
        transformed = m.transform_inputs(sequence, prep, valid)
        summary_transformed = m.transform_inputs(summary, prep, valid[:, :1])
        self.assertEqual(len(transformed), n)
        np.testing.assert_array_equal(transformed[0], 0.)
        np.testing.assert_array_equal(summary_transformed[0], 0.)
        proxies = m._proxy(summary_transformed, transformed, valid)
        self.assertTrue(np.isfinite(proxies).all())
        np.testing.assert_array_equal(proxies[0], 0.)
        diagnostics = m.distribution_diagnostics(summary_transformed, summary_transformed[:1], prep["selected_names"],
                                                  valid_train=valid[:, :1], valid_validation=valid[:1, :1])
        self.assertEqual(diagnostics[0]["train_observed_rows"], n-1)
        self.assertEqual(diagnostics[0]["validation_observed_rows"], 0)
        self.assertIsNone(diagnostics[0]["validation_mean"])

    def test_declared_count_does_not_silently_accept_fraction(self):
        summary = np.asarray([[[.5]], [[2.]]])
        with self.assertRaisesRegex(ValueError, "Noninteger"):
            m._fit_transform(summary, np.tile(summary, (1, 12, 1)), np.ones((2, 12), bool), ["count"], ["count"])

    def test_train_zero_one_values_do_not_invent_a_binary_source_contract(self):
        summary = np.asarray([[[0.]], [[1.]]])
        sequence = np.tile(summary, (1, 12, 1))
        valid = np.ones((2, 12), bool)
        inferred = m._fit_transform(summary, sequence, valid, ["rare_count"], None)
        self.assertEqual(m._transform_all(np.asarray([[[2.]]]), inferred)[0, 0, 0], 2.)
        declared = m._fit_transform(summary, sequence, valid, ["flag"], ["binary"])
        with self.assertRaisesRegex(ValueError, "binary semantics"):
            m._transform_all(np.asarray([[[2.]]]), declared)

    def test_class_weight_modes_respect_patient_weights(self):
        y = np.array([1, 1, 0, 0, 0])
        groups = np.array(["a", "a", "b", "c", "d"])
        w = m.sample_weights(groups, "patient")
        self.assertAlmostEqual(m.class_weight(y, w, "full"), 3.)
        self.assertAlmostEqual(m.class_weight(y, w, "sqrt"), np.sqrt(3.))
        self.assertEqual(m.class_weight(y, w, "none"), 1.)
        np.testing.assert_allclose(m.sample_weights(groups, "snapshot"), 1.)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.a = rng.normal(size=(48, 12, 3)).astype(np.float32)
        self.b = rng.normal(size=(24, 12, 3)).astype(np.float32)
        self.y = np.tile([0, 1], 24)
        self.vy = np.tile([0, 1], 12)
        self.groups = np.array([f"tr{i}" for i in range(48)])
        self.vgroups = np.array([f"va{i}" for i in range(24)])
        self.cfg = dict(d_model=8, heads=2, ff=16, epochs=2, patience=2,
                        batch_size=16, threads=1, dropout=.1)

    def test_both_arms_fit_restore_mask_and_reproduce(self):
        for steps in (1, 12):
            a, b = self.a[:, :steps], self.b[:, :steps]
            ma, mb = np.ones(a.shape[:2], bool), np.ones(b.shape[:2], bool)
            if steps == 12:
                ma[:, 9:] = False; mb[:, 9:] = False
            run = m.train_encoder(a, self.y, self.groups, b, self.vy, ma, mb,
                                  self.cfg, 42, self.vgroups)
            restored = m.predict_encoder(run, b, mb)
            np.testing.assert_allclose(restored, run["val_scores"], rtol=0, atol=1e-7)
            self.assertTrue(run["summary"]["patient_disjointness_verified"])
            self.assertFalse(run["summary"]["test_used"])
            self.assertEqual(run["summary"]["best_epoch"],
                             max(run["history"], key=lambda r: r["validation_lift"])["epoch"])
            if steps == 12:
                changed = b.copy(); changed[~mb] = 100000
                np.testing.assert_allclose(m.predict_encoder(run, changed, mb), restored, rtol=0, atol=1e-7)
            repeated = m.train_encoder(a, self.y, self.groups, b, self.vy, ma, mb,
                                       self.cfg, 42, self.vgroups)
            np.testing.assert_array_equal(repeated["val_scores"], run["val_scores"])
            damaged = copy.deepcopy(run)
            damaged["model_blob"] += b"damage"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                m.predict_encoder(damaged, b, mb)

    def test_rejects_patient_overlap(self):
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            m.train_encoder(self.a, self.y, self.groups, self.b, self.vy,
                            config=self.cfg, groups_val=self.groups[:24])

    def test_checkpoint_uses_lift_even_when_ap_improves(self):
        yv = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        early = np.array([1., .9, .1, .8, .7, .6, .5, .4, .3, .2])
        later = np.array([1., .8, .7, .9, .6, .5, .4, .3, .2, .1])
        train_scores = np.full(48, .5)
        with patch.object(m, "_predict", side_effect=[train_scores, early, train_scores, later, train_scores, early]):
            run = m.train_encoder(self.a, self.y, self.groups, self.b[:10], yv,
                                  config={**self.cfg, "primary_lift_fraction": .2}, groups_val=self.vgroups[:10])
        self.assertGreater(run["history"][1]["validation_ap"], run["history"][0]["validation_ap"])
        self.assertLess(run["history"][1]["validation_lift"], run["history"][0]["validation_lift"])
        self.assertEqual(run["summary"]["best_epoch"], 1)
        self.assertEqual(run["summary"]["selection_metric"], "validation_lift_at_primary_capacity")

    def test_attention_pooling_ignores_masked_values(self):
        import torch
        model = m.make_encoder(3, 12, {**self.cfg, "pooling": "attention"})
        model.eval()
        valid = torch.ones((2, 12), dtype=torch.bool); valid[:, 3:] = False
        x = torch.tensor(self.a[:2])
        changed = x.clone(); changed[:, 3:] = 1e4
        with torch.inference_mode():
            torch.testing.assert_close(model(x, valid), model(changed, valid))

    def test_static_fusion_initial_equivalence_gradients_and_saved_prediction(self):
        import torch
        m.seed_run(42)
        base = m.make_encoder(3, 12, self.cfg)
        m.seed_run(42)
        fusion = m.make_encoder(3, 12, self.cfg, static_dim=2)
        base.eval(); fusion.eval()
        x = torch.tensor(self.a[:4]); mask = torch.ones((4, 12), dtype=torch.bool)
        static = torch.ones((4, 2))
        torch.testing.assert_close(base(x, mask), fusion(x, mask, static))
        fusion(x, mask, static).sum().backward()
        self.assertGreater(float(fusion.static_branch[0].weight.grad.abs().sum()), 0.)
        sa = np.stack([self.y, 1-self.y], axis=1).astype(np.float32)
        sb = np.stack([self.vy, 1-self.vy], axis=1).astype(np.float32)
        run = m.train_encoder(self.a, self.y, self.groups, self.b, self.vy,
                              config=self.cfg, groups_val=self.vgroups, static_train=sa, static_val=sb)
        self.assertEqual(run["summary"]["static_dim"], 2)
        np.testing.assert_allclose(m.predict_encoder(run, self.b, static=sb), run["val_scores"], atol=1e-7, rtol=0)
        with self.assertRaisesRegex(ValueError, "Static feature dimension"):
            m.predict_encoder(run, self.b)


if __name__ == "__main__":
    unittest.main()
