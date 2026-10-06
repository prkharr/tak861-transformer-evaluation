"""Synthetic tests of leakage boundaries, count semantics and shortlist rules."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("rigorous_features", Path(__file__).with_name("features.py"))
f = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = f
_SPEC.loader.exec_module(f)


def fixture():
    rng = np.random.default_rng(101)
    patients, snapshots, steps, features = 48, 2, 12, 8
    ids = np.repeat([f"synthetic_{i:03d}" for i in range(patients)], snapshots)
    y = np.repeat(np.arange(patients) % 2, snapshots)
    splits = np.repeat(["train"] * 36 + ["validation"] * 6 + ["test"] * 6, snapshots)
    X = rng.poisson(0.4, (patients * snapshots, steps, features)).astype(np.float32)
    X[:, :3, 0] += (3 * y[:, None]).astype(np.float32)
    X[:, :, 3] = X[:, :, 0]
    X[:, :, 4] = 0
    X[:, :, 5] = 0
    X[0, 0, 5] = 1  # Rare legitimate event: diagnose, do not discard globally.
    X[:, :, 6] = np.nan
    meta = pd.DataFrame(dict(PATIENT_ID=ids, END_DT=np.tile(["2024-01-20", "2024-06-20"], patients),
                             RESP=y, SPLIT=splits))
    fmap = pd.DataFrame(dict(FEATURE_INDEX=np.arange(features), FEATURE_COLUMN=[f"F{i:04d}" for i in range(features)],
                             FEATURE_NAME=[f"PX_CATEGORY_{i}" for i in range(features)]))
    config = dict(expected_features=features, expected_steps=steps, candidate_sizes=(1, 2, 4),
                  proxy_max_iter=300, batch_rows=11)
    return X, meta, fmap, config


class CountContracts(unittest.TestCase):
    def test_identity_order_and_counts_fail_loudly(self):
        X, meta, fmap, cfg = fixture()
        audit = f.validate_count_inputs(X, meta, fmap, expected_features=8)
        self.assertGreater(audit["missing_values"], 0)
        self.assertEqual(audit["upstream_provenance"], "NOT_VERIFIED_BY_THIS_CHECK")
        for bad in (-1.0, 0.5, np.inf):
            a = X.copy()
            a[0, 0, 1] = bad
            with self.assertRaises(ValueError):
                f.validate_count_inputs(a, meta, fmap, expected_features=8)
        with self.assertRaisesRegex(ValueError, "order"):
            f.validate_count_inputs(X, meta, fmap.iloc[::-1], expected_features=8)
        changed = meta.copy()
        changed.loc[0, "SPLIT"] = "validation"
        with self.assertRaisesRegex(ValueError, "crosses"):
            f.validate_count_inputs(X, changed, fmap, expected_features=8)
        changed = meta.copy()
        changed.loc[0, "END_DT"] = changed.loc[1, "END_DT"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            f.validate_count_inputs(X, changed, fmap, expected_features=8)

    def test_zero_activity_is_not_padding_and_padded_nan_rejected(self):
        X, meta, fmap, _ = fixture()
        X[0] = 0
        audit = f.validate_count_inputs(X, meta, fmap, expected_features=8)
        self.assertEqual(audit["mask_source"], "all_positions_retained")
        valid = np.ones(X.shape[:2], bool)
        valid[0] = False
        X[0, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "padded"):
            f.validate_count_inputs(X, meta, fmap, valid, expected_features=8)

    def test_quality_keeps_rare_and_missingness_and_drops_exact_noise(self):
        X, meta, _, cfg = fixture()
        train = np.flatnonzero(meta.SPLIT.eq("train"))
        state = f.fit_count_preprocessor(X, train, meta.PATIENT_ID, config=cfg)
        ledger = state.quality.set_index("FEATURE_INDEX")
        self.assertTrue(ledger.loc[5, "ELIGIBLE"])
        self.assertTrue(ledger.loc[5, "NEAR_CONSTANT"])
        self.assertEqual(ledger.loc[5, "NONZERO_PATIENTS"], 1)
        self.assertEqual(ledger.loc[3, "QUALITY_REASON"], "exact_duplicate_in_fit")
        self.assertEqual(ledger.loc[3, "DUPLICATE_OF"], 0)
        self.assertEqual(ledger.loc[4, "QUALITY_REASON"], "constant_in_fit")
        self.assertEqual(ledger.loc[6, "QUALITY_REASON"], "all_missing_in_fit")
        a = X.copy()
        a[:, :, 7] = 0
        a[0, 0, 7] = np.nan
        missing = f.fit_count_preprocessor(a, train, meta.PATIENT_ID, config=cfg)
        self.assertIn(7, missing.eligible_indices)
        self.assertTrue(missing.quality.set_index("FEATURE_INDEX").loc[7, "MISSINGNESS_VARIES"])

    def test_state_json_roundtrip_default_scaling_mask_and_no_mutation(self):
        X, meta, _, cfg = fixture()
        X[0, 1, 1] = np.nan
        valid = np.ones(X.shape[:2], bool)
        valid[0, -1] = False
        X[0, -1] = 0
        original = X.copy()
        train = np.flatnonzero(meta.SPLIT.eq("train"))
        state = f.fit_count_preprocessor(X, train, meta.PATIENT_ID, valid, cfg)
        restored = f.CountPreprocessor.from_dict(json.loads(json.dumps(state.to_dict(), allow_nan=False)))
        actual = f.transform_counts(X, restored, valid)
        self.assertTrue(np.array_equal(X, original, equal_nan=True))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual[~valid] == 0).all())
        self.assertTrue((state.mean == 0).all())
        self.assertTrue((state.scale == 1).all())
        j = list(state.eligible_indices).index(0)
        np.testing.assert_allclose(actual[1, :, j], np.log1p(X[1, :, 0]), rtol=1e-6)
        cfg2 = {**cfg, "scaling": "floored_standard"}
        scaled = f.fit_count_preprocessor(X, train, meta.PATIENT_ID, valid, cfg2)
        self.assertTrue((scaled.scale >= 1).all())
        preallocated = np.empty(actual.shape, dtype=np.float32)
        self.assertIs(f.transform_counts(X, restored, valid, out=preallocated), preallocated)
        np.testing.assert_equal(actual, preallocated)

    def test_quarter_window_sums_and_padding(self):
        X, meta, _, cfg = fixture()
        X[:, :, 0] = np.arange(12)
        valid = np.ones(X.shape[:2], bool)
        valid[0, 11] = False
        X[0, 11] = 0
        state = f.fit_count_preprocessor(X, np.flatnonzero(meta.SPLIT.eq("train")), meta.PATIENT_ID, valid, cfg)
        a, names = f.temporal_count_summaries(X, state, np.array([0, 1]), valid, np.array([0]))
        self.assertEqual(names, ["block_0", "block_1", "block_2", "block_3", "newest_1", "newest_3", "newest_6", "newest_12"])
        np.testing.assert_allclose(a[1, :, 0], np.log1p([3, 12, 21, 30, 0, 3, 15, 66]), rtol=1e-6)
        self.assertAlmostEqual(a[0, -1, 0], np.log1p(55), places=6)

    def test_patient_weights_and_grouped_folds(self):
        ids = np.array(["a", "a", "a", "b", "c", "c"])
        weights = f.patient_snapshot_weights(ids)
        totals = pd.Series(weights).groupby(ids).sum()
        np.testing.assert_allclose(totals, totals.iloc[0])
        self.assertAlmostEqual(weights.mean(), 1)
        X, meta, _, cfg = fixture()
        rows = np.flatnonzero(meta.SPLIT.eq("train"))
        seen = []
        for fit, score in f._patient_folds(meta.RESP.to_numpy()[rows], meta.PATIENT_ID.to_numpy()[rows], f._config(cfg)):
            self.assertFalse(set(meta.PATIENT_ID.iloc[rows[fit]]) & set(meta.PATIENT_ID.iloc[rows[score]]))
            seen.extend(rows[score])
        np.testing.assert_equal(np.sort(seen), rows)

    def test_production_width_and_blocked_transform_preserve_original_axis(self):
        rng = np.random.default_rng(78)
        X = rng.poisson(.2, (36, 12, 1028)).astype(np.float32)
        metadata = pd.DataFrame(dict(PATIENT_ID=[f"width_{i}" for i in range(36)],
                                     END_DT=["2024-01-20"] * 36, RESP=np.arange(36) % 2,
                                     SPLIT=["train"] * 24 + ["validation"] * 6 + ["test"] * 6))
        fmap = pd.DataFrame(dict(FEATURE_INDEX=np.arange(1028), FEATURE_COLUMN=[f"F{i:04d}" for i in range(1028)],
                                 FEATURE_NAME=[f"DX_CATEGORY_{i}" for i in range(1028)]))
        self.assertEqual(f.validate_count_inputs(X, metadata, fmap)["shape"], [36, 12, 1028])
        state = f.fit_count_preprocessor(X, np.arange(24), metadata.PATIENT_ID, config={"batch_rows": 5})
        columns = np.array([0, 527, 1027])
        result = f.transform_counts(X, state, feature_indices=columns)
        np.testing.assert_allclose(result, np.log1p(X[:, :, columns]), rtol=1e-6)


class ScientificSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.meta, cls.fmap, cls.cfg = fixture()
        cls.result = f.select_ranked_features(cls.X, cls.meta, cls.fmap, config=cls.cfg)

    def test_one_se_rule_and_control(self):
        table = pd.DataFrame(dict(CANDIDATE=["top_2", "top_4", "all_eligible"],
                                  MEAN_AP=[.40, .42, .41], SE_AP=[.01, .03, .02], FINAL_FEATURE_COUNT=[2, 4, 8]))
        choices, boundary = f.one_se_candidates(table)
        self.assertEqual(choices["proxy_best"], "top_4")
        self.assertEqual(choices["proxy_one_se"], "top_2")
        self.assertEqual(choices["all_eligible"], "all_eligible")
        self.assertAlmostEqual(boundary, .39)
        self.assertIn("all_eligible", self.result.candidate_indices)
        self.assertFalse(self.result.input_audit["proxy_is_transformer_optimum"])
        for indices in self.result.candidate_indices.values():
            np.testing.assert_equal(indices, np.sort(indices))
        self.assertEqual(len({tuple(x) for x in self.result.candidate_indices.values()}), len(self.result.candidate_indices))

    def test_rank_known_signal_and_folds(self):
        self.assertEqual(int(self.result.ranking.iloc[0].FEATURE_INDEX), 0)
        self.assertTrue((self.result.fold_audit.PATIENT_OVERLAP == 0).all())
        self.assertEqual(len(self.result.quality), 8)
        self.assertEqual(set(self.result.cv_scores.METRIC), {"patient_weighted_proxy_AP"})
        self.assertIn("all_eligible", set(self.result.cv_scores.CANDIDATE))

    def test_outer_values_and_labels_do_not_change_selection(self):
        a, meta = self.X.copy(), self.meta.copy()
        heldout = ~meta.SPLIT.eq("train").to_numpy()
        a[heldout] = -999  # Selection must not even inspect outer raw values.
        meta.loc[heldout, "RESP"] = 1 - meta.loc[heldout, "RESP"]
        changed = f.select_ranked_features(a, meta, self.fmap, config=self.cfg)
        pd.testing.assert_frame_equal(changed.ranking, self.result.ranking)
        pd.testing.assert_frame_equal(changed.cv_scores, self.result.cv_scores)
        pd.testing.assert_frame_equal(changed.quality, self.result.quality)
        self.assertEqual(changed.preprocessor.to_dict(), self.result.preprocessor.to_dict())

    def test_fold_preprocessing_does_not_use_heldout_patient_outlier(self):
        rows = np.flatnonzero(self.meta.SPLIT.eq("train"))
        fit, score = next(f._patient_folds(self.meta.RESP.to_numpy()[rows], self.meta.PATIENT_ID.to_numpy()[rows], f._config(self.cfg)))
        cfg = {**self.cfg, "scaling": "floored_standard"}
        first = f.fit_count_preprocessor(self.X, rows[fit], self.meta.PATIENT_ID, config=cfg)
        changed = self.X.copy()
        changed[rows[score], :, 1] = 1_000_000
        second = f.fit_count_preprocessor(changed, rows[fit], self.meta.PATIENT_ID, config=cfg)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_diagnostics_train_only_and_test_shift_refused(self):
        state = self.result.preprocessor
        diag = f.feature_diagnostics(self.X, self.meta, self.fmap, state)
        self.assertEqual(diag["summary"]["patients"], 36)
        self.assertFalse(diag["summary"]["redundancy_auto_removal"])
        shift = f.shift_summary(self.X, self.meta, self.fmap, state)
        self.assertFalse(shift.USED_FOR_SELECTION.any())
        with self.assertRaisesRegex(ValueError, "TEST"):
            f.shift_summary(self.X, self.meta, self.fmap, state, comparison="test")

    def test_correlated_counts_are_reported_without_automatic_removal(self):
        X = self.X.copy()
        X[:, :, 2] = 2 * X[:, :, 0]
        state = f.fit_count_preprocessor(X, np.flatnonzero(self.meta.SPLIT.eq("train")), self.meta.PATIENT_ID,
                                         config=self.cfg)
        self.assertIn(2, state.eligible_indices)
        diagnostic = f.feature_diagnostics(X, self.meta, self.fmap, state)
        pairs = diagnostic["correlation_pairs"]
        pair = pairs.loc[pairs.FEATURE_INDEX_A.eq(0) & pairs.FEATURE_INDEX_B.eq(2)]
        self.assertEqual(len(pair), 1)
        self.assertAlmostEqual(pair.SPEARMAN.iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
