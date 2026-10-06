"""Synthetic alignment and frozen-validation permutation checks; no project data."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("rigorous_reliance", Path(__file__).with_name("reliance.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def fixture(names=None, columns=None, magnitudes=None, family="transformer", representation="monthly"):
    names = names or ["DX_SIGNAL", "PX_UNUSED", "DX__PARTNER", "RX_UNUSED", "AGE_COUNT", "OTHER"]
    columns = list(range(len(names))) if columns is None else columns
    fm = pd.DataFrame(dict(FEATURE_INDEX=range(len(names)), FEATURE_NAME=names))
    ranking = fm.assign(RANK=np.arange(1, len(names)+1), RANKING_SCORE=np.linspace(1, .1, len(names)),
                        FOLD_ELIGIBILITY_FRACTION=1.0)
    cfg = dict(family=family, representation=representation, columns=columns)
    seeds = [11, 23, 41]
    magnitudes = magnitudes or [np.arange(len(columns), 0, -1).tolist() for _ in seeds]
    blocks = 1 if family == "transformer" else dict(flattened=12, windows=4, annual=1)[representation]
    runs = []
    for seed, importance in zip(seeds, magnitudes):
        blob = ("synthetic model " + str(seed)).encode()
        values = np.asarray(importance, dtype=float)
        runs.append(dict(model_blob=blob, summary=dict(seed=seed, config=copy.deepcopy(cfg),
            config_sha256=r._hash(cfg), model_sha256=hashlib.sha256(blob).hexdigest(), training_complete=True,
            importance=(np.tile(values / blocks, blocks)).tolist(), aggregated_original_importance=values.tolist(),
            importance_blocks_per_original_feature=blocks)))
    return fm, ranking, runs, columns, seeds


def joined_fixture(**kwargs):
    return r.join_rank_and_internal_diagnostics(*fixture(**kwargs))


def synthetic_data(n=40):
    patient = np.arange(n)
    y = (patient % 2).astype(int)
    x = np.empty((n, 12, 4), dtype=float)
    x[:, :, 0] = y[:, None] * 100 + np.arange(12)
    x[:, :, 1] = 2 * x[:, :, 0]
    x[:, :, 2] = patient[:, None] * 1000 + np.arange(12)
    x[:, :, 3] = 7
    meta = pd.DataFrame(dict(PATIENT_ID=["synthetic_" + str(i) for i in patient],
        END_DT="2024-12-31", RESP=y, SPLIT="validation"))
    return x, meta


def diagnostic_fixture(**probe_options):
    names = ["DX_SIGNAL", "DX__PARTNER", "RX_DISTRACTOR", "OTHER"]
    # The strongest internal magnitude is deliberately assigned to the irrelevant feature.
    joined = joined_fixture(names=names, magnitudes=[[1, 2, 100, .5], [2, 1, 90, .1], [1, 3, 80, .2]])
    return names, r.prespecify_probes(joined, **probe_options)


def frozen_predictor(x):
    return np.where(x[:, 0, 0] >= 100, .9, .1)


class RankingAlignment(unittest.TestCase):
    def test_original_indices_survive_gapped_selection_and_every_seed_is_reported(self):
        inputs = fixture(columns=[0, 2, 5], magnitudes=[[3, 2, 1], [1, 2, 3], [2, 3, 1]])
        inputs = list(inputs)
        inputs[1] = inputs[1].iloc[::-1].reset_index(drop=True)
        result = r.join_rank_and_internal_diagnostics(*inputs)
        full, seed = result["full_table"], result["seed_table"]
        self.assertEqual(full.FEATURE_INDEX.tolist(), list(range(6)))
        self.assertEqual(full.loc[full.SELECTED_IN_MODEL, "MODEL_POSITION"].tolist(), [0, 1, 2])
        self.assertEqual(result["summary"]["selected_names"], ["DX_SIGNAL", "DX__PARTNER", "OTHER"])
        self.assertEqual(seed.SEED.value_counts().to_dict(), {11: 3, 23: 3, 41: 3})
        self.assertEqual(seed.loc[seed.SEED.eq(23), "INTERNAL_MAGNITUDE"].tolist(), [1, 2, 3])
        self.assertTrue(seed.NOT_HELDOUT_RELIANCE.all())
        self.assertTrue(full.loc[~full.SELECTED_IN_MODEL, "MEAN_INTERNAL_MAGNITUDE"].isna().all())
        self.assertTrue(full.TRAIN_FOLD_ELIGIBILITY_FRACTION.eq(1).all())

    def test_disagreement_is_descriptive_never_forced_to_match(self):
        joined = joined_fixture(magnitudes=[list(range(1, 7))] * 3)
        self.assertAlmostEqual(joined["summary"]["aggregate_agreement"]["spearman_rank_agreement"], -1)
        self.assertIn("never a model failure", joined["summary"]["aggregate_agreement"]["interpretation"])

    def test_wrong_names_order_width_missing_seeds_and_hashes_rejected(self):
        cases = []
        a = list(fixture()); a[1].loc[0, "FEATURE_NAME"] = "WRONG"; cases.append(a)
        a = list(fixture()); a[3] = [1, 0, 2, 3, 4, 5]; cases.append(a)
        a = list(fixture()); a[2][0]["summary"]["aggregated_original_importance"] = [1]; cases.append(a)
        a = list(fixture()); a[2].pop(); cases.append(a)
        a = list(fixture()); a[2][0]["summary"]["seed"] = 23; cases.append(a)
        a = list(fixture()); a[2][0]["summary"]["config_sha256"] = "wrong"; cases.append(a)
        a = list(fixture()); a[2][0]["model_blob"] += b"tampered"; cases.append(a)
        a = list(fixture()); a[2][0]["summary"]["importance"][0] += 10; cases.append(a)
        for index, args in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                r.join_rank_and_internal_diagnostics(*args)

    def test_temporal_lightgbm_gain_aggregation_checks_axis(self):
        joined = joined_fixture(columns=[0, 2, 5], family="lightgbm", representation="flattened")
        self.assertEqual(joined["seed_table"].INTERNAL_DIAGNOSTIC.unique().tolist(), ["temporal_block_gain_sum"])
        np.testing.assert_array_equal(joined["seed_table"].INTERNAL_MAGNITUDE.to_numpy(), [3, 2, 1] * 3)
        args = fixture(columns=[0, 2, 5], family="lightgbm", representation="flattened")
        args[2][0]["summary"]["importance_blocks_per_original_feature"] = 4
        with self.assertRaisesRegex(ValueError, "temporal-block"):
            r.join_rank_and_internal_diagnostics(*args)

    def test_scale_aware_projection_uses_each_seeds_ordered_train_sd_only(self):
        args = fixture(columns=[0, 2, 5], magnitudes=[[3, 2, 1]] * 3)
        args[2][0]["summary"]["transform"] = dict(training_transformed_sd=[10., .1, 3.])
        args[2][1]["summary"]["transform"] = dict(training_transformed_sd=[1., 2., 5.])
        result = r.join_rank_and_internal_diagnostics(*args)
        first = result["seed_table"].loc[result["seed_table"].SEED.eq(11)]
        self.assertEqual(first.FEATURE_INDEX.tolist(), [0, 2, 5])
        np.testing.assert_allclose(first.PROJECTION_NORM_X_TRAIN_SD, [30., .2, 3.])
        self.assertEqual(first.SCALE_AWARE_INTERNAL_RANK.tolist(), [1., 3., 2.])
        old = result["seed_table"].loc[result["seed_table"].SEED.eq(41)]
        self.assertTrue(old.PROJECTION_NORM_X_TRAIN_SD.isna().all())
        self.assertFalse(old.SCALE_AWARE_INTERNAL_AVAILABLE.any())
        selected = result["full_table"].loc[result["full_table"].SELECTED_IN_MODEL]
        np.testing.assert_allclose(selected.MEAN_PROJECTION_NORM_X_TRAIN_SD, [16.5, 2.1, 4.])
        self.assertEqual(selected.SCALE_AWARE_SEEDS_REPORTED.tolist(), [2, 2, 2])
        self.assertEqual(result["summary"]["scale_aware_available_seeds"], [11, 23])
        for invalid in ([1.], [1., np.nan, 3.], [1., -1., 3.]):
            args[2][0]["summary"]["transform"]["training_transformed_sd"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "TRAIN transformed SD"):
                r.join_rank_and_internal_diagnostics(*args)

    def test_probe_union_bounded_reproducible_and_names_preserved(self):
        names = ["DX_" + str(i) if i % 3 == 0 else "PX__" + str(i) if i % 3 == 1 else "RX_" + str(i) for i in range(60)]
        joined = joined_fixture(names=names)
        a, b = r.prespecify_probes(joined), r.prespecify_probes(joined)
        self.assertEqual(a, b)
        self.assertLessEqual(a["individual_probe_count"], 30)
        self.assertEqual(a["family_probe_count"], 3)
        self.assertEqual(a["selected_names"], names)
        reasons = [reason for p in a["probes"] for reason in p["reasons"]]
        self.assertIn("label_independent_random_remaining", reasons)
        self.assertEqual(len(a["model_identities"]), 3)

    def test_train_terciles_and_correlations_are_deterministic_and_bounded(self):
        names = ["FEATURE_" + str(i) for i in range(24)]
        joined = joined_fixture(names=names)
        pairs = pd.DataFrame(dict(FEATURE_INDEX_A=list(range(0, 24, 2)), FEATURE_INDEX_B=list(range(1, 24, 2)),
                                  SPEARMAN=[.95, -.96] * 6, SPLIT="train"))
        pairs["FEATURE_NAME_A"] = pairs.FEATURE_INDEX_A.map(dict(enumerate(names)))
        pairs["FEATURE_NAME_B"] = pairs.FEATURE_INDEX_B.map(dict(enumerate(names)))
        plan = r.prespecify_probes(joined, correlation_pairs=pairs)
        same = r.prespecify_probes(joined, correlation_pairs=pairs.iloc[::-1])
        self.assertEqual(plan, same)
        self.assertEqual(plan["correlation_probe_count"], 10)
        self.assertEqual(plan["omitted_correlation_components"], 2)
        self.assertEqual(plan["correlation_component_coverage"][-1]["status"], "OMITTED_BY_PRESPECIFIED_GROUP_CAP")
        terciles = {p["probe_id"]:p["feature_indices"] for p in plan["probes"] if p["kind"] == "train_tercile"}
        self.assertEqual(terciles["train_tercile:top"], list(range(8)))
        self.assertEqual(terciles["train_tercile:bottom"], list(range(16, 24)))
        bad = pairs.copy(); bad.loc[0, "SPLIT"] = "validation"
        with self.assertRaisesRegex(ValueError, "TRAIN only"):
            r.prespecify_probes(joined, correlation_pairs=bad)
        bad = pairs.copy(); bad.loc[0, "FEATURE_NAME_A"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "name/index"):
            r.prespecify_probes(joined, correlation_pairs=bad)

    def test_correlation_components_use_only_selected_nodes_and_transitive_edges(self):
        joined = joined_fixture(columns=[0, 2, 4, 5])
        fm, _, _, _, _ = fixture()
        pairs = pd.DataFrame([(0, 1, .99), (1, 2, .99), (2, 4, .95), (4, 5, -.92)],
                             columns=["FEATURE_INDEX_A", "FEATURE_INDEX_B", "SPEARMAN"])
        for side in ("A", "B"):
            pairs["FEATURE_NAME_" + side] = pairs["FEATURE_INDEX_" + side].map(fm.set_index("FEATURE_INDEX").FEATURE_NAME)
        plan = r.prespecify_probes(joined, correlation_pairs=pairs)
        components = [p["feature_indices"] for p in plan["probes"] if p["kind"] == "train_correlation"]
        self.assertEqual(components, [[2, 4, 5]])


class FrozenValidationPermutation(unittest.TestCase):
    def run_diagnostic(self, x=None, meta=None, names=None, plan=None, predictor=frozen_predictor, **kwargs):
        xx, mm = synthetic_data()
        nn, pp = diagnostic_fixture()
        return r.validation_permutation_diagnostic(xx if x is None else x, mm if meta is None else meta,
            nn if names is None else names, predictor, pp if plan is None else plan, "locked-decision", **kwargs)

    def test_measured_reliance_distinguishes_signal_despite_misleading_norms(self):
        result = self.run_diagnostic()
        table = result["coverage"].set_index("FEATURE_INDEX")
        self.assertGreater(table.loc[0, "MEAN_AP_DROP"], .25)
        self.assertEqual(table.loc[2, "MEAN_AP_DROP"], 0)
        self.assertEqual(table.loc[2, "MEAN_ABS_PREDICTION_CHANGE"], 0)
        self.assertGreater(table.loc[0, "MEAN_ABS_PREDICTION_CHANGE"], .1)
        self.assertTrue(result["protocol"]["all_features_individually_tested"])
        self.assertFalse(result["protocol"]["test_rows_used"])
        self.assertIn("not a confidence interval", result["protocol"]["uncertainty"])
        self.assertTrue(result["probe_summary"].REPEATS.eq(3).all())

    def test_whole_trajectories_and_family_columns_share_patient_permutation(self):
        x, meta = synthetic_data()
        original = x.copy()
        names, plan = diagnostic_fixture()
        observed = []
        def spy(values):
            self.assertFalse(values.flags.writeable)
            observed.append(values.copy())
            return frozen_predictor(values)
        result = self.run_diagnostic(x=x, meta=meta, names=names, plan=plan, predictor=spy)
        np.testing.assert_array_equal(x, original)
        self.assertEqual(len(observed), result["protocol"]["predictor_calls"])
        for i, probe in enumerate(plan["probes"]):
            for j, seed in enumerate(result["protocol"]["repeat_seeds"]):
                actual = observed[1 + 3*i + j]
                order = np.random.default_rng(seed).permutation(len(x))
                expected = original.copy()
                expected[:, :, probe["model_positions"]] = original[np.ix_(order, np.arange(12), probe["model_positions"])]
                np.testing.assert_array_equal(actual, expected)
                if probe["probe_id"] == "family:DX":
                    np.testing.assert_array_equal(actual[:, :, 1], 2 * actual[:, :, 0])

    def test_untested_features_remain_unknown_even_in_joint_group(self):
        names, plan = diagnostic_fixture(max_features=1, top_train=1, top_internal=0, lowest_train=0, random_remaining=0)
        result = self.run_diagnostic(names=names, plan=plan)
        table = result["coverage"].set_index("FEATURE_INDEX")
        self.assertEqual(table.loc[1, "PERMUTATION_STATUS"], "NOT_TESTED_UNKNOWN")
        self.assertTrue(pd.isna(table.loc[1, "MEAN_AP_DROP"]))
        self.assertIn("family:DX", table.loc[1, "JOINT_GROUP_PROBES"])
        self.assertFalse(result["protocol"]["all_features_individually_tested"])

    def test_test_train_missing_lock_and_wrong_metadata_rejected_before_prediction(self):
        x, meta = synthetic_data(); names, plan = diagnostic_fixture()
        def prohibited(values):
            self.fail("Predictor should not run before input guards pass")
        for split, lock in [("test", "lock"), ("train", "lock"), ("validation", "")]:
            with self.subTest(split=split, lock=lock), self.assertRaises(ValueError):
                r.validation_permutation_diagnostic(x, meta, names, prohibited, plan, lock, split=split)
        bad = meta.copy(); bad.loc[0, "SPLIT"] = "test"
        with self.assertRaisesRegex(ValueError, "VALIDATION"):
            self.run_diagnostic(meta=bad, predictor=prohibited)
        bad = meta.copy(); bad.loc[1, "PATIENT_ID"] = bad.loc[0, "PATIENT_ID"]
        with self.assertRaisesRegex(ValueError, "distinct patient"):
            self.run_diagnostic(meta=bad, predictor=prohibited)

    def test_plan_tampering_and_feature_name_order_mismatch_rejected(self):
        names, plan = diagnostic_fixture()
        with self.assertRaisesRegex(ValueError, "names/order"):
            self.run_diagnostic(names=names[::-1], plan=plan)
        bad = copy.deepcopy(plan); bad["probes"][0]["feature_names"] = ["wrong"]
        with self.assertRaisesRegex(ValueError, "hash"):
            self.run_diagnostic(names=names, plan=bad)
        bad["plan_sha256"] = r._hash({k:v for k,v in bad.items() if k != "plan_sha256"})
        with self.assertRaisesRegex(ValueError, "name/index/position"):
            self.run_diagnostic(names=names, plan=bad)

    def test_optional_full_validation_reference_verifies_latest_dates_and_labels(self):
        x, meta = synthetic_data()
        old = meta.assign(END_DT="2024-11-30")
        reference = pd.concat([old, meta], ignore_index=True)
        result = self.run_diagnostic(all_validation_metadata=reference)
        self.assertTrue(result["protocol"]["latest_selection_verified"])
        with self.assertRaisesRegex(ValueError, "not the latest"):
            self.run_diagnostic(meta=old, all_validation_metadata=reference)
        bad = reference.copy(); bad.loc[len(old), "RESP"] = 1 - int(bad.loc[len(old), "RESP"])
        with self.assertRaisesRegex(ValueError, "not the latest"):
            self.run_diagnostic(all_validation_metadata=bad)

    def test_invalid_counts_and_predictor_output_rejected_missing_counts_preserved(self):
        x, meta = synthetic_data()
        for value in (-1., .25, np.inf):
            bad = x.copy(); bad[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integer raw counts"):
                self.run_diagnostic(x=bad)
        for callback in (lambda x: np.zeros((len(x), 1)), lambda x: np.full(len(x), np.nan), lambda x: np.full(len(x), 2.)):
            with self.assertRaisesRegex(ValueError, "finite probability"):
                self.run_diagnostic(predictor=callback)
        missing = x.copy(); missing[0, 0, 3] = np.nan
        self.run_diagnostic(x=missing)
        self.assertTrue(np.isnan(missing[0, 0, 3]))

    def test_aggregate_outputs_do_not_export_patient_keys_or_predictions(self):
        result = self.run_diagnostic()
        for key in ("probe_summary", "repeats", "coverage"):
            self.assertFalse({"PATIENT_ID", "END_DT", "RESP", "PREDICTION"} & set(result[key].columns))

    def test_calendar_month_strata_preserved_with_unmoved_singleton(self):
        x, meta = synthetic_data()
        meta.loc[:18, "END_DT"] = "2024-10-31"
        meta.loc[19:38, "END_DT"] = "2024-11-30"
        observed = []
        def spy(values):
            observed.append(values.copy())
            return frozen_predictor(values)
        names, plan = diagnostic_fixture()
        result = self.run_diagnostic(x=x, meta=meta, names=names, plan=plan, predictor=spy)
        for values in observed:
            # Distinct raw counts identify the donor without using patient metadata in the predictor.
            donors = (values[:, 0, 2] / 1000).astype(int)
            self.assertEqual(meta.END_DT.tolist(), meta.iloc[donors].END_DT.tolist())
            np.testing.assert_array_equal(values[39], x[39])
        self.assertEqual(result["protocol"]["calendar_strata"], 3)
        self.assertEqual(result["protocol"]["singleton_strata"], 1)
        self.assertEqual(result["protocol"]["patients_in_singleton_strata"], 1)

    def test_sparse_constant_feature_reports_value_changes_separately_from_movement(self):
        result = self.run_diagnostic()
        constant = result["repeats"].loc[result["repeats"].PROBE_ID.eq("feature:3")]
        self.assertTrue(constant.MOVED_PATIENT_FRACTION.gt(.5).all())
        self.assertTrue(constant.RAW_VALUE_CHANGED_FRACTION.eq(0).all())
        self.assertTrue(constant.RAW_VALUE_CHANGED_PATIENTS.eq(0).all())
        self.assertTrue((result["repeats"].RAW_VALUE_CHANGED_FRACTION <= result["repeats"].MOVED_PATIENT_FRACTION).all())

    def test_lift_drop_and_repeat_sd_are_deterministic_secondary_metrics(self):
        a, b = self.run_diagnostic(), self.run_diagnostic()
        pd.testing.assert_frame_equal(a["repeats"], b["repeats"])
        pd.testing.assert_frame_equal(a["probe_summary"], b["probe_summary"])
        signal = a["probe_summary"].set_index("PROBE_ID").loc["feature:0"]
        rows = a["repeats"].loc[a["repeats"].PROBE_ID.eq("feature:0")]
        self.assertEqual(signal.BASELINE_LIFT_AT10, 2.)
        self.assertGreater(signal.MEAN_LIFT_AT10_DROP, 0)
        self.assertAlmostEqual(signal.REPEAT_SD_LIFT_AT10_DROP, rows.LIFT_AT10_DROP.std(ddof=1))
        self.assertAlmostEqual(signal.REPEAT_SD_AP_DROP, rows.AP_DROP.std(ddof=1))
        self.assertIn("Secondary", a["protocol"]["comparison_scope"])


if __name__ == "__main__":
    unittest.main()
