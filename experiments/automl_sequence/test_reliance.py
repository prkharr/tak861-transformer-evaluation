"""Synthetic functional-reliance tests; no clinical conclusions."""
import copy
import types
import unittest

import numpy as np
import pandas as pd

from experiments.automl_sequence import reliance, modeling
from experiments.rigorous_transformer import comparison


class RelianceTests(unittest.TestCase):
    def fixture(self):
        n = 16
        labels = np.tile([0, 1], 8)
        meta = pd.DataFrame(dict(PATIENT_ID=[f"v{i//2}" for i in range(n)],
                                 END_DT=["2026-05-31", "2026-06-30"]*8,
                                 SPLIT=["validation"]*n))
        # Latest labels alternate across patients; paired earlier labels irrelevant.
        labels[1::2] = np.tile([0, 1], 4)
        x = np.zeros((n, 12, 7), dtype=np.float32)
        x[:, :, 0] = labels[:, None]
        x[:, :, 3] = labels[:, None]
        x[:, :, -1] = 1.  # AutoML availability must remain untouched.
        static = np.column_stack([labels, np.zeros(n), np.ones(n)]).astype(np.float32)
        data = dict(indices={"validation": np.arange(n), "test": np.array([999])}, metadata=meta,
                    y=labels, sequence_mask=np.ones((n, 12), bool), count_mask=np.ones((n, 12), bool),
                    summary_mask=np.ones((n, 1), bool))
        prep = {"count": {"selected_names": ["count_a", "count_b", "count_c"]},
                "automl": {"selected_names": ["auto_a", "auto_b", "auto_c"]}}
        protocol = dict(seeds=[42, 142, 242], arms={name: {"family": "transformer", "name": name}
                        for name in ("COUNT", "STATIC_FUSION", "TEMPORAL_FUSION")})
        results = {name: [{"summary": {"seed": seed}, "arm": name} for seed in protocol["seeds"]]
                   for name in protocol["arms"]}
        calls = []
        def arrays(data, prep, config, split, subset=None):
            self.assertEqual(split, "validation")
            arm = config["name"]
            rows=np.arange(n) if subset is None else subset
            return ((x[rows, :, :3] if arm != "TEMPORAL_FUSION" else x[rows]),
                    np.ones((len(rows), 12), bool), (static[rows] if arm == "STATIC_FUSION" else None))
        def predict(run, values, valid=None, static=None):
            self.assertEqual(len(values), 8)
            if run["arm"] == "TEMPORAL_FUSION":
                np.testing.assert_array_equal(values[:, :, -1], 1.)
            calls.append((run["arm"], run["summary"]["seed"]))
            score = values[:, 0, 0].copy()
            if run["arm"] == "STATIC_FUSION":
                score += static[:, 0]
            if run["arm"] == "TEMPORAL_FUSION":
                score += values[:, 0, 3]
            return 1 / (1 + np.exp(-score))
        return data, prep, protocol, results, types.SimpleNamespace(arm_arrays=arrays), types.SimpleNamespace(
            predict_encoder=predict, lift_at_fraction=modeling.lift_at_fraction,
            take_rows=modeling.take_rows, release_array=modeling.release_array), calls

    def test_all_seeds_temporal_groups_and_latest_validation_only(self):
        data, prep, protocol, results, wf, model, calls = self.fixture()
        original = copy.deepcopy(data)
        report = reliance.run(data, prep, protocol, results, wf, model, comparison)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["patients"], 8)
        self.assertEqual(report["shuffleable_strata"], 1)
        self.assertFalse(report["test_accessed"])
        self.assertFalse(report["model_selection_performed"])
        self.assertEqual(len(report["rows"]), 45)
        important = [r for r in report["rows"] if r["arm"] == "COUNT" and r["rank_group"] == "top"]
        self.assertTrue(all(r["ap_drop"] > 0 for r in important))
        unimportant = [r for r in report["rows"] if r["arm"] == "COUNT" and r["rank_group"] != "top"]
        self.assertTrue(all(r["ap_drop"] == 0 for r in unimportant))
        self.assertTrue(all(r["lift_drop"] == 0 for r in unimportant))
        self.assertTrue(all(r["primary_lift_fraction"] == .10 for r in important))
        self.assertTrue(any(r["lift_drop"] > 0 for r in important))
        self.assertEqual(set(seed for _, seed in calls), {42, 142, 242})
        np.testing.assert_array_equal(data["y"], original["y"])
        pd.testing.assert_frame_equal(data["metadata"], original["metadata"])

    def test_incomplete_latest_is_excluded_without_falling_back(self):
        data, prep, protocol, results, wf, model, calls = self.fixture()
        data["sequence_mask"][1::2] = False
        report = reliance.run(data, prep, protocol, results, wf, model, comparison)
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["reason"], "no_complete_latest_validation_snapshots")
        self.assertEqual(report["excluded_incomplete_latest"], 8)
        self.assertEqual(calls, [])

    def test_no_comparable_date_stratum_is_explicit(self):
        data, prep, protocol, results, wf, model, calls = self.fixture()
        data["metadata"]["END_DT"] = pd.date_range("2026-01-01", periods=16)
        report = reliance.run(data, prep, protocol, results, wf, model, comparison)
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["reason"], "no_date_stratum_has_two_patients")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
