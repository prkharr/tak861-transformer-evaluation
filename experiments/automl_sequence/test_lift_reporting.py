import copy
import unittest

import numpy as np
import pandas as pd

from experiments.automl_sequence import lift_reporting as reporting, workflow
from experiments.rigorous_transformer import baseline49, comparison


class ExistingScoreLiftTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"PATIENT_ID": [f"p{i//2}" for i in range(40)],
                                   "END_DT": pd.to_datetime(["2026-01-31", "2026-02-28"] * 20),
                                   "RESP": [0, 1] * 20, "SPLIT": "validation"})
        self.source = self.frame.drop(columns="SPLIT").assign(SCORE=np.linspace(.01, .99, 40))
        self.scores = {name: self.frame.RESP.to_numpy() * .8 + .1 for name in reporting.ARMS}
        self.baseline = baseline49.load_original_v63_scores(lambda _: self.source.sample(frac=1, random_state=42), self.frame)

    def report(self, **kwargs):
        return reporting.build_report(self.frame, self.scores, self.baseline, "validation",
                                      comparison=comparison, workflow=workflow, n_bootstrap=20, **kwargs)

    def test_shared_loader_aligns_and_report_uses_exact_capacity(self):
        report = self.report()
        self.assertEqual(report["interpretation"], "DESCRIPTIVE_ONLY")
        self.assertEqual(report["snapshots"], 40)
        self.assertEqual(report["positives"], 20)
        for name in reporting.ARMS:
            for metric in report["metrics"][name]["topk"]:
                self.assertEqual(metric["lift"], 2.)
                self.assertEqual(metric["selected"], int(np.ceil(metric["fraction"] * 40)))
        np.testing.assert_array_equal(self.baseline["scores"], self.source.SCORE)
        self.assertEqual(len(reporting.interval_table(report)), 15)
        rank = reporting.ranking_table(report)
        self.assertEqual(rank.columns.tolist(), ["Rank", "Model", "Lift@10%"])
        self.assertEqual(rank.Rank.tolist(), [1, 1, 1, 4])
        self.assertEqual(set(rank.loc[rank.Rank.eq(1), "Model"]), set(reporting.ARMS))
        self.assertNotIn("PATIENT_ID", reporting.report_table(report))
        self.assertFalse(report["baseline_provenance"]["fit_membership_verified"])

    def test_missing_duplicate_or_wrong_label_source_stops(self):
        for source in (self.source.iloc[:-1], pd.concat([self.source, self.source.iloc[:1]]),
                       self.source.assign(RESP=1-self.source.RESP)):
            with self.subTest(rows=len(source)), self.assertRaises(ValueError):
                baseline49.load_original_v63_scores(lambda _: source, self.frame)

    def test_wrong_order_split_or_nonfinite_scores_stops(self):
        original = copy.deepcopy(self.baseline)
        for mutation in (lambda b: b.update(metadata=b["metadata"].iloc[::-1]),
                         lambda b: b.update(metadata=b["metadata"].assign(SPLIT="train")),
                         lambda b: b.update(scores=np.full(40, np.nan)),
                         lambda b: b.update(scores=np.full(40, 1.1))):
            self.baseline = copy.deepcopy(original)
            mutation(self.baseline)
            with self.assertRaises(ValueError):
                self.report()

    def test_all_arms_required_and_scores_are_one_vector(self):
        self.scores.pop("COUNT")
        with self.assertRaises(ValueError):
            self.report()
        self.scores["COUNT"] = np.zeros((3, 40))
        with self.assertRaises(ValueError):
            self.report()

    def test_test_requires_both_gates(self):
        frame = self.frame.assign(SPLIT="test")
        baseline = baseline49.load_original_v63_scores(lambda _: self.source, frame,
                                                       split="test", allow_test=True, selection_locked=True)
        for allow, lock in ((False, False), (False, True), (True, False)):
            with self.assertRaises(ValueError):
                reporting.build_report(frame, self.scores, baseline, "test", comparison=comparison,
                                       workflow=workflow, allow_test=allow, selection_locked=lock, n_bootstrap=20)
        report = reporting.build_report(frame, self.scores, baseline, "test", comparison=comparison,
                                        workflow=workflow, allow_test=True, selection_locked=True, n_bootstrap=20)
        self.assertIn("Retrospective", report["evaluation_scope"])

    def test_identity_changes_with_scores_not_physical_source_order(self):
        report = self.report()
        old_identity = report["prediction_identity_sha256"]
        self.baseline = baseline49.load_original_v63_scores(lambda _: self.source, self.frame)
        self.assertEqual(self.report()["prediction_identity_sha256"], old_identity)
        self.scores["COUNT"] = self.scores["COUNT"].copy()
        self.scores["COUNT"][0] += .01
        self.assertNotEqual(self.report()["prediction_identity_sha256"], old_identity)


if __name__ == "__main__":
    unittest.main()
