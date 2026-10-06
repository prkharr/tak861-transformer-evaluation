"""Synthetic tests of lineage signal scope and the no-held-out-inspection boundary."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


_SPEC = importlib.util.spec_from_file_location("rigorous_lineage", Path(__file__).with_name("lineage.py"))
l = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(l)


def fixture(orientation="newest_first", patients=8):
    rng = np.random.default_rng(613)
    steps, features = 12, 6
    X = np.zeros((2 * patients + 2, steps, features), dtype=np.float32)
    records = []
    for patient in range(patients):
        calendar = rng.integers(0, 20, (30, features))
        for snapshot, (anchor, date) in enumerate(((15, "2024-01-20"), (18, "2024-04-20"))):
            row = 2 * patient + snapshot
            sequence = calendar[anchor - np.arange(steps)]
            X[row] = sequence if orientation == "newest_first" else sequence[::-1]
            records.append(dict(PATIENT_ID=f"synthetic_{patient:04d}", END_DT=date,
                                RESP=patient % 2, SPLIT="train"))
    records.extend([dict(PATIENT_ID="validation_private", END_DT="2025-01-20", RESP=1, SPLIT="validation"),
                    dict(PATIENT_ID="test_private", END_DT="2025-01-20", RESP=0, SPLIT="test")])
    fmap = pd.DataFrame(dict(FEATURE_INDEX=np.arange(features),
                            FEATURE_NAME=[f"DX_CATEGORY_{i}" for i in range(features)],
                            FEATURE_COLUMN=[f"F{i:04d}" for i in range(features)]))
    return X, pd.DataFrame(records), fmap


class LineageSignals(unittest.TestCase):
    def test_newest_and_oldest_hypotheses_are_scoped_not_certified(self):
        for orientation, other in (("newest_first", "oldest_first"), ("oldest_first", "newest_first")):
            X, meta, fmap = fixture(orientation)
            report = l.overlap_month_consistency(X, meta, fmap)
            self.assertEqual(report["hypotheses"][orientation]["status"], "CORROBORATED")
            self.assertEqual(report["hypotheses"][orientation]["agreement_ratio"], 1.0)
            self.assertEqual(report["hypotheses"][other]["status"], "CONTRADICTED")
            self.assertEqual(report["orientation_status"], "INCONCLUSIVE")
            self.assertFalse(report["orientation_verified"])
            self.assertFalse(report["historical_provenance_verified"])
            self.assertFalse(report["partial_month_risk"]["event_cutoff_verified"])

    def test_capped_and_whole_month_edges_do_not_prove_cutoff(self):
        X, meta, fmap = fixture()
        full = l.overlap_month_consistency(X, meta, fmap)
        altered = X.copy()
        altered[:-2, 0, :] = 0
        altered[:-2, -1, :] = 999
        capped = l.overlap_month_consistency(altered, meta, fmap)
        self.assertEqual(full, capped)  # Both edge positions are excluded.
        risk = capped["partial_month_risk"]
        self.assertEqual(risk["status"], "INCONCLUSIVE")
        self.assertEqual(risk["anchors_before_calendar_month_end"], 16)
        self.assertTrue(risk["without_step0_arm_advisable"])
        self.assertFalse(risk["whole_month_aggregation_verified"])

    def test_zero_and_stable_activity_are_uninformative(self):
        for value in (0, 7):
            X, meta, fmap = fixture()
            X[:] = value
            report = l.overlap_month_consistency(X, meta, fmap)
            for result in report["hypotheses"].values():
                self.assertEqual(result["agreement_ratio"], 1.0)
                self.assertEqual(result["status"], "INCONCLUSIVE")
                self.assertEqual(result["varying_active_comparisons"], 0)
            self.assertEqual(report["mask_source"], "all_positions_retained_no_coverage_claim")

    def test_naive_copied_dynamic_sequences_do_not_pass_window_shift(self):
        X, meta, fmap = fixture()
        X[1:-2:2] = X[:-2:2]
        report = l.overlap_month_consistency(X, meta, fmap)
        self.assertEqual(report["hypotheses"]["newest_first"]["status"], "CONTRADICTED")
        self.assertFalse(report["historical_provenance_verified"])

    def test_missing_is_not_zero_or_inferred_padding(self):
        X, meta, fmap = fixture()
        X[:-2, :, 0] = np.nan
        report = l.overlap_month_consistency(X, meta, fmap)
        result = report["hypotheses"]["newest_first"]
        self.assertGreater(result["missing_comparisons"], 0)
        self.assertEqual(result["agreement_ratio"], 1)
        screen = l.step0_label_screen(X, meta, fmap).set_index("FEATURE_INDEX")
        self.assertEqual(screen.loc[0, "paired_observed_snapshots"], 0)
        self.assertTrue(pd.isna(screen.loc[0, "step0_ap"]))
        self.assertEqual(screen.loc[0, "status"], "INCONCLUSIVE")
        report, _ = l.audit_count_lineage(X, meta, fmap)
        json.dumps(report, allow_nan=False)

    def test_no_held_out_values_dates_identifiers_or_labels_inspected(self):
        X, meta, fmap = fixture()
        expected, expected_screen = l.audit_count_lineage(X, meta, fmap)
        X[-2:] = -np.inf
        changed = meta.astype(object)
        changed.loc[len(meta)-2:, "RESP"] = "unreadable_held_out_label"
        changed.loc[len(meta)-2:, "END_DT"] = "unreadable_held_out_date"
        changed.loc[len(meta)-2:, "PATIENT_ID"] = None
        actual, actual_screen = l.audit_count_lineage(X, changed, fmap)
        self.assertEqual(expected, actual)
        pd.testing.assert_frame_equal(expected_screen, actual_screen)

    def test_pair_cap_is_deterministic_under_row_permutation(self):
        X, meta, fmap = fixture(patients=260)
        report = l.overlap_month_consistency(X, meta, fmap)
        self.assertEqual(report["candidate_adjacent_pair_count"], 260)
        self.assertEqual(report["sampled_patient_pair_count"], 200)
        order = np.random.default_rng(56).permutation(len(X))
        reordered = l.overlap_month_consistency(X[order], meta.iloc[order].reset_index(drop=True), fmap)
        self.assertEqual(report, reordered)
        with self.assertRaisesRegex(ValueError, "1..200"):
            l.overlap_month_consistency(X, meta, fmap, max_pairs=201)

    def test_insufficient_pairs_and_no_overlap_remain_inconclusive(self):
        X, meta, fmap = fixture(patients=1)
        report = l.overlap_month_consistency(X, meta, fmap)
        self.assertEqual(report["hypotheses"]["newest_first"]["status"], "INCONCLUSIVE")
        meta.loc[1, "END_DT"] = "2026-01-20"
        report = l.overlap_month_consistency(X, meta, fmap)
        self.assertEqual(report["sampled_patient_pair_count"], 0)
        self.assertIsNone(report["hypotheses"]["newest_first"]["agreement_ratio"])

    def test_step0_enrichment_is_ranked_but_not_called_leakage(self):
        X, meta, fmap = fixture()
        X[:-2, :, -1] = 0
        X[:-2, 0, -1] = 10 * meta.iloc[:-2].RESP.to_numpy()
        screen = l.step0_label_screen(X, meta, fmap, feature_batch_size=2)
        top = screen.iloc[0]
        self.assertEqual(top.FEATURE_INDEX, X.shape[-1] - 1)
        self.assertEqual(top.step0_ap, 1)
        self.assertEqual(top.step1_ap, 0.5)
        self.assertEqual(top.step0_positive_nonzero_rate, 1)
        self.assertEqual(top.step1_positive_nonzero_rate, 0)
        self.assertEqual(top.claim, "descriptive_step0_enrichment_only_not_leakage_proof")
        self.assertEqual(screen.attrs["scope"], "TRAIN_ONLY")
        self.assertFalse(screen.attrs["historical_provenance_verified"])

    def test_supplied_mask_is_not_certified_and_excludes_pairs(self):
        X, meta, fmap = fixture()
        valid = np.ones(X.shape[:2], dtype=bool)
        valid[:-2] = False
        report, screen = l.audit_count_lineage(X, meta, fmap, valid)
        self.assertEqual(report["mask_source"], "caller_supplied_not_certified_here")
        self.assertEqual(report["hypotheses"]["newest_first"]["finite_comparisons"], 0)
        self.assertEqual(screen.paired_observed_snapshots.sum(), 0)

    def test_count_errors_and_feature_order_are_rejected_on_train(self):
        X, meta, fmap = fixture()
        for bad in (-1, 0.5, np.inf):
            changed = X.copy()
            changed[0, 1, 0] = bad
            with self.assertRaises(ValueError):
                l.step0_label_screen(changed, meta, fmap)
            with self.assertRaises(ValueError):
                l.overlap_month_consistency(changed, meta, fmap)
        with self.assertRaisesRegex(ValueError, "order"):
            l.overlap_month_consistency(X, meta, fmap.iloc[::-1])


if __name__ == "__main__":
    unittest.main()
