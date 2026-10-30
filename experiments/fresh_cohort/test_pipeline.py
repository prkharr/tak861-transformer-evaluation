import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from experiments.automl_sequence import inputs, modeling, raw_automl, workflow as wf
from experiments.automl_sequence import test_workflow as workflow_fixture
from experiments.automl_sequence import test_raw_automl as raw_fixture
from experiments.fresh_cohort import pipeline


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = workflow_fixture.WorkflowTest()
        self.fixture.setUp()
        self.original_sequence = self.fixture.data["X_sequence"]

    def tearDown(self):
        if self.fixture.data["X_sequence"] is not self.original_sequence:
            inputs.release_array(self.original_sequence)
        wf.release_inputs(self.fixture.data)

    def test_summary_phase_and_nlp_use_real_separate_shapes_and_resume(self):
        f = self.fixture
        config = dict(f.config, experiment_phase="SUMMARY", include_nlp=True, raw_automl={"request_steps": 1})
        data = f.data
        data["X_sequence"] = np.array(data["X_sequence"][:, :1])
        data["sequence_mask"] = data["sequence_mask"][:, :1]
        data["X_nlp"] = np.random.default_rng(5).random((96, 12, 3), dtype=np.float32)
        data["nlp_mask"] = np.ones((96, 12), dtype=bool)
        data["nlp_mask"][0] = False
        data["X_nlp"][0] = 0
        data["nlp_manifest"] = dict(feature_names=["sleep", "type 1", "no cataplexy"],
            fit_scope="train_only", orientation="0=newest", metadata_hash=inputs.frame_hash(data["metadata"]),
            train_patients_sha256=inputs.train_patient_hash(data["metadata"]))
        wf.save_inputs(f.prefix, data, f.panel_manifest, config, f.impl)
        loaded = wf.load_inputs(f.prefix, f.impl)
        try:
            prep, protocol = wf.prepare_run(loaded, f.prefix, config, f.impl)
            self.assertEqual(set(protocol["arms"]), {"COUNT", "STATIC_FUSION", "NLP_FUSION"})
            self.assertEqual(loaded["X_sequence"].shape[1], 1)
            self.assertEqual(loaded["X_nlp"].shape[1], 12)
            x, mask, static = wf.arm_arrays(loaded, prep, protocol["arms"]["NLP_FUSION"], "train")
            self.assertTrue(mask[0].all())
            np.testing.assert_array_equal(x[0, :, -4:], 0)
            self.assertIsNone(static)
            modeling.release_array(x)
            results = wf.train_registered(loaded, prep, protocol, f.prefix)
            with patch.object(modeling, "train_encoder", side_effect=AssertionError("Refit")):
                resumed = wf.train_registered(loaded, prep, protocol, f.prefix)
            for arm in results:
                np.testing.assert_array_equal(resumed[arm][0]["val_scores"], results[arm][0]["val_scores"])
            lock = wf.make_lock(loaded, prep, protocol, results, f.prefix)
            report = wf.compare_run(loaded, prep, protocol, results, lock, f.prefix, n_bootstrap=20)
            self.assertEqual(report["split"], "validation")
            self.assertIn("NLP_FUSION_MINUS_COUNT", report["comparisons"])
            self.assertIn("NLP_FUSION_MINUS_STATIC_FUSION", report["comparisons"])
            self.assertFalse(lock["test_prediction_performed"])
        finally:
            wf.release_inputs(loaded)

    def test_phase_mismatch_and_implicit_short_history_rejected(self):
        with self.assertRaises(ValueError):
            wf.phase_steps({"experiment_phase": "SUMMARY", "raw_automl": {"request_steps": 12}})
        f = self.fixture
        data = dict(f.data, X_sequence=f.data["X_sequence"][:, :1], sequence_mask=f.data["sequence_mask"][:, :1])
        with self.assertRaises(ValueError):
            wf.save_inputs(f.prefix, data, f.panel_manifest, f.config, f.impl)

    def test_current_anchor_extraction_does_not_query_historical_keys(self):
        f = raw_fixture.RawAutoMLTest()
        f.setUp()
        current = f.long[f.long.AS_OF_DATE.eq(f.meta.END_DT.iloc[0])].copy()
        result = raw_automl.build_raw_automl_inputs(lambda q: current, f.meta, {"request_steps": 1})
        try:
            self.assertEqual(len(result["cached_panel"]), 2)
            arrays = inputs.assemble_inputs(f.meta, result["cached_panel"], result["feature_specs"],
                                           steps=1, date_policy="month_end_history")
            self.assertEqual(arrays["X_sequence"].shape[1], 1)
            inputs.release_array(arrays["X_sequence"])
            inputs.release_array(arrays["X_summary"])
        finally:
            panel = result["cached_panel"]
            if isinstance(panel, inputs.DiskPanel):
                panel.close()


class AuditGateTests(unittest.TestCase):
    def test_missing_provenance_blocks_before_any_extraction(self):
        for cfg in ({}, {"cohort_evidence": {"producer_sha256": "a"*64}}):
            with self.assertRaises(ValueError):
                pipeline.validate_evidence(cfg)

    def test_only_explicit_retention_and_forward_window_accepted(self):
        cfg = dict(cohort_evidence={k:"verified-local-test" for k in
            ("eligibility_reference", "first_initiation_reference", "negative_followup_reference",
             "effective_cutoff_reference", "source_snapshot_reference")},
             client_universe={"expected_population": {"snapshots": 20}}, experiment_phase="SUMMARY")
        cfg["cohort_evidence"].update(producer_sha256=hashlib.sha256(b"fixture").hexdigest(),
            retention_rule="all_eligible_preinitiation_snapshots", outcome_days=90,
            outcome_boundary="cutoff < first_event <= cutoff + 90 days", historical_availability="retrospective_latest_sources")
        self.assertEqual(pipeline.validate_evidence(cfg)["outcome_days"], 90)
        changed = copy.deepcopy(cfg)
        changed["cohort_evidence"]["retention_rule"] = "all_dates_for_positive_patient"
        with self.assertRaises(ValueError):
            pipeline.validate_evidence(changed)

    def test_candidate_development_requires_explicit_acceptance_without_fabricated_producer(self):
        cfg = dict(cohort_evidence={k:"documented but implementation unverified" for k in
            ("eligibility_reference", "first_initiation_reference", "negative_followup_reference",
             "effective_cutoff_reference", "source_snapshot_reference")},
             client_universe={"expected_population": {"snapshots": 20}}, experiment_phase="SUMMARY")
        cfg["cohort_evidence"].update(lineage_status="candidate_as_delivered_development", producer_sha256=None,
            candidate_labels_accepted=True, automatic_relabel=False, unresolved_rules=["paid status unresolved"],
            retention_rule="all_eligible_preinitiation_snapshots", outcome_days=90,
            outcome_boundary="cutoff < first_event <= cutoff + 90 days", historical_availability="retrospective_latest_sources")
        self.assertIsNone(pipeline.validate_evidence(cfg)["producer_sha256"])
        for key, value in (("candidate_labels_accepted", False), ("automatic_relabel", True), ("unresolved_rules", [])):
            changed = copy.deepcopy(cfg); changed["cohort_evidence"][key] = value
            with self.assertRaises(ValueError):
                pipeline.validate_evidence(changed)


if __name__ == "__main__":
    unittest.main()
