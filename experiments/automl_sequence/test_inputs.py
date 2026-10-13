"""Synthetic safety/assembly tests; no clinical data or warehouse access."""
import hashlib
import unittest

import numpy as np
import pandas as pd

try:
    from . import inputs as m
except ImportError:
    import inputs as m


class InputsTest(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({
            "PATIENT_ID": ["a", "a", "b", "c"],
            "END_DT": ["2026-03-31", "2026-04-30", "2026-03-31", "2026-03-31"],
            "RESP": [0, 1, 0, 1], "SPLIT": ["TRAIN", "TRAIN", "VALIDATION", "TEST"],
        })
        self.specs = [
            {"name": "DX_RESPIRATORY", "kind": "count", "definition": "Count through as-of date"},
            {"name": "PSYCHOTHERAPY", "kind": "binary", "definition": "Observed procedure flag through cutoff"},
            {"name": "SIGNED_RATE", "kind": "continuous", "definition": "Continuous signed rate"},
        ]
        self.contract = {"source_id": "synthetic_fixture", "source_version": "v1",
                         "extraction_sha256": hashlib.sha256(b"fixture").hexdigest(),
                         "cutoff_rule": "event and availability <= AS_OF_DATE", "fit_scope": "deterministic_raw",
                         "feature_semantics_evidence": "test fixture declaration"}
        self.calls = []

    def extractor(self, req):
        self.calls.append(req.copy())
        out = req.copy()
        out["IS_OBSERVED"] = True
        out["DX_RESPIRATORY"] = out.AS_OF_DATE.dt.month
        out["PSYCHOTHERAPY"] = out.AS_OF_DATE.dt.month % 2
        out["SIGNED_RATE"] = out.AS_OF_DATE.dt.month - 6.5
        out["MAX_EVENT_DATE"] = out.AS_OF_DATE
        out["MAX_AVAILABLE_DATE"] = out.AS_OF_DATE
        return out

    def panel(self):
        req, _ = m.make_requests(self.metadata)
        return m.obtain_panel(req, self.specs, self.contract, extractor=self.extractor)

    def test_exact_anchor_month_clamping_and_dedup(self):
        req, grid = m.make_requests(self.metadata)
        a = grid.loc[grid.SNAPSHOT_ROW.eq(0)]
        self.assertEqual(a.iloc[0].AS_OF_DATE, pd.Timestamp("2026-03-31"))
        self.assertEqual(a.iloc[1].AS_OF_DATE, pd.Timestamp("2026-02-28"))
        self.assertEqual(a.iloc[2].AS_OF_DATE, pd.Timestamp("2026-01-31"))
        self.assertEqual(len(grid), 48)
        self.assertLess(len(req), 48)
        self.assertFalse(req.duplicated(m.KEYS).any())

    def test_no_claim_evidence_preserves_cohort_and_explicit_missing_mask(self):
        panel, _ = self.panel()
        absent = panel.PATIENT_ID.eq("b")
        panel.loc[absent, "IS_OBSERVED"] = False
        panel.loc[absent, [s["name"] for s in self.specs]] = np.nan
        panel.loc[absent, ["MAX_EVENT_DATE", "MAX_AVAILABLE_DATE"]] = pd.NaT
        data = m.assemble_inputs(self.metadata, panel, self.specs)
        self.assertEqual(len(data["metadata"]), len(self.metadata))
        self.assertFalse(data["sequence_mask"][2].any())
        self.assertTrue(np.isnan(data["X_sequence"][2]).all())
        self.assertEqual(data["report"]["snapshots_without_current_automl_evidence"], 1)

    def test_midmonth_anchor_not_advanced(self):
        meta = self.metadata.iloc[:1].copy()
        meta["END_DT"] = "2026-03-12"
        _, grid = m.make_requests(meta)
        self.assertEqual(grid.iloc[1].AS_OF_DATE, pd.Timestamp("2026-02-12"))
        _, monthends = m.make_requests(meta, date_policy="month_end_history")
        self.assertEqual(monthends.iloc[0].AS_OF_DATE, pd.Timestamp("2026-03-12"))
        self.assertEqual(monthends.iloc[1].AS_OF_DATE, pd.Timestamp("2026-02-28"))

    def test_per_snapshot_historical_values_and_order(self):
        panel, _ = self.panel()
        panel = panel.sample(frac=1, random_state=17)
        data = m.assemble_inputs(self.metadata, panel, self.specs)
        self.assertEqual(data["X_sequence"].shape, (4, 12, 3))
        self.assertEqual(data["X_summary"].shape, (4, 1, 3))
        np.testing.assert_array_equal(data["X_summary"][:, 0], data["X_sequence"][:, 0])
        self.assertEqual(data["X_sequence"][0, 1, 0], 2)
        self.assertEqual(data["X_sequence"][1, 1, 0], 3)
        self.assertNotEqual(data["X_sequence"][0, 0, 0], data["X_sequence"][0, 1, 0])
        np.testing.assert_array_equal(data["indices"]["train"], [0, 1])
        self.assertEqual(data["feature_names"], [s["name"] for s in self.specs])
        self.assertEqual(list(self.calls[0]), m.KEYS)

    def test_cache_reuses_exact_keys_and_extracts_only_missing(self):
        req, _ = m.make_requests(self.metadata)
        first = req.iloc[:3].copy()
        cached, manifest = m.obtain_panel(first, self.specs, self.contract, extractor=self.extractor)
        panel, second = m.obtain_panel(req, self.specs, self.contract, cached, manifest, self.extractor)
        self.assertEqual(second["extraction_report"]["cache_keys_reused"], 3)
        self.assertEqual(second["extraction_report"]["keys_extracted"], len(req) - 3)
        full, third = m.obtain_panel(req, self.specs, self.contract, panel, second)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(third["extraction_report"]["keys_extracted"], 0)
        pd.testing.assert_frame_equal(panel, full)

    def test_mutated_cache_values_rejected(self):
        panel, manifest = self.panel()
        req, _ = m.make_requests(self.metadata)
        panel.loc[0, "SIGNED_RATE"] += 1
        with self.assertRaisesRegex(ValueError, "panel_sha256"):
            m.obtain_panel(req, self.specs, self.contract, panel, manifest)

    def test_changed_source_and_feature_definition_rejected(self):
        panel, manifest = self.panel()
        req, _ = m.make_requests(self.metadata)
        contract = dict(self.contract, source_version="v2")
        with self.assertRaisesRegex(ValueError, "contract_sha256"):
            m.obtain_panel(req, self.specs, contract, panel, manifest)
        specs = [dict(s) for s in self.specs]
        specs[0]["definition"] = "Different window"
        with self.assertRaisesRegex(ValueError, "feature_specs_sha256"):
            m.obtain_panel(req, specs, self.contract, panel, manifest)

    def test_duplicate_keys_fail_and_patient_split_leakage_fails(self):
        bad = pd.concat([self.metadata, self.metadata.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate snapshot"):
            m.normalize_snapshots(bad)
        bad = self.metadata.copy()
        bad.loc[1, "SPLIT"] = "TEST"
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            m.normalize_snapshots(bad)
        panel, _ = self.panel()
        with self.assertRaisesRegex(ValueError, "Duplicate patient/date"):
            m.validate_panel(pd.concat([panel, panel.iloc[:1]], ignore_index=True), self.specs)

    def test_no_substring_feature_deletion(self):
        names = [s["name"] for s in m.validate_feature_specs(self.specs)]
        self.assertIn("DX_RESPIRATORY", names)
        self.assertIn("PSYCHOTHERAPY", names)
        for name in ("RESP", "PATIENT_ID", "MAX_EVENT_DATE", "PREDICTION"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "cannot be a feature"):
                m.validate_feature_specs([{"name": name, "kind": "continuous", "definition": "bad"}])

    def test_domain_and_future_date_checks(self):
        panel, _ = self.panel()
        for field, value, message in [
            ("DX_RESPIRATORY", -1, "invalid count"),
            ("DX_RESPIRATORY", 0.5, "invalid count"),
            ("PSYCHOTHERAPY", 2, "invalid binary"),
            ("SIGNED_RATE", np.inf, "infinite"),
        ]:
            bad = panel.copy()
            bad.loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                m.validate_panel(bad, self.specs)
        for field in ("MAX_EVENT_DATE", "MAX_AVAILABLE_DATE"):
            bad = panel.copy()
            bad.loc[0, field] = bad.loc[0, "AS_OF_DATE"] + pd.Timedelta(days=1)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "future information"):
                m.validate_panel(bad, self.specs)

    def test_unavailable_history_explicit_mask_not_fake_zeros(self):
        panel, _ = self.panel()
        row = panel.index[(panel.PATIENT_ID == "a") & (panel.AS_OF_DATE == pd.Timestamp("2026-02-28"))][0]
        panel.loc[row, "IS_OBSERVED"] = False
        for spec in self.specs:
            panel.loc[row, spec["name"]] = np.nan
        panel.loc[row, ["MAX_EVENT_DATE", "MAX_AVAILABLE_DATE"]] = pd.NaT
        data = m.assemble_inputs(self.metadata, panel, self.specs)
        self.assertFalse(data["sequence_mask"][0, 1])
        self.assertTrue(np.isnan(data["X_sequence"][0, 1]).all())
        panel.loc[row, "DX_RESPIRATORY"] = 0
        with self.assertRaisesRegex(ValueError, "unobserved history must remain missing"):
            m.validate_panel(panel, self.specs)

    def test_missing_rows_require_extraction_never_snapshot_broadcast(self):
        req, _ = m.make_requests(self.metadata)
        with self.assertRaisesRegex(ValueError, "current snapshot table cannot fill history"):
            m.obtain_panel(req, self.specs, self.contract)
        partial = self.extractor(req.iloc[:1])
        with self.assertRaisesRegex(ValueError, "exactly the requested keys"):
            m.obtain_panel(req, self.specs, self.contract, extractor=lambda _: partial)
        with self.assertRaisesRegex(ValueError, "every requested patient/date"):
            m.assemble_inputs(self.metadata, partial, self.specs)

    def test_intraday_and_invalid_label_rejected(self):
        bad = self.metadata.copy()
        bad.loc[0, "END_DT"] = "2026-03-31 13:15:00"
        with self.assertRaises(ValueError):
            m.normalize_snapshots(bad)
        bad = self.metadata.copy()
        bad.loc[0, "RESP"] = 2
        with self.assertRaisesRegex(ValueError, "RESP must be binary"):
            m.normalize_snapshots(bad)

    def test_query_adapter_only_keys_correct_literal_escaping(self):
        query_calls = []
        req = pd.DataFrame({"PATIENT_ID": ["x' OR '1'='1"], "AS_OF_DATE": [pd.Timestamp("2026-01-31")]})
        def reader(query):
            query_calls.append(query)
            return self.extractor(req)
        template = "WITH request AS ({request_sql}) SELECT * FROM request"
        extractor = m.make_query_extractor(reader, template)
        result = extractor(req)
        self.assertEqual(len(result), 1)
        self.assertIn("'x'' OR ''1''=''1' AS PATIENT_ID", query_calls[0])
        self.assertNotIn("RESP", query_calls[0])
        self.assertEqual(extractor.extraction_sha256, hashlib.sha256(template.encode()).hexdigest())

    def test_learned_upstream_requires_fit_scope_fingerprint(self):
        bad = dict(self.contract, fit_scope="train_only_fitted")
        with self.assertRaisesRegex(ValueError, "fitting_patients_sha256"):
            m.validate_contract(bad)
        bad["fitting_patients_sha256"] = hashlib.sha256(b"trainpatients").hexdigest()
        self.assertEqual(m.validate_contract(bad)["fit_scope"], "train_only_fitted")
        with self.assertRaisesRegex(ValueError, "differ from current TRAIN"):
            m.verify_fitting_scope(bad, self.metadata)
        bad["fitting_patients_sha256"] = m.train_patient_hash(self.metadata)
        self.assertEqual(m.verify_fitting_scope(bad, self.metadata), bad)

    def test_float32_overflow_blocked(self):
        panel, _ = self.panel()
        panel.loc[0, "SIGNED_RATE"] = 1e100
        with np.errstate(over="ignore"), self.assertRaisesRegex(ValueError, "overflow float32"):
            m.assemble_inputs(self.metadata, panel, self.specs)


if __name__ == "__main__":
    unittest.main()
