import copy
import re
import tempfile
import unittest
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse

from experiments.fresh_cohort import nlp_features as nlp


class NLPTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame(dict(PATIENT_ID=["train", "train", "held"],
            END_DT=pd.to_datetime(["2026-03-31", "2026-04-15", "2026-03-31"]),
            SPLIT=["train", "train", "validation"], RESP=[0, 1, 1]))
        self.cfg = dict(medical="DB.SCH.MED", reference="DB.SCH.MAP", event_id_column="MEDICAL_EVENT_ID",
                        max_terms=256, chunk_size=7, max_source_rows_per_chunk=1000, run_identity="synthetic-v1")
        self.calls = []

    def read(self, sql):
        self.calls.append(sql)
        self.assertNotIn("RESP", sql)
        if "HASH_AGG" in sql:
            return pd.DataFrame({"ROWS": [2], "MAPPING_HASH": ["synthetic-v1"]})
        values = re.findall(r"\((\d+),'([^']+)','(\d{4}-\d{2}-\d{2})'\)", sql)
        rows = []
        for doc, patient, date in values:
            observed = pd.Timestamp(date) >= pd.Timestamp("2025-07-01")
            common = dict(DOC_ID=int(doc), NULL_EVENT_ID_ROWS=0,
                          MAX_EVENT_DATE=date if observed else None)
            rows.append(dict(common, ROW_KIND="COVERAGE", FAMILY=None, CODE=None, DESCRIPTION=None,
                             DESCRIPTION_VARIANTS=0, EVENT_COUNT=int(observed)))
            if observed:
                description = "type 1 narcolepsy without cataplexy" if patient == "train" else "heldoutmystery"
                rows.append(dict(common, ROW_KIND="FEATURE", FAMILY="DX", CODE=patient, DESCRIPTION=description,
                                 DESCRIPTION_VARIANTS=1, EVENT_COUNT=2 if pd.Timestamp(date).month == 3 else 1))
        return pd.DataFrame(rows)

    def test_training_vocabulary_reuse_orientation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            out = nlp.extract_and_fit(self.read, self.meta, self.cfg, directory)
            try:
                self.assertEqual(out["X_sequence"].shape[:2], (3, 12))
                self.assertIn("type 1", out["feature_names"])
                self.assertIn("without cataplexy", out["feature_names"])
                self.assertNotIn("heldoutmystery", out["feature_names"])
                self.assertEqual(out["receipt"]["unique_monthly_documents"], 25)
                self.assertEqual(out["receipt"]["reused_positions"], 11)
                np.testing.assert_array_equal(out["X_sequence"][0, 0], out["X_sequence"][1, 1])
                self.assertFalse(out["mask"][0, -1])
                self.assertTrue(out["mask"][2, 0])
                self.assertFalse(out["X_sequence"][2].any())
                state = out["fitted_state"]
                n = state["training_documents"]
                synthetic_counts = np.zeros((n, len(state["terms"])))
                for j, df in enumerate(state["document_frequency"]):
                    synthetic_counts[:df, j] = 1
                reference = TfidfTransformer(sublinear_tf=True).fit(sparse.csr_matrix(synthetic_counts))
                np.testing.assert_allclose(state["idf"], reference.idf_, rtol=0, atol=0)
                def forbid(sql):
                    if "HASH_AGG" in sql:
                        return self.read(sql)
                    raise AssertionError("Validated chunk queried again")
                resumed = nlp.extract_and_fit(forbid, self.meta, self.cfg, directory)
                try:
                    self.assertEqual(state, resumed["fitted_state"])
                    np.testing.assert_array_equal(out["X_sequence"], resumed["X_sequence"])
                    self.assertEqual(resumed["receipt"]["checkpoint"]["queries"], 0)
                finally:
                    nlp.release_result(resumed)
            finally:
                nlp.release_result(out)

    def test_validation_only_changes_do_not_change_fit(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = nlp.extract_and_fit(self.read, self.meta, self.cfg, a)
            def changed(sql):
                frame = self.read(sql)
                if "CODE" not in frame:
                    return frame
                frame.loc[frame.CODE.eq("held"), "DESCRIPTION"] = "novelvalidationsecret type 2"
                return frame
            second = nlp.extract_and_fit(changed, self.meta.assign(RESP=1-self.meta.RESP), self.cfg, b)
            try:
                self.assertEqual(first["fitted_state"], second["fitted_state"])
                np.testing.assert_array_equal(first["X_sequence"][:2], second["X_sequence"][:2])
            finally:
                nlp.release_result(first); nlp.release_result(second)

    def test_chunk_overflow_splits_queries_without_subsampling(self):
        cfg = dict(self.cfg, max_source_rows_per_chunk=4)
        with tempfile.TemporaryDirectory() as directory:
            out = nlp.extract_and_fit(self.read, self.meta, cfg, directory)
            try:
                self.assertGreater(out["receipt"]["checkpoint"]["split_queries"], 0)
                self.assertEqual(out["receipt"]["unique_monthly_documents"], 25)
            finally:
                nlp.release_result(out)

    def test_future_event_missing_coverage_and_changed_membership_stop(self):
        for kind in ("future", "missing", "duplicate", "null_id"):
            def bad(sql):
                frame = self.read(sql)
                if "CODE" not in frame:
                    return frame
                if kind == "future":
                    frame.loc[0, "MAX_EVENT_DATE"] = "2030-01-01"
                elif kind == "missing":
                    frame = frame.iloc[1:]
                elif kind == "duplicate":
                    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
                else:
                    frame.loc[0, "NULL_EVENT_ID_ROWS"] = 1
                return frame
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                nlp.extract_and_fit(bad, self.meta, self.cfg, directory)
        with tempfile.TemporaryDirectory() as directory:
            out = nlp.extract_and_fit(self.read, self.meta, self.cfg, directory)
            nlp.release_result(out)
            changed = self.meta.copy(); changed.loc[2, "SPLIT"] = "test"
            with self.assertRaisesRegex(ValueError, "identity"):
                nlp.extract_and_fit(self.read, changed, self.cfg, directory)

    def test_patient_cross_split_and_empty_mapping_fail(self):
        bad = self.meta.copy(); bad.loc[1, "SPLIT"] = "test"
        with self.assertRaises(ValueError):
            nlp.make_requests(bad)
        def unmapped(sql):
            frame = self.read(sql)
            if "CODE" in frame:
                frame["DESCRIPTION"] = None
            return frame
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "No tokenized"):
            nlp.extract_and_fit(unmapped, self.meta, self.cfg, directory)

    def test_partial_months_are_not_merged_with_future_month_end(self):
        meta = pd.DataFrame(dict(PATIENT_ID=["a"]*3, SPLIT=["train"]*3,
            END_DT=pd.to_datetime(["2026-10-21", "2026-10-31", "2026-11-21"])))
        _, requests, positions = nlp.make_requests(meta)
        self.assertNotEqual(positions[0, 0], positions[1, 0])
        self.assertEqual(positions[1, 0], positions[2, 1])
        self.assertEqual(requests.iloc[positions[0, 0]].AS_OF_DATE, pd.Timestamp("2026-10-21"))

    def test_mapping_drift_and_checkpoint_tampering_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            out = nlp.extract_and_fit(self.read, self.meta, self.cfg, directory)
            nlp.release_result(out)
            def drift(sql):
                frame = self.read(sql)
                if "HASH_AGG" in sql:
                    frame["MAPPING_HASH"] = "synthetic-v2"
                return frame
            with self.assertRaisesRegex(ValueError, "identity"):
                nlp.extract_and_fit(drift, self.meta, self.cfg, directory)
            with closing(sqlite3.connect(Path(directory)/"monthly.sqlite")) as db:
                db.execute("UPDATE features SET n=n+1 WHERE doc=(SELECT MIN(doc) FROM features)")
                db.commit()
            with self.assertRaisesRegex(ValueError, "integrity"):
                nlp.extract_and_fit(self.read, self.meta, self.cfg, directory)


if __name__ == "__main__":
    unittest.main()
