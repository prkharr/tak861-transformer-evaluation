"""Synthetic read-only checkpoint/export/recovery checks."""
import hashlib
import json
from contextlib import closing, contextmanager
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from experiments.automl_sequence import inputs, raw_automl
from experiments.cached_pilot import cache


@contextmanager
def connected(path):
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            yield connection


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source_dir = self.root / "extraction"
        self.source_dir.mkdir()
        self.source = self.source_dir / "raw.sqlite"
        self.dest = self.root / "pilot"
        self.meta = pd.DataFrame({
            "PATIENT_ID": [p for p in ("001", "002", "003", "004") for _ in range(2)],
            "END_DT": pd.to_datetime(["2026-01-19", "2025-12-19"] * 4),
            "RESP": [0, 1, 0, 0, 1, 0, 0, 1],
            "SPLIT": [s for s in ("train", "validation", "test", "train") for _ in range(2)]})
        self.requests, _ = inputs.make_requests(self.meta, steps=1, date_policy="month_end_history")
        self.train = set(self.meta.loc[self.meta.SPLIT.eq("train"), "PATIENT_ID"])
        raw_automl.inputs = inputs
        self.template = "WITH REQUESTS AS ({request_sql}) SELECT * FROM REQUESTS"

    def tearDown(self):
        self.tmp.cleanup()

    def populate(self, max_chunks=None, steps=1):
        request, _ = inputs.make_requests(self.meta, steps=steps, date_policy="month_end_history")
        calls = 0

        def reader(sql):
            nonlocal calls
            if max_chunks is not None and calls >= max_chunks:
                raise InterruptedError("synthetic interruption")
            calls += 1
            keys = re.findall(r"\('([^']+)','(\d{4}-\d{2}-\d{2})'\)", sql)
            rows = []
            for patient, date in keys:
                rows.append(dict(PATIENT_ID=patient, AS_OF_DATE=date, ROW_KIND="COVERAGE", CATEGORY=None,
                                 VALUE=None, RECENT_VALUE=None, MAX_EVENT_DATE=date, AGE=40, AGE_CONFLICT=0))
                rows.append(dict(PATIENT_ID=patient, AS_OF_DATE=date, ROW_KIND="FEATURE", CATEGORY="STUDY",
                                 VALUE=8, RECENT_VALUE=3, MAX_EVENT_DATE=date, AGE=None, AGE_CONFLICT=None))
            return pd.DataFrame(rows)

        with connected(self.source) as connection:
            try:
                raw_automl._spool_long(reader, request, self.template, 3, self.train, connection,
                                       run_identity="SYNTHETIC_SUMMARY")
            except InterruptedError:
                pass

    def test_export_matches_source_receipts_and_does_not_mutate_source(self):
        self.populate(max_chunks=2)
        before = cache._hash_file(self.source)
        output = cache.freeze_cache(self.source, self.dest, self.meta, page_rows=1)
        manifest = cache.load_frozen_cache(output)
        self.assertEqual(before, cache._hash_file(self.source))
        self.assertEqual(manifest["frozen_request_count"], 6)
        self.assertEqual(manifest["total_requests"], 8)
        with connected(self.source) as source, connected(manifest["analysis_db_path"]) as target:
            original = source.execute("SELECT p,d,k,c,v,r,e,a,ac,chunk_no FROM raw ORDER BY p,d,k,c").fetchall()
            exported = target.execute("SELECT * FROM raw ORDER BY p,d,k,c").fetchall()
        self.assertEqual(original, exported)
        self.assertEqual(len(manifest["chunks"]), 2)
        self.assertTrue(all(x["sha256"] and x["export_sha256"] for x in manifest["chunks"]))

    def test_split_boundary_patient_is_excluded_not_partially_kept(self):
        self.populate(max_chunks=2)
        output = cache.freeze_cache(self.source, self.dest, self.meta, chunk_limit=1)
        complete = cache.complete_patient_metadata(output)
        self.assertEqual(complete.PATIENT_ID.tolist(), ["001", "001"])
        self.assertEqual(complete.SPLIT.tolist(), ["train", "train"])
        self.assertEqual(complete.END_DT.tolist(), self.meta.iloc[:2].END_DT.tolist())

    def test_resume_does_not_reread_original_raw_rows_and_new_commits_ignored(self):
        self.populate(max_chunks=1)
        first = cache.freeze_cache(self.source, self.dest, self.meta)
        original_bytes = first.read_bytes()
        self.populate()
        real_pages = cache._pages

        def forbid_source_rows(connection, chunk_no, columns, page_rows):
            if columns == cache.SOURCE_COLUMNS:
                raise AssertionError("Unnecessary repeated raw export")
            yield from real_pages(connection, chunk_no, columns, page_rows)

        with patch.object(cache, "_pages", side_effect=forbid_source_rows):
            second = cache.freeze_cache(self.source, self.dest, self.meta)
        self.assertEqual(second.read_bytes(), original_bytes)
        self.assertEqual(cache.load_frozen_cache(second)["frozen_request_count"], 3)

    def test_interrupted_export_rolls_back_current_chunk_and_resumes(self):
        self.populate()
        real_pages = cache._pages

        def interrupt(connection, chunk_no, columns, page_rows):
            for page in real_pages(connection, chunk_no, columns, page_rows):
                yield page
                if chunk_no == 1 and columns == cache.SOURCE_COLUMNS:
                    raise InterruptedError("export interrupted")

        with patch.object(cache, "_pages", side_effect=interrupt), self.assertRaises(InterruptedError):
            cache.freeze_cache(self.source, self.dest, self.meta, page_rows=1)
        with connected(self.dest / "analysis.sqlite") as connection:
            self.assertEqual(connection.execute("SELECT chunk_no FROM chunks").fetchall(), [(0,)])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM raw WHERE chunk_no=1").fetchone()[0], 0)
        result = cache.freeze_cache(self.source, self.dest, self.meta, page_rows=2)
        self.assertEqual(len(cache.load_frozen_cache(result)["chunks"]), 3)

    def test_refresh_imports_old_export_without_rereading_source_rows(self):
        self.populate(max_chunks=1)
        previous = cache.freeze_cache(self.source, self.dest, self.meta)
        self.populate()
        real_pages = cache._pages

        def guard(connection, chunk_no, columns, page_rows):
            if columns == cache.SOURCE_COLUMNS and chunk_no == 0:
                raise AssertionError("Old source chunk must be imported from verified export")
            yield from real_pages(connection, chunk_no, columns, page_rows)

        with patch.object(cache, "_pages", side_effect=guard):
            result = cache.freeze_cache(self.source, self.root / "refresh", self.meta,
                                        previous_destination=previous, page_rows=2)
        manifest = cache.load_frozen_cache(result)
        self.assertEqual(manifest["frozen_request_count"], 8)
        self.assertEqual(len(cache.complete_patient_metadata(manifest)), 8)

    def test_source_content_tamper_fails_receipt_hash(self):
        self.populate(max_chunks=1)
        with connected(self.source) as connection:
            connection.execute("UPDATE raw SET v=v+1 WHERE k='FEATURE'")
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.freeze_cache(self.source, self.dest, self.meta)

    def test_original_raw_json_is_hashed_even_when_compact_values_unchanged(self):
        self.populate(max_chunks=1)
        with connected(self.source) as connection:
            connection.execute("UPDATE raw SET raw_json=? WHERE k='FEATURE'", (b"tampered original bytes",))
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.freeze_cache(self.source, self.dest, self.meta)

    def test_analysis_content_or_metadata_tamper_is_rejected(self):
        self.populate(max_chunks=1)
        result = cache.freeze_cache(self.source, self.dest, self.meta)
        with connected(self.dest / "analysis.sqlite") as connection:
            connection.execute("UPDATE raw SET v=v+1 WHERE k='FEATURE'")
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.load_frozen_cache(result)
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.freeze_cache(self.source, self.dest, self.meta)
        (self.dest / "metadata.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            cache.load_frozen_cache(result)

    def test_wrong_population_or_temporal_cache_or_membership_rejected(self):
        self.populate(max_chunks=1)
        wrong = self.meta.copy()
        wrong.loc[0, "END_DT"] += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "SUMMARY"):
            cache.freeze_cache(self.source, self.dest, wrong)
        wrong = self.meta.copy()
        wrong.loc[wrong.PATIENT_ID.eq("002"), "SPLIT"] = "train"
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            cache.freeze_cache(self.source, self.dest, wrong)
        with connected(self.source) as connection:
            identity = json.loads(connection.execute("SELECT value FROM state WHERE key='identity'").fetchone()[0])
            temporal_requests, _ = inputs.make_requests(self.meta, steps=12, date_policy="month_end_history")
            identity["requests_sha256"] = inputs.frame_hash(temporal_requests)
            connection.execute("UPDATE state SET value=? WHERE key='identity'", (json.dumps(identity),))
        with self.assertRaisesRegex(ValueError, "SUMMARY"):
            cache.freeze_cache(self.source, self.dest, self.meta)

    def test_uncommitted_writer_rows_are_not_exported(self):
        self.populate(max_chunks=1)
        writer = sqlite3.connect(self.source)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("INSERT INTO raw VALUES ('unfinished','2026-01-19','COVERAGE','',?,NULL,NULL,NULL,40,0,1)", (b"[]",))
            result = cache.freeze_cache(self.source, self.dest, self.meta, page_rows=1)
            self.assertEqual(cache.load_frozen_cache(result)["frozen_request_count"], 3)
        finally:
            writer.rollback()
            writer.close()

    def test_source_directory_and_uncommitted_boundary_refused(self):
        self.populate(max_chunks=1)
        with self.assertRaisesRegex(ValueError, "isolated"):
            cache.freeze_cache(self.source, self.source_dir, self.meta)
        with self.assertRaisesRegex(ValueError, "isolated"):
            cache.freeze_cache(self.source, self.source_dir / "pilot", self.meta)
        with self.assertRaisesRegex(ValueError, "committed"):
            cache.freeze_cache(self.source, self.dest, self.meta, chunk_limit=2)


if __name__ == "__main__":
    unittest.main()
