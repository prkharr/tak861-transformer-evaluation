"""Synthetic crash/restart and integrity tests for the private extraction spool."""
import decimal
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.automl_sequence import inputs, raw_automl as raw


@contextmanager
def database(path):
    with closing(sqlite3.connect(path)) as db, db:
        yield db


class RawResumeTests(unittest.TestCase):
    def test_query_budget_checks_all_chunks_before_any_source_read(self):
        raw.inputs = inputs
        requests = pd.DataFrame(dict(PATIENT_ID=['a', 'z' * 900001],
            AS_OF_DATE=pd.to_datetime(['2026-01-31'] * 2)))
        with closing(sqlite3.connect(':memory:')) as db:
            with self.assertRaisesRegex(ValueError, '900000-byte'):
                raw._spool_long(lambda _: self.fail('Source read before query-size preflight'),
                    requests, raw.raw_sql_template(), 1, {'a'}, db)

    def setUp(self):
        raw.inputs = inputs
        self.meta = pd.DataFrame(dict(PATIENT_ID=['p0', 'p1'],
            END_DT=pd.to_datetime(['2026-10-21']*2), RESP=[0, 1], SPLIT=['train', 'validation']))
        self.requests, _ = inputs.make_requests(self.meta, date_policy='month_end_history')
        self.config = dict(chunk_size=7, max_features=5, date_policy='month_end_history')
        self.calls = []

    def reader(self, sql):
        pairs = re.findall(r"\('([^']+)','(\d{4}-\d{2}-\d{2})'\)", sql)
        self.assertTrue(pairs)
        self.calls.append(pairs)
        rows = []
        for patient, date in pairs:
            anchor = pd.Timestamp(date)
            rows.append(dict(PATIENT_ID=patient, AS_OF_DATE=anchor, ROW_KIND='COVERAGE',
                CATEGORY=None, VALUE=None, RECENT_VALUE=None, MAX_EVENT_DATE=anchor,
                AGE=decimal.Decimal('47.000'), AGE_CONFLICT=decimal.Decimal('0')))
            rows.append(dict(PATIENT_ID=patient, AS_OF_DATE=anchor, ROW_KIND='FEATURE',
                CATEGORY='TEST_FEATURE', VALUE=decimal.Decimal(str(anchor.month+7)),
                RECENT_VALUE=decimal.Decimal('3'), MAX_EVENT_DATE=anchor, AGE=None, AGE_CONFLICT=None))
        return pd.DataFrame(rows)

    def build(self, checkpoint, reader=None, **kwargs):
        return raw.build_raw_automl_inputs(reader or self.reader, self.meta, self.config,
            checkpoint_directory=checkpoint, run_identity='SYNTHETIC_RESUME', **kwargs)

    def assert_equal(self, first, second):
        self.assertEqual(first['feature_specs'], second['feature_specs'])
        self.assertEqual(first['contract'], second['contract'])
        self.assertEqual(first['cached_manifest']['panel_sha256'], second['cached_manifest']['panel_sha256'])
        self.assertEqual(first['receipt']['source_content_sha256'], second['receipt']['source_content_sha256'])
        pd.testing.assert_frame_equal(first['cached_panel'], second['cached_panel'])

    def test_restart_resumes_only_validated_chunks_and_preserves_exact_results(self):
        reference = raw.build_raw_automl_inputs(self.reader, self.meta, self.config)
        self.assertGreater(reference['receipt']['max_chunk_query_bytes'], 0)
        self.assertLessEqual(reference['receipt']['max_chunk_query_bytes'], raw.MAX_REQUEST_QUERY_BYTES)
        with tempfile.TemporaryDirectory() as checkpoint:
            self.calls.clear()
            def crash(sql):
                if len(self.calls) == 1:
                    raise RuntimeError('Synthetic lost connection')
                return self.reader(sql)
            with self.assertRaisesRegex(RuntimeError, 'Synthetic lost connection'):
                self.build(checkpoint, crash)
            with database(Path(checkpoint)/'raw.sqlite') as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0], 1)
            self.calls.clear()
            resumed = self.build(checkpoint)
            self.assertEqual(len(self.calls), 3)
            self.assertEqual(resumed['receipt']['checkpoint']['resumed_chunks'], 1)
            self.assert_equal(reference, resumed)
            self.calls.clear()
            cached = self.build(checkpoint, lambda _: self.fail('Completed source chunk was queried again'))
            self.assertEqual(cached['receipt']['checkpoint']['resumed_chunks'], 4)
            self.assert_equal(reference, cached)

    def test_partial_stream_chunk_rolls_back_and_requeries(self):
        class BrokenStream:
            columns = ['PATIENT_ID','AS_OF_DATE','ROW_KIND','CATEGORY','VALUE','RECENT_VALUE','MAX_EVENT_DATE','AGE','AGE_CONFLICT']
            def toLocalIterator(self, **kwargs):
                for index in range(4096):
                    yield ('not-requested', pd.Timestamp('2026-10-21'), 'FEATURE', str(index), 7, 3,
                           pd.Timestamp('2026-10-21'), None, None)
                raise RuntimeError('Synthetic stream interruption')
        with tempfile.TemporaryDirectory() as checkpoint:
            with self.assertRaisesRegex(RuntimeError, 'Synthetic stream interruption'):
                self.build(checkpoint, lambda _: BrokenStream())
            with database(Path(checkpoint)/'raw.sqlite') as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM raw').fetchone()[0], 0)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0], 0)
            result = self.build(checkpoint)
            self.assertEqual(result['receipt']['checkpoint']['resumed_chunks'], 0)
            self.assertEqual(len(self.calls), 4)

    def test_corrupted_numeric_payload_or_manifest_stops_before_reuse(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            self.build(checkpoint)
            with database(Path(checkpoint)/'raw.sqlite') as db:
                db.execute("UPDATE raw SET v=v+1 WHERE k='FEATURE' AND chunk_no=0")
            with self.assertRaisesRegex(ValueError, 'content hash mismatch'):
                self.build(checkpoint, lambda _: self.fail('Corrupt checkpoint queried source'))
        with tempfile.TemporaryDirectory() as checkpoint:
            self.build(checkpoint)
            for changed in (dict(self.config, chunk_size=6), dict(self.config, source_schema='DB.NEW_SCHEMA')):
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    raw.build_raw_automl_inputs(lambda _: self.fail('Changed identity queried source'), self.meta,
                        changed, checkpoint_directory=checkpoint, run_identity='SYNTHETIC_RESUME')
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                raw.build_raw_automl_inputs(self.reader, self.meta, self.config,
                    checkpoint_directory=checkpoint, run_identity='OTHER_RUN')

    def test_frozen_previous_implementation_has_identical_content_and_panel_hashes(self):
        repo = Path(__file__).resolve().parents[2]
        previous_source = subprocess.check_output(['git', '-c', 'safe.directory='+repo.as_posix(),
            'show', '3c35c5c:experiments/automl_sequence/raw_automl.py'], cwd=repo).decode()
        previous = types.ModuleType('frozen_raw_automl')
        exec(previous_source, previous.__dict__)
        previous.inputs = inputs
        baseline = previous.build_raw_automl_inputs(self.reader, self.meta, self.config)
        with tempfile.TemporaryDirectory() as checkpoint:
            current = self.build(checkpoint)
        self.assert_equal(baseline, current)

    def test_checkpoint_survives_a_fresh_python_process(self):
        repo = Path(__file__).resolve().parents[2]
        setup = '''import json, sys
from pathlib import Path
from experiments.automl_sequence.test_raw_resume import RawResumeTests
fixture=RawResumeTests()
fixture.setUp()
'''
        first = setup + '''def fail_on_second_chunk(sql):
    if len(fixture.calls)==1:
        raise RuntimeError('synthetic interrupted extraction')
    return fixture.reader(sql)
try:
    fixture.build(sys.argv[1], fail_on_second_chunk)
except RuntimeError:
    assert len(fixture.calls)==1
else:
    raise AssertionError('Expected interruption')
'''
        second = setup + '''result=fixture.build(sys.argv[1])
assert len(fixture.calls)==3
assert result['receipt']['checkpoint']['resumed_chunks']==1
Path(sys.argv[2]).write_text(json.dumps({'source_hash':result['receipt']['source_content_sha256'],
    'panel_hash':result['cached_manifest']['panel_sha256']}))
'''
        reference = raw.build_raw_automl_inputs(self.reader, self.meta, self.config)
        with tempfile.TemporaryDirectory() as checkpoint:
            output = Path(checkpoint)/'receipt.json'
            for script in (first, second):
                completed = subprocess.run([sys.executable, '-c', script, checkpoint, str(output)],
                    cwd=repo, capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt['source_hash'], reference['receipt']['source_content_sha256'])
            self.assertEqual(receipt['panel_hash'], reference['cached_manifest']['panel_sha256'])


if __name__ == '__main__':
    unittest.main()
