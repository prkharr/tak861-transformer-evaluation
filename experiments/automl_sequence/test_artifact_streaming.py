"""Artifact byte/hash equivalence and bounded publication; no warehouse calls."""
import base64
import hashlib
import importlib.util
import io
from pathlib import Path
import random
import re
import tempfile
import tracemalloc
import unittest
from itertools import islice

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "temporal_selection" / "input_helpers.py"


def module():
    spec = importlib.util.spec_from_file_location("artifact_streaming_under_test", SOURCE)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Rows:
    def __init__(self, rows): self.rows = rows
    def limit(self, count): return Rows(self.rows[:count])
    def collect(self): return list(self.rows)
    def toLocalIterator(self, prefetchPartitions=False):
        assert prefetchPartitions is False
        yield from self.rows


class FakeStore:
    def __init__(self, wh, has_jvm=True):
        self.wh, self.tables, self.batch_sizes, self.renames = wh, {}, [], []
        self.queries, self.query_limits = [], []
        self.unordered_reads = False
        if has_jvm:
            self._jvm = object()
        self.fail_write_number = None
        wh.spark = self
        wh.sf_options_dl_poc = {}
        wh.table_exists = self.tables.__contains__
        wh._query_artifacts = self.query
        wh._artifact_ddl = self.ddl

    def createDataFrame(self, rows, schema):
        self.batch_sizes.append(len(rows))
        owner = self
        class Writer:
            def format(self, value): return self
            def options(self, **kwargs): return self
            def option(self, name, value):
                if name == "dbtable": self.table = value
                return self
            def mode(self, value): self.save_mode = value; return self
            def save(self):
                if owner.fail_write_number == len(owner.batch_sizes):
                    raise RuntimeError("Injected transfer failure")
                if self.save_mode == "errorifexists" and self.table in owner.tables:
                    raise FileExistsError(self.table)
                owner.tables.setdefault(self.table, []).extend(rows)
        class Frame: write = Writer()
        return Frame()

    def query(self, query):
        self.queries.append(query)
        table = re.search(r" FROM ([A-Z0-9_]+)", query)[1]
        records = self.tables[table]
        if "GROUP BY" in query:
            groups = {}
            for name, index, count, size, digest, payload in records:
                key = (name, count, size, digest)
                groups[key] = groups.get(key, 0) + 1
            return Rows([(*key, value) for key, value in groups.items()])
        name, start, end = re.search(r"ARTIFACT_NAME='([^']+)' AND CHUNK_INDEX >= (\d+) AND CHUNK_INDEX < (\d+)", query).groups()
        self.query_limits.append(int(end)-int(start))
        rows = sorted((r for r in records if r[0] == name and int(start) <= r[1] < int(end)), key=lambda r:r[1])
        if self.unordered_reads:
            random.Random(42).shuffle(rows)
        return Rows(rows)

    def ddl(self, statement):
        old, new = re.fullmatch(r"ALTER TABLE ([A-Z0-9_]+) RENAME TO ([A-Z0-9_]+)", statement).groups()
        if new in self.tables: raise FileExistsError(new)
        self.tables[new] = self.tables.pop(old)
        self.renames.append((old, new))


class ArtifactStreamingTests(unittest.TestCase):
    def setUp(self): self.wh = module()

    def test_rows_and_hashes_equal_legacy_packer_for_odd_boundaries(self):
        artifacts = {"empty": b"", "small": bytes(range(99)), "unicode.json": '{"x":"é"}'.encode()}
        for chunk_size in (1, 3, 4, 7, 16, 50000):
            expected = []
            for name, blob in artifacts.items():
                encoded = base64.b64encode(blob).decode("ascii")
                pieces = [encoded[i:i+chunk_size] for i in range(0,len(encoded),chunk_size)] or [""]
                expected.extend((name,i,len(pieces),len(blob),hashlib.sha256(blob).hexdigest(),part)
                                for i,part in enumerate(pieces))
            self.assertEqual(self.wh.pack_artifacts(artifacts,chunk_size), expected)
            random.Random(42).shuffle(expected)
            self.assertEqual(self.wh.unpack_artifacts(expected,artifacts),artifacts)

    def test_file_and_bytes_payloads_roundtrip_identical(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"arrays.npz"
            arrays = {"X":np.arange(1024,dtype=np.float32).reshape(16,64),"mask":np.ones(16,bool)}
            in_memory = io.BytesIO();np.savez_compressed(in_memory,**arrays)
            np.savez_compressed(path,**arrays)
            self.assertEqual(path.read_bytes(),in_memory.getvalue())
            artifacts = {"arrays.npz":path,"manifest.json":b"{}"}
            rows = self.wh.iter_artifact_rows(artifacts,31)
            loaded = self.wh.unpack_artifacts(rows,artifacts,disk_names={"arrays.npz"},output_dir=root,ordered=True)
            self.assertEqual(loaded["arrays.npz"].read_bytes(),in_memory.getvalue())
            self.assertEqual(loaded["manifest.json"],b"{}")

    def test_bounded_writes_atomic_publish_and_existing_identity(self):
        store = FakeStore(self.wh)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"arrays.npz";path.write_bytes(bytes(range(256))*100)
            artifacts={"arrays.npz":path,"manifest.json":b"{}"}
            self.wh.save_artifacts("RUN_INPUTS",artifacts,chunk_size=64,batch_rows=5)
            self.assertLessEqual(max(store.batch_sizes),5)
            self.assertEqual(len(store.renames),1)
            self.assertEqual(set(store.tables),{"RUN_INPUTS"})
            # Only one complete read-back before the atomic object rename.
            self.assertEqual(sum("GROUP BY" in q for q in store.queries),1)
            self.assertLessEqual(max(store.query_limits),5)
            loaded=self.wh.read_artifacts("RUN_INPUTS",artifacts,disk_names={"arrays.npz"},output_dir=root,batch_rows=3)
            self.assertEqual(loaded['arrays.npz'].read_bytes(),path.read_bytes())
            count=len(store.batch_sizes)
            self.wh.save_artifacts("RUN_INPUTS",artifacts)
            self.assertEqual(len(store.batch_sizes),count)
            with self.assertRaises(FileExistsError):
                self.wh.save_artifacts("RUN_INPUTS",dict(artifacts,**{"manifest.json":b"changed"}))

    def test_interrupted_transfer_never_publishes_final_stage(self):
        store=FakeStore(self.wh);store.fail_write_number=2
        with self.assertRaisesRegex(RuntimeError,"Injected"):
            self.wh.save_artifacts("RUN_INPUTS",{"data":b"a"*2000},chunk_size=40,batch_rows=2)
        self.assertNotIn("RUN_INPUTS",store.tables)
        self.assertFalse(store.renames)
        store.fail_write_number=None
        self.wh.save_artifacts("RUN_INPUTS",{"data":b"a"*2000},chunk_size=40,batch_rows=2)
        self.assertIn("RUN_INPUTS",store.tables)

    def test_corrupt_chunks_rejected_and_partial_disk_outputs_cleaned(self):
        with tempfile.TemporaryDirectory() as root:
            original=self.wh.pack_artifacts({"large":b"abcd"*20},8)
            for rows in (original[:-1], original+[original[-1]],
                         [(*original[0][:-1],"!!!!")]+original[1:]):
                with self.assertRaises(ValueError):
                    self.wh.unpack_artifacts(rows,{"large"},disk_names={"large"},output_dir=root)
                self.assertEqual(list(Path(root).iterdir()),[])

    def test_unordered_connector_batches_are_sorted_locally(self):
        store=FakeStore(self.wh);store.unordered_reads=True
        artifacts={"data":bytes(range(256))*100,"manifest":b"{}"}
        self.wh.save_artifacts("RUN_INPUTS",artifacts,chunk_size=64,batch_rows=11)
        self.assertEqual(self.wh.read_artifacts("RUN_INPUTS",artifacts,batch_rows=7),artifacts)
        self.assertLessEqual(max(store.query_limits),11)

    def test_no_jvm_direct_write_and_failed_upload_remain_immutable(self):
        store=FakeStore(self.wh,has_jvm=False)
        artifacts={"data":b"a"*2000}
        store.fail_write_number=2
        with self.assertRaisesRegex(RuntimeError,"Injected"):
            self.wh.save_artifacts("RUN_INPUTS",artifacts,chunk_size=40,batch_rows=2)
        self.assertIn("RUN_INPUTS",store.tables)
        self.assertFalse(store.renames)
        partial=list(store.tables["RUN_INPUTS"])
        with self.assertRaises(ValueError):
            self.wh.read_artifacts("RUN_INPUTS",artifacts)
        store.fail_write_number=None
        with self.assertRaisesRegex(FileExistsError,"incomplete"):
            self.wh.save_artifacts("RUN_INPUTS",artifacts,chunk_size=40,batch_rows=2)
        self.assertEqual(store.tables["RUN_INPUTS"],partial)
        before=len(store.queries)
        self.wh.save_artifacts("NEW_RUN_INPUTS",artifacts,chunk_size=40,batch_rows=2)
        self.assertEqual(sum("GROUP BY" in q for q in store.queries[before:]),1)
        self.assertEqual(self.wh.read_artifacts("NEW_RUN_INPUTS",artifacts),artifacts)
        self.assertFalse(store.renames)

    def test_spark_connect_raising_jvm_attribute_detected_before_upload(self):
        class ConnectStore(FakeStore):
            @property
            def _jvm(self):
                self.jvm_checks.append(len(self.batch_sizes))
                raise AttributeError("JVM_ATTRIBUTE_NOT_SUPPORTED")
        store=ConnectStore(self.wh,has_jvm=False);store.jvm_checks=[]
        self.wh.save_artifacts("RUN_INPUTS",{"data":b"abc"})
        self.assertEqual(store.jvm_checks,[0])
        self.assertFalse(store.renames)

    def test_default_batches_are_1024_and_memory_remains_bounded(self):
        self.assertEqual(self.wh.ARTIFACT_BATCH_ROWS,1024)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"large.bin"
            with path.open('wb') as output:
                for _ in range(48):output.write(b'x'*(1024*1024))
            rows=self.wh.iter_artifact_rows({"large":path})
            tracemalloc.start()
            sizes=[]
            while True:
                batch=list(islice(rows,self.wh.ARTIFACT_BATCH_ROWS))
                if not batch:break
                sizes.append(len(batch))
                del batch
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            self.assertEqual(max(sizes),self.wh.ARTIFACT_BATCH_ROWS)
            self.assertGreater(len(sizes),1)
            # One 51.2 MB encoded batch plus bounded encoder/hash bookkeeping.
            self.assertLess(peak,self.wh.ARTIFACT_BATCH_ROWS*50000+5*1024*1024)

    def test_file_row_iterator_peak_is_independent_of_file_size(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"large.bin"
            with path.open('wb') as output:
                for _ in range(24):output.write(b'x'*(1024*1024))
            tracemalloc.start()
            count=sum(1 for _ in self.wh.iter_artifact_rows({"large":path}))
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            self.assertGreater(count,600)
            self.assertLess(peak,3*1024*1024)


if __name__ == '__main__': unittest.main()
