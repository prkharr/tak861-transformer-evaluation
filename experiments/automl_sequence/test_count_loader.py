"""Synthetic connector-double tests; do not imply a real Spark/warehouse run."""
import hashlib
import io
import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from . import count_loader, inputs
except ImportError:
    import count_loader, inputs


class Expression:
    def cast(self, *args): return self
    def alias(self, *args): return self


class Frame:
    def __init__(self, frame=None, rows=None, columns=None):
        self.frame, self.rows = frame, rows
        self.columns = list(frame.columns) if frame is not None else columns
    def orderBy(self, *args): return self
    def select(self, *names): return Frame(self.frame[list(names)].copy())
    def toPandas(self): return self.frame.copy()
    def collect(self):
        return [types.SimpleNamespace(asDict=lambda r=row: r) for row in self.rows]
    def groupBy(self, *args): return self
    def agg(self, *args): return self
    def toLocalIterator(self, **kwargs): return iter(self.rows)


class CountLoaderTest(unittest.TestCase):
    def setUp(self):
        count_loader.inputs = inputs
        self.names = [f"DX__COUNT_{i}" for i in range(1028)]
        self.aliases = [f"F{i:04d}" for i in range(1028)]
        creation = {"n_timesteps": 12, "feature_order_sha256": hashlib.sha256(json.dumps(self.names).encode()).hexdigest()}
        self.metadata = pd.DataFrame({"PATIENT_ID": ["01", "02"], "END_DT": [pd.Timestamp("2026-03-21")] * 2,
                                      "RESP": [0, 1], "SPLIT": ["TRAIN", "VALIDATION"],
                                      "SPLIT_CONFIG": [json.dumps(creation)] * 2})
        mapping = [{"FEATURE_INDEX": i, "FEATURE_NAME": name, "FEATURE_COLUMN": self.aliases[i]} for i, name in enumerate(self.names)]
        self.rows = [{"PATIENT_ID": r.PATIENT_ID, "END_DT": r.END_DT, "RESP": r.RESP,
                      "SEQUENCE": [{"T": step, "V": [float(step)] * 1028} for step in reversed(range(12))]}
                     for r in reversed(list(self.metadata.itertuples()))]
        self.tables = {"FEATURE_MAP": Frame(rows=mapping), "PATIENT_SPLIT": Frame(self.metadata),
                       "SNAPSHOTS": Frame(self.metadata[["PATIENT_ID", "END_DT", "RESP"]]),
                       "TENSOR_MONTHLY": Frame(rows=self.rows, columns=["PATIENT_ID", "END_DT", "RESP", "TIME_STEP"] + self.aliases)}
        functions = types.ModuleType("pyspark.sql.functions")
        for name in ("col", "array", "collect_list", "struct"):
            setattr(functions, name, lambda *args: Expression())
        sql = types.ModuleType("pyspark.sql"); sql.functions = functions
        package = types.ModuleType("pyspark"); package.sql = sql
        self.modules = {"pyspark": package, "pyspark.sql": sql, "pyspark.sql.functions": functions}

    def test_source_only_count_reuse_and_calendar_contract(self):
        seen = []
        def read(name):
            seen.append(name)
            return self.tables[name]
        with patch.dict(sys.modules, self.modules):
            data = count_loader.load_count_inputs(read, "SYNTHETIC_COUNTS", "fixture_v1")
        try:
            self.assertEqual(seen, ["FEATURE_MAP", "PATIENT_SPLIT", "SNAPSHOTS", "TENSOR_MONTHLY"])
            self.assertEqual(data["metadata"].PATIENT_ID.tolist(), ["01", "02"])
            np.testing.assert_array_equal(data["X"][0, :, 0], np.arange(12))
            self.assertEqual(data["contract"]["date_policy"], "month_end_history")
            self.assertEqual(data["contract"]["mask_kind"], "saved_calendar_position_not_clinical_observation")
            self.assertTrue(data["valid"].all())
            memory = io.BytesIO(); np.save(memory, data["X"], allow_pickle=False)
            self.assertEqual(data["contract"]["content_sha256"], hashlib.sha256(memory.getvalue()).hexdigest())
            _, grid = inputs.make_requests(data["metadata"], date_policy="month_end_history")
            self.assertEqual(grid.iloc[0].AS_OF_DATE, pd.Timestamp("2026-03-21"))
            self.assertEqual(grid.iloc[1].AS_OF_DATE, pd.Timestamp("2026-02-28"))
        finally:
            count_loader.release_count_inputs(data)

    def test_missing_calendar_row_fails(self):
        self.rows[0]["SEQUENCE"].pop()
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, "incomplete calendar"):
            count_loader.load_count_inputs(self.tables.__getitem__, "SYNTHETIC", "v1")

    def test_fractional_count_fails_without_rounding(self):
        self.rows[0]["SEQUENCE"][0]["V"][0] = 1.5
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, "Invalid raw count"):
            count_loader.load_count_inputs(self.tables.__getitem__, "SYNTHETIC", "v1")


if __name__ == "__main__":
    unittest.main()
