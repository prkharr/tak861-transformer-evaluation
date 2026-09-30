"""Full synthetic notebook smoke test. The warehouse and Spark writer are test doubles.

This does not certify Snowflake SQL execution or real connector permissions.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import types
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'notebooks' / 'feature analysis' / '01_gather_and_select_features.ipynb'


def test_notebook_structure_and_embedded_sources():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    code = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            assert index > 0 and nb.cells[index - 1].cell_type == 'markdown'
            assert not cell.outputs and cell.execution_count is None
            compile(cell.source, f'cell_{index}', 'exec')
            code.append(cell.source)
    assert len(code) == 12
    joined = '\n'.join(code)
    for filename in ('source_catalog.py', 'warehouse.py', 'selection.py'):
        source = (Path(__file__).parent / filename).read_text(encoding='utf-8')
        assert source in joined
    assert '%run' not in joined and 'from experiments' not in joined
    assert 'Co-authored-by' not in joined


def test_synthetic_notebook_all_cells_and_three_exports():
    rng = np.random.default_rng(42)
    n = 240
    names = ['RX__SIGNAL', 'RX__NOISE', 'DX__A', 'DX__B', 'PX__A', 'PX__B']
    aliases = [f'F{i:04d}' for i in range(len(names))]
    labels = np.tile([1, 0, 0, 0, 0], n // 5)
    source = pd.DataFrame({'PATIENT_ID': [f'P{i:04d}' for i in range(n)],
                           'END_DT': '2024-11-21', 'RESP': labels})
    creation = json.dumps({'n_timesteps': 12, 'feature_order_sha256':
                           hashlib.sha256(json.dumps(names, ensure_ascii=False).encode()).hexdigest()})
    frozen = source.assign(SPLIT=['train'] * 180 + ['validation'] * 30 + ['test'] * 30,
                           SPLIT_CONFIG=creation)
    feature_map = pd.DataFrame({'FEATURE_INDEX': range(len(names)), 'FEATURE_NAME': names,
                                'FEATURE_COLUMN': aliases})
    train = source.iloc[:180].copy()
    counts = train.assign(N_ROWS=12, N_STEPS=12, MIN_STEP=0, MAX_STEP=11,
                          BAD_STEPS=0, BAD_LABELS=0, BAD_VALUES=0)
    for j, alias in enumerate(aliases):
        counts[alias + '_M12'] = rng.poisson(1 + (6 * train.RESP if j == 0 else 0), size=len(train))
    prefix = 'DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_DL_POC'
    inventory_rows = []
    schemas = {'FEATURE_MAP': list(feature_map.columns), 'SNAPSHOTS': list(source.columns),
               'PATIENT_SPLIT': list(frozen.columns),
               'TENSOR_MONTHLY': ['PATIENT_ID', 'END_DT', 'RESP', 'TIME_STEP'] + aliases}
    for suffix, cols in schemas.items():
        for index, column in enumerate(cols):
            inventory_rows.append({'TABLE_CATALOG': 'DSVC_TAKEDA_TA_PRIVATE', 'TABLE_SCHEMA': 'DS_ML',
                'TABLE_NAME': prefix.split('.')[-1] + '_' + suffix, 'COLUMN_NAME': column,
                'DATA_TYPE': 'NUMBER' if column in aliases else 'VARCHAR',
                'ORDINAL_POSITION': index + 1, 'IS_NULLABLE': 'YES'})
    inventory = pd.DataFrame(inventory_rows)
    queries, stored = [], {}

    def query_result(query):
        queries.append(query)
        if 'INFORMATION_SCHEMA.COLUMNS' in query:
            return inventory.copy() if "'DS_ML'" in query else inventory.iloc[:0].copy()
        if 'INFORMATION_SCHEMA.TABLES' in query:
            name = re.search(r"TABLE_NAME = '([^']+)'", query).group(1)
            return pd.DataFrame({'TABLE_NAME': [name] if any(t.endswith('.' + name) for t in stored) else []})
        if query.startswith('WITH '):
            assert "SPLIT = 'train'" in query and 'ROW_NUMBER() OVER (PARTITION BY PATIENT_ID' in query
            assert 'SUM(IFF(t.TIME_STEP < 12' in query
            return counts.copy()
        if '_FEATURE_MAP"' in query:
            return feature_map.copy()
        if '_SNAPSHOTS"' in query:
            return source.copy()
        if '_PATIENT_SPLIT"' in query:
            return frozen.copy()
        if '_MODEL_TYPE"' in query:
            return pd.DataFrame({'FEATURES': [json.dumps(['SIGNAL', 'AGE'])]})
        for table, frame in stored.items():
            if '.'.join('"' + p + '"' for p in table.split('.')) in query:
                return frame[['FEATURE_NAME', 'RANK', 'SELECTED', 'RUN_ID']].copy()
        raise AssertionError('Unexpected warehouse query: ' + query[:100])

    class Reader:
        def format(self, name):
            assert name == 'snowflake'
            return self
        def options(self, **opts):
            return self
        def option(self, key, value):
            assert key == 'query'
            self.query = value
            return self
        def load(self):
            return types.SimpleNamespace(toPandas=lambda: query_result(self.query))

    class Writer:
        def __init__(self, frame):
            self.frame = frame
        def format(self, name):
            return self
        def options(self, **opts):
            return self
        def option(self, key, value):
            assert key == 'dbtable'
            self.destination = value
            return self
        def mode(self, mode):
            assert mode == 'errorifexists'
            return self
        def save(self):
            assert self.destination not in stored
            stored[self.destination] = self.frame.copy()

    class FakeSpark:
        @property
        def read(self):
            return Reader()
        def createDataFrame(self, records, schema):
            frame = pd.DataFrame(records, columns=[f.name for f in schema.fields])
            return types.SimpleNamespace(write=Writer(frame))

    class StructType:
        def __init__(self, fields):
            self.fields = fields
    class StructField:
        def __init__(self, name, kind, nullable):
            self.name = name

    mock_types = types.ModuleType('pyspark.sql.types')
    mock_types.StructType, mock_types.StructField = StructType, StructField
    for name in ('StringType', 'DoubleType', 'LongType', 'BooleanType'):
        setattr(mock_types, name, type(name, (), {}))
    modules = {'pyspark': types.ModuleType('pyspark'), 'pyspark.sql': types.ModuleType('pyspark.sql'),
               'pyspark.sql.types': mock_types}
    shown = []
    scope = {'spark': FakeSpark(), 'sf_options': {}, 'display': lambda x: shown.append(x)}
    code = [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == 'code']
    with patch.dict(sys.modules, modules), patch.object(plt, 'show'), contextlib.redirect_stdout(io.StringIO()):
        for index, cell in enumerate(code):
            exec(compile(cell, f'notebook_cell_{index+1}', 'exec'), scope)
            if index == 0:
                # Small synthetic population/vocabulary and bounded trees only; runtime logic is unchanged.
                scope['EXPECTED_POPULATION'] = (n, n, int(labels.sum()))
                scope['EXPECTED_GROUPS'] = {'RX': 2, 'DX': 2, 'PX': 2}
                scope['SELECTION_CONFIG'].update(tree_estimators=12, tree_min_samples_leaf=4,
                    min_nonzero_patients=3, min_observed_patients=10,
                    filter_max_features=3, wrapper_shortlist=5, wrapper_max_features=2,
                    embedded_max_features=3, n_jobs=1)
    plt.close('all')
    assert len(stored) == 3 and len(scope['saved_results']) == 3
    assert all(len(t) == len(names) for t in stored.values())
    for frame in stored.values():
        assert 'PATIENT_ID' not in frame and 'RESP' not in frame
        assert frame.RUN_MANIFEST_JSON.nunique() == 1
        assert frame.SELECTED.any()
        manifest = json.loads(frame.RUN_MANIFEST_JSON.iloc[0])
        assert manifest['analysis_patients'] == 180
        assert len(manifest['diagnostics']) == 12
    assert all(r['STATUS'] == 'SAVED_AND_VERIFIED' for r in scope['saved_results'])
    assert set(scope['comparison'].STRATEGY) == {'all_eligible', 'filter', 'wrapper', 'embedded'}
    assert not any('_MODEL_DATA' in q or 'CHECKPOINT' in q for q in queries)
