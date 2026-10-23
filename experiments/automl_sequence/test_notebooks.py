"""Execute generated notebook code against synthetic sources and private-store doubles.

These are notebook integration tests, not a Spark/warehouse execution certificate
and not clinical performance evidence. Delivered notebooks remain unexecuted.
"""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tokenize
import types
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout

import numpy as np
import pandas as pd

from experiments.automl_sequence import test_workflow as fixture_module
from experiments.automl_sequence import source_recovery
from experiments.automl_sequence import build_notebooks
from experiments.automl_sequence.test_count_loader import Expression, Frame


BOOKS = Path(__file__).resolve().parents[2] / "notebooks" / "AutoML Sequential Features"
FILENAMES = ["01_tensor_initialization_AutoML.ipynb", "02_patient_level_split_AutoML.ipynb",
             "03_transformer_training_AutoML.ipynb", "04_transformer_evaluation_AutoML.ipynb",
             "05_model_comparison_AutoML.ipynb"]


def notebook(name):
    return json.loads((BOOKS / name).read_text(encoding="utf-8-sig"))


def notebook_code_parts(book):
    """Locate setup cells by their contents, independently of display-cell count."""
    cells = [(i, "".join(cell["source"])) for i, cell in enumerate(book["cells"])
             if cell["cell_type"] == "code"]
    config = next((i, source) for i, source in cells
                  if "sf_options =" in source and "CONFIG =" in source)
    helper = next((i, source) for i, source in cells if "EMBEDDED_SOURCES =" in source)
    remaining = [(i, source) for i, source in cells if i not in {config[0], helper[0]}]
    return config, helper, remaining


def comment_tokens(source):
    return [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT]


@contextmanager
def temporary_module_entries(entries):
    """Restore only test doubles; keep newly imported native modules cached.

    patch.dict(sys.modules) would remove torch after importing it inside the
    notebook and make a later native-extension reload fail in a combined suite.
    """
    missing = object()
    original = {name: sys.modules.get(name, missing) for name in entries}
    sys.modules.update(entries)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class NotebookDeliveryTests(unittest.TestCase):
    def test_blank_connection_options_and_separate_extraction_cell(self):
        config = notebook_code_parts(notebook(FILENAMES[0]))[0][1]
        declaration = next(node for node in ast.parse(config).body if isinstance(node, ast.Assign)
                           and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'sf_options')
        self.assertTrue(all(value == '' for value in ast.literal_eval(declaration.value).values()))
        self.assertNotIn('dbutils.secrets', config)
        with self.assertRaisesRegex(ValueError, 'Fill the blank sf_options'):
            exec(config.replace('"sfPassword": ""', '"sfPassword": "SYNTHETIC"'), dict(spark=object()))
        sources = [''.join(cell['source']) for cell in notebook(FILENAMES[0])['cells']]
        count_cell = next(i for i, text in enumerate(sources) if 'counts = population.load_client_counts' in text)
        raw_cell = next(i for i, text in enumerate(sources) if 'AUTOML_SETUP = raw.build_raw_automl_inputs' in text)
        self.assertLess(count_cell, raw_cell)
        self.assertNotIn('population.load_client_counts', sources[raw_cell])

    def test_notebook_source_strips_only_comments_and_preserves_ast(self):
        source = ('"""Runtime documentation."""\n'
                  '# A development comment\n'
                  'value = "# remains a string"  # an inline comment\n'
                  'sql = "SELECT 1 -- SQL payload stays unchanged"\n')
        cleaned = build_notebooks.notebook_source(source)
        self.assertEqual(comment_tokens(cleaned), [])
        self.assertEqual(ast.dump(ast.parse(cleaned), include_attributes=False),
                         ast.dump(ast.parse(source), include_attributes=False))
        self.assertIn('# remains a string', cleaned)
        self.assertIn('SQL payload stays unchanged', cleaned)

    def test_default_source_uses_embedded_raw_extractor_without_legacy_procedure_calls(self):
        source = next("".join(cell["source"]) for cell in notebook(FILENAMES[0])["cells"]
                      if cell["cell_type"] == "code" and "AUTOML_SETUP =" in "".join(cell["source"]))
        ns = dict(RUN_PREFIX="SYNTHETIC_RAW", wh=types.SimpleNamespace(table_exists=lambda _: False))
        exec(compile(source, "01-source-configuration", "exec"), ns)
        self.assertIsNone(ns["AUTOML_SETUP"])
        self.assertFalse(ns["REUSE_SAVED_INPUTS"])
        all_code = "\n".join("".join(c["source"]) for c in notebook(FILENAMES[0])["cells"]
                             if c["cell_type"] == "code" and "EMBEDDED_SOURCES =" not in "".join(c["source"]))
        self.assertIn("raw.build_raw_automl_inputs", all_code)
        self.assertNotIn("recover_procedure_sources(", all_code)

    def test_embedded_sources_configuration_and_clean_outputs_match(self):
        source_maps, hashes, configs = [], [], []
        for filename in FILENAMES:
            book = notebook(filename)
            self.assertEqual(book["nbformat"], 4)
            self.assertTrue(all(cell["cell_type"] == "code" for cell in book["cells"]), filename)
            config_cell, helper_cell, remaining = notebook_code_parts(book)
            self.assertEqual((config_cell[0], helper_cell[0]), (0, 1), filename)
            configs.append(config_cell[1])
            helper = helper_cell[1]
            assignments = {node.targets[0].id: ast.literal_eval(node.value)
                           for node in ast.parse(helper).body
                           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                           and node.targets[0].id in {"EMBEDDED_SOURCES", "EMBEDDED_SOURCE_SHA256", "IMPLEMENTATION_SHA256"}}
            sources = assignments["EMBEDDED_SOURCES"]
            self.assertEqual(set(sources), {"inputs", "count_loader", "modeling", "workflow", "reliance", "source_recovery", "raw_automl", "fresh_counts", "client_universe", "comparison", "warehouse"})
            repo = BOOKS.parent.parent
            paths = {name: repo / "experiments" / "automl_sequence" / (name + ".py")
                     for name in ("inputs", "count_loader", "modeling", "workflow", "reliance", "source_recovery", "raw_automl", "fresh_counts", "client_universe")}
            paths.update(comparison=repo / "experiments" / "rigorous_transformer" / "comparison.py",
                         warehouse=repo / "experiments" / "temporal_selection" / "input_helpers.py")
            for name, path in paths.items():
                original = path.read_text(encoding="utf-8-sig")
                self.assertEqual(sources[name], build_notebooks.notebook_source(original),
                                 "Stale embedded source; rebuild all notebooks: " + name)
                self.assertEqual(ast.dump(ast.parse(sources[name]), include_attributes=False),
                                 ast.dump(ast.parse(original), include_attributes=False),
                                 "Source cleaning changed executable structure: " + name)
            self.assertEqual(hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                             assignments["EMBEDDED_SOURCE_SHA256"])
            source_maps.append(sources); hashes.append(assignments["IMPLEMENTATION_SHA256"])
            for name, source in sources.items():
                self.assertEqual(comment_tokens(source), [], name)
                parsed = ast.parse(source)
                for node in ast.walk(parsed):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith("experiments."), name)
                    elif isinstance(node, ast.Import):
                        self.assertFalse(any(alias.name.startswith("experiments.") for alias in node.names), name)
            for cell in book["cells"]:
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                self.assertEqual(comment_tokens(source), [], filename)
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        # Word boundaries allow required technical identifiers
                        # such as client_universe; SQLite commit is unrelated.
                        self.assertIsNone(re.search(r"\b(?:codex|claude|git|client|brian)\b", node.value, re.I),
                                          filename + ": development narrative in a text constant")
        self.assertTrue(all(source == source_maps[0] for source in source_maps))
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(len(set(configs)), 1)
        nb5code = "\n".join("".join(cell["source"]) for cell in notebook(FILENAMES[-1])["cells"] if cell["cell_type"] == "code")
        self.assertIn("RUN_FINAL_TEST = False", nb5code)

    def test_all_five_notebooks_execute_with_synthetic_inputs(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        fixture = fixture_module.WorkflowTest(methodName="test_three_arm_end_to_end_freeze_restore_and_test_gate")
        fixture.setUp()
        meta = fixture.data["metadata"].copy()
        names = [('DX','PX','RX')[i%3] + '__COUNT_' + str(i) for i in range(1028)]
        aliases = [f"F{i:04d}" for i in range(1028)]
        creation = dict(n_timesteps=12, feature_order_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest())
        meta["SPLIT_CONFIG"] = json.dumps(creation)
        raw_counts = fixture.data["X_count"]
        rows = [dict(PATIENT_ID=row.PATIENT_ID, END_DT=row.END_DT, RESP=row.RESP,
                     SEQUENCE=[{"T": step, "V": raw_counts[i, step].astype(float).tolist()} for step in range(12)])
                for i, row in enumerate(meta.itertuples())]
        tables = {
            "FEATURE_MAP": Frame(frame=pd.DataFrame([dict(FEATURE_INDEX=i, FEATURE_NAME=name, FEATURE_COLUMN=aliases[i]) for i, name in enumerate(names)]), rows=[dict(FEATURE_INDEX=i, FEATURE_NAME=name, FEATURE_COLUMN=aliases[i]) for i, name in enumerate(names)]),
            "PATIENT_SPLIT": Frame(meta),
            "SNAPSHOTS": Frame(meta[["PATIENT_ID", "END_DT", "RESP"]]),
            "TENSOR_MONTHLY": Frame(rows=rows, columns=["PATIENT_ID", "END_DT", "RESP", "TIME_STEP"] + aliases),
        }
        functions = types.ModuleType("pyspark.sql.functions")
        for name in ("col", "array", "collect_list", "struct"):
            setattr(functions, name, lambda *args: Expression())
        sql = types.ModuleType("pyspark.sql"); sql.functions = functions
        package = types.ModuleType("pyspark"); package.sql = sql
        fake_modules = {"pyspark": package, "pyspark.sql": sql, "pyspark.sql.functions": functions}
        # Display transport is a notebook-host concern; plots use real Agg.
        ipython = types.ModuleType("IPython")
        ipython.get_ipython = lambda: None
        ipython.version_info = (8, 0, 0)
        display_module = types.ModuleType("IPython.display")
        display_module.display = lambda *args, **kwargs: None
        ipython.display = display_module
        fake_modules.update({"IPython": ipython, "IPython.display": display_module})
        store, displays, queried, extraction_calls = {}, [], [], []
        def save(table, artifacts):
            artifacts = {k: v.read_bytes() if isinstance(v, Path) else v for k, v in artifacts.items()}
            if table in store and store[table] != artifacts:
                raise ValueError("Immutable artifact replacement")
            store[table] = copy.deepcopy(artifacts)
        def read(table, artifact_names, *, disk_names=(), output_dir=None):
            self.assertEqual(set(store[table]), set(artifact_names))
            result = copy.deepcopy(store[table])
            for name in disk_names:
                target = Path(output_dir) / name
                target.write_bytes(result[name]); result[name] = target
            return result
        def read_sf(suffix):
            queried.append(suffix)
            return tables[suffix]
        def extractor(keys):
            extraction_calls.append(keys.copy())
            panel = keys.copy()
            p = panel.PATIENT_ID.str[1:].astype(int).to_numpy()
            month = panel.AS_OF_DATE.dt.month.to_numpy()
            for j in range(5):
                panel[f"automl{j}"] = np.sin(p * (j+1) + month/(j+1)) + (p % 2)*.2
            panel["IS_OBSERVED"] = True
            panel["MAX_EVENT_DATE"] = panel.AS_OF_DATE
            panel["MAX_AVAILABLE_DATE"] = panel.AS_OF_DATE
            return panel
        setup = dict(feature_specs=fixture.specs, contract=fixture.panel_manifest["contract"], extractor=extractor)
        config = copy.deepcopy(fixture.config)
        config["date_policy"] = "month_end_history"
        config["raw_automl"] = dict(date_policy="month_end_history", chunk_size=5000, max_features=512)
        config['client_universe'] = dict(table='DB.SC.TAK861_TX_READY_PATIENT_UNIVERSE_DL',split_seed=42)
        config['client_universe']['expected_population'] = dict(snapshots=96, patients=96, positives=48,
            per_split={s: dict(snapshots=len(p), patients=p.PATIENT_ID.nunique(), positives=int(p.RESP.sum()))
                       for s,p in meta.groupby('SPLIT')})
        raw_calls = []
        from experiments.automl_sequence import inputs
        req,_=inputs.make_requests(meta,date_policy='month_end_history')
        long_rows=[]
        for patient,date in req.itertuples(index=False,name=None):
            p=int(patient[1:]); month=date.month
            long_rows.append(dict(PATIENT_ID=patient,AS_OF_DATE=date,ROW_KIND='COVERAGE',CATEGORY=None,
                VALUE=None,RECENT_VALUE=None,MAX_EVENT_DATE=date,AGE=30+p%40,AGE_CONFLICT=0))
            for j in range(3):
                value=7+(p*(j+1)+month)%13
                long_rows.append(dict(PATIENT_ID=patient,AS_OF_DATE=date,ROW_KIND='FEATURE',CATEGORY='C'+str(j),
                    VALUE=value,RECENT_VALUE=3+(p+j)%3,MAX_EVENT_DATE=date,AGE=None,AGE_CONFLICT=None))
        def read_query(query):
            if query.startswith('SELECT PATIENT_ID,END_DT,RESP FROM '):
                return meta[['PATIENT_ID','END_DT','RESP']]
            raw_calls.append(query)
            return pd.DataFrame(long_rows)
        output = io.StringIO()
        last_namespace = None
        with tempfile.TemporaryDirectory() as scratch, temporary_module_entries(fake_modules), redirect_stdout(output):
            for filename in FILENAMES:
                config_cell, helper_cell, remaining = notebook_code_parts(notebook(filename))
                namespace = {"__name__": "__synthetic_notebook__", "spark": object(),
                             "sf_options_dl_poc": {"synthetic": True}}
                config_source = config_cell[1]
                with self.assertRaisesRegex(ValueError, "Enter your PAT"):
                    exec(compile(config_source, filename+":blank_pat", "exec"), namespace)
                config_source = config_source.replace('"sfPassword": ""', '"sfPassword": "SYNTHETIC_NOT_A_CREDENTIAL"')
                for field in ('sfUrl', 'sfUser', 'sfRole', 'sfWarehouse', 'sfDatabase', 'sfSchema'):
                    config_source = config_source.replace('"' + field + '": ""', '"' + field + '": "SYNTHETIC"')
                exec(compile(config_source, filename+":config", "exec"), namespace)
                self.assertEqual(namespace['sf_options_dl_poc']['sfPassword'], 'SYNTHETIC_NOT_A_CREDENTIAL')
                self.assertNotIn('synthetic', namespace['sf_options_dl_poc'])
                namespace["CONFIG"] = copy.deepcopy(config)
                namespace["SCRATCH_DIRECTORY"] = scratch
                namespace["RUN_ID"] = "SYNTHETICNOTEBOOKS"
                exec(compile(helper_cell[1], filename+":embedded", "exec"), namespace)
                namespace["display"] = lambda obj: displays.append(copy.deepcopy(obj))
                wh, wf = namespace["wh"], namespace["wf"]
                wh.read_sf = read_sf
                wh.save_artifacts, wh.read_artifacts, wh.table_exists = save, read, store.__contains__
                wf.save_artifacts, wf.read_artifacts, wf.table_exists = save, read, store.__contains__
                namespace['read_query'] = read_query
                # Double only live count aggregation, not client alignment or the raw AutoML builder.
                fc=namespace['modules']['fresh_counts']
                fc.resolve_source_contract=lambda *args,**kwargs: {'synthetic':True}
                fc.build_source_frames=lambda *args,**kwargs: dict(tables,manifest={'synthetic_spark_aggregate':True},
                    cached_aggregates=types.SimpleNamespace(unpersist=lambda **kw:None))
                for index, source in remaining:
                    if 'def existing_lift_report(' in source:
                        marker = ('for evaluation_split in ' if filename.startswith(('03', '04'))
                                  else 'evaluation_split = report["split"]')
                        setup_source, body_source = source.split(marker, 1)
                        exec(compile(setup_source, filename+':lift_setup', 'exec'), namespace)
                        source_scores = meta[['PATIENT_ID', 'END_DT', 'RESP']].copy()
                        source_scores['SCORE'] = .1 + .7 * source_scores.RESP
                        if filename.startswith('03'):
                            source_scores = source_scores.loc[~meta.SPLIT.eq('train')]
                        namespace['read_existing_score_table'] = lambda table: source_scores.iloc[::-1].copy()
                        source = marker + body_source
                    exec(compile(source, filename+f":cell{index}", "exec"), namespace)
                last_namespace = namespace
        self.assertEqual(queried, ["PATIENT_SPLIT", "FEATURE_MAP"])
        self.assertEqual(len(raw_calls), 1)
        self.assertIn('FROM VALUES',raw_calls[0])
        self.assertNotIn('RESP',raw_calls[0])
        self.assertFalse(last_namespace["RUN_FINAL_TEST"])
        self.assertEqual(last_namespace["report"]["split"], "validation")
        self.assertEqual(set(last_namespace["report"]["metrics"]), {"COUNT", "STATIC_FUSION", "TEMPORAL_FUSION"})
        self.assertTrue(any(name.endswith("_VALIDATION_COMPARISON") for name in store))
        self.assertTrue(any(name.endswith("_RELIANCE") for name in store))
        self.assertFalse(any(name.endswith("_FINAL_TEST") for name in store))
        if 'lift_comparison' in last_namespace:
            self.assertIn('Existing scores unavailable for TRAIN', output.getvalue())
            self.assertEqual(last_namespace['lift_comparison']['split'], 'validation')
            ranking = last_namespace['lift_reporting'].ranking_table(last_namespace['lift_comparison'])
            self.assertEqual(len(ranking), 4)
            self.assertEqual(list(ranking.columns), ['Rank', 'Model', 'Lift@10%'])
            self.assertFalse(any('_EXISTING_V63_TEST' in name for name in store))
        self.assertIn("No independent OOT conclusion", output.getvalue())
        self.assertGreater(len(displays), 10)
        # Displayed frames are aggregates or feature ledgers, never patient rows.
        for obj in displays:
            if isinstance(obj, pd.DataFrame):
                self.assertNotIn("PATIENT_ID", obj.columns)


if __name__ == "__main__":
    unittest.main()
