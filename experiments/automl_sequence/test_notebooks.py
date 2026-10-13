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
import sys
import types
import unittest
from contextlib import contextmanager, redirect_stdout

import numpy as np
import pandas as pd

from experiments.automl_sequence import test_workflow as fixture_module
from experiments.automl_sequence import source_recovery
from experiments.automl_sequence.test_count_loader import Expression, Frame


BOOKS = Path(__file__).resolve().parents[2] / "notebooks" / "V63 Snapshot Padding"
FILENAMES = ["01_tensor_initialization.ipynb", "02_patient_level_split.ipynb",
             "03_transformer_training.ipynb", "04_transformer_evaluation.ipynb",
             "05_model_comparison.ipynb"]


def notebook(name):
    return json.loads((BOOKS / name).read_text(encoding="utf-8-sig"))


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
            configs.append("".join(book["cells"][1]["source"]))
            helper = "".join(book["cells"][2]["source"])
            assignments = {node.targets[0].id: ast.literal_eval(node.value)
                           for node in ast.parse(helper).body
                           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                           and node.targets[0].id in {"EMBEDDED_SOURCES", "EMBEDDED_SOURCE_SHA256", "IMPLEMENTATION_SHA256"}}
            sources = assignments["EMBEDDED_SOURCES"]
            self.assertEqual(set(sources), {"inputs", "count_loader", "modeling", "workflow", "reliance", "source_recovery", "raw_automl", "comparison", "warehouse"})
            repo = BOOKS.parent.parent
            paths = {name: repo / "experiments" / "automl_sequence" / (name + ".py")
                     for name in ("inputs", "count_loader", "modeling", "workflow", "reliance", "source_recovery", "raw_automl")}
            paths.update(comparison=repo / "experiments" / "rigorous_transformer" / "comparison.py",
                         warehouse=repo / "experiments" / "temporal_selection" / "input_helpers.py")
            for name, path in paths.items():
                self.assertEqual(sources[name], path.read_text(encoding="utf-8-sig"),
                                 "Stale embedded source; rebuild all notebooks: " + name)
            self.assertEqual(hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                             assignments["EMBEDDED_SOURCE_SHA256"])
            source_maps.append(sources); hashes.append(assignments["IMPLEMENTATION_SHA256"])
            for name, source in sources.items():
                parsed = ast.parse(source)
                for node in ast.walk(parsed):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith("experiments."), name)
                    elif isinstance(node, ast.Import):
                        self.assertFalse(any(alias.name.startswith("experiments.") for alias in node.names), name)
            for cell in book["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    ast.parse("".join(cell["source"]))
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
        names = fixture.data["count_contract"]["feature_names"]
        aliases = [f"F{i:04d}" for i in range(1028)]
        creation = dict(n_timesteps=12, feature_order_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest())
        meta["SPLIT_CONFIG"] = json.dumps(creation)
        raw_counts = fixture.data["X_count"]
        rows = [dict(PATIENT_ID=row.PATIENT_ID, END_DT=row.END_DT, RESP=row.RESP,
                     SEQUENCE=[{"T": step, "V": raw_counts[i, step].astype(float).tolist()} for step in range(12)])
                for i, row in enumerate(meta.itertuples())]
        tables = {
            "FEATURE_MAP": Frame(rows=[dict(FEATURE_INDEX=i, FEATURE_NAME=name, FEATURE_COLUMN=aliases[i]) for i, name in enumerate(names)]),
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
            if table in store and store[table] != artifacts:
                raise ValueError("Immutable artifact replacement")
            store[table] = copy.deepcopy(artifacts)
        def read(table, artifact_names):
            self.assertEqual(set(store[table]), set(artifact_names))
            return copy.deepcopy(store[table])
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
        raw_calls = []
        def raw_builder(read_query, metadata, source_config):
            raw_calls.append((metadata.copy(), source_config.copy()))
            return dict(setup, receipt={"source": "synthetic raw adapter fixture"})
        output = io.StringIO()
        last_namespace = None
        with temporary_module_entries(fake_modules), redirect_stdout(output):
            for filename in FILENAMES:
                cells = notebook(filename)["cells"]
                namespace = {"__name__": "__synthetic_notebook__", "spark": object(),
                             "sf_options_dl_poc": {"synthetic": True}}
                exec(compile("".join(cells[1]["source"]), filename+":config", "exec"), namespace)
                namespace["CONFIG"] = copy.deepcopy(config)
                namespace["RUN_ID"] = "SYNTHETICNOTEBOOKS"
                exec(compile("".join(cells[2]["source"]), filename+":embedded", "exec"), namespace)
                namespace["display"] = lambda obj: displays.append(copy.deepcopy(obj))
                wh, wf = namespace["wh"], namespace["wf"]
                wh.read_sf = read_sf
                wh.save_artifacts, wh.read_artifacts, wh.table_exists = save, read, store.__contains__
                wf.save_artifacts, wf.read_artifacts, wf.table_exists = save, read, store.__contains__
                namespace["raw"].build_raw_automl_inputs = raw_builder
                for index, cell in enumerate(cells[3:], 3):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), filename+f":cell{index}", "exec"), namespace)
                last_namespace = namespace
        self.assertEqual(queried, ["FEATURE_MAP", "PATIENT_SPLIT", "SNAPSHOTS", "TENSOR_MONTHLY"])
        self.assertEqual(len(extraction_calls), 1)
        self.assertEqual(len(raw_calls), 1)
        self.assertEqual(set(extraction_calls[0]), {"PATIENT_ID", "AS_OF_DATE"})
        self.assertFalse(extraction_calls[0].duplicated(["PATIENT_ID", "AS_OF_DATE"]).any())
        self.assertFalse(last_namespace["RUN_FINAL_TEST"])
        self.assertEqual(last_namespace["report"]["split"], "validation")
        self.assertEqual(set(last_namespace["report"]["metrics"]), {"COUNT", "STATIC_FUSION", "TEMPORAL_FUSION"})
        self.assertTrue(any(name.endswith("_VALIDATION_COMPARISON") for name in store))
        self.assertTrue(any(name.endswith("_RELIANCE") for name in store))
        self.assertFalse(any(name.endswith("_FINAL_TEST") for name in store))
        self.assertIn("No independent OOT conclusion", output.getvalue())
        self.assertGreater(len(displays), 10)
        # Displayed frames are aggregates or feature ledgers, never patient rows.
        for obj in displays:
            if isinstance(obj, pd.DataFrame):
                self.assertNotIn("PATIENT_ID", obj.columns)


if __name__ == "__main__":
    unittest.main()
