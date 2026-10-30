import ast
import copy
import io
import json
import re
import tempfile
import types
import unittest
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.automl_sequence import inputs
from experiments.automl_sequence import test_workflow as fixture_module
from experiments.automl_sequence.test_notebooks import temporary_module_entries
from experiments.fresh_cohort import build_notebooks as builder
from experiments.fresh_cohort import build_production_reporting as delivery


class NotebookTests(unittest.TestCase):
    def test_presentation_source_hashes_and_protocol_defaults(self):
        manifest = delivery.build()
        for entry in manifest["notebooks"]:
            path = builder.base.REPO/entry["path"]
            book = json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(all(c["cell_type"] == "code" and c["outputs"] == [] and c["execution_count"] is None for c in book["cells"]))
            text = path.read_text(encoding='utf-8')
            self.assertIsNone(re.search(r"\b(codex|claude|github|git|client|brian)\b", text, re.I))
            first = "".join(book["cells"][0]["source"])
            self.assertTrue(all(v == "" for v in builder.assigned(first, "sf_options").values()))
            config = builder.assigned(first, "CONFIG")
            self.assertEqual(config["seeds"], [42, 142, 242])
            self.assertEqual(config["model"]["epochs"], 30)
            self.assertEqual(config["model"]["batch_size"], 256)
            self.assertIsNone(config["cohort_universe"]["expected_population"])
            self.assertEqual(config['cohort_plan']['months'], 48)
            for c in book["cells"]:
                ast.parse("".join(c["source"]))

    def test_all_five_execute_with_synthetic_sources(self):
        delivery.build()
        f = fixture_module.WorkflowTest(); f.setUp()
        metadata = f.data["metadata"].copy()
        store, displays, queries, populations = {}, [], [], {}
        expected = dict(snapshots=len(metadata), patients=len(metadata), positives=int(metadata.RESP.sum()),
            per_split={s:dict(snapshots=len(g), patients=len(g), positives=int(g.RESP.sum())) for s,g in metadata.groupby("SPLIT")})
        def save(table, values):
            values = {k:v.read_bytes() if isinstance(v, Path) else v for k,v in values.items()}
            if table in store and store[table] != values:
                raise ValueError("Immutable store changed")
            store[table] = copy.deepcopy(values)
        def read(table, names, *, disk_names=(), output_dir=None):
            self.assertEqual(set(store[table]), set(names))
            result = copy.deepcopy(store[table])
            for name in disk_names:
                target = Path(output_dir)/name; target.write_bytes(result[name]); result[name] = target
            return result
        def query(sql):
            queries.append(sql)
            if sql.endswith('WHERE 1=0'):
                return pd.DataFrame(columns=['PATIENT_ID','END_DT','RESP','FILL_DATE','IS_GENERIC','SERVICE_DATE'])
            if 'AS CANDIDATE_SNAPSHOTS' in sql:
                return pd.DataFrame([dict(CANDIDATE_SNAPSHOTS=len(metadata),BOUNDED_SNAPSHOTS=len(metadata),
                    OBSERVED_THROUGH='2026-12-31',UPPER_END='2026-09-30',LOWER_EXCLUSIVE='2022-09-30',
                    NULL_CLASSIFICATION_ROWS=0,NULL_ADVANCED_DATE_ROWS=0,
                    SNAPSHOTS=len(metadata),PATIENTS=metadata.PATIENT_ID.nunique(),POSITIVES=int(metadata.RESP.sum()))])
            if 'SELECT * FROM decisions ORDER BY' in sql:
                d = metadata[['PATIENT_ID','END_DT','RESP']].copy()
                d['ORIGINAL_RESP'] = d.RESP
                d['FIRST_EVENT'] = d.END_DT+pd.Timedelta(days=20)
                d.loc[d.RESP.eq(0),'FIRST_EVENT'] = pd.NaT
                return d.assign(PRIOR_INITIATION=0,OBSERVED_NEGATIVE=1,RETAIN=1)
            if "HASH_AGG" in sql:
                return pd.DataFrame({"ROWS": [1], "MAPPING_HASH": ["synthetic-v1"]})
            if sql.startswith("SELECT PATIENT_ID"):
                return metadata[["PATIENT_ID", "END_DT", "RESP"]].copy()
            rows = []
            if "AS DOC_ID" in sql:
                for doc, patient, date in re.findall(r"\((\d+),'([^']+)','(\d{4}-\d{2}-\d{2})'\)", sql):
                    common = dict(DOC_ID=int(doc),NULL_EVENT_ID_ROWS=0,MAX_EVENT_DATE=date)
                    rows.append(dict(common,ROW_KIND="COVERAGE",FAMILY=None,CODE=None,DESCRIPTION=None,DESCRIPTION_VARIANTS=0,EVENT_COUNT=1))
                    rows.append(dict(common,ROW_KIND="FEATURE",FAMILY="DX",CODE="DX_A",DESCRIPTION="type 1 narcolepsy without cataplexy",
                                     DESCRIPTION_VARIANTS=1,EVENT_COUNT=1+int(patient[1:])%3))
            else:
                for patient, date in re.findall(r"\('([^']+)','(\d{4}-\d{2}-\d{2})'\)", sql):
                    p = int(patient[1:])
                    rows.append(dict(PATIENT_ID=patient,AS_OF_DATE=date,ROW_KIND="COVERAGE",CATEGORY=None,VALUE=None,RECENT_VALUE=None,
                                     MAX_EVENT_DATE=date,AGE=30+p%40,AGE_CONFLICT=0))
                    for j in range(3):
                        rows.append(dict(PATIENT_ID=patient,AS_OF_DATE=date,ROW_KIND="FEATURE",CATEGORY="C"+str(j),VALUE=7+(p*(j+1))%13,
                                         RECENT_VALUE=3+(p+j)%3,MAX_EVENT_DATE=date,AGE=None,AGE_CONFLICT=None))
            return pd.DataFrame(rows)
        ipy = types.ModuleType("IPython")
        ipy.version_info = (8,0,0)
        ipy.get_ipython = lambda: None
        display_module = types.ModuleType("IPython.display")
        display_module.display = lambda *a, **k: None
        ipy.display = display_module
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), temporary_module_entries(
                {"IPython": ipy, "IPython.display": display_module}):
            for path in sorted(builder.DEST.glob("0*.ipynb")):
                book = json.loads(path.read_text(encoding='utf-8'))
                ns = dict(spark=object(), __name__="__synthetic_fresh_notebook__")
                config_source = "".join(book["cells"][0]["source"])
                config_source = builder.replace_assignment(config_source, "sf_options", {k:"SYNTHETIC" for k in
                    ("sfUrl","sfUser","sfPassword","sfRole","sfWarehouse","sfDatabase","sfSchema")})
                exec(config_source, ns)
                cfg = ns["CONFIG"]
                cfg["model"] = f.config["model"].copy(); cfg["selection"] = f.config["selection"].copy(); cfg["seeds"] = [42]
                for key in ("eligibility_reference", "first_initiation_reference", "negative_followup_reference", "effective_cutoff_reference", "source_snapshot_reference"):
                    cfg["cohort_evidence"][key] = "synthetic fixture only"
                ns["SCRATCH_DIRECTORY"] = directory
                exec("".join(book["cells"][1]["source"]), ns)
                self.assertIn("population", ns)
                self.assertTrue(callable(ns["population"].load_cohort_counts))
                ns["display"] = lambda value: displays.append(copy.deepcopy(value))
                wh, wf, pop, inp = [ns[k] for k in ("wh", "wf", "population", "inp")]
                wh.save_artifacts, wh.read_artifacts = save, read
                wh.table_exists = lambda table: table in store or table in populations
                wf.save_artifacts, wf.read_artifacts, wf.table_exists = save, read, store.__contains__
                wh.read_sf = lambda suffix: metadata.copy()
                ns["read_query"] = query
                ns['write_population'] = lambda table, values: populations.update({table.split('.')[-1]: values.copy()})
                aligned, audit = pop.align_universe(metadata, metadata)
                contract = copy.deepcopy(f.data["count_contract"])
                contract["date_policy"] = "month_end_history"
                _, grid = inp.make_requests(aligned, date_policy="month_end_history")
                contract["asof_grid_sha256"] = inp.frame_hash(grid)
                contract["metadata_sha256"] = inp.frame_hash(aligned)
                contract["cohort_universe_audit"] = audit
                pop.load_cohort_counts = lambda *a, **k: dict(metadata=aligned.copy(),X=f.data["X_count"],valid=f.data["count_mask"],
                                                             features=contract["feature_names"],contract=copy.deepcopy(contract))
                ns["fresh_lgbm"].SETTINGS.update(seeds=[42],n_estimators=10,stopping_rounds=3,n_jobs=1)
                for i, c in enumerate(book["cells"][2:],2):
                    source = "".join(c["source"]).replace("n_bootstrap=1000", "n_bootstrap=20")
                    exec(compile(source, str(path)+":"+str(i), "exec"), ns)
            ranking = ns["fresh_lgbm"].ranking(ns["lift_reports"]["validation"])
            self.assertEqual(set(ranking.Model), {"COUNT", "STATIC_FUSION", "NLP_FUSION", "LIGHTGBM_TRAIN_ONLY_STATIC"})
            self.assertFalse(ns["EVALUATE_TEST"])
            self.assertFalse(any(k.endswith("_FINAL_TEST") for k in store))
            self.assertTrue(any(k.endswith("_NLP_INPUTS") for k in store))
            self.assertEqual(ns['production_reports']['validation']['production']['status'],'BLOCKED')
            self.assertIsNone(ns['prodcmp'].ranking_table(ns['production_reports']['validation']).iloc[-1]['Rank'])
            ns['read_query'] = lambda sql: metadata[['PATIENT_ID','END_DT','RESP']].assign(SCORE=np.linspace(.1,.9,len(metadata)))
            for cell in delivery.cells()[0][1:]:
                source=''.join(cell['source']).replace('n_bootstrap=1000','n_bootstrap=20')
                exec(compile(source,'synthetic_production_reporting','exec'),ns)
            self.assertEqual(ns['production_reports']['validation']['production']['status'],'ALIGNED')
            self.assertEqual(ns['production_reports']['validation']['benchmark']['status'],'COMMON_POPULATION_DESCRIPTIVE')
            self.assertEqual(len(ns['prodcmp'].ranking_table(ns['production_reports']['validation'])),5)
        wf.release_inputs(f.data)
        for display in displays:
            if isinstance(display, pd.DataFrame):
                self.assertNotIn("PATIENT_ID", display.columns)


if __name__ == "__main__":
    unittest.main()
