import copy
import hashlib
import json
import types
import unittest

import numpy as np
import pandas as pd

from experiments.automl_sequence import build_lift_reporting as delivery, workflow
from experiments.rigorous_transformer import comparison


class ReportingDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"PATIENT_ID": [f"p{i}" for i in range(20)],
                                   "END_DT": pd.Timestamp("2026-02-28"),
                                   "RESP": [0, 1] * 10, "SPLIT": "validation"})
        self.source = self.frame.drop(columns="SPLIT").assign(SCORE=np.linspace(.01, .99, 20))
        self.loader = types.ModuleType("pinned_score_loader")
        self.loader_text = delivery.loader_source()
        exec(self.loader_text, self.loader.__dict__)

    def test_pinned_loader_aligns_and_rejects_invalid_sources(self):
        result = self.loader.load_original_v63_scores(
            lambda _: self.source.sample(frac=1, random_state=12), self.frame)
        np.testing.assert_array_equal(result["scores"], self.source.SCORE)
        for source in (self.source.iloc[:-1], pd.concat([self.source, self.source.iloc[:1]]),
                       self.source.assign(RESP=1-self.source.RESP)):
            with self.assertRaises(ValueError):
                self.loader.load_original_v63_scores(lambda _: source, self.frame)
        for allow, locked in ((False, False), (True, False), (False, True)):
            with self.assertRaises(ValueError):
                self.loader.load_original_v63_scores(lambda _: self.source, self.frame.assign(SPLIT="test"),
                    "test", allow_test=allow, selection_locked=locked)

    def runtime(self):
        store, source_reads, builds = {}, [], []
        def save(table, artifacts):
            self.assertNotIn(table, store)
            store[table] = artifacts
        warehouse = types.SimpleNamespace(table_exists=lambda table: table in store,
            read_artifacts=lambda table, names: {name: store[table][name] for name in names}, save_artifacts=save)
        scope = dict(np=np, pd=pd, json=json, wf=workflow, cmp=comparison, wh=warehouse,
                     RUN_PREFIX="SYNTHETIC_REPORT", lift_metadata=self.frame.copy(), aggregate=lambda *args: None)
        reporting_source = (delivery.core.HERE / "lift_reporting.py").read_text(encoding="utf-8")
        exec("".join(delivery.reporting_setup(self.loader_text, reporting_source, "reporting-identity")["source"]), scope)
        def read(table):
            source_reads.append(table)
            return self.source
        scope["read_existing_score_table"] = read
        def build(metadata, scores, baseline, split, **kwargs):
            builds.append(kwargs)
            return dict(split=split, marker="fixed synthetic report")
        scope["lift_reporting"].build_report = build
        scope["lift_reporting"].report_table = lambda report: pd.DataFrame()
        scope["lift_reporting"].interval_table = lambda report: pd.DataFrame()
        return scope, store, source_reads, builds

    def test_reporting_cache_reuses_frozen_scores_and_rejects_drift(self):
        scope, store, reads, builds = self.runtime()
        scores = {arm: self.source.SCORE.to_numpy().copy() for arm in ("COUNT", "STATIC_FUSION", "TEMPORAL_FUSION")}
        first = scope["existing_lift_report"]("validation", scores)
        self.source["SCORE"] = .5
        self.assertEqual(scope["existing_lift_report"]("validation", scores), first)
        self.assertEqual(len(reads), 1)
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0]["n_bootstrap"], 1000)
        altered = copy.deepcopy(scores)
        altered["COUNT"][0] += .001
        with self.assertRaisesRegex(ValueError, "different predictions"):
            scope["existing_lift_report"]("validation", altered)
        scope["lift_metadata"].loc[0, "RESP"] = 1
        with self.assertRaisesRegex(ValueError, "snapshot identity"):
            scope["existing_lift_report"]("validation", scores)

    def test_reporting_test_gate_precedes_any_reads(self):
        scope, store, reads, builds = self.runtime()
        for allow, locked in ((False, False), (True, False), (False, True)):
            with self.assertRaisesRegex(ValueError, "TEST reporting"):
                scope["existing_lift_report"]("test", {}, allow_test=allow, selection_locked=locked)
        self.assertFalse(store)
        self.assertFalse(reads)
        self.assertFalse(builds)

    def test_missing_scores_do_not_block_and_do_not_invent_baseline_rank(self):
        scope, store, reads, builds = self.runtime()
        self.source = self.source.iloc[:-1]
        scores = {arm: np.linspace(.1, .9, len(self.frame))
                  for arm in ("COUNT", "STATIC_FUSION", "TEMPORAL_FUSION")}
        from experiments.automl_sequence import lift_reporting
        scope['lift_reporting'].build_report = lift_reporting.build_report
        report = scope['try_existing_lift_report']('validation', scores)
        self.assertEqual(set(report['metrics']), set(scores))
        self.assertEqual(len(lift_reporting.ranking_table(report)), 3)
        self.assertEqual(report['interpretation'], 'TRANSFORMER_ONLY_EXISTING_SCORES_UNAVAILABLE')
        self.assertFalse(store)
        with self.assertRaisesRegex(ValueError, 'TEST reporting'):
            scope['try_existing_lift_report']('test', scores)

    def test_notebook_end_cells_and_core_identity(self):
        manifest = json.loads((delivery.core.HERE / "build_manifest.json").read_text())
        core_builder = delivery.core.HERE / "build_notebooks.py"
        self.assertEqual(manifest["implementation_sha256"], hashlib.sha256(
            (manifest["source_sha256"] + core_builder.read_text(encoding="utf-8")).encode()).hexdigest())
        self.assertEqual(manifest["existing_score_loader_sha256"], hashlib.sha256(self.loader_text.encode()).hexdigest())
        for item in manifest["notebooks"]:
            path = delivery.core.REPO / item["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            book = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(book["metadata"]["tak861"]["implementation_sha256"], manifest["implementation_sha256"])
            sources = ["".join(cell["source"]) for cell in book["cells"]]
            stage = path.name[:2]
            if stage == "03":
                self.assertIn('for evaluation_split in ("train", "validation"):', sources[-1])
                self.assertIn("existing_lift_report(evaluation_split, ensemble_scores)", sources[-1])
            elif stage == "04":
                self.assertIn('for evaluation_split in ("validation",):', sources[-1])
            elif stage == "05":
                self.assertTrue(any("ranking_table(lift_comparison)" in source for source in sources))
                self.assertTrue(any("allow_test=RUN_FINAL_TEST, selection_locked=True" in source for source in sources))
                self.assertIn("fresh_lgbm.ranking(fresh_result)", sources[-1])
                self.assertTrue(any("FRESH_LGBM_EVALUATE_TEST = False" in source for source in sources))
                self.assertTrue(any("RUN_FINAL_TEST = False" in source for source in sources))
            if stage in ("03", "04", "05"):
                retained = next(source for source in sources if "lift_metadata = data" in source)
                self.assertLess(retained.index("lift_metadata = data"), retained.index("wf.release_inputs(data)"))


if __name__ == "__main__":
    unittest.main()
