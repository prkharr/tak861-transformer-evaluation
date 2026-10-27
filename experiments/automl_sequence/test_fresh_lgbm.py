import copy
import hashlib
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.automl_sequence import fresh_lgbm as f, modeling as m, workflow as w, inputs
from experiments.rigorous_transformer import comparison as c


class FreshBaselineTests(unittest.TestCase):
    def setUp(self):
        w.inputs = inputs
        w.modeling = m
        rng = np.random.default_rng(4)
        x = rng.normal(size=(120, 1, 4)).astype(np.float32)
        y = (x[:, 0, 0] > 0).astype(int)
        meta = pd.DataFrame(dict(PATIENT_ID=[str(i) for i in range(120)],
            END_DT=pd.Timestamp("2025-01-31"), RESP=y,
            SPLIT=["train"] * 80 + ["validation"] * 20 + ["test"] * 20))
        state = m._fit_transform(x[:80], x[:80], np.ones((80, 1), bool),
                                 ["a", "b", "c", "d"], ["continuous"] * 4)
        self.prep = {"automl": dict(transform=state, selected_indices=[0, 1, 2], selected_names=["a", "b", "c"],
                                     selection_used_test=False, selection_used_external_validation=False)}
        self.data = dict(metadata=meta, y=y, X_summary=x, summary_mask=np.ones((120, 1), bool),
                         indices={s: np.flatnonzero(meta.SPLIT.eq(s)) for s in ("train", "validation", "test")},
                         manifest={"synthetic": True})
        self.protocol = {"arms": {a: {} for a in ("COUNT", "STATIC_FUSION", "TEMPORAL_FUSION")}}
        self.lock = dict(protocol_hash=w.digest(self.protocol), preparation_hash=w.digest(self.prep),
                         input_manifest_hash=w.digest(self.data["manifest"]))
        self.store = {"S_LOCK": {"lock.json": w.json_bytes(self.lock)}}
        def save(table, blob):
            self.assertNotIn(table, self.store)
            self.store[table] = copy.deepcopy(blob)
        def read(table, names):
            self.assertEqual(set(self.store[table]), set(names))
            return {k: self.store[table][k] for k in names}
        self.wh = types.SimpleNamespace(table_exists=self.store.__contains__, save_artifacts=save,
             read_artifacts=read)
        self.cfg = dict(f.SETTINGS, n_estimators=20, stopping_rounds=4, min_child_samples=3, n_jobs=1)

    def fit(self):
        return f.fit_or_resume(self.data, self.prep, self.protocol, self.lock,
                              w, m, self.wh, "S", "synthetic", self.cfg)

    def test_real_fit_resume_and_no_test_input(self):
        with patch.object(f, "design", wraps=f.design) as design:
            result = self.fit()
            self.assertEqual([call.args[2] for call in design.call_args_list], ["train", "validation"])
        with patch.object(lgb.LGBMClassifier, "fit", side_effect=AssertionError("must resume")):
            resumed = self.fit()
        np.testing.assert_array_equal(result["validation_scores"], resumed["validation_scores"])
        self.assertEqual(len(result["models"]), 3)
        self.assertEqual(result["identity"]["features"], ["a", "b", "c", "AUTOML_CURRENT_AVAILABLE"])

    def test_lift_report_gate_and_resume(self):
        baseline = self.fit()
        scores = {a: baseline["validation_scores"].copy() for a in self.protocol["arms"]}
        with self.assertRaisesRegex(ValueError, "authorization"):
            f.compare(self.data, self.prep, baseline, scores, "test", w, m, c, self.wh)
        report = f.compare(self.data, self.prep, baseline, scores, "validation", w, m, c, self.wh, n_bootstrap=100)
        self.assertEqual(len(report["intervals"]), 9)
        self.assertTrue(all(row["Delta@10%"] == 0 for row in report["rows"]))
        self.assertTrue((f.ranking(report).Rank == 1).all())
        with patch.object(w, "paired_lift_bootstrap", side_effect=AssertionError("must resume")):
            self.assertEqual(report, f.compare(self.data, self.prep, baseline, scores, "validation", w, m, c, self.wh, n_bootstrap=100))
        xt = f.design(self.data, self.prep, "test", m)
        test_scores = {a: np.mean([b.predict(xt) for b in baseline["models"]], axis=0) for a in scores}
        result = f.compare(self.data, self.prep, baseline, test_scores, "test", w, m, c, self.wh,
                           allow_test=True, n_bootstrap=100)
        self.assertIn("Retrospective", result["scope"])

    def test_drift_and_corruption_stop(self):
        self.fit()
        self.cfg["num_leaves"] = 7
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.fit()
        self.cfg["num_leaves"] = f.SETTINGS["num_leaves"]
        self.store["S_FRESH_LGBM_STATIC_V1_SEED_42"]["model.txt"] += b"broken"
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.fit()

    def test_overlap_and_labels_stop(self):
        self.data["metadata"].loc[90, "PATIENT_ID"] = "0"
        with self.assertRaises(ValueError):
            self.fit()
        self.data["metadata"].loc[90, "PATIENT_ID"] = "90"
        self.data["metadata"].loc[90, "RESP"] = 1 - self.data["y"][90]
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            self.fit()

    def test_actual_notebook_extension_executes(self):
        path = Path(__file__).resolve().parents[2] / "notebooks/AutoML Sequential Features/05_model_comparison_AutoML.ipynb"
        book = json.loads(path.read_text(encoding="utf-8"))
        extension = ["".join(cell["source"]) for cell in book["cells"] if "FRESH_LGBM_EXTENSION" in "".join(cell["source"])]
        self.assertEqual(len(extension), 4)
        self.store["S_PREPARATION"] = {"preparation.json": w.json_bytes(self.prep), "protocol.json": w.json_bytes(self.protocol)}
        scores = {a + ".npy": w.array_bytes(np.linspace(.05, .95, 20)) for a in self.protocol["arms"]}
        self.store["S_VALIDATION_COMPARISON"] = dict(scores, **{"report.json": w.json_bytes(dict(lock_hash=w.digest(self.lock), split="validation"))})
        scope = dict(types=types, hashlib=hashlib, json=json, np=np, pd=pd,
                     wf=w, mdl=m, cmp=c, wh=self.wh, RUN_PREFIX="S", load_stage=lambda: self.data,
                     display=lambda obj: None)
        exec(extension[0], scope)
        scope["fresh_lgbm"].SETTINGS = self.cfg
        for cell in extension[1:]:
            exec(cell, scope)
        self.assertEqual(set(scope["fresh_reports"]), {"validation"})
        self.assertNotIn("fresh_data", scope)


if __name__ == "__main__":
    unittest.main()
