"""Synthetic tests for frozen production comparison: exact alignment, blocking, gating and lift-only status."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from experiments.automl_sequence import workflow
from experiments.fresh_cohort import production_comparison as pc
from experiments.rigorous_transformer import comparison

TABLE = "DB.SC.PRODUCTION_SCORES_CURRENT"


def evidence(**overrides):
    base = dict(production_model_fingerprint="sha256:fixture", production_training_window_end="2024-12-31",
                production_training_patient_exclusion_reference="receipt://fixture",
                target_definition_match_reference="receipt://fixture", historical_input_vintage_reference="receipt://fixture",
                patient_disjoint_from_production_training=True, same_target_definition=True, historical_inputs_frozen=True)
    base.update(overrides)
    return base


class ProductionComparisonTests(unittest.TestCase):
    def setUp(self):
        n = 40
        self.meta = pd.DataFrame({"PATIENT_ID": [f"p{i // 2}" for i in range(n)],
                                  "END_DT": pd.to_datetime(["2026-01-31", "2026-02-28"] * (n // 2)),
                                  "RESP": [0, 1] * (n // 2), "SPLIT": "validation"})
        rng = np.random.default_rng(3)
        self.models = {"COUNT": np.clip(self.meta.RESP * .5 + rng.random(n) * .4, 0, 1),
                       "STATIC_FUSION": np.clip(self.meta.RESP * .6 + rng.random(n) * .3, 0, 1),
                       "NLP_FUSION": np.clip(self.meta.RESP * .7 + rng.random(n) * .3, 0, 1),
                       "LIGHTGBM_TRAIN_ONLY_STATIC": np.clip(self.meta.RESP * .4 + rng.random(n) * .5, 0, 1)}
        self.scored = self.meta[["PATIENT_ID", "END_DT", "RESP"]].assign(SCORE=np.linspace(.05, .95, n))
        self.queries = []

    def reader(self, frame):
        def read(sql):
            self.queries.append(sql)
            self.assertTrue(sql.upper().startswith("SELECT"))
            return frame.copy()
        return read

    def config(self, table=TABLE, label="RESP", **ev):
        return dict(score_table=table, score_column="SCORE", label_column=label, evidence=evidence(**ev))

    def run_report(self, frame=None, config=None, **kwargs):
        return pc.run(self.reader(self.scored if frame is None else frame), self.meta, config or self.config(),
                      self.models, "validation", comparison=comparison, workflow=workflow, n_bootstrap=20, **kwargs)

    def test_blocked_without_table_reports_models_only_and_issues_no_query(self):
        report = self.run_report(config=dict(score_table="", evidence={}))
        self.assertEqual(report["production"]["status"], "BLOCKED")
        self.assertIn("score_table", report["production"]["required"])
        self.assertTrue(set(pc.REQUIRED_EVIDENCE_FIELDS) <= set(report["production"]["required"]))
        self.assertEqual(set(report["metrics"]), set(self.models))
        self.assertEqual(report["comparisons"], {})
        self.assertEqual(report["benchmark"]["status"], "NOT_COMPARABLE")
        self.assertEqual(self.queries, [])
        ranking = pc.ranking_table(report)
        self.assertEqual(ranking.Status.tolist()[-1], "BLOCKED")
        self.assertTrue(pd.isna(ranking.Rank.iloc[-1]))
        self.assertEqual(ranking.Rank.notna().sum(), len(self.models), "every registered model is ranked; only production is blocked")
        self.assertEqual(len(pc.interval_table(report)), 0)

    def test_exact_key_alignment_rejects_missing_duplicate_shifted_and_invalid_scores(self):
        missing = self.scored.iloc[1:]
        self.assertEqual(self.run_report(frame=missing)["production"]["status"], "BLOCKED")
        self.assertIn("1 of 40", self.run_report(frame=missing)["production"]["reason"])
        shifted = self.scored.copy(); shifted.loc[0, "END_DT"] = shifted.loc[0, "END_DT"] + pd.Timedelta(days=1)
        self.assertIn("no production score on the exact key", self.run_report(frame=shifted)["production"]["reason"])
        duplicate = pd.concat([self.scored, self.scored.iloc[:1]], ignore_index=True)
        self.assertIn("duplicate", self.run_report(frame=duplicate)["production"]["reason"])
        bad = self.scored.copy(); bad.loc[3, "SCORE"] = 1.2
        self.assertIn("finite probabilities", self.run_report(frame=bad)["production"]["reason"])
        nan = self.scored.copy(); nan.loc[3, "SCORE"] = np.nan
        self.assertIn("finite probabilities", self.run_report(frame=nan)["production"]["reason"])
        wide = pd.concat([self.scored, pd.DataFrame({"PATIENT_ID": ["zz"], "END_DT": [pd.Timestamp("2026-01-31")],
                                                     "RESP": [0], "SCORE": [.4]})], ignore_index=True)
        report = self.run_report(frame=wide)
        self.assertEqual(report["production"]["status"], "ALIGNED")
        self.assertEqual(report["production"]["provenance"]["extra_scored_rows_ignored"], 1)

    def test_scored_keys_are_never_repaired(self):
        padded = self.scored.copy(); padded.loc[0, "PATIENT_ID"] = " " + padded.loc[0, "PATIENT_ID"]
        report = self.run_report(frame=padded)
        self.assertEqual(report["production"]["status"], "BLOCKED")
        self.assertIn("whitespace", report["production"]["reason"])
        numeric = self.scored.copy(); numeric["PATIENT_ID"] = range(len(numeric))
        self.assertIn("exact nonempty strings", self.run_report(frame=numeric)["production"]["reason"])
        empty = self.scored.copy(); empty.loc[0, "PATIENT_ID"] = None
        self.assertIn("Missing PATIENT_ID", self.run_report(frame=empty)["production"]["reason"])
        clock = self.scored.copy(); clock.loc[0, "END_DT"] = clock.loc[0, "END_DT"] + pd.Timedelta(hours=9)
        self.assertIn("midnight", self.run_report(frame=clock)["production"]["reason"])
        zoned = self.scored.copy(); zoned["END_DT"] = zoned.END_DT.dt.tz_localize("UTC")
        self.assertIn("Timezone", self.run_report(frame=zoned)["production"]["reason"])
        shadow = self.scored.copy(); shadow["score"] = shadow.SCORE
        self.assertIn("duplicate columns", self.run_report(frame=shadow)["production"]["reason"])
        for frame in (padded, numeric, empty, clock, zoned, shadow):
            self.assertEqual(set(self.run_report(frame=frame)["metrics"]), set(self.models))

    def test_source_failures_block_with_class_name_only(self):
        class VaultError(RuntimeError):
            pass

        def failing(sql):
            self.queries.append(sql)
            raise VaultError("password=secret host=private")
        report = pc.run(failing, self.meta, self.config(), self.models, "validation",
                        comparison=comparison, workflow=workflow, n_bootstrap=20)
        self.assertEqual(report["production"]["status"], "BLOCKED")
        self.assertEqual(report["production"]["reason"], "Scored table read failed: VaultError")
        self.assertNotIn("secret", json.dumps(report, default=str))
        self.assertEqual(len(self.queries), 1)
        with self.assertRaisesRegex(TypeError, "Spark or pandas"):
            pc.run(lambda sql: [1, 2, 3], self.meta, self.config(), self.models, "validation",
                   comparison=comparison, workflow=workflow, n_bootstrap=20)

    def test_aligned_documented_lift_only_metrics_and_direction(self):
        report = self.run_report()
        self.assertEqual(report["production"]["status"], "ALIGNED")
        self.assertEqual(report["benchmark"]["status"], "MATCHED_POPULATION_PROVENANCE_DOCUMENTED")
        self.assertIn("not independently verified", report["benchmark"]["statement"])
        self.assertIn("not an untouched holdout", report["benchmark"]["statement"])
        self.assertIn("checkpoint selection", report["benchmark"]["evaluation_scope"])
        self.assertTrue(report["benchmark"]["evidence_is_assertion"])
        self.assertNotIn("FAIR", json.dumps(report, default=str))
        self.assertNotIn("HELDOUT", json.dumps(report, default=str))
        self.assertEqual(set(report["metrics"]), set(self.models) | {pc.PRODUCTION})
        for name, pair in report["comparisons"].items():
            self.assertEqual(set(pair), {"lift", "direction"})
            self.assertEqual(pair["direction"], "model_minus_production")
            self.assertEqual(set(pair["lift"]["lift_by_fraction"]), {"0.1", "0.2", "0.3"})
            lifts = {float(i["fraction"]): i["lift"] for i in report["metrics"][name]["topk"]}
            base = {float(i["fraction"]): i["lift"] for i in report["metrics"][pc.PRODUCTION]["topk"]}
            self.assertAlmostEqual(pair["lift"]["lift_by_fraction"]["0.1"]["delta"], lifts[.1] - base[.1])
        table = pc.report_table(report)
        self.assertEqual(len(table), 5)
        self.assertEqual(list(table.columns), ["Model", "Split", "Lift@10%", "Lift@20%", "Lift@30%", "Snapshots", "Positives"])
        ranking = pc.ranking_table(report)
        self.assertTrue(ranking["Lift@10%"].is_monotonic_decreasing)
        self.assertTrue((ranking.Status == "MATCHED_POPULATION_PROVENANCE_DOCUMENTED").all())
        intervals = pc.interval_table(report)
        self.assertEqual(len(intervals), 4 * 3)
        self.assertTrue(intervals.Metric.str.startswith("Lift@").all())
        self.assertFalse(report["production"]["provenance"]["fit_membership_verified"])
        self.assertFalse(report["production"]["provenance"]["refit_performed"])
        self.assertEqual(report["production"]["provenance"]["scored_label_mismatches"], 0)
        self.assertEqual(self.queries, ["SELECT PATIENT_ID, END_DT, SCORE, RESP FROM " + TABLE])
        text = json.dumps(report, default=str)
        self.assertNotIn('"p1"', text)

    def test_descriptive_when_evidence_incomplete_or_labels_differ(self):
        for change in (dict(production_model_fingerprint=""), dict(patient_disjoint_from_production_training=False),
                       dict(same_target_definition=None), dict(historical_inputs_frozen="yes")):
            report = self.run_report(config=self.config(**change))
            self.assertEqual(report["benchmark"]["status"], "COMMON_POPULATION_DESCRIPTIVE", change)
            self.assertNotIn("verified", report["benchmark"]["statement"].lower())
            self.assertTrue(report["benchmark"]["reasons"])
        relabelled = self.scored.copy(); relabelled.loc[0, "RESP"] = 1 - relabelled.loc[0, "RESP"]
        report = self.run_report(frame=relabelled)
        self.assertEqual(report["benchmark"]["status"], "COMMON_POPULATION_DESCRIPTIVE")
        self.assertEqual(report["production"]["provenance"]["scored_label_mismatches"], 1)
        self.assertEqual(report["metrics"][pc.PRODUCTION]["positives"], int(self.meta.RESP.sum()), "current labels, not scored labels")
        report = self.run_report(config=self.config(label=None))
        self.assertEqual(report["benchmark"]["status"], "COMMON_POPULATION_DESCRIPTIVE")
        self.assertIn("scored labels not supplied", "; ".join(report["benchmark"]["reasons"]))

    def test_test_gate_precedes_any_read_and_model_validation(self):
        test_meta = self.meta.assign(SPLIT="test")
        reader = self.reader(self.scored)
        for allow, locked in ((False, False), (True, False), (False, True)):
            with self.assertRaisesRegex(ValueError, "TEST reporting"):
                pc.run(reader, test_meta, self.config(), self.models, "test", comparison=comparison, workflow=workflow,
                       allow_test=allow, selection_locked=locked, n_bootstrap=20)
        self.assertEqual(self.queries, [], "gate must run before any source read")
        with self.assertRaisesRegex(ValueError, "Unrecognized evaluation split"):
            pc.run(reader, test_meta, self.config(), self.models, "holdout", comparison=comparison, workflow=workflow)
        self.assertEqual(self.queries, [])
        report = pc.run(reader, test_meta, self.config(), self.models, "test", comparison=comparison, workflow=workflow,
                        allow_test=True, selection_locked=True, n_bootstrap=20)
        self.assertIn("Retrospective", report["evaluation_scope"])
        self.assertIn("already inspected", report["benchmark"]["statement"])
        self.assertEqual(len(self.queries), 1)
        with self.assertRaisesRegex(ValueError, "one finite probability"):
            pc.compare(self.meta, {"COUNT": np.zeros((3, 40))}, None, "validation", comparison=comparison, workflow=workflow)
        with self.assertRaisesRegex(ValueError, "reserved"):
            pc.compare(self.meta, dict(self.models, **{pc.PRODUCTION: self.models["COUNT"]}), None, "validation",
                       comparison=comparison, workflow=workflow)
        production = pc.load_production_scores(self.reader(self.scored), self.meta, self.config(), "validation")
        for bad in (production["scores"][:-1], np.where(np.arange(40) == 3, 1.5, production["scores"]),
                    np.where(np.arange(40) == 3, np.nan, production["scores"])):
            with self.assertRaisesRegex(ValueError, "one finite probability"):
                pc.compare(self.meta, self.models, dict(production, scores=bad), "validation",
                           comparison=comparison, workflow=workflow, config=self.config(), n_bootstrap=20)
        with self.assertRaisesRegex(ValueError, "Unknown production comparison configuration"):
            pc.run(reader, self.meta, dict(score_table=TABLE, bogus=1), self.models, "validation", comparison=comparison, workflow=workflow)
        with self.assertRaisesRegex(ValueError, "Only probability scores"):
            pc.run(reader, self.meta, dict(score_table=TABLE, score_kind="margin"), self.models, "validation", comparison=comparison, workflow=workflow)

    def test_identity_tracks_scores_and_production(self):
        first = self.run_report()
        second = self.run_report(frame=self.scored.assign(SCORE=self.scored.SCORE.to_numpy()[::-1]))
        self.assertNotEqual(first["prediction_identity_sha256"], second["prediction_identity_sha256"])
        models = copy.deepcopy(self.models); models["COUNT"][0] = min(1., models["COUNT"][0] + .01)
        third = pc.run(self.reader(self.scored), self.meta, self.config(), models, "validation",
                       comparison=comparison, workflow=workflow, n_bootstrap=20)
        self.assertNotEqual(first["prediction_identity_sha256"], third["prediction_identity_sha256"])
        self.assertEqual(first["prediction_identity_sha256"], self.run_report()["prediction_identity_sha256"])


if __name__ == "__main__":
    unittest.main()
