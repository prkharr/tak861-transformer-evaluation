"""Independent gate and adversarial tests for cohort_binding.bind on pandas doubles.

A small Python oracle reproduces the SQL contract (horizon, window, first non-generic
event, days 1..90 label, prior-initiation and observed-negative retention) so that
bind()'s runtime gates can be exercised without a warehouse. No clinical data.
"""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from experiments.automl_sequence import client_universe, inputs, workflow as wf
from experiments.fresh_cohort import cohort_binding as cb

CAND = "DB.SC.CANDIDATE_UNIVERSE"
BASKET = "DB.SC.MARKET_BASKET_CLAIMS"
MEDICAL = "DB.SC.MEDICAL_EVENTS"
PREFIX = "SYNTHETIC_BIND"
POPULATION = "DSVC_TAKEDA_TA_PRIVATE.DS_ML." + PREFIX + "_POPULATION"
D = pd.Timestamp


def world():
    """Candidate snapshots, basket fills and medical events designed to hit every rule."""
    candidate = pd.DataFrame([
        ("p1", "2025-01-31", 0), ("p1", "2025-02-28", 1),   # event on day 1 after 02-28 -> positive; 01-31 negative observed
        ("p2", "2025-01-31", 0), ("p2", "2025-03-31", 0),   # event exactly +90 after 01-31 -> positive; 03-31 prior initiation
        ("p3", "2025-02-28", 1),                             # event at +91 -> relabelled 0, observed negative
        ("p4", "2025-02-28", 0),                             # never initiates; medical activity -> observed negative
        ("p5", "2025-02-28", 0),                             # never initiates; no activity -> unobserved, excluded
        ("p6", "2022-06-30", 0),                             # outside the window
        ("p7", "2025-02-28", 0),                             # event on day 0 -> prior initiation (same day)
        ("p8", "2025-02-28", 0),                             # only generic fills; medical activity -> observed negative
        ("p10", "2025-02-28", 1),                            # inherited TEST positive, event at +15 days
    ], columns=["PATIENT_ID", "END_DT", "RESP"])
    candidate["END_DT"] = pd.to_datetime(candidate.END_DT)
    basket = pd.DataFrame([
        ("p1", "2025-03-01", False), ("p2", "2025-05-01", False), ("p3", "2025-05-30", False),
        ("p7", "2025-02-28", False), ("p8", "2025-03-10", True), ("p1", "2024-01-01", True),
        ("p9", "2025-12-31", True),   # late fill: horizon 2025-12-31 - 90 d keeps every 2025 snapshot in window
        ("p10", "2025-03-15", False),
    ], columns=["PATIENT_ID", "FILL_DATE", "IS_GENERIC"])
    basket["FILL_DATE"] = pd.to_datetime(basket.FILL_DATE)
    medical = pd.DataFrame([
        ("p1", "2025-02-10"), ("p2", "2025-02-15"), ("p3", "2025-04-01"), ("p4", "2025-03-15"),
        ("p8", "2025-04-20"), ("p9", "2025-12-31"),
    ], columns=["PATIENT_ID", "SERVICE_DATE"])
    medical["SERVICE_DATE"] = pd.to_datetime(medical.SERVICE_DATE)
    prior = pd.DataFrame([
        ("p1", "2025-01-31", 0, "train"), ("p1", "2025-02-28", 1, "train"),
        ("p2", "2025-01-31", 0, "validation"), ("p4", "2025-02-28", 0, "validation"),
        ("p3", "2025-02-28", 1, "test"), ("p10", "2025-02-28", 1, "test"), ("p5", "2025-02-28", 0, "train"),
    ], columns=["PATIENT_ID", "END_DT", "RESP", "SPLIT"])
    prior["END_DT"] = pd.to_datetime(prior.END_DT)
    return candidate, basket, medical, prior


def oracle(candidate, basket, medical, months):
    observed_through = min(basket.FILL_DATE.max(), medical.SERVICE_DATE.max())
    upper = min(candidate.END_DT.max(), observed_through - pd.Timedelta(days=90))
    lower = upper - pd.DateOffset(months=months)
    bounded = candidate[(candidate.END_DT > lower) & (candidate.END_DT <= upper)].copy()
    bounded = bounded.rename(columns={"RESP": "ORIGINAL_RESP"})
    patients = set(bounded.PATIENT_ID)
    advanced = basket[basket.IS_GENERIC.eq(False) & basket.PATIENT_ID.isin(patients)]
    first = advanced.groupby("PATIENT_ID").FILL_DATE.min().rename("FIRST_EVENT")
    d = bounded.merge(first, left_on="PATIENT_ID", right_index=True, how="left")
    d["RESP"] = ((d.FIRST_EVENT > d.END_DT) & (d.FIRST_EVENT <= d.END_DT + pd.Timedelta(days=90))).astype(int)
    d["PRIOR_INITIATION"] = (d.FIRST_EVENT <= d.END_DT).fillna(False).astype(int)
    observed = []
    for row in d.itertuples(index=False):
        events = medical[(medical.PATIENT_ID == row.PATIENT_ID) & (medical.SERVICE_DATE > row.END_DT)
                         & (medical.SERVICE_DATE <= row.END_DT + pd.Timedelta(days=90))]
        observed.append(int(row.RESP == 0 and row.PRIOR_INITIATION == 0 and len(events) > 0))
    d["OBSERVED_NEGATIVE"] = observed
    d["RETAIN"] = ((d.PRIOR_INITIATION == 0) & ((d.RESP == 1) | (d.OBSERVED_NEGATIVE == 1))).astype(int)
    d = d.sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    kept = d[d.RETAIN.eq(1)]
    in_basket = basket[basket.PATIENT_ID.isin(patients)]
    audit = pd.DataFrame([dict(
        CANDIDATE_SNAPSHOTS=len(candidate), BOUNDED_SNAPSHOTS=len(bounded), OBSERVED_THROUGH=observed_through,
        UPPER_END=upper, LOWER_EXCLUSIVE=lower, SNAPSHOTS=len(kept), PATIENTS=kept.PATIENT_ID.nunique(),
        POSITIVES=int(kept.RESP.sum()), POSITIVE_PATIENTS=kept.loc[kept.RESP.eq(1), "PATIENT_ID"].nunique(),
        POSITIVES_GAINED=int((kept.ORIGINAL_RESP.eq(0) & kept.RESP.eq(1)).sum()),
        POSITIVES_LOST=int((kept.ORIGINAL_RESP.eq(1) & kept.RESP.eq(0)).sum()),
        NULL_CLASSIFICATION_ROWS=int(in_basket.IS_GENERIC.isna().sum()),
        NULL_ADVANCED_DATE_ROWS=int((in_basket.IS_GENERIC.eq(False) & in_basket.FILL_DATE.isna()).sum()),
        PRIOR_INITIATION_EXCLUSIONS=int(d.PRIOR_INITIATION.sum()),
        UNOBSERVED_NEGATIVE_EXCLUSIONS=int(((d.PRIOR_INITIATION == 0) & (d.RESP == 0) & (d.OBSERVED_NEGATIVE == 0)).sum()))])
    return d, audit


class Warehouse:
    def __init__(self, candidate, basket, medical, prior, months):
        self.candidate, self.basket, self.medical, self.prior, self.months = candidate, basket, medical, prior, months
        self.tables, self.artifacts, self.queries = {}, {}, []
        self.audit_override, self.decision_override = None, None

    def read_query(self, sql):
        self.queries.append(sql)
        if "WHERE 1=0" in sql:
            table = sql.split("FROM ")[1].split(" ")[0]
            columns = {CAND: ["PATIENT_ID", "END_DT", "RESP"], BASKET: ["PATIENT_ID", "FILL_DATE", "IS_GENERIC"],
                       MEDICAL: ["PATIENT_ID", "SERVICE_DATE"]}[table]
            return pd.DataFrame(columns=columns)
        if sql == "SELECT PATIENT_ID,END_DT,RESP FROM " + CAND:
            return self.candidate.copy()
        if sql == "SELECT PATIENT_ID,END_DT,RESP FROM " + POPULATION:
            return self.tables[POPULATION].copy()
        decisions, audit = oracle(self.candidate, self.basket, self.medical, self.months)
        if "CANDIDATE_SNAPSHOTS" in sql:
            return self.audit_override(audit) if self.audit_override else audit
        if "SELECT * FROM decisions" in sql:
            return self.decision_override(decisions) if self.decision_override else decisions
        raise AssertionError("unexpected query: " + sql[:80])

    def read_sf(self, suffix):
        assert suffix == "PATIENT_SPLIT"
        return self.prior.copy()

    def write_population(self, table, metadata):
        assert table == POPULATION and table not in self.tables
        self.tables[table] = metadata[["PATIENT_ID", "END_DT", "RESP"]].copy()

    def table_exists(self, name):
        return name in self.artifacts or name in {t.split(".")[-1] for t in self.tables}

    def read_artifacts(self, table, names):
        return {k: v for k, v in self.artifacts[table].items() if k in names}

    def save_artifacts(self, table, artifacts):
        assert table not in self.artifacts, "immutable artifact replaced"
        self.artifacts[table] = dict(artifacts)


def make_config(months=24):
    return {"cohort_plan": dict(candidate=CAND, basket=BASKET, medical=MEDICAL, months=months, max_missing_original_snapshots=16),
            "client_universe": dict(split_seed=42), "cohort_evidence": {}}


class BindingGateTests(unittest.TestCase):
    def setUp(self):
        client_universe.inputs = inputs
        self.candidate, self.basket, self.medical, self.prior = world()
        self.wh = Warehouse(self.candidate, self.basket, self.medical, self.prior, 24)
        self.config = make_config()

    def bind(self, config=None, create=True, wh=None):
        wh = wh or self.wh
        return cb.bind(wh.read_query, wh.read_sf, wh.write_population, config or self.config, "impl-synthetic",
                       PREFIX, inputs, client_universe, wf, wh, create=create)

    def test_oracle_and_sql_contract_boundaries(self):
        decisions, audit = oracle(self.candidate, self.basket, self.medical, 24)
        by_key = decisions.set_index(["PATIENT_ID", decisions.END_DT.dt.strftime("%Y-%m-%d")])
        self.assertEqual(by_key.loc[("p1", "2025-02-28"), "RESP"], 1)          # day 1
        self.assertEqual(by_key.loc[("p2", "2025-01-31"), "RESP"], 1)          # day 90
        self.assertEqual(by_key.loc[("p3", "2025-02-28"), "RESP"], 0)          # day 91 -> relabelled
        self.assertEqual(by_key.loc[("p7", "2025-02-28"), "PRIOR_INITIATION"], 1)  # day 0 -> prior
        self.assertEqual(by_key.loc[("p2", "2025-03-31"), "PRIOR_INITIATION"], 0)
        self.assertEqual(by_key.loc[("p5", "2025-02-28"), "RETAIN"], 0)        # unobserved negative
        self.assertEqual(by_key.loc[("p8", "2025-02-28"), "RETAIN"], 1)        # generic-only, observed
        self.assertNotIn("p6", set(decisions.PATIENT_ID))                        # outside window
        self.assertEqual(int(audit.POSITIVES_LOST.iloc[0]), 1)
        population_sql, audit_sql = cb.queries(self.config["cohort_plan"])
        for text in (population_sql, audit_sql):
            self.assertTrue(text.strip().upper().startswith("WITH"))
            self.assertIsNone(__import__("re").search(r"\b(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|MERGE|CALL)\b", text, __import__("re").I))
            self.assertIn("IFF(f.FIRST_EVENT > c.END_DT AND f.FIRST_EVENT <= DATEADD(DAY,90,c.END_DT),1,0) AS RESP", text)
            self.assertIn("IFF(f.FIRST_EVENT <= c.END_DT,1,0) AS PRIOR_INITIATION", text)
            self.assertIn("WHERE b.IS_GENERIC=FALSE", text)
            self.assertIn("DATEADD(DAY,-90,s.OBSERVED_THROUGH)", text)
            self.assertIn("c.RESP AS ORIGINAL_RESP", text)
            self.assertNotIn("ORIGINAL_RESP=", text.split("decisions AS")[0].split("labeled AS")[1])

    def test_happy_path_freezes_population_and_restores_in_later_stage(self):
        saved = self.bind()
        decisions, audit = oracle(self.candidate, self.basket, self.medical, 24)
        kept = decisions[decisions.RETAIN.eq(1)]
        expected = self.config["client_universe"]["expected_population"]
        self.assertEqual((expected["snapshots"], expected["positives"]), (len(kept), int(kept.RESP.sum())))
        self.assertEqual(self.config["client_universe"]["table"], POPULATION)
        self.assertEqual(self.config["cohort_evidence"]["lineage_status"], "bounded_reconstruction_development")
        self.assertFalse(saved["original_producer_verified"])
        self.assertEqual(saved["expected_population_origin"], "measured_at_binding_reconciled_with_SQL_aggregates")
        reasons = {(r["split"], r["reason"]): r for r in saved["prior_changes_by_split"]}
        self.assertEqual(reasons[("train", "unobserved_negative")]["snapshots"], 1)        # p5
        self.assertEqual(reasons[("test", "retained")]["labels_1_to_0"], 1)                # p3 relabelled
        self.assertEqual(reasons[("validation", "retained")]["labels_0_to_1"], 1)          # p2 01-31 gained
        text = json.dumps(saved)
        for patient in ("p1", "p2", "p3", "p9"):
            self.assertNotIn('"' + patient + '"', text, "patient identifier leaked into cohort.json")
        self.assertIn(PREFIX + "_COHORT_AUDIT", self.wh.artifacts)
        later = make_config()
        restored = self.bind(config=later, create=False)
        self.assertEqual(restored, saved)
        self.assertEqual(later["client_universe"]["expected_population"], expected)
        self.assertEqual(sum("SELECT * FROM decisions" in q for q in self.wh.queries), 1, "later stage re-extracted")

    def test_later_stage_requires_binding_and_matching_identity(self):
        with self.assertRaisesRegex(ValueError, "Run notebook 01"):
            self.bind(create=False)
        self.bind()
        with self.assertRaisesRegex(ValueError, "different configuration"):
            self.bind(config=make_config(months=36), create=False)
        with self.assertRaisesRegex(ValueError, "different configuration"):
            cb.bind(self.wh.read_query, self.wh.read_sf, self.wh.write_population, make_config(), "impl-other",
                    PREFIX, inputs, client_universe, wf, self.wh, create=False)

    def test_aggregate_disagreement_and_candidate_drift_stop(self):
        self.wh.audit_override = lambda a: a.assign(POSITIVES=a.POSITIVES + 1)
        with self.assertRaisesRegex(ValueError, "disagree: positives"):
            self.bind()
        self.wh.audit_override = lambda a: a.assign(CANDIDATE_SNAPSHOTS=a.CANDIDATE_SNAPSHOTS + 1)
        with self.assertRaisesRegex(ValueError, "Candidate population changed"):
            self.bind()
        self.wh.audit_override = lambda a: a.assign(UPPER_END=None)
        with self.assertRaisesRegex(ValueError, "cannot support"):
            self.bind()

    def test_null_classification_and_null_advanced_date_stop(self):
        self.wh.audit_override = lambda a: a.assign(NULL_CLASSIFICATION_ROWS=1)
        with self.assertRaisesRegex(ValueError, "unclassified"):
            self.bind()
        self.wh.audit_override = lambda a: a.assign(NULL_ADVANCED_DATE_ROWS=2)
        with self.assertRaisesRegex(ValueError, "unclassified"):
            self.bind()

    def test_tampered_decisions_stop(self):
        def flip(d):
            d = d.copy(); d.loc[0, "RESP"] = 1 - d.loc[0, "RESP"]; return d
        self.wh.decision_override = flip
        with self.assertRaisesRegex(ValueError, "days1..90"):
            self.bind()
        self.wh.decision_override = lambda d: d.iloc[1:]
        with self.assertRaisesRegex(ValueError, "differ from candidate snapshot keys"):
            self.bind()
        def retain_unobserved(d):
            d = d.copy(); d.loc[d.PATIENT_ID.eq("p5"), "RETAIN"] = 1; return d
        self.wh.decision_override = retain_unobserved
        with self.assertRaisesRegex(ValueError, "observability retention"):
            self.bind()

    def test_missing_original_keys_limit_and_persisted_population_mismatch(self):
        extra = pd.DataFrame({"PATIENT_ID": [f"old{i}" for i in range(17)], "END_DT": pd.Timestamp("2024-06-30"),
                              "RESP": [0, 1] * 8 + [0], "SPLIT": "train"})
        wh = Warehouse(self.candidate, self.basket, self.medical, pd.concat([self.prior, extra], ignore_index=True), 24)
        with self.assertRaisesRegex(ValueError, "missing historical keys"):
            self.bind(wh=wh)
        tampered = Warehouse(self.candidate, self.basket, self.medical, self.prior, 24)
        original_write = tampered.write_population
        def corrupt(table, metadata):
            bad = metadata.copy(); bad.loc[bad.index[0], "RESP"] = 1 - bad.loc[bad.index[0], "RESP"]
            original_write(table, bad)
        tampered.write_population = corrupt
        with self.assertRaises(ValueError):
            self.bind(wh=tampered)
        self.assertNotIn(PREFIX + "_COHORT_AUDIT", tampered.artifacts, "audit must not be frozen after a persistence failure")

    def test_change_policy_is_enforced_in_later_alignment(self):
        self.bind()
        policy = dict(self.config["client_universe"]["change_policy"])
        incoming = self.wh.tables[POPULATION].copy()
        incoming["SPLIT"] = "train"
        client_universe.align_universe(incoming, self.prior, 42, change_policy=policy)
        wrong = dict(policy, removed_keys_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "removes prior snapshot keys|differs from its frozen audit"):
            client_universe.align_universe(incoming, self.prior, 42, change_policy=wrong)
        with self.assertRaisesRegex(ValueError, "removes prior snapshot keys"):
            client_universe.align_universe(incoming, self.prior, 42)


if __name__ == "__main__":
    unittest.main()
