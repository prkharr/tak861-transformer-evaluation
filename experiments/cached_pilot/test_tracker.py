import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.cached_pilot import tracker as tr


def report(version="v1", train=100):
    row = {"Model": "AUTOML", "Split": "VALIDATION", "Snapshots": 100, "Patients": 80,
           "Positive_snapshots": 20, "Lift@10%": 2., "Captured@10%": 4, "Selected@10%": 10}
    second = dict(row, Model="AUTOML_TIME", **{"Lift@10%": 3., "Captured@10%": 6})
    return {"schema": "cached_pilot_report_v1", "identity": {"version": version,
            "evaluation_keys_sha256": "a"*64, "evaluation_labels_sha256": "b"*64,
            "preparation_hash": "c"*64, "feature_definition_hash": "d"*64,
            "protocol_hash": "e"*64,
            "settings": {"seed": 42, "n_estimators": 300}, "feature_names": {"automl": ["A"], "time": ["T"]},
            "train_snapshots": train}, "rows": [row, second], "paired_lift": {"AUTOML_TIME": {
            "paired": True, "unit": "patient_cluster", "interval_reliable": True,
            "lift_by_fraction": {"0.1": {"delta": 1., "lower": -.1, "upper": 2.}}}}}


def receipt(sha="f"*64):
    return {"source": "synthetic/report.json", "sha256": sha, "integrity_verified": True}


def empty_seed():
    return {"schema": "experiment_evidence_seed_v1", "experiments": [], "engineering": []}


class TrackerTests(unittest.TestCase):
    def test_pilot_aggregate_lift_and_capture_contrast(self):
        rows = tr.pilot_rows(report(), receipt())
        self.assertEqual(rows[1]["Matched delta@10%"], 1.)
        self.assertEqual(rows[1]["Additional captures@10%"], 2)
        self.assertIn("not equivalence", rows[1]["Conclusion"])

    def test_inconsistent_lift_rejected(self):
        data = report(); data["rows"][1]["Lift@10%"] = 4.
        with self.assertRaisesRegex(ValueError, "aggregate counts"):
            tr.pilot_rows(data, receipt())

    def test_wrong_population_rejected(self):
        data = report(); data["rows"][1]["Patients"] = 81
        with self.assertRaisesRegex(ValueError, "different evaluation"):
            tr.pilot_rows(data, receipt())

    def test_test_and_duplicate_rows_rejected(self):
        for mutation in ("test", "duplicate"):
            data = report()
            if mutation == "test": data["rows"][0]["Split"] = "TEST"
            else: data["rows"].append(copy.deepcopy(data["rows"][0]))
            with self.assertRaises(ValueError): tr.pilot_rows(data, receipt())

    def test_wrong_bootstrap_direction_rejected(self):
        data = report(); data["paired_lift"]["AUTOML_TIME"]["lift_by_fraction"]["0.1"]["delta"] = -1
        with self.assertRaisesRegex(ValueError, "direction"):
            tr.pilot_rows(data, receipt())

    def test_unreliable_interval_not_promoted(self):
        data = report(); data["paired_lift"]["AUTOML_TIME"]["interval_reliable"] = False
        row = tr.pilot_rows(data, receipt())[1]
        self.assertIsNone(row["lower"])
        self.assertIn("unavailable", row["Conclusion"])

    def test_missing_identity_rejected(self):
        data = report(); del data["identity"]["evaluation_labels_sha256"]
        with self.assertRaisesRegex(ValueError, "identity"):
            tr.pilot_rows(data, receipt())

    def test_local_receipt_and_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"report.json"
            blob = json.dumps(report()).encode(); path.write_bytes(blob)
            path.with_name("report_receipt.json").write_text(json.dumps({"sha256": hashlib.sha256(blob).hexdigest()}))
            data, checked = tr.load_local_report(folder)
            self.assertTrue(checked["integrity_verified"])
            self.assertEqual(data["schema"], "cached_pilot_report_v1")
            path.write_bytes(blob+b" ")
            with self.assertRaisesRegex(ValueError, "does not match"):
                tr.load_local_report(folder)

    def test_remote_only_aggregate_artifact_and_bounds(self):
        calls = []
        blob = json.dumps(report()).encode()
        def read(query, limit):
            calls.append((query,limit))
            if len(calls)==1:
                return pd.DataFrame([dict(MIN_BYTES=len(blob), MAX_BYTES=len(blob), MAX_CHUNKS=1, STORED_CHUNKS=1, ENCODED_BYTES=len(blob)*2)])
            return pd.DataFrame([dict(CHUNK_INDEX=0)])
        result, checked = tr.load_remote_report(read, "DB.SCHEMA.REPORT", lambda f,b: blob)
        self.assertEqual(len(calls),2)
        self.assertTrue(all("ARTIFACT_NAME = 'report.json'" in c[0] for c in calls))
        self.assertFalse(any("SELECT *" in c[0] for c in calls))
        self.assertEqual(result["schema"], "cached_pilot_report_v1")
        with self.assertRaises(ValueError): tr.load_remote_report(read, "DB.SCHEMA.REPORT;DROP X", None)

    def test_remote_oversize_stops_before_payload(self):
        calls = []
        def read(query,limit):
            calls.append(query)
            return pd.DataFrame([dict(MIN_BYTES=tr.MAX_REPORT_BYTES+1,MAX_BYTES=tr.MAX_REPORT_BYTES+1)])
        with self.assertRaisesRegex(ValueError,"oversized"):
            tr.load_remote_report(read, "DB.SCHEMA.REPORT", None)
        self.assertEqual(len(calls),1)

    def test_refresh_groups_require_stable_keys_labels_protocol(self):
        first=report(); second=report("v2", 200)
        second["identity"]["preparation_hash"]="e"*64
        table,_,_ = tr.ledger(empty_seed(), [(first,receipt()),(second,receipt("0"*64))])
        self.assertEqual(len(tr.progress_groups(table)),2)
        for field in ("evaluation_keys_sha256", "evaluation_labels_sha256", "feature_definition_hash", "protocol_hash"):
            changed=copy.deepcopy(second); changed["identity"][field]="9"*64
            table,_,_ = tr.ledger(empty_seed(),[(first,receipt()),(changed,receipt("0"*64))])
            self.assertEqual(tr.progress_groups(table),[])

    def test_incomplete_definition_identity_cannot_make_curve(self):
        first=report(); second=report("v2",200)
        for item in [first,second]: del item["identity"]["feature_definition_hash"]
        table,_,_=tr.ledger(empty_seed(),[(first,receipt()),(second,receipt("0"*64))])
        self.assertEqual(tr.progress_groups(table),[])

    def test_seed_has_missing_results_and_separate_groups(self):
        seed=json.loads(Path(__file__).with_name("evidence_seed.json").read_text(encoding="utf-8"))
        table,engineering,errors=tr.ledger(seed)
        self.assertTrue(table["Lift@10%"].isna().any())
        self.assertTrue((table["Evidence"]=="DOCUMENTED_SCREENSHOT_REPORTED").any())
        self.assertTrue(engineering.shape[0]>0)
        self.assertTrue(errors.empty)
        self.assertEqual(tr.progress_groups(table),[])
        for (block,split,metric),group in tr.chart_groups(table):
            self.assertEqual(group.comparison_block.nunique(),1)
            self.assertEqual(group.split.nunique(),1)

    def test_duplicate_reports_not_double_counted(self):
        table,_,_=tr.ledger(empty_seed(),[(report(),receipt()),(report(),receipt())])
        self.assertEqual(len(table),2)

    def test_unsupported_report_is_visible_blocker(self):
        table,_,errors=tr.ledger(empty_seed(),[({"metric":3},receipt())])
        self.assertTrue(table.empty)
        self.assertEqual(len(errors),1)

    def test_nonfinite_json_and_metric_rejected(self):
        with self.assertRaises(ValueError): tr._json(b'{"lift": NaN}')
        for value in [float("inf"), True]:
            with self.assertRaises(ValueError): tr.number(value)

    def test_generic_saved_rows_not_assumed_matched(self):
        data={"rows": report()["rows"], "identity": {}, "scope": "exploratory"}
        rows=tr.saved_rows(data, receipt())
        self.assertTrue(all(row["comparison_block"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
