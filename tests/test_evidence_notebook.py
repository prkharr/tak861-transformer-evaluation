"""Offline contract checks for the standalone read-only evidence handoff."""
import ast
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest

import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/Project Evidence/00_collect_missing_inputs.ipynb"


def load_namespace():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # Runtime hardware probing is not needed for offline evidence logic.
            if source.startswith("def runtime_evidence"):
                continue
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return notebook, namespace, output.getvalue()


class EvidenceNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, cls.ns, cls.output = load_namespace()

    def test_portable_documented_and_output_cleared(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(self.notebook["cells"][index - 1]["cell_type"], "markdown")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                ast.parse("".join(cell["source"]))

    def test_offline_collection_finishes_without_claiming_live_pass(self):
        self.assertIn("EVIDENCE_END E13", self.output)
        reports = self.ns["reports"]
        self.assertTrue(any(r["status"] == "UNAVAILABLE" for r in reports))
        self.assertFalse(any(r["status"] == "PASS" and r["section"] != "E00" for r in reports))
        self.assertFalse(self.ns["FULL_TENSOR_AUDIT"])

    def test_sql_rejects_writes_and_stacked_queries(self):
        check = self.ns["_read_only_sql"]
        for query in ["DELETE FROM X", "SELECT 1; DROP TABLE X", "WITH x AS (SELECT 1) INSERT INTO X SELECT * FROM x", "CALL build()"]:
            with self.assertRaises(ValueError):
                check(query)
        self.assertEqual(check("SELECT 'DROP TABLE X; embedded literal' AS TEXT"), "SELECT 'DROP TABLE X; embedded literal' AS TEXT")
        with self.assertRaises(ValueError):
            self.ns["qi"]('T; DELETE FROM T')
        self.assertEqual(self.ns["qi"]('DB.SCHEMA.TABLE'), '"DB"."SCHEMA"."TABLE"')

    def test_report_rejects_raw_sensitive_fields(self):
        for key in ["PATIENT_ID", "PAYLOAD_BASE64", "PROCEDURE_DEFINITION", "SFPASSWORD"]:
            with self.assertRaises(ValueError):
                self.ns["_json_safe"]({key: "must never appear"})

    def test_report_pagination_preserves_all_records_and_unique_sections(self):
        records = [{"feature_order": i} for i in range(67)]
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            self.ns["emit"]("TEST.paging", "READ_OK", "test", tables=records)
        text = capture.getvalue()
        self.assertEqual(text.count("EVIDENCE_BEGIN"), 3)
        self.assertIn('"feature_order": 66', text)
        saved = next(r for r in self.ns["reports"] if r["section"] == "TEST.paging")
        self.assertEqual(saved["tables"]["records"], records)

    def test_integrity_reader_verifies_full_blob_encoding_and_corruption(self):
        blob = b'{"average_precision":0.123456789012345,"population_sha256":"' + b'a'*64 + b'"}'
        encoded = base64.b64encode(blob).decode()
        chunks = [encoded[i:i+13] for i in range(0, len(encoded), 13)]
        frame = pd.DataFrame([{"CHUNK_INDEX": i, "CHUNK_COUNT": len(chunks), "BYTE_LENGTH": len(blob),
                               "SHA256": hashlib.sha256(blob).hexdigest(), "PAYLOAD_BASE64": chunk}
                              for i, chunk in enumerate(chunks)])
        read = self.ns["decode_checked_artifact"]
        self.assertEqual(read(frame.sample(frac=1, random_state=2), 2000), blob)
        with self.assertRaises(ValueError):
            read(frame.iloc[:-1], 2000)
        broken = frame.copy()
        broken.loc[0, "SHA256"] = "0"*64
        with self.assertRaises(ValueError):
            read(broken, 2000)
        with self.assertRaises(ValueError):
            read(frame, 10)

    def test_report_reader_preserves_precision_and_excludes_patient_rows(self):
        extract = self.ns["extract_safe_report"]
        rows, _ = extract(b'{"metrics":{"average_precision":0.123456789012345,"pr_auc":0.234567890123456}}', "summary.json")
        fields = {r["field"]: r["value"] for r in rows}
        self.assertEqual(fields["metrics.average_precision"], 0.123456789012345)
        self.assertEqual(fields["metrics.pr_auc"], 0.234567890123456)
        rows, omitted = extract(b'[{"patient_id":"secret-person","average_precision":0.8}]', "summary.json")
        self.assertEqual(rows, [])
        self.assertGreater(omitted, 0)

    def test_manifest_reader_returns_declared_contract_without_certifying_execution(self):
        features = ["PRIVATE_FEATURE_ALPHA", "PRIVATE_FEATURE_BETA"]
        manifest = {
            "schema": 5,
            "dataset_id": "D20260927S",
            "source_vintage": "20260825",
            "features": features,
            "source_features": features,
            "excluded_features": [],
            "feature_policy": "v63_snapshot_only_valid_timestep_broadcast_v1",
            "population_sha256": "A" * 64,
            "source_tables": {"model_data": "DB.SCHEMA.MODEL_DATA"},
            "business_encoding": {
                "encoding_applied_here": False,
                "snapshot_MODEL_DATA_already_encoded": True,
                "upstream_fitting_uses_RESP": True,
                "parameter_sha256": "B" * 64,
            },
            "representations": {
                "MONTHLY": {
                    "shape": [23151, 12, 49],
                    "value_space": "V63 encoded",
                    "X_sha256": "C" * 64,
                    "coverage_audit": {
                        "available_timesteps": 200000,
                        "unavailable_timesteps": 77812,
                        "all_padded_snapshots": 37,
                    },
                }
            },
            "audit": [{"VALUE_SOURCE": "V63_SNAPSHOT"}, {"VALUE_SOURCE": "TEMPORAL"}],
        }
        rows, omitted = self.ns["extract_safe_report"](
            json.dumps(manifest).encode(), "manifest.json"
        )
        fields = {row["field"]: row["value"] for row in rows}
        self.assertEqual(omitted, 0)
        self.assertEqual(fields["schema"], 5)
        self.assertEqual(fields["dataset_id"], "D20260927S")
        self.assertEqual(fields["source_vintage_declared"], "20260825")
        self.assertEqual(fields["features_count_declared"], 2)
        feature_bytes = json.dumps(features, ensure_ascii=False, separators=(",", ":")).encode()
        self.assertEqual(fields["features_ordered_sha256"], hashlib.sha256(feature_bytes).hexdigest())
        self.assertEqual(fields["source_tables.model_data_declared"], "DB.SCHEMA.MODEL_DATA")
        self.assertEqual(fields["population_sha256"], "a" * 64)
        self.assertEqual(fields["business_encoding.parameter_sha256"], "b" * 64)
        self.assertIs(fields["business_encoding.upstream_fitting_uses_RESP_declared"], True)
        self.assertEqual(fields["representations.MONTHLY.shape_timesteps_declared"], 12)
        self.assertEqual(fields["representations.MONTHLY.coverage_audit.all_padded_snapshots_declared"], 37)
        self.assertEqual(fields["feature_audit.V63_SNAPSHOT_count_declared"], 1)
        self.assertEqual(fields["feature_audit.TEMPORAL_count_declared"], 1)
        self.assertEqual(fields["evidence_kind"], "SAVED_CONTRACT_DECLARATION_NOT_INDEPENDENT_VALIDATION")
        self.assertIs(fields["historical_execution_and_fit_provenance_verified"], False)

    def test_manifest_reader_omits_patient_records_feature_names_and_free_text(self):
        feature_name = "CONFIDENTIAL_FEATURE_SENTINEL"
        patient_value = "private-person-sentinel"
        free_text = "unreviewed historical explanation must not be exported"
        manifest = {
            "features": [feature_name],
            "feature_policy": free_text,
            "population": [{"PATIENT_ID": patient_value, "END_DT": "2024-02-01", "RESP": 1}],
            "notes": free_text,
            "business_encoding": {
                "parameters": [{"FEATURES": feature_name, "VALUE_P": 100}],
                "upstream_fit_population": free_text,
            },
            "source_tables": {"unexpected_source": free_text},
            "audit": [{"FEATURE_NAME": feature_name, "VALUE_SOURCE": "V63_SNAPSHOT", "SOURCE": free_text}],
            "representations": {
                "MONTHLY": {
                    "X": [[[9.123456789]]],
                    "valid": [[True]],
                    "provenance": [{"PATIENT_ID": patient_value, "description": free_text}],
                    "coverage_audit": {"evidence": free_text},
                }
            },
        }
        rows, omitted = self.ns["extract_safe_report"](
            json.dumps(manifest).encode(), "manifest.json"
        )
        serialized = json.dumps(rows)
        for prohibited in (feature_name, patient_value, free_text, "9.123456789", "PATIENT_ID", "END_DT"):
            self.assertNotIn(prohibited, serialized)
        fields = {row["field"]: row["value"] for row in rows}
        self.assertEqual(fields["features_count_declared"], 1)
        self.assertIs(fields["feature_policy_recognized"], False)
        self.assertEqual(fields["feature_policy_unrecognized_text_sha256"], hashlib.sha256(free_text.encode()).hexdigest())
        self.assertIs(fields["patient_arrays_and_free_text_exported"], False)
        self.assertGreater(omitted, 0)

    def test_manifest_reader_does_not_accept_boolean_counts_or_dimensions(self):
        manifest = {
            "schema": True,
            "business_encoding": {"encoding_applied_here": False},
            "representations": {
                "MONTHLY": {
                    "shape": [True, 12, 49],
                    "coverage_audit": {
                        "available_timesteps": True,
                        "unavailable_timesteps": False,
                        "unknown_timesteps": 0,
                        "all_padded_snapshots": True,
                        "closed_claims_filter_applied": True,
                    },
                }
            },
        }
        rows, _ = self.ns["extract_safe_report"](json.dumps(manifest).encode(), "manifest.json")
        fields = {row["field"]: row["value"] for row in rows}
        self.assertNotIn("schema", fields)
        self.assertFalse(any("shape_" in field for field in fields))
        for field in ("available_timesteps", "unavailable_timesteps", "all_padded_snapshots"):
            self.assertNotIn("representations.MONTHLY.coverage_audit." + field + "_declared", fields)
        self.assertEqual(fields["representations.MONTHLY.coverage_audit.unknown_timesteps_declared"], 0)
        self.assertIs(fields["representations.MONTHLY.coverage_audit.closed_claims_filter_applied_declared"], True)
        self.assertIs(fields["business_encoding.encoding_applied_here_declared"], False)

    def test_integrity_reader_rejects_boolean_chunk_metadata(self):
        blob = b"1"
        baseline = {"CHUNK_INDEX": 0, "CHUNK_COUNT": 1, "BYTE_LENGTH": 1,
                    "SHA256": hashlib.sha256(blob).hexdigest(),
                    "PAYLOAD_BASE64": base64.b64encode(blob).decode()}
        for field in ("CHUNK_INDEX", "CHUNK_COUNT", "BYTE_LENGTH"):
            for boolean_type in (bool, self.ns["np"].bool_):
                with self.subTest(field=field, boolean_type=boolean_type.__name__):
                    row = {**baseline, field: boolean_type(baseline[field])}
                    frame = pd.DataFrame([row], dtype=object)
                    with self.assertRaises(ValueError):
                        self.ns["decode_checked_artifact"](frame, 100)

    def test_checklist_includes_underscore_sections(self):
        checklist = next(r for r in self.ns["reports"] if r["section"] == "E13")
        sections = [r["section"] for r in checklist["tables"]["records"]]
        self.assertIn("E04_POPULATION", sections)
        self.assertIn("E05B_TENSOR_VALUES", sections)


if __name__ == "__main__":
    unittest.main()
