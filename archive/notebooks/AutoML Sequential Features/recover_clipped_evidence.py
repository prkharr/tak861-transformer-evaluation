"""Run as a cell in the existing approved collector session; default is zero queries.

Only sanitized aggregate fields are printed. Do not print private_errors,
private_procedure_sources, credentials, patient data or raw artifact payloads.
The optional checks reuse the collector's bounded run_query helper.
"""
import json
import math

RUN_BOUNDED_WAREHOUSE_CHECKS = False
PAGE_SIZE = 20
REPORT_INDEX = 0
FIELD_OFFSET = 0
PRINT_REPORT_INDEX = False  # Optional compact index; never reprints every report at once.
SOURCE = "DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_"


def _safe_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _safe_value(value.item())
    return str(value)


def _emit_small(record):
    print(json.dumps(record, ensure_ascii=True, default=_safe_value, allow_nan=False))


def reprint_existing_reports():
    stored = globals().get("recovered_safe_reports")
    if not isinstance(stored, list):
        _emit_small({"recovery": "UNAVAILABLE", "reason": "No recovered_safe_reports in this kernel; no queries issued."})
        return
    allowed_names = {"training_summary.json", "training_history.csv", "summary.json", "history.json",
                     "evaluation_metadata.json", "evaluation.json", "global_metrics.csv",
                     "topk.csv", "deciles.csv", "ties.csv", "selection.json", "comparison.csv"}
    selected = [r for r in stored if isinstance(r, dict) and r.get("artifact") in allowed_names
                and str(r.get("source", "")).startswith(SOURCE + "DL_POC_")]
    def priority(report):
        source, artifact = report.get("source", ""), report.get("artifact", "")
        first = {"training_summary.json": 0, "training_history.csv": 1,
                 "global_metrics.csv": 2, "evaluation_metadata.json": 3}
        return (0 if source.endswith(("_MODEL_RUN_001", "_EVAL_RUN_001")) else 1,
                first.get(artifact, 4), source, artifact)
    selected.sort(key=priority)
    _emit_small({"recovery": "EXISTING_SANITIZED_MEMORY", "reports": len(selected), "new_queries": 0,
                 "report_index": REPORT_INDEX, "field_offset": FIELD_OFFSET})
    if PRINT_REPORT_INDEX:
        _emit_small({"report_index": [{"index": i, "source": r.get("source"), "artifact": r.get("artifact"),
                                       "fields": len(r.get("fields", []))} for i, r in enumerate(selected)]})
    if not selected:
        return
    if not isinstance(REPORT_INDEX, int) or not 0 <= REPORT_INDEX < len(selected):
        raise ValueError("REPORT_INDEX outside the available report range")
    if not isinstance(FIELD_OFFSET, int) or FIELD_OFFSET < 0:
        raise ValueError("FIELD_OFFSET must be a nonnegative integer")
    # Exactly ONE report page per execution avoids the original cell clipping.
    for report in [selected[REPORT_INDEX]]:
        fields = report.get("fields", [])
        if not isinstance(fields, list):
            continue
        # These are already allowlisted scalar fields, never raw JSON/CSV records.
        safe = [{"field": row["field"], "value": _safe_value(row.get("value"))}
                for row in fields if isinstance(row, dict) and isinstance(row.get("field"), str)
                and not isinstance(row.get("value"), (dict, list, tuple))]
        meta = {k: report.get(k) for k in ("source", "artifact", "sha256", "integrity_verified", "omitted_fields_or_records")}
        next_offset = FIELD_OFFSET + PAGE_SIZE
        _emit_small({**meta, "recovery_offset": FIELD_OFFSET, "total_recovered_fields": len(safe),
                     "fields": safe[FIELD_OFFSET:next_offset], "comparison_validity": "NOT_ESTABLISHED",
                     "next_field_offset": next_offset if next_offset < len(safe) else None,
                     "next_report_index": REPORT_INDEX + 1 if next_offset >= len(safe) and REPORT_INDEX + 1 < len(selected) else None})


def _quote_physical(name):
    # Names originate only from this exact source's INFORMATION_SCHEMA metadata.
    return '"' + str(name).replace('"', '""') + '"'


def bounded_existing_metrics():
    table = SOURCE + "MODEL_COMPARISON"
    schema = run_query("SELECT COLUMN_NAME FROM DSVC_TAKEDA_TA_PRIVATE.INFORMATION_SCHEMA.COLUMNS "
                       "WHERE TABLE_SCHEMA='DS_ML' AND TABLE_NAME='TAK861_TX_READY_V63_MODEL_COMPARISON' "
                       "ORDER BY ORDINAL_POSITION", max_rows=100)
    mapping = {"MODEL": "MODEL", "MODEL_NAME": "MODEL", "ACCURACY": "ACCURACY", "AUC": "AUC",
               "ROC_AUC": "ROC_AUC", "RECALL": "RECALL", "PREC.": "PRECISION",
               "PRECISION": "PRECISION", "F1": "F1", "KAPPA": "KAPPA", "MCC": "MCC",
               "TT (SEC)": "TRAINING_SECONDS", "AP": "AP", "AVERAGE_PRECISION": "AVERAGE_PRECISION",
               "PR_AUC": "PR_AUC"}
    physical = schema["COLUMN_NAME"].astype(str).tolist()
    if any(x.upper() in {"PATIENT_ID", "END_DT", "RESP"} for x in physical):
        raise ValueError("Refuse metric rows: source includes patient-level column names.")
    chosen = [(name, mapping[name.upper()]) for name in physical if name.upper() in mapping]
    if not chosen or len({alias for _, alias in chosen}) != len(chosen):
        raise ValueError("No safe unique aggregate metric schema recognized.")
    sql = "SELECT " + ",".join(_quote_physical(n) + " AS " + _quote_physical(a) for n, a in chosen)
    sql += ' FROM "DSVC_TAKEDA_TA_PRIVATE"."DS_ML"."TAK861_TX_READY_V63_MODEL_COMPARISON"'
    frame = run_query(sql, max_rows=100)
    records = []
    for row in frame.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if key == "MODEL":
                # No arbitrary free text beyond model labels.
                import re
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_ .%()/+-]{0,99}", value):
                    clean[key] = value
            else:
                try:
                    number = float(value)
                    clean[key] = number if math.isfinite(number) else None
                except (TypeError, ValueError):
                    clean[key] = None
        records.append(clean)
    _emit_small({"source": table, "recovery": "CASE_PRESERVING_EXISTING_METRICS", "records": records,
                 "fit_membership": "UNKNOWN", "metric_definitions": "UNVERIFIED"})


def reconcile_backtest_once():
    # One source scan computes both raw and parsed date bounds under one query.
    # No individual identifiers, dates per patient, scores or labels are returned.
    sql = '''SELECT COUNT(*) AS N_ROWS, COUNT(DISTINCT PATIENT_ID) AS PATIENTS,
      COUNT(DISTINCT PATIENT_ID, END_DT) AS SNAPSHOT_KEYS,
      COUNT_IF(PATIENT_ID IS NULL OR END_DT IS NULL) AS NULL_KEYS,
      COUNT_IF(RESP=1) AS POSITIVES,
      MIN(END_DT) AS RAW_MIN_END_DT, MAX(END_DT) AS RAW_MAX_END_DT,
      MIN(TRY_TO_DATE(TO_VARCHAR(END_DT))) AS PARSED_MIN_END_DT,
      MAX(TRY_TO_DATE(TO_VARCHAR(END_DT))) AS PARSED_MAX_END_DT,
      COUNT_IF(END_DT IS NOT NULL AND TRY_TO_DATE(TO_VARCHAR(END_DT)) IS NULL) AS UNPARSEABLE_DATES
      FROM "DSVC_TAKEDA_TA_PRIVATE"."DS_ML"."TAK861_TX_READY_V63_BACK_TESTING_SCORED"'''
    frame = run_query(sql, max_rows=1)
    _emit_small({"source": SOURCE + "BACK_TESTING_SCORED", "recovery": "SINGLE_QUERY_COHORT_RECONCILIATION",
                 "records": frame.to_dict("records"), "untouched_holdout_verified": False,
                 "note": "Current observation does not erase conflicting historical receipts."})


reprint_existing_reports()
if RUN_BOUNDED_WAREHOUSE_CHECKS:
    if not callable(globals().get("run_query")):
        raise RuntimeError("Run only in the existing authorized collector session with run_query available.")
    # Three bounded read-only statements total: schema, existing metrics, aggregate reconciliation.
    bounded_existing_metrics()
    reconcile_backtest_once()
