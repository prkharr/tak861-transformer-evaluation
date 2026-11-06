"""Build the standalone aggregate tracker with curated evidence and checked decoder."""
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "expanded cohort" / "08_experiment_tracker_AutoML.ipynb"


def build():
    source = (HERE / "tracker.py").read_text(encoding="utf-8")
    seed_text = (HERE / "evidence_seed.json").read_text(encoding="utf-8")
    existing = json.loads((ROOT / "notebooks" / "Project Evidence" / "00_collect_missing_inputs.ipynb").read_text(encoding="utf-8-sig"))
    joined = "\n".join("".join(c["source"]) for c in existing["cells"] if c["cell_type"] == "code")
    node = next(n for n in ast.parse(joined).body if isinstance(n, ast.FunctionDef) and n.name == "decode_checked_artifact")
    decoder = ast.get_source_segment(joined, node)
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in {"tracker": source, "evidence": seed_text, "decoder": decoder}.items()}
    cells = []
    def cell(text):
        ast.parse(text)
        cells.append({"id": "08-" + str(len(cells) + 1), "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)})
    cell('''import base64
import hashlib
import json
import re
import types
from pathlib import Path
import numpy as np
import pandas as pd

sf_options = {"sfUrl": "", "sfUser": "", "sfPassword": "", "sfRole": "", "sfWarehouse": "", "sfDatabase": "", "sfSchema": ""}
LOCAL_REPORT_PATHS = []
REMOTE_REPORT_TABLES = []
OUTPUT_DIRECTORY = ""
MAX_REPORTS = 100
print("Aggregate experiment tracker: no training, prediction arrays or raw-claim reads.")
print("Historic results remain reported evidence. Missing outcomes remain unavailable.")
''')
    cell("TRACKER_SOURCE = " + repr(source) + "\n" +
         "DECODER_SOURCE = " + repr(decoder) + "\n" +
         "EVIDENCE_SOURCE = " + repr(seed_text) + "\n" +
         "EXPECTED_HASHES = " + repr(hashes) + "\n" +
         '''for name, text in [("tracker", TRACKER_SOURCE), ("decoder", DECODER_SOURCE), ("evidence", EVIDENCE_SOURCE)]:
    if hashlib.sha256(text.encode()).hexdigest() != EXPECTED_HASHES[name]:
        raise ValueError("Embedded component checksum mismatch: " + name)
tracker = types.ModuleType("aggregate_evidence_tracker")
exec(compile(TRACKER_SOURCE, "aggregate_evidence_tracker", "exec"), tracker.__dict__)
exec(compile(DECODER_SOURCE, "checked_aggregate_decoder", "exec"), globals())
EVIDENCE = json.loads(EVIDENCE_SOURCE)
print("Embedded aggregate reader and evidence checksums verified.")
''')
    cell('''reports, ingest_errors = [], []
if len(LOCAL_REPORT_PATHS) + len(REMOTE_REPORT_TABLES) > MAX_REPORTS:
    raise ValueError("Too many explicit reports for this bounded review")
for location in LOCAL_REPORT_PATHS:
    try:
        reports.append(tracker.load_local_report(location))
    except Exception as exc:
        ingest_errors.append({"Source": str(location), "Status": "BLOCKED", "Reason": str(exc)})
if REMOTE_REPORT_TABLES:
    if not all(sf_options.values()):
        raise ValueError("Fill the connection settings before requesting remote aggregate reports")
    if "spark" not in globals():
        raise RuntimeError("The remote aggregate reader requires the existing Spark session")
    def read_aggregate_query(query, maximum_rows):
        frame = spark.read.format("snowflake").options(**sf_options).option("query", query).load().limit(maximum_rows + 1).toPandas()
        frame.columns = [str(c).upper() for c in frame.columns]
        if len(frame) > maximum_rows:
            raise ValueError("Aggregate query exceeded its row bound")
        return frame
    for table in REMOTE_REPORT_TABLES:
        try:
            reports.append(tracker.load_remote_report(read_aggregate_query, table, decode_checked_artifact))
        except Exception as exc:
            ingest_errors.append({"Source": str(table), "Status": "BLOCKED", "Reason": str(exc)})
print("Checked aggregate reports:", len(reports))
''')
    cell('''evidence_table, engineering_table, parse_errors = tracker.ledger(EVIDENCE, reports)
blocked_table = pd.concat([pd.DataFrame(ingest_errors), parse_errors], ignore_index=True)
display(evidence_table[tracker.DISPLAY_COLUMNS])
print("Feature subgroup association and model Lift@10% are different measurements.")
print("A verified file checksum establishes integrity, not an unbiased evaluation.")
if not blocked_table.empty:
    display(blocked_table)
print("Implemented or pending experiments:")
display(evidence_table[evidence_table["Lift@10%"].isna()][tracker.DISPLAY_COLUMNS])
print("Engineering changes are listed separately from predictive improvements:")
display(engineering_table)
''')
    cell('''import matplotlib.pyplot as plt
for figure in tracker.plot_matched(evidence_table):
    display(figure)
    plt.close(figure)
progress = tracker.plot_progress(evidence_table)
if not progress:
    print("No refresh series with identical validation keys, labels and feature definitions is available yet.")
for figure in progress:
    display(figure)
    plt.close(figure)
print("Comparisons are grouped by recorded population, split and metric. No overall ranking is produced.")
''')
    cell('''if OUTPUT_DIRECTORY:
    destination = Path(OUTPUT_DIRECTORY)
    destination.mkdir(parents=True, exist_ok=True)
    evidence_table.to_csv(destination / "experiment_evidence.csv", index=False)
    engineering_table.to_csv(destination / "engineering_evidence.csv", index=False)
    blocked_table.to_csv(destination / "blocked_reports.csv", index=False)
    payload = {"schema": "experiment_tracker_export_v1", "embedded_hashes": EXPECTED_HASHES,
               "report_receipts": [receipt for _, receipt in reports],
               "experiments": json.loads(evidence_table.to_json(orient="records")),
               "engineering": json.loads(engineering_table.to_json(orient="records"))}
    blob = json.dumps(payload, sort_keys=True, allow_nan=False, indent=2).encode()
    (destination / "tracker.json").write_bytes(blob)
    (destination / "tracker_receipt.json").write_text(json.dumps({"sha256": hashlib.sha256(blob).hexdigest()}), encoding="utf-8")
    print("Aggregate tracker saved:", str(destination))
else:
    print("Display complete. Set OUTPUT_DIRECTORY to save this aggregate ledger.")
''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, "embedded_sha256": hashes}, "nbformat": 4, "nbformat_minor": 5}
    raw = json.dumps(notebook, ensure_ascii=False, indent=1)
    import re
    if re.search(r"\b(?:codex|claude|github|git|client|brian)\b", raw, re.I):
        raise ValueError("Disallowed presentation reference")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(raw + "\n", encoding="utf-8", newline="\n")
    return hashes


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
