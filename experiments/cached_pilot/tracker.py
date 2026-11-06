"""Bounded aggregate evidence ledger; never trains or reads prediction arrays."""
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


MAX_REPORT_BYTES = 8 * 1024 * 1024
DISPLAY_COLUMNS = ["Experiment", "What changed", "Population/split", "Lift@10%", "Matched delta@10%",
                   "Additional captures@10%", "Conclusion", "Remaining limitation", "Evidence", "Status", "Source"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("A metric must be finite or explicitly missing")
    return float(value)


def _json(blob):
    if len(blob) > MAX_REPORT_BYTES:
        raise ValueError("Aggregate report exceeds the configured byte bound")
    result = json.loads(blob.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(result, dict):
        raise ValueError("An aggregate report must be an object")
    return result


def load_local_report(path):
    path = Path(path)
    if path.is_dir():
        path = path / "report.json"
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Aggregate report exceeds the configured byte bound")
    blob = path.read_bytes()
    receipt_path = path.with_name("report_receipt.json")
    if receipt_path.stat().st_size > 256 * 1024:
        raise ValueError("Oversized report receipt")
    receipt = _json(receipt_path.read_bytes())
    expected = receipt.get("sha256")
    if isinstance(expected, dict):
        expected = expected.get("report.json")
    if expected is None:
        expected = receipt.get("hashes", {}).get("report.json")
    actual = hashlib.sha256(blob).hexdigest()
    if expected != actual:
        raise ValueError("Aggregate report receipt does not match; retry after the active writer commits")
    return _json(blob), {"source": str(path), "sha256": actual, "integrity_verified": True}


def load_remote_report(read_query, table, decoder):
    """Read only the bounded report.json artifact with the existing checked decoder."""
    if not isinstance(table, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9$]*\.[A-Za-z_][A-Za-z_0-9$]*\.[A-Za-z_][A-Za-z_0-9$]*", table):
        raise ValueError("Use one explicit fully qualified artifact table")
    quoted = ".".join('"' + part.upper() + '"' for part in table.split("."))
    inventory = read_query(f"SELECT MIN(BYTE_LENGTH) AS MIN_BYTES, MAX(BYTE_LENGTH) AS MAX_BYTES, "
                           f"MAX(CHUNK_COUNT) AS MAX_CHUNKS, COUNT(*) AS STORED_CHUNKS, "
                           f"SUM(LENGTH(PAYLOAD_BASE64)) AS ENCODED_BYTES FROM {quoted} "
                           "WHERE ARTIFACT_NAME = 'report.json'", 1)
    if len(inventory) != 1:
        raise ValueError("Missing aggregate report inventory")
    item = inventory.iloc[0]
    if not (0 <= int(item.MIN_BYTES) <= int(item.MAX_BYTES) <= MAX_REPORT_BYTES):
        raise ValueError("Missing or oversized report")
    chunks = int(item.MAX_CHUNKS)
    if not 1 <= chunks <= 10000 or int(item.STORED_CHUNKS) != chunks:
        raise ValueError("Invalid report chunk inventory")
    if int(item.ENCODED_BYTES) > 4 * ((MAX_REPORT_BYTES + 2) // 3):
        raise ValueError("Oversized encoded report")
    cols = "CHUNK_INDEX,CHUNK_COUNT,BYTE_LENGTH,SHA256,PAYLOAD_BASE64"
    frame = read_query(f"SELECT {cols} FROM {quoted} WHERE ARTIFACT_NAME = 'report.json' ORDER BY CHUNK_INDEX", chunks)
    blob = decoder(frame, MAX_REPORT_BYTES)
    return _json(blob), {"source": table + "/report.json", "sha256": hashlib.sha256(blob).hexdigest(),
                        "integrity_verified": True}


def interval_conclusion(delta, lower, upper):
    if delta is None:
        return "No matched result available"
    if lower is None or upper is None:
        return "Observed difference only; paired uncertainty unavailable"
    if lower > upper or not lower <= delta <= upper:
        # A percentile bootstrap need not contain the point estimate.
        if lower > upper:
            raise ValueError("Reversed interval")
    if lower > 0:
        return "Higher observed development lift; interval excludes zero"
    if upper < 0:
        return "Lower observed development lift; interval excludes zero"
    return "Ordering unresolved; interval includes zero (not equivalence)"


def _row(experiment, changed, population, lift, source, **extra):
    row = {"Experiment": experiment, "What changed": changed, "Population/split": population,
           "Lift@10%": number(lift), "Matched delta@10%": None, "Additional captures@10%": None,
           "Conclusion": "Result reported; no matched contrast established", "Remaining limitation": "",
           "Evidence": "SAVED_AGGREGATE_INTEGRITY_VERIFIED", "Status": "COMPLETED_REPORTED",
           "Source": source, "comparison_block": None, "split": None, "metric": "Lift@10%",
           "lower": None, "upper": None, "train_snapshots": None, "validation_identity": None,
           "protocol_identity": None, "version": None, "arm": None}
    row.update(extra)
    return row


def pilot_rows(report, receipt):
    if report.get("schema") != "cached_pilot_report_v1" or not receipt.get("integrity_verified"):
        raise ValueError("A checked pilot report is required")
    identity = report["identity"]
    for name in ("evaluation_keys_sha256", "evaluation_labels_sha256", "preparation_hash"):
        if not re.fullmatch("[0-9a-f]{64}", str(identity.get(name, ""))):
            raise ValueError("Missing pilot evaluation/preparation identity: " + name)
    evaluation = digest({k: identity[k] for k in ("evaluation_keys_sha256", "evaluation_labels_sha256")})
    definition_hash = identity.get("feature_definition_hash")
    implementation_protocol = identity.get("protocol_hash")
    protocol = (digest({"settings": identity["settings"], "feature_names": identity["feature_names"],
                        "feature_definition_hash": definition_hash, "protocol_hash": implementation_protocol})
                if all(re.fullmatch("[0-9a-f]{64}", str(value)) for value in (definition_hash, implementation_protocol)) else None)
    grouped = {}
    for entry in report["rows"]:
        split = str(entry["Split"]).upper()
        if split not in {"TRAIN", "VALIDATION"}:
            raise ValueError("Pilot reports cannot expose TEST")
        key = (split, str(entry["Model"]))
        if key in grouped:
            raise ValueError("Duplicate pilot result row")
        grouped[key] = entry
    results = []
    for (split, arm), entry in grouped.items():
        n, positives = int(entry["Snapshots"]), int(entry["Positive_snapshots"])
        if n <= 0 or not 0 < positives < n:
            raise ValueError("Pilot result requires both classes")
        lift = number(entry["Lift@10%"])
        captured, selected = int(entry["Captured@10%"]), int(entry["Selected@10%"])
        if selected != math.ceil(.1 * n) or not 0 <= captured <= min(positives, selected):
            raise ValueError("Pilot top-decile counts are inconsistent")
        if lift is None or not math.isclose(lift, captured / selected / (positives / n), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("Pilot lift does not match its aggregate counts")
        row = _row("Pilot " + str(identity.get("version", "")) + " / " + arm, arm,
                   f"{n:,} snapshots / {int(entry['Patients']):,} patients / {positives:,} positives / {split}",
                   lift, receipt["source"], split=split,
                   comparison_block="pilot:" + receipt["sha256"] + ":" + split,
                   validation_identity=evaluation if split == "VALIDATION" else None,
                   protocol_identity=protocol, train_snapshots=int(identity["train_snapshots"]),
                   version=identity.get("version"), arm=arm, artifact_sha256=receipt["sha256"],
                   **{"Evidence": "SAVED_PILOT_AGGREGATE_INTEGRITY_VERIFIED",
                      "Remaining limitation": "Convenience pilot; repeated development validation; no out-of-time claim. "
                                              "Integrity checks do not establish historical source availability.",
                      "Conclusion": "Baseline" if arm == "AUTOML" else "Matched point difference pending"})
        baseline = grouped.get((split, "AUTOML"))
        if baseline is not None:
            if any(entry[k] != baseline[k] for k in ("Snapshots", "Patients", "Positive_snapshots", "Selected@10%")):
                raise ValueError("Pilot arms have different evaluation populations")
            row["Matched delta@10%"] = lift - float(baseline["Lift@10%"])
            row["Additional captures@10%"] = captured - int(baseline["Captured@10%"])
        pair = report.get("paired_lift", {}).get(arm, {}) if split == "VALIDATION" else {}
        interval = pair.get("lift_by_fraction", {}).get("0.1", {})
        if interval:
            if pair.get("paired") is not True or pair.get("unit") != "patient_cluster":
                raise ValueError("Expected paired patient-cluster uncertainty")
            delta = number(interval["delta"])
            if row["Matched delta@10%"] is None or not math.isclose(delta, row["Matched delta@10%"], abs_tol=1e-8):
                raise ValueError("Paired interval direction differs from candidate minus baseline")
            lower = number(interval.get("lower")) if pair.get("interval_reliable") else None
            upper = number(interval.get("upper")) if pair.get("interval_reliable") else None
            row.update(lower=lower, upper=upper)
            row["Conclusion"] = interval_conclusion(delta, lower, upper)
        elif arm != "AUTOML":
            row["Conclusion"] = interval_conclusion(row["Matched delta@10%"], None, None)
        results.append(row)
    return results


def saved_rows(report, receipt):
    if report.get("schema") == "cached_pilot_report_v1":
        return pilot_rows(report, receipt)
    if not receipt.get("integrity_verified"):
        raise ValueError("Saved report integrity must be verified")
    # Existing timing experiment reports are aggregate rows plus a paired-lift bundle.
    if "rows" in report and "identity" in report and "scope" in report:
        out = []
        for entry in report["rows"]:
            if not {"Model", "Lift@10%", "Snapshots", "Split"} <= set(entry):
                raise ValueError("Unsupported saved result row; add an explicit schema adapter")
            out.append(_row(str(entry["Model"]), "Saved aggregate feature comparison",
                       f"{int(entry['Snapshots']):,} snapshots / {entry['Split']}", entry["Lift@10%"], receipt["source"],
                       comparison_block=None, split=str(entry["Split"]).upper(),
                       **{"Remaining limitation": str(report["scope"]),
                          "Conclusion": "Saved reported result; comparison identity requires its protocol receipt"}))
        return out
    # Existing production comparison reports encode metric dictionaries, not arrays.
    if {"metrics", "snapshots", "split", "evaluation_scope", "prediction_identity_sha256"} <= set(report):
        out = []
        for model, metric in report["metrics"].items():
            top = {float(x["fraction"]): x["lift"] for x in metric.get("topk", [])}
            if .1 not in top:
                continue
            out.append(_row(model, "Saved comparison on aligned prediction keys",
                       f"{int(report['snapshots']):,} snapshots / {str(report['split']).upper()}", top[.1], receipt["source"],
                       comparison_block="saved:" + receipt["sha256"], split=str(report["split"]).upper(),
                       **{"Remaining limitation": "; ".join(report.get("limitations", [])),
                          "Conclusion": report.get("benchmark", {}).get("statement", str(report["evaluation_scope"]))}))
        return out
    raise ValueError("Unsupported aggregate report schema; no result was inferred")


def ledger(seed, reports=()):
    if seed.get("schema") != "experiment_evidence_seed_v1":
        raise ValueError("Unsupported evidence seed")
    rows = []
    for entry in seed["experiments"]:
        row = _row(entry["experiment"], entry["changed"], entry["population"], entry.get("lift10"), entry["source"],
                   comparison_block=entry.get("comparison_block"), split=entry.get("split"),
                   metric=entry.get("metric", "Lift@10%"),
                   **{"Evidence": entry["evidence"], "Status": entry["status"], "Conclusion": entry["conclusion"],
                      "Remaining limitation": entry["limitation"], "precision": entry.get("precision", "unavailable")})
        if entry.get("delta10") is not None:
            if not entry.get("comparison_block"):
                raise ValueError("A contrast requires a declared comparison block")
            row["Matched delta@10%"] = number(entry["delta10"])
        row["lower"], row["upper"] = number(entry.get("lower")), number(entry.get("upper"))
        rows.append(row)
    seen = set()
    errors = []
    for report, receipt in reports:
        sha = receipt.get("sha256")
        if sha in seen:
            continue
        try:
            rows.extend(saved_rows(report, receipt))
            seen.add(sha)
        except (KeyError, ValueError, TypeError) as exc:
            errors.append({"Source": receipt.get("source"), "Status": "BLOCKED", "Reason": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(seed.get("engineering", [])), pd.DataFrame(errors)


def chart_groups(frame):
    groups = []
    for key, group in frame.dropna(subset=["comparison_block", "Lift@10%"]).groupby(["comparison_block", "split", "metric"], dropna=False, sort=False):
        if len(group) >= 2:
            groups.append((key, group.copy()))
    return groups


def progress_groups(frame):
    selected = frame[(frame["split"] == "VALIDATION") & frame["validation_identity"].notna() & frame["protocol_identity"].notna()]
    result = []
    for key, group in selected.groupby(["validation_identity", "protocol_identity", "arm"], sort=False):
        group = group.drop_duplicates(["version", "train_snapshots"]).sort_values("train_snapshots", kind="stable")
        if len(group) >= 2:
            result.append((key, group))
    return result


def plot_matched(frame):
    import matplotlib.pyplot as plt
    figures = []
    for (block, split, metric), group in chart_groups(frame):
        fig, ax = plt.subplots(figsize=(10, max(2.6, len(group) * .48)))
        labels = group["Experiment"].tolist()
        ax.barh(labels, group["Lift@10%"], color="#007F86")
        ax.invert_yaxis()
        ax.set_xlabel(metric + " (higher is better)")
        ax.set_title(str(block)[:75] + " | " + str(split), loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(group["Lift@10%"]):
            ax.text(value, i, f"  {value:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        figures.append(fig)
    return figures


def plot_progress(frame):
    import matplotlib.pyplot as plt
    figures = []
    for (_, _, arm), group in progress_groups(frame):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.plot(group["train_snapshots"], group["Lift@10%"], marker="o", color="#007F86")
        ax.set(xlabel="TRAIN snapshots", ylabel="Validation Lift@10%", title=arm + " | fixed validation panel and feature definitions")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        figures.append(fig)
    return figures
