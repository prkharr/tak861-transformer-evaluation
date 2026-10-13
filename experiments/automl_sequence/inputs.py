"""Historical AutoML input construction. Never derives history from snapshot values.

Only extraction/assembly live here. Every learned transform belongs to TRAIN-only
preparation. Cached panels are reused by exact patient/date and checked definition
and content fingerprints. This module does not assert unavailable source lineage.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Callable

import numpy as np
import pandas as pd


KEYS = ["PATIENT_ID", "AS_OF_DATE"]
RESERVED_COLUMNS = frozenset({
    "PATIENT_ID", "AS_OF_DATE", "END_DT", "RESP", "SPLIT", "TARGET", "LABEL",
    "OUTCOME", "SCORE", "PREDICTION", "IS_OBSERVED", "MAX_EVENT_DATE",
    "MAX_AVAILABLE_DATE", "OBSERVATION_EVIDENCE", "TIME_STEP", "SNAPSHOT_ROW",
})
DATE_POLICIES = ("same_day_months", "month_end_history")
RECOVERY_QUERY = """SELECT PROCEDURE_CATALOG, PROCEDURE_SCHEMA, PROCEDURE_NAME,
ARGUMENT_SIGNATURE, CREATED, LAST_ALTERED, PROCEDURE_DEFINITION
FROM DSVC_TAKEDA_TA_PRIVATE.INFORMATION_SCHEMA.PROCEDURES
WHERE PROCEDURE_SCHEMA IN ('DS_ML_DEV', 'DS_ML', 'DS_ML_PRO', 'DS_ML_PRD')
AND PROCEDURE_NAME IN ('FEATURE_CREATION_PROD', 'FEATURE_CREATION_PROC', 'FEATURE_ENGINEERING_PROD')
ORDER BY PROCEDURE_NAME, ARGUMENT_SIGNATURE"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False, default=str).encode()).hexdigest()


def _dates(values, label):
    result = pd.to_datetime(values, errors="raise")
    _require(result.dt.tz is None, label + ": timezone conversion must be explicit")
    _require(result.notna().all(), label + ": missing dates")
    _require(result.eq(result.dt.normalize()).all(), label + ": expected exact dates, not intraday timestamps")
    return result


def _patients(values):
    _require(values.notna().all(), "Missing patient keys")
    # Preserve identifiers verbatim: do not coerce to a numeric type or drop leading zeros.
    out = values.map(str)
    _require(out.str.len().gt(0).all(), "Empty patient keys")
    return out


def normalize_snapshots(metadata):
    """Keep input order exactly; reject duplicate snapshots and cross-split patients."""
    needed = ["PATIENT_ID", "END_DT", "RESP", "SPLIT"]
    _require(isinstance(metadata, pd.DataFrame) and not metadata.columns.duplicated().any(),
             "Snapshot metadata must be a DataFrame with unique columns")
    _require(set(needed).issubset(metadata.columns), "Metadata requires PATIENT_ID, END_DT, RESP, SPLIT")
    frame = metadata[needed].copy().reset_index(drop=True)
    _require(len(frame) > 0, "No snapshots")
    frame["PATIENT_ID"] = _patients(frame.PATIENT_ID)
    frame["END_DT"] = _dates(frame.END_DT, "END_DT")
    _require(not frame.duplicated(["PATIENT_ID", "END_DT"]).any(), "Duplicate snapshot keys")
    _require(frame.RESP.notna().all() and frame.RESP.isin([0, 1]).all(), "RESP must be binary without missing labels")
    frame["RESP"] = frame.RESP.astype(np.int64)
    _require(frame.SPLIT.notna().all(), "Missing split labels")
    frame["SPLIT"] = frame.SPLIT.astype(str).str.lower().replace({"val": "validation", "valid": "validation"})
    _require(frame.SPLIT.isin(["train", "validation", "test"]).all(), "Unknown split label")
    _require(frame.groupby("PATIENT_ID").SPLIT.nunique().eq(1).all(), "Patient appears in multiple splits")
    return frame


def train_patient_hash(metadata):
    """The canonical scope fingerprint for any upstream TRAIN-fitted encoding."""
    meta = normalize_snapshots(metadata)
    return stable_hash(sorted(meta.loc[meta.SPLIT.eq("train"), "PATIENT_ID"].unique().tolist()))


def verify_fitting_scope(contract, metadata):
    contract = validate_contract(contract)
    if contract["fit_scope"] == "train_only_fitted":
        _require(contract["fitting_patients_sha256"] == train_patient_hash(metadata),
                 "Upstream learned feature fitting patients differ from current TRAIN patients")
    return contract


def make_requests(metadata, steps=12, date_policy="same_day_months"):
    """Return deduplicated extraction keys and a row-aligned snapshot/date grid.

    TIME_STEP zero is newest, preserving the source tensor convention. Older
    dates are independently derived from the original END_DT, not iterated from
    previously clamped dates. No label or split is sent to the extractor.
    """
    _require(isinstance(steps, int) and 1 <= steps <= 120, "Invalid number of steps")
    _require(date_policy in DATE_POLICIES, "Unknown historical date policy")
    meta = normalize_snapshots(metadata)
    parts = []
    for step in range(steps):
        part = meta[["PATIENT_ID", "END_DT"]].copy()
        part["SNAPSHOT_ROW"] = np.arange(len(meta), dtype=np.int64)
        part["TIME_STEP"] = step
        if step == 0:
            part["AS_OF_DATE"] = part.END_DT
        elif date_policy == "same_day_months":
            part["AS_OF_DATE"] = part.END_DT.map(lambda date: date - pd.DateOffset(months=step))
        else:
            part["AS_OF_DATE"] = (part.END_DT.dt.to_period("M") - step).dt.to_timestamp(how="end").dt.normalize()
        parts.append(part)
    grid = pd.concat(parts, ignore_index=True).sort_values(["SNAPSHOT_ROW", "TIME_STEP"]).reset_index(drop=True)
    _require(grid.AS_OF_DATE.le(grid.END_DT).all(), "Historical request exceeds snapshot cutoff")
    requests = grid[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    return requests, grid


def validate_feature_specs(specs):
    """Explicit, ordered numeric feature contract; no clinical-name substring filtering."""
    _require(isinstance(specs, (list, tuple)) and len(specs) > 0, "Feature specifications are required")
    clean = []
    for original in specs:
        _require(isinstance(original, dict), "Feature specification must be a dictionary")
        item = dict(original)
        name = item.get("name")
        _require(isinstance(name, str) and bool(name.strip()), "Missing feature name")
        _require(name.upper() not in RESERVED_COLUMNS, "Identifier/target/provenance column cannot be a feature: " + name)
        _require(item.get("kind") in ("count", "binary", "continuous"),
                 name + ": declare count, binary or continuous; categorical fields require a TRAIN-only encoder")
        _require(isinstance(item.get("definition"), str) and bool(item["definition"].strip()),
                 name + ": source feature definition/evidence required")
        clean.append(item)
    names = [f["name"] for f in clean]
    _require(len(names) == len(set(names)), "Duplicate feature names")
    return clean


def infer_numeric_feature_specs(frame, definitions, exact_exclusions=()):
    """Create continuous declarations only for explicitly documented numeric fields.

    This does not infer encoding, count or binary semantics from feature names.
    Definitions must map every selected name to a meaningful source description.
    Use explicit specs for validated count/binary domains.
    """
    excluded = RESERVED_COLUMNS | {str(x).upper() for x in exact_exclusions}
    selected = [c for c in frame.columns if str(c).upper() not in excluded]
    specs = []
    for name in selected:
        _require(name in definitions, "Missing feature definition: " + name)
        pd.to_numeric(frame[name], errors="raise")
        specs.append({"name": name, "kind": "continuous", "definition": definitions[name]})
    return validate_feature_specs(specs)


def validate_contract(contract):
    _require(isinstance(contract, dict), "Historical extraction contract is required")
    result = dict(contract)
    required = ["source_id", "source_version", "extraction_sha256", "cutoff_rule", "fit_scope", "feature_semantics_evidence"]
    for key in required:
        _require(isinstance(result.get(key), str) and bool(result[key].strip()), "Missing extraction contract field: " + key)
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", result["extraction_sha256"])), "extraction_sha256 must hash actual source extraction code")
    _require(result["fit_scope"] in ("deterministic_raw", "train_only_fitted"), "Unknown or unsafe AutoML fitting scope")
    if result["fit_scope"] == "train_only_fitted":
        _require(bool(re.fullmatch(r"[0-9a-f]{64}", result.get("fitting_patients_sha256", ""))),
                 "Learned upstream features require fitting_patients_sha256")
    return result


def _normalize_keys(frame):
    _require(isinstance(frame, pd.DataFrame) and not frame.columns.duplicated().any(), "Invalid panel/request columns")
    _require(set(KEYS).issubset(frame.columns), "Historical panel requires PATIENT_ID and AS_OF_DATE")
    result = frame.copy().reset_index(drop=True)
    result["PATIENT_ID"] = _patients(result.PATIENT_ID)
    result["AS_OF_DATE"] = _dates(result.AS_OF_DATE, "AS_OF_DATE")
    _require(not result.duplicated(KEYS).any(), "Duplicate patient/date panel keys")
    return result


def validate_panel(panel, specs):
    specs = validate_feature_specs(specs)
    frame = _normalize_keys(panel)
    names = [s["name"] for s in specs]
    _require(set(names + ["IS_OBSERVED"]).issubset(frame.columns), "Panel lacks feature fields or explicit IS_OBSERVED coverage")
    _require(frame.IS_OBSERVED.notna().all() and frame.IS_OBSERVED.isin([0, 1, False, True]).all(),
             "Every panel row requires explicit observation coverage")
    frame["IS_OBSERVED"] = frame.IS_OBSERVED.astype(bool)
    observed = frame.IS_OBSERVED.to_numpy()
    for spec in specs:
        name = spec["name"]
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
        _require(not np.isinf(values).any(), name + ": infinite feature value")
        _require(np.isnan(values[~observed]).all(), name + ": unobserved history must remain missing, not copied or zero-filled")
        finite = values[np.isfinite(values)]
        if spec["kind"] == "count":
            _require(np.all((finite >= 0) & (finite == np.floor(finite))), name + ": invalid count values")
        elif spec["kind"] == "binary":
            _require(np.isin(finite, [0., 1.]).all(), name + ": invalid binary values")
        frame[name] = values
    for col in ("MAX_EVENT_DATE", "MAX_AVAILABLE_DATE"):
        if col in frame:
            dates = pd.to_datetime(frame[col], errors="raise")
            _require(dates.dt.tz is None, col + ": timezone conversion must be explicit")
            _require(dates.dropna().eq(dates.dropna().dt.normalize()).all(), col + ": expected exact dates")
            _require((dates.isna() | dates.le(frame.AS_OF_DATE)).all(), col + ": future information at historical cutoff")
            _require(dates[~observed].isna().all(), col + ": unavailable rows cannot contain event provenance")
            frame[col] = dates
    keep = KEYS + ["IS_OBSERVED"] + names + [c for c in ("MAX_EVENT_DATE", "MAX_AVAILABLE_DATE", "OBSERVATION_EVIDENCE") if c in frame]
    return frame[keep].sort_values(KEYS).reset_index(drop=True)


def frame_hash(frame):
    """Fingerprint content and ordering without returning any patient-level data."""
    h = hashlib.sha256(json.dumps(list(frame.columns), separators=(",", ":")).encode())
    for row in frame.itertuples(index=False, name=None):
        canonical = []
        for value in row:
            if pd.isna(value):
                canonical.append(None)
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                canonical.append(pd.Timestamp(value).isoformat())
            elif isinstance(value, (np.bool_, bool)):
                canonical.append(bool(value))
            elif isinstance(value, (float, np.floating)):
                canonical.append(float(value))
            elif isinstance(value, (int, np.integer)):
                canonical.append(int(value))
            else:
                canonical.append(str(value))
        h.update(json.dumps(canonical, separators=(",", ":"), allow_nan=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def make_panel_manifest(panel, specs, contract):
    specs = validate_feature_specs(specs)
    contract = validate_contract(contract)
    frame = validate_panel(panel, specs)
    return {"schema": 1, "contract": contract, "contract_sha256": stable_hash(contract),
            "feature_specs": specs, "feature_specs_sha256": stable_hash(specs),
            "panel_sha256": frame_hash(frame), "key_sha256": frame_hash(frame[KEYS]),
            "rows": len(frame), "observed_rows": int(frame.IS_OBSERVED.sum()),
            "availability_provenance": "MAX_AVAILABLE_DATE present; nulls remain unresolved" if "MAX_AVAILABLE_DATE" in frame else "Still unresolved: source availability clock not supplied",
            "event_provenance": "MAX_EVENT_DATE checked where supplied" if "MAX_EVENT_DATE" in frame else "Contract-declared only; event maxima not supplied"}


def obtain_panel(requests, specs, contract, cached_panel=None, cached_manifest=None,
                 extractor: Callable | None = None):
    """Reuse checked patient/date rows and extract missing keys once.

    An extractor receives *only* keys and must explicitly return every requested
    row, including unavailable history. It may not silently omit incomplete rows.
    The supplied manifest is provenance evidence, not automatic verification that
    the warehouse extraction really implemented the declared semantics.
    """
    started = time.perf_counter()
    req = _normalize_keys(requests[KEYS]).sort_values(KEYS).reset_index(drop=True)
    _require(len(req) > 0, "No historical extraction keys")
    specs = validate_feature_specs(specs)
    contract = validate_contract(contract)
    if cached_panel is None:
        _require(cached_manifest is None, "Manifest supplied without cached values")
        cached = None
        missing = req
    else:
        _require(isinstance(cached_manifest, dict), "Cached data requires its original definition/content manifest")
        cached = validate_panel(cached_panel, specs)
        actual = make_panel_manifest(cached, specs, contract)
        for key in ("schema", "contract_sha256", "feature_specs_sha256", "panel_sha256", "key_sha256", "rows"):
            _require(actual[key] == cached_manifest.get(key), "Cached panel mismatch: " + key)
        missing = req.merge(cached[KEYS], on=KEYS, how="left", indicator=True, validate="one_to_one")
        missing = missing.loc[missing._merge.eq("left_only"), KEYS].reset_index(drop=True)
    if len(missing):
        _require(callable(extractor),
                 f"Historical AutoML extraction required for {len(missing)} unique patient/date keys. "
                 "Provide the recovered source query/procedure with exact historical cutoff semantics; a current snapshot table cannot fill history.")
        fresh = validate_panel(extractor(missing.copy()), specs)
        coverage = missing.merge(fresh[KEYS], on=KEYS, how="outer", indicator=True, validate="one_to_one")
        _require(coverage._merge.eq("both").all(), "Extractor must return exactly the requested keys, including explicit unavailable rows")
        if cached is not None:
            _require(list(cached.columns) == list(fresh.columns), "Fresh/cached provenance schemas differ; resolve before combining")
            combined = pd.concat([cached, fresh], ignore_index=True)
        else:
            combined = fresh
    else:
        combined = cached
    combined = validate_panel(combined, specs)
    manifest = make_panel_manifest(combined, specs, contract)
    manifest["extraction_report"] = {
        "requested_unique_keys": len(req), "cache_keys_reused": len(req) - len(missing),
        "keys_extracted": len(missing), "panel_cache_rows": len(combined),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return combined, manifest


def assemble_inputs(metadata, panel, specs, steps=12, date_policy="same_day_months"):
    """Assemble S=Nx1xF and T=Nx12xF with common ordering and coverage masks."""
    meta = normalize_snapshots(metadata)
    specs = validate_feature_specs(specs)
    frame = validate_panel(panel, specs)
    requests, grid = make_requests(meta, steps, date_policy)
    aligned = grid.merge(frame, on=KEYS, how="left", sort=False, validate="many_to_one", indicator=True)
    _require(aligned._merge.eq("both").all(), "Historical panel does not cover every requested patient/date")
    aligned = aligned.sort_values(["SNAPSHOT_ROW", "TIME_STEP"]).reset_index(drop=True)
    names = [s["name"] for s in specs]
    x = aligned[names].to_numpy(dtype=np.float32).reshape(len(meta), steps, len(names))
    _require(not np.isinf(x).any(), "Numeric features overflow float32; inspect source scale before modeling")
    mask = aligned.IS_OBSERVED.astype(bool).to_numpy().reshape(len(meta), steps)
    # Missing historical rows are explicit NaNs; TRAIN preprocessing supplies
    # finite model padding while the attention/pooling mask retains missingness.
    _require(np.isnan(x[~mask]).all(), "Unobserved steps unexpectedly have values")
    report = {"snapshots": len(meta), "feature_count": len(names), "steps": steps,
              "unique_patient_dates": len(requests), "sequence_positions": len(grid),
              "reused_positions": len(grid) - len(requests),
              "unobserved_steps": int((~mask).sum()),
              "snapshots_without_current_automl_evidence": int((~mask[:, 0]).sum()),
              "snapshots_with_incomplete_history": int((~mask).any(axis=1).sum()),
              "date_policy": date_policy, "orientation": "0=newest",
              "sequence_kind": "Historical as-of feature summaries; lookbacks may overlap, not disjoint monthly counts",
              "feature_order_sha256": stable_hash(names),
              "feature_specs_sha256": stable_hash(specs)}
    return {"metadata": meta, "y": meta.RESP.to_numpy(dtype=np.int64),
            "indices": {s: np.flatnonzero(meta.SPLIT.to_numpy() == s) for s in ("train", "validation", "test")},
            "X_summary": x[:, :1, :].copy(), "X_sequence": x,
            "summary_mask": mask[:, :1].copy(), "sequence_mask": mask,
            "feature_names": names, "feature_specs": specs,
            "hashes": {"metadata_sha256": frame_hash(meta), "panel_sha256": frame_hash(frame),
                       "requests_sha256": frame_hash(requests), "feature_specs_sha256": stable_hash(specs),
                       "grid_sha256": frame_hash(grid)}, "report": report}


def make_query_extractor(read_query, sql_template, chunk_size=1000):
    """Bind recovered read-only SQL using a keys-only request CTE.

    sql_template must contain the literal ``{request_sql}`` exactly once, e.g.
    ``WITH request AS ({request_sql}) SELECT ...``. It must return the standard
    panel columns. The actual feature calculation is supplied/recovered source
    code, never an invented fallback. Reader can return pandas or Spark frames.
    """
    _require(callable(read_query), "A source query reader is required")
    _require(isinstance(sql_template, str) and sql_template.count("{request_sql}") == 1,
             "Recovered SQL must contain exactly one {request_sql} placeholder")
    _require(isinstance(chunk_size, int) and 1 <= chunk_size <= 10000, "Invalid query chunk size")

    def extractor(requests):
        req = _normalize_keys(requests[KEYS])
        outputs = []
        for start in range(0, len(req), chunk_size):
            values = []
            for patient, date in req.iloc[start:start + chunk_size].itertuples(index=False, name=None):
                # Standard SQL literal escaping; never interpolate credentials/labels.
                patient_sql = patient.replace("'", "''")
                values.append(f"SELECT '{patient_sql}' AS PATIENT_ID, TO_DATE('{date:%Y-%m-%d}') AS AS_OF_DATE")
            query = sql_template.replace("{request_sql}", " UNION ALL ".join(values))
            result = read_query(query)
            outputs.append(result if isinstance(result, pd.DataFrame) else result.toPandas())
        _require(outputs, "No historical keys requested")
        return pd.concat(outputs, ignore_index=True)

    extractor.extraction_sha256 = hashlib.sha256(sql_template.encode()).hexdigest()
    return extractor
