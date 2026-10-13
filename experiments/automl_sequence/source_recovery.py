"""Read-only retrieval of explicitly supplied procedure signatures.

Returned DDL is evidence for private review, never executed or automatically
approved as a feature extractor. No source text is printed by this module.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

import pandas as pd


DEFAULT_PROCEDURES = {
    "FEATURE_CREATION_PROC": (
        "DSVC_TAKEDA_TA_PRIVATE.DS_ML_DEV.FEATURE_CREATION_PROC("
        "VARCHAR,VARCHAR,VARCHAR,VARCHAR,VARCHAR,VARCHAR,VARCHAR)"),
    "FEATURE_ENGINEERING_PROD": (
        "DSVC_TAKEDA_TA_PRIVATE.DS_ML_DEV.FEATURE_ENGINEERING_PROD("
        "VARCHAR,VARCHAR,VARCHAR)"),
    "FEATURE_CREATION_PROD": (
        "DSVC_TAKEDA_TA_PRIVATE.DS_ML_DEV.FEATURE_CREATION_PROD("
        "VARCHAR,VARCHAR,VARCHAR,VARCHAR)"),
}

_SIGNATURE = re.compile(
    r"([A-Z_][A-Z0-9_$]*)\.([A-Z_][A-Z0-9_$]*)\."
    r"([A-Z_][A-Z0-9_$]*)\((VARCHAR(?:,VARCHAR)*)\)", re.IGNORECASE)
_DDL_HEADER = re.compile(
    r'\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:SECURE\s+)?PROCEDURE\s+'
    r'((?:"?[A-Z_][A-Z0-9_$]*"?\.){0,2}"?[A-Z_][A-Z0-9_$]*"?)\s*\(',
    re.IGNORECASE)
_OMISSION = re.compile(
    r"\[\s*(?:\.{3}|\u2026)[^\]]{0,2000}\]"
    r"|\[(?:[^\]\r\n]{0,100}\b(?:omitted|truncated|redacted)\b[^\]\r\n]{0,100})\]",
    re.IGNORECASE)


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _requests(procedures, expected_sha256):
    if not isinstance(procedures, Mapping) or not procedures:
        raise ValueError("Procedure signatures must be a nonempty explicit mapping")
    result = []
    signatures = set()
    for name, signature in procedures.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_$]*", name):
            raise ValueError("Procedure evidence names must be uppercase SQL identifiers")
        if not isinstance(signature, str) or not (match := _SIGNATURE.fullmatch(signature)):
            raise ValueError("Use an exact fully qualified supplied VARCHAR procedure signature")
        canonical = signature.upper()
        if name != match.group(3).upper() or canonical in signatures:
            raise ValueError("Procedure name/signature mismatch or duplicate request")
        signatures.add(canonical)
        result.append((name, canonical))
    expected = {} if expected_sha256 is None else expected_sha256
    if not isinstance(expected, Mapping) or set(expected) - set(procedures):
        raise ValueError("Expected source hashes must name requested procedures only")
    for value in expected.values():
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("Expected source hashes must be SHA-256 hex strings")
    return result, {name: value.lower() for name, value in expected.items()}


def _ddl_value(response, name, signature):
    # SELECT GET_DDL returns one scalar. Column spelling differs by connector;
    # positional access is safe only after verifying the exact scalar shape.
    if hasattr(response, "toPandas"):
        response = response.toPandas()
    if not isinstance(response, pd.DataFrame) or response.shape != (1, 1):
        raise ValueError("GET_DDL must return exactly one row and one column: " + name)
    text = response.iloc[0, 0]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("GET_DDL returned empty or non-text source: " + name)
    header = _DDL_HEADER.search(text)
    if header is None:
        raise ValueError("GET_DDL response has no recognizable procedure declaration: " + name)
    returned_parts = header.group(1).replace('"', '').upper().split('.')
    requested_parts = signature.split('(', 1)[0].split('.')
    if returned_parts != requested_parts[-len(returned_parts):]:
        raise ValueError("GET_DDL declaration does not match requested procedure: " + name)
    return text


def recover_procedure_sources(read_query, procedures=None, expected_sha256=None):
    """Return ``{'ddl_texts': {name: text}, 'receipt': {...}}`` without printing.

    ``read_query(sql)`` must use an existing authorized connection and return a
    pandas DataFrame or a Spark-like DataFrame exposing ``toPandas``. Only SELECT
    GET_DDL statements are issued; no procedure CALL, writes or source execution
    occur. Custom signatures must come from supplied evidence or read-only
    metadata, never guesses. Optional expected hashes detect changed retrieved
    source. Partial retrieval or a mismatch raises instead of returning a bundle
    that could be mistaken for complete recovery.

    Absence of an omission marker does not prove completeness, correct historical
    deployment, fitting provenance, cutoff semantics or safe execution. Keep DDL
    private; the aggregate receipt is suitable for display.
    """
    if not callable(read_query):
        raise TypeError("read_query must be a callable authorized query adapter")
    requests, expected = _requests(
        DEFAULT_PROCEDURES if procedures is None else procedures, expected_sha256)
    ddl_texts, rows = {}, []
    for name, signature in requests:
        # Signature validation excludes quotes, comments and statement separators.
        query = "SELECT GET_DDL('PROCEDURE', '" + signature + "') AS PROCEDURE_DDL"
        try:
            response = read_query(query)
        except Exception:
            # Do not include connector details, SQL response or credentials in
            # the public error. Inspect connection failures privately if needed.
            raise RuntimeError("Read-only GET_DDL retrieval failed for " + name) from None
        text = _ddl_value(response, name, signature)
        encoded = text.encode("utf-8")
        digest = _sha256(encoded)
        if name in expected and expected[name] != digest:
            raise ValueError("Recovered source SHA-256 changed for " + name)
        markers = list(_OMISSION.finditer(text))
        ddl_texts[name] = text
        rows.append({
            "name": name, "requested_signature": signature,
            "sha256": digest, "utf8_bytes": len(encoded),
            "lines": len(text.splitlines()),
            "query_sha256": _sha256(query.encode("utf-8")),
            "expected_sha256_verified": name in expected,
            "explicit_omission_marker_detected": bool(markers),
            "omission_marker_count": len(markers),
            "omission_marker_lines": [text.count("\n", 0, m.start()) + 1 for m in markers],
            "completeness": "not_certified",
        })
    # Stable identity excludes display/runtime timestamps and preserves all exact
    # source bytes. Request ordering does not alter the aggregate identity.
    identity = [{"requested_signature": r["requested_signature"], "sha256": r["sha256"]}
                for r in sorted(rows, key=lambda r: r["requested_signature"])]
    receipt = {
        "schema_version": 1,
        "retrieval_method": "supplied_read_query_SELECT_GET_DDL",
        "classification": "Recovered from existing artifact",
        "scope": "Returned deployed DDL evidence; not historical execution verification",
        "read_only": True, "procedure_count": len(rows), "procedures": rows,
        "aggregate_sha256": _sha256(json.dumps(identity, sort_keys=True,
                                                 separators=(",", ":")).encode("utf-8")),
        "explicit_omission_marker_detected": any(r["explicit_omission_marker_detected"] for r in rows),
        "completeness": "not_certified", "approved_for_feature_binding": False,
        "historical_execution_verified": False, "source_executed": False,
        "clinical_results_produced": False,
        "next_step": "Privately review recovered source and bind an explicit raw historical extractor; do not execute DDL",
    }
    return {"ddl_texts": ddl_texts, "receipt": receipt}
