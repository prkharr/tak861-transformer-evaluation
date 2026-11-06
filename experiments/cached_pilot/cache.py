"""Read-only, bounded handoff of completed raw-feature checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from contextlib import closing
from functools import lru_cache

import pandas as pd

try:
    from experiments.automl_sequence import inputs
except ModuleNotFoundError:
    inputs = None


COMPACT_COLUMNS = ("p", "d", "k", "c", "v", "r", "e", "a", "ac", "chunk_no")
SOURCE_COLUMNS = ("p", "d", "k", "c", "raw_json", "v", "r", "e", "a", "ac")
SCHEMA = 1


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _atomic(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _readonly(path):
    path = Path(path).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2,
                                 isolation_level=None)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=2000")
    return connection


def _read_source_state(connection):
    state = dict(connection.execute("SELECT key,value FROM state WHERE key IN ('identity','columns')"))
    if set(state) != {"identity", "columns"}:
        raise ValueError("No completed extraction checkpoint identity and column declaration")
    identity = json.loads(state["identity"])
    columns = json.loads(state["columns"])
    required = {"PATIENT_ID", "AS_OF_DATE", "ROW_KIND", "CATEGORY", "VALUE", "RECENT_VALUE",
                "MAX_EVENT_DATE", "AGE", "AGE_CONFLICT"}
    if identity.get("schema") != 1 or not required.issubset(columns) or len(columns) != len(set(columns)):
        raise ValueError("Unsupported raw checkpoint schema")
    receipts = [dict(chunk_no=int(n), row_count=int(count), sha256=sha)
                for n, count, sha in connection.execute("SELECT chunk_no,row_count,sha256 FROM chunks ORDER BY chunk_no")]
    if not receipts:
        raise ValueError("No fully committed chunks are available")
    if [x["chunk_no"] for x in receipts] != list(range(len(receipts))):
        raise ValueError("Completed chunk receipts must form a contiguous prefix")
    if any(x["row_count"] <= 0 or len(x["sha256"]) != 64 for x in receipts):
        raise ValueError("Invalid completed-chunk receipt")
    return state, identity, receipts


def _pages(connection, chunk_no, columns, page_rows):
    """Finish each short SELECT before processing its page; no long source transaction."""
    last = None
    names = ",".join(columns)
    while True:
        sql = "SELECT " + names + " FROM raw WHERE chunk_no=?"
        parameters = [chunk_no]
        if last is not None:
            sql += " AND (p,d,k,c) > (?,?,?,?)"
            parameters.extend(last)
        sql += " ORDER BY p,d,k,c LIMIT ?"
        parameters.append(page_rows)
        cursor = connection.execute(sql, parameters)
        try:
            rows = cursor.fetchall()
        finally:
            cursor.close()
        if not rows:
            return
        last = tuple(rows[-1][:4])
        yield rows


def _compact_digest(connection, chunk_no, page_rows=4096):
    digest = hashlib.sha256(_json_bytes(COMPACT_COLUMNS))
    count = 0
    for page in _pages(connection, chunk_no, COMPACT_COLUMNS, page_rows):
        for row in page:
            digest.update(_json_bytes(row) + b"\n")
        count += len(page)
    return count, digest.hexdigest()


@lru_cache(maxsize=65536)
def _validate_dates(date, event):
    cutoff = pd.Timestamp(date)
    if cutoff.strftime("%Y-%m-%d") != date:
        raise ValueError("Raw checkpoint dates must be exact days")
    if event is not None:
        event_date = pd.Timestamp(event)
        if event_date.strftime("%Y-%m-%d") != event or not cutoff - pd.DateOffset(months=12) <= event_date <= cutoff:
            raise ValueError("Cached source event exceeds its feature window")


def _validate_row(row):
    patient, date, kind, category, value, recent, event, age, conflict, chunk_no = row
    if not isinstance(patient, str) or not patient or kind not in {"COVERAGE", "FEATURE"}:
        raise ValueError("Invalid cached raw record key/kind")
    _validate_dates(date, event)
    if kind == "FEATURE":
        if not category or event is None or value is None or recent is None:
            raise ValueError("Feature rows require category, counts and event evidence")
        if not all(math.isfinite(x) and x == math.floor(x) for x in (value, recent)) or not 0 <= recent <= value or value <= 0:
            raise ValueError("Invalid annual/recent distinct counts")
    elif category != "" or value is not None or recent is not None:
        raise ValueError("Invalid coverage row")
    if age is not None and (not math.isfinite(age) or age < 0):
        raise ValueError("Invalid cached age")
    if conflict is not None and not math.isfinite(conflict):
        raise ValueError("Invalid cached age-conflict indicator")


def _coverage_check(connection, chunk_no, requests):
    returned = list(connection.execute("SELECT p,d FROM raw WHERE chunk_no=? AND k='COVERAGE' ORDER BY p,d", (chunk_no,)))
    expected = [(p, d.strftime("%Y-%m-%d")) for p, d in requests.itertuples(index=False, name=None)]
    if returned != expected:
        raise ValueError("Completed chunk does not cover exactly its original requested keys")
    invalid = connection.execute("SELECT 1 FROM raw f LEFT JOIN raw c ON f.p=c.p AND f.d=c.d AND c.k='COVERAGE' "
                                 "AND f.chunk_no=c.chunk_no WHERE f.chunk_no=? AND f.k='FEATURE' "
                                 "AND (c.p IS NULL OR c.e IS NULL) LIMIT 1", (chunk_no,)).fetchone()
    if invalid:
        raise ValueError("Feature rows lack observed coverage evidence")


def _base_path(location):
    path = Path(location).resolve()
    return path.parent if path.name == "manifest.json" else path


def freeze_cache(source_path, destination, metadata, *, chunk_limit=None, page_rows=4096,
                 previous_destination=None):
    """Freeze committed SUMMARY chunks without changing the full extraction.

    Run after the writer has completed a chunk, then let it resume. A frozen
    export plan ignores later commits. Repeating this call resumes that same
    plan; use a new destination for additional chunks. Existing exported chunks
    can be imported with previous_destination without rereading raw source rows.
    """
    if inputs is None:
        raise RuntimeError("Inject the existing inputs module")
    if type(page_rows) is not int or not 1 <= page_rows <= 65536:
        raise ValueError("page_rows must be between 1 and 65536")
    if chunk_limit is not None and (type(chunk_limit) is not int or chunk_limit < 1):
        raise ValueError("chunk_limit must be a positive integer")
    source = Path(source_path).resolve(strict=True)
    dest = _base_path(destination)
    if dest.is_relative_to(source.parent) or source.is_relative_to(dest):
        raise ValueError("Use an isolated export directory outside the extraction checkpoint directory")
    meta = inputs.normalize_snapshots(metadata)
    requests, _ = inputs.make_requests(meta, steps=1, date_policy="month_end_history")
    metadata_hash = inputs.frame_hash(meta)
    source_db = _readonly(source)
    target = None
    previous_db = None
    try:
        state, identity, receipts = _read_source_state(source_db)
        chunk_size = identity.get("chunk_size")
        if type(chunk_size) is not int or not 1 <= chunk_size <= 10000:
            raise ValueError("Invalid source request chunk size")
        if identity.get("requests_sha256") != inputs.frame_hash(requests):
            raise ValueError("Source request identity differs: only the matching SUMMARY population is supported")
        if identity.get("train_patients_sha256") != inputs.train_patient_hash(meta):
            raise ValueError("Source TRAIN patient membership differs")
        number_chunks = (len(requests) + chunk_size - 1) // chunk_size
        if len(receipts) > number_chunks:
            raise ValueError("Source contains unexpected chunk indices")
        dest.mkdir(parents=True, exist_ok=True)
        plan_path = dest / "frozen_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if (plan["identity"] != identity or plan["source_columns_json"] != state["columns"]
                    or plan["parent_metadata_sha256"] != metadata_hash):
                raise ValueError("Frozen handoff identity changed; choose a new export directory")
            if chunk_limit is not None and chunk_limit != len(plan["chunks"]):
                raise ValueError("Cannot change the frozen chunk boundary in an existing export")
            if receipts[:len(plan["chunks"])] != plan["chunks"]:
                raise ValueError("Previously frozen source receipts changed or disappeared")
        else:
            if chunk_limit is not None and chunk_limit > len(receipts):
                raise ValueError("Requested chunk boundary is not fully committed")
            chosen = receipts[:chunk_limit] if chunk_limit else receipts
            meta_bytes = meta.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%d").encode()
            request_file = requests.copy()
            request_file["chunk_no"] = request_file.index // chunk_size
            request_bytes = request_file.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%d").encode()
            plan = dict(schema=SCHEMA, identity=identity, source_columns_json=state["columns"],
                        parent_metadata_sha256=metadata_hash, chunks=chosen, request_steps=1,
                        total_requests=len(requests), frozen_request_count=min(len(chosen)*chunk_size, len(requests)),
                        analysis_db="analysis.sqlite", metadata_file="metadata.csv", requests_file="requests.csv",
                        metadata_file_sha256=hashlib.sha256(meta_bytes).hexdigest(),
                        requests_file_sha256=hashlib.sha256(request_bytes).hexdigest(),
                        sampling="completed sorted request prefix; complete patients only; not a random sample",
                        source_vintage="checkpoint cache may combine source vintages; historical availability unresolved")
            _atomic(dest / "metadata.csv", meta_bytes)
            _atomic(dest / "requests.csv", request_bytes)
            _atomic(plan_path, _json_bytes(plan))
        for filename, key in (("metadata.csv", "metadata_file_sha256"), ("requests.csv", "requests_file_sha256")):
            if _hash_file(dest / filename) != plan[key]:
                raise ValueError("Frozen metadata or request file checksum mismatch")
        previous = None
        if previous_destination is not None:
            previous = load_frozen_cache(previous_destination)
            if (previous["identity"] != identity or previous["parent_metadata_sha256"] != metadata_hash
                    or previous["source_columns_json"] != state["columns"]):
                raise ValueError("Previous export has a different parent identity")
            previous_db = _readonly(previous["analysis_db_path"])
        target = sqlite3.connect(dest / "analysis.sqlite", timeout=2)
        target.execute("PRAGMA temp_store=FILE")
        target.execute("PRAGMA cache_size=-16384")
        with target:
            target.execute("CREATE TABLE IF NOT EXISTS raw(p TEXT,d TEXT,k TEXT,c TEXT,v REAL,r REAL,e TEXT,a REAL,ac REAL,chunk_no INTEGER,PRIMARY KEY(p,d,k,c)) WITHOUT ROWID")
            target.execute("CREATE INDEX IF NOT EXISTS raw_by_chunk ON raw(chunk_no,p,d,k,c)")
            target.execute("CREATE TABLE IF NOT EXISTS chunks(chunk_no INTEGER PRIMARY KEY,row_count INTEGER,source_sha256 TEXT,export_sha256 TEXT)")
        saved_chunks = dict((n, (count, source_sha, export_sha)) for n, count, source_sha, export_sha in target.execute("SELECT * FROM chunks"))
        allowed = {x["chunk_no"] for x in plan["chunks"]}
        if set(saved_chunks) - allowed or target.execute("SELECT 1 FROM raw LEFT JOIN chunks USING(chunk_no) WHERE chunks.chunk_no IS NULL LIMIT 1").fetchone():
            raise ValueError("Analysis cache contains unreceipted or unexpected rows")
        completed = []
        previous_receipts = {} if previous is None else {x["chunk_no"]: x for x in previous["chunks"]}
        for receipt in plan["chunks"]:
            chunk_no = receipt["chunk_no"]
            batch = requests.iloc[chunk_no*chunk_size:(chunk_no+1)*chunk_size]
            saved = saved_chunks.get(chunk_no)
            if saved:
                count, exported = _compact_digest(target, chunk_no, page_rows)
                if saved != (count, receipt["sha256"], exported) or count != receipt["row_count"]:
                    raise ValueError("Analysis cache checkpoint checksum mismatch")
                _coverage_check(target, chunk_no, batch)
                mode = "reused"
            else:
                original_digest = hashlib.sha256(state["columns"].encode())
                export_digest = hashlib.sha256(_json_bytes(COMPACT_COLUMNS))
                count = 0
                inherited = previous_receipts.get(chunk_no)
                if inherited and any(inherited[k] != receipt[k] for k in ("chunk_no", "row_count", "sha256")):
                    raise ValueError("Previous export chunk differs from source receipt")
                try:
                    target.execute("BEGIN IMMEDIATE")
                    reader = previous_db if inherited else source_db
                    columns = COMPACT_COLUMNS if inherited else SOURCE_COLUMNS
                    for page in _pages(reader, chunk_no, columns, page_rows):
                        compact = []
                        for row in page:
                            if inherited:
                                item = row
                            else:
                                original = list(row)
                                original[4] = original[4].hex()
                                original_digest.update(json.dumps(original, separators=(",", ":"), allow_nan=False).encode() + b"\n")
                                item = tuple(row[:4]) + tuple(row[5:]) + (chunk_no,)
                            _validate_row(item)
                            export_digest.update(_json_bytes(item) + b"\n")
                            compact.append(item)
                        target.executemany("INSERT INTO raw VALUES (?,?,?,?,?,?,?,?,?,?)", compact)
                        count += len(compact)
                    exported = export_digest.hexdigest()
                    if count != receipt["row_count"] or (not inherited and original_digest.hexdigest() != receipt["sha256"]):
                        raise ValueError("Source checkpoint content checksum mismatch")
                    if inherited and exported != inherited["export_sha256"]:
                        raise ValueError("Previous export content checksum mismatch")
                    _coverage_check(target, chunk_no, batch)
                    target.execute("INSERT INTO chunks VALUES (?,?,?,?)", (chunk_no, count, receipt["sha256"], exported))
                    target.commit()
                except BaseException:
                    target.rollback()
                    raise
                mode = "imported" if inherited else "exported"
            completed.append(dict(receipt, export_sha256=exported))
            print("Analysis chunk", chunk_no+1, "/", len(plan["chunks"]), mode, "rows", count, flush=True)
        end_state, end_identity, end_receipts = _read_source_state(source_db)
        if end_identity != identity or end_state != state or end_receipts[:len(plan["chunks"])] != plan["chunks"]:
            raise ValueError("Source checkpoint changed while exporting frozen chunks")
        with target:
            target.execute("CREATE INDEX IF NOT EXISTS raw_by_category ON raw(k,c,p,d)")
        manifest = dict(plan, chunks=completed, status="verified", plan_sha256=_hash_file(plan_path),
                        compact_columns=list(COMPACT_COLUMNS))
        _atomic(dest / "manifest.json", _json_bytes(manifest))
        return dest / "manifest.json"
    finally:
        if previous_db is not None:
            previous_db.close()
        if target is not None:
            target.close()
        source_db.close()


def load_frozen_cache(location, *, verify_chunks=True):
    """Verify a completed handoff without opening or depending on its source cache."""
    dest = _base_path(location)
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "verified" or manifest.get("request_steps") != 1:
        raise ValueError("Incomplete or unsupported analysis handoff")
    for filename, key in (("metadata.csv", "metadata_file_sha256"), ("requests.csv", "requests_file_sha256"), ("frozen_plan.json", "plan_sha256")):
        if _hash_file(dest / filename) != manifest[key]:
            raise ValueError("Analysis handoff file checksum mismatch")
    plan = json.loads((dest / "frozen_plan.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in plan.items() if key != "chunks"):
        raise ValueError("Analysis manifest differs from frozen export plan")
    if [{k: row[k] for k in ("chunk_no", "row_count", "sha256")} for row in manifest["chunks"]] != plan["chunks"]:
        raise ValueError("Analysis manifest changed its frozen chunk list")
    if manifest.get("compact_columns") != list(COMPACT_COLUMNS):
        raise ValueError("Unexpected analysis cache columns")
    with closing(_readonly(dest / "analysis.sqlite")) as connection:
        actual = list(connection.execute("SELECT chunk_no,row_count,source_sha256,export_sha256 FROM chunks ORDER BY chunk_no"))
        expected = [(x["chunk_no"], x["row_count"], x["sha256"], x["export_sha256"]) for x in manifest["chunks"]]
        if actual != expected or connection.execute("SELECT 1 FROM raw LEFT JOIN chunks USING(chunk_no) WHERE chunks.chunk_no IS NULL LIMIT 1").fetchone():
            raise ValueError("Analysis cache receipt mismatch")
        if verify_chunks:
            for receipt in manifest["chunks"]:
                if _compact_digest(connection, receipt["chunk_no"]) != (receipt["row_count"], receipt["export_sha256"]):
                    raise ValueError("Analysis cache content checksum mismatch")
    return dict(manifest, analysis_db_path=str(dest / "analysis.sqlite"), metadata_path=str(dest / "metadata.csv"),
                requests_path=str(dest / "requests.csv"), manifest_path=str(dest / "manifest.json"))


def complete_patient_metadata(location):
    """Keep every parent snapshot for a patient, or none; never infer row order."""
    manifest = location if isinstance(location, dict) else load_frozen_cache(location)
    meta = inputs.normalize_snapshots(pd.read_csv(manifest["metadata_path"], dtype={"PATIENT_ID": str}))
    if inputs.frame_hash(meta) != manifest["parent_metadata_sha256"]:
        raise ValueError("Parent metadata identity mismatch")
    with closing(_readonly(manifest["analysis_db_path"])) as connection:
        coverage = pd.read_sql_query("SELECT p AS PATIENT_ID,d AS END_DT FROM raw WHERE k='COVERAGE' ORDER BY p,d", connection)
    coverage["END_DT"] = pd.to_datetime(coverage.END_DT)
    joined = meta.merge(coverage.assign(available=True), on=["PATIENT_ID", "END_DT"], how="left", validate="one_to_one")
    complete = joined.groupby("PATIENT_ID", sort=False).available.count().eq(joined.groupby("PATIENT_ID", sort=False).size())
    return meta.loc[meta.PATIENT_ID.isin(complete.index[complete])].reset_index(drop=True)
