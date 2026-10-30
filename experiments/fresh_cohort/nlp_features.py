"""Bounded monthly medical-description TF-IDF with TRAIN-only fitted state.

Source and token semantics follow medical_nlp_demo: distinct medical event IDs,
current DX/PX descriptions, unigram/bigrams, numerals and negation preserved,
sublinear TF, smoothed IDF and L2 normalization. This predictive adapter uses
every configured snapshot, deduplicates exact monthly requests and fits only
TRAIN documents. It does not read free-text notes or infer clinical enrollment.
"""
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS


STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS)-{"no", "not", "nor", "without", "neither", "never", "with"})
TOKEN_PATTERN = r"(?u)\b\w+\b"


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def mapping_select(cfg):
    return f'''SELECT UPPER(TRIM(C_TYPE)) AS FAMILY,
        CASE WHEN LOWER(TRIM(C_TYPE))='dx' THEN REPLACE(UPPER(TRIM(TO_VARCHAR(CODE))),'.','')
             ELSE UPPER(TRIM(TO_VARCHAR(CODE))) END AS CODE,
        MIN(NULLIF(TRIM(TO_VARCHAR("DESC")),'')) AS DESCRIPTION,
        COUNT(DISTINCT NULLIF(TRIM(TO_VARCHAR("DESC")),'')) AS DESCRIPTION_VARIANTS
    FROM {cfg['reference']} WHERE CURR_IND='Y' AND LOWER(TRIM(C_TYPE)) IN ('dx','px') GROUP BY 1,2'''


def mapping_fingerprint(reader, cfg):
    query = ("SELECT COUNT(*) AS ROWS, TO_VARCHAR(HASH_AGG(HASH(FAMILY,CODE,DESCRIPTION,DESCRIPTION_VARIANTS))) "
             "AS MAPPING_HASH FROM (" + mapping_select(cfg) + ")")
    value = reader(query)
    value = value if isinstance(value, pd.DataFrame) else value.limit(2).toPandas()
    value = value.copy(); value.columns = [str(c).upper() for c in value.columns]
    if len(value) != 1 or not {"ROWS", "MAPPING_HASH"}.issubset(value):
        raise ValueError("NLP reference audit did not return a single aggregate fingerprint")
    row = value.iloc[0]
    if pd.isna(row.ROWS) or int(row.ROWS) <= 0 or pd.isna(row.MAPPING_HASH):
        raise ValueError("NLP reference mapping is empty or unauditable")
    return dict(rows=int(row.ROWS), warehouse_hash_agg=str(row.MAPPING_HASH))


def source_sql(cfg, requests):
    for key in ("medical", "reference"):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){2}", cfg[key]):
            raise ValueError("Invalid qualified NLP source identifier")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", cfg["event_id_column"]):
        raise ValueError("Invalid medical event identifier column")
    values = []
    for row in requests.itertuples(index=False):
        patient = row.PATIENT_ID.replace("'", "''")
        values.append(f"({int(row.DOC_ID)},'{patient}','{row.AS_OF_DATE:%Y-%m-%d}')")
    return f'''WITH requests AS (
 SELECT column1::NUMBER AS DOC_ID,column2::VARCHAR AS PATIENT_ID,column3::DATE AS AS_OF_DATE
 FROM VALUES {','.join(values)}
), med AS (
 SELECT r.DOC_ID,r.AS_OF_DATE,m.{cfg['event_id_column']} AS EVENT_ID,
        m.SERVICE_DATE::DATE AS SERVICE_DATE,m.DIAGNOSIS_CODES,m.PROCEDURE_CODE
 FROM {cfg['medical']} m JOIN requests r ON TO_VARCHAR(m.PATIENT_ID)=r.PATIENT_ID
 WHERE m.SERVICE_DATE>=DATE_TRUNC('month',r.AS_OF_DATE)
   AND m.SERVICE_DATE<DATEADD(day,1,r.AS_OF_DATE)
   AND m.SERVICE_DATE>=(SELECT MIN(DATE_TRUNC('month',AS_OF_DATE)) FROM requests)
   AND m.SERVICE_DATE<(SELECT DATEADD(day,1,MAX(AS_OF_DATE)) FROM requests)
), events AS (
 SELECT m.DOC_ID,m.EVENT_ID,m.SERVICE_DATE,'DX' AS FAMILY,
        REPLACE(UPPER(TRIM(d.VALUE)),'.','') AS CODE
 FROM med m,LATERAL SPLIT_TO_TABLE(COALESCE(m.DIAGNOSIS_CODES,''),'|') d
 WHERE NULLIF(TRIM(d.VALUE),'') IS NOT NULL
 UNION ALL
 SELECT DOC_ID,EVENT_ID,SERVICE_DATE,'PX',UPPER(TRIM(TO_VARCHAR(PROCEDURE_CODE)))
 FROM med WHERE NULLIF(TRIM(TO_VARCHAR(PROCEDURE_CODE)),'') IS NOT NULL
), code_map AS (
 {mapping_select(cfg)}
), combined AS (
 SELECT r.DOC_ID,'COVERAGE' AS ROW_KIND,NULL::VARCHAR AS FAMILY,NULL::VARCHAR AS CODE,
        NULL::VARCHAR AS DESCRIPTION,0 AS DESCRIPTION_VARIANTS,
        COUNT(DISTINCT m.EVENT_ID) AS EVENT_COUNT,
        SUM(IFF(m.SERVICE_DATE IS NOT NULL AND m.EVENT_ID IS NULL,1,0)) AS NULL_EVENT_ID_ROWS,
        MAX(m.SERVICE_DATE) AS MAX_EVENT_DATE
 FROM requests r LEFT JOIN med m ON r.DOC_ID=m.DOC_ID GROUP BY r.DOC_ID
 UNION ALL
 SELECT e.DOC_ID,'FEATURE',e.FAMILY,e.CODE,r.DESCRIPTION,COALESCE(r.DESCRIPTION_VARIANTS,0),
        COUNT(DISTINCT e.EVENT_ID),SUM(IFF(e.EVENT_ID IS NULL,1,0)),MAX(e.SERVICE_DATE)
 FROM events e LEFT JOIN code_map r ON e.FAMILY=r.FAMILY AND e.CODE=r.CODE
 GROUP BY 1,2,3,4,5,6
)
SELECT * FROM combined LIMIT {int(cfg['max_source_rows_per_chunk'])+1}'''


def make_requests(metadata):
    meta = metadata.copy()
    if not {"PATIENT_ID", "END_DT", "SPLIT"}.issubset(meta):
        raise ValueError("NLP requires snapshot keys and frozen patient partitions")
    if meta.empty or meta[["PATIENT_ID", "END_DT", "SPLIT"]].isna().any().any():
        raise ValueError("Empty or null NLP metadata")
    if not meta.PATIENT_ID.map(lambda v: isinstance(v, str) and v.strip() == v and bool(v)).all():
        raise ValueError("Patient identifiers must be nonempty canonical strings")
    meta["END_DT"] = pd.to_datetime(meta.END_DT, errors="raise").dt.normalize()
    if meta.duplicated(["PATIENT_ID", "END_DT"]).any() or meta.groupby("PATIENT_ID").SPLIT.nunique().gt(1).any():
        raise ValueError("Duplicate snapshots or patient overlap between partitions")
    if set(meta.SPLIT) - {"train", "validation", "test"} or not meta.SPLIT.eq("train").any():
        raise ValueError("Invalid or absent TRAIN partition")
    dates = np.empty((len(meta), 12), dtype="datetime64[ns]")
    dates[:, 0] = meta.END_DT.to_numpy()
    month_start = meta.END_DT.dt.to_period("M").dt.to_timestamp()
    for step in range(1, 12):
        dates[:, step] = (month_start-pd.offsets.MonthEnd(step)).to_numpy()
    grid = pd.DataFrame({"PATIENT_ID": np.repeat(meta.PATIENT_ID.to_numpy(), 12),
                         "AS_OF_DATE": dates.reshape(-1), "SPLIT": np.repeat(meta.SPLIT.to_numpy(), 12)})
    requests = grid.drop_duplicates(["PATIENT_ID", "AS_OF_DATE"]).sort_values(["PATIENT_ID", "AS_OF_DATE"]).reset_index(drop=True)
    requests.insert(0, "DOC_ID", np.arange(len(requests), dtype=np.int64))
    index = pd.MultiIndex.from_frame(requests[["PATIENT_ID", "AS_OF_DATE"]])
    positions = index.get_indexer(pd.MultiIndex.from_frame(grid[["PATIENT_ID", "AS_OF_DATE"]])).reshape(len(meta), 12)
    return meta, requests, positions


def _query_parts(reader, batch, cfg, telemetry):
    sql = source_sql(cfg, batch)
    telemetry["queries"] += 1
    value = reader(sql)
    limit = cfg["max_source_rows_per_chunk"]
    if not isinstance(value, pd.DataFrame):
        value = value.limit(limit+1).toPandas()
    if len(value) > limit:
        if len(batch) <= 1:
            raise ValueError("One monthly document exceeds the aggregate row budget; no truncation is permitted")
        telemetry["split_queries"] += 1
        midpoint = len(batch)//2
        yield from _query_parts(reader, batch.iloc[:midpoint], cfg, telemetry)
        yield from _query_parts(reader, batch.iloc[midpoint:], cfg, telemetry)
        return
    value = value.copy()
    value.columns = [str(k).upper() for k in value.columns]
    needed = {"DOC_ID", "ROW_KIND", "FAMILY", "CODE", "DESCRIPTION", "DESCRIPTION_VARIANTS",
              "EVENT_COUNT", "NULL_EVENT_ID_ROWS", "MAX_EVENT_DATE"}
    if not needed.issubset(value) or value.empty:
        raise ValueError("NLP extraction omitted required coverage rows or source fields")
    for key in ("DOC_ID", "EVENT_COUNT", "NULL_EVENT_ID_ROWS", "DESCRIPTION_VARIANTS"):
        value[key] = pd.to_numeric(value[key], errors="raise")
        a = value[key].to_numpy(dtype=float)
        if not np.isfinite(a).all() or (a < 0).any() or (a != np.floor(a)).any():
            raise ValueError("Invalid NLP event counts or document identifiers")
    if not value.DOC_ID.isin(batch.DOC_ID).all() or value.NULL_EVENT_ID_ROWS.ne(0).any():
        raise ValueError("Unrequested document or medical event without identifier")
    coverage = value[value.ROW_KIND.eq("COVERAGE")]
    if coverage.DOC_ID.duplicated().any() or set(coverage.DOC_ID) != set(batch.DOC_ID):
        raise ValueError("Monthly request coverage is incomplete or duplicated")
    if not value.ROW_KIND.isin(["COVERAGE", "FEATURE"]).all():
        raise ValueError("Unknown NLP row kind")
    cutoff = batch.set_index("DOC_ID").AS_OF_DATE
    events = pd.to_datetime(value.MAX_EVENT_DATE, errors="raise")
    ends = value.DOC_ID.map(cutoff)
    starts = ends.dt.to_period("M").dt.to_timestamp()
    if ((events.notna() & ((events > ends) | (events < starts))) | (value.EVENT_COUNT.gt(0) & events.isna())).any():
        raise ValueError("NLP source event is outside its monthly cutoff or missing event provenance")
    rows = value[value.ROW_KIND.eq("FEATURE")]
    if rows[["FAMILY", "CODE"]].isna().any().any() or rows.EVENT_COUNT.le(0).any():
        raise ValueError("Invalid code-level monthly counts")
    if not rows.FAMILY.isin(["DX", "PX"]).all() or rows.duplicated(["DOC_ID", "FAMILY", "CODE"]).any():
        raise ValueError("Unexpected family or duplicate monthly code count")
    yield value


def _chunk_hash(db, start, end):
    h = hashlib.sha256()
    for table, fields in (("docs", "doc,observed"), ("features", "doc,family,code,description,n,variants")):
        cursor = db.execute(f"SELECT {fields} FROM {table} WHERE doc>=? AND doc<? ORDER BY {fields}", (start, end))
        for row in cursor:
            h.update(json.dumps(row, ensure_ascii=False, allow_nan=False).encode()+b"\n")
    return h.hexdigest()


def _spool(reader, requests, cfg, directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory/"monthly.sqlite")
    db.executescript('''PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS identity (value TEXT);
      CREATE TABLE IF NOT EXISTS docs (doc INTEGER PRIMARY KEY, observed INTEGER);
      CREATE TABLE IF NOT EXISTS features (doc INTEGER,family TEXT,code TEXT,description TEXT,n INTEGER,variants INTEGER,
                                         PRIMARY KEY(doc,family,code));
      CREATE TABLE IF NOT EXISTS mapping (family TEXT,code TEXT,description TEXT,PRIMARY KEY(family,code));
      CREATE TABLE IF NOT EXISTS chunks (start INTEGER PRIMARY KEY,end INTEGER,sha TEXT);
    ''')
    telemetry = dict(queries=0, split_queries=0, reused_chunks=0, new_chunks=0)
    try:
        existing = db.execute("SELECT value FROM identity").fetchall()
        serialized = json.dumps(identity, sort_keys=True)
        if existing and existing != [(serialized,)]:
            raise ValueError("NLP checkpoint source, membership, request or implementation identity changed")
        if not existing:
            db.execute("INSERT INTO identity VALUES (?)", (serialized,)); db.commit()
        for start in range(0, len(requests), cfg["chunk_size"]):
            end = min(start+cfg["chunk_size"], len(requests))
            previous = db.execute("SELECT end,sha FROM chunks WHERE start=?", (start,)).fetchone()
            if previous:
                if previous != (end, _chunk_hash(db, start, end)):
                    raise ValueError("NLP checkpoint content integrity failure")
                telemetry["reused_chunks"] += 1
                continue
            with db:
                for part in _query_parts(reader, requests.iloc[start:end], cfg, telemetry):
                    for row in part.itertuples(index=False):
                        doc = int(row.DOC_ID)
                        if row.ROW_KIND == "COVERAGE":
                            db.execute("INSERT INTO docs VALUES (?,?)", (doc, int(row.EVENT_COUNT > 0)))
                        else:
                            description = None if pd.isna(row.DESCRIPTION) else str(row.DESCRIPTION).strip() or None
                            old = db.execute("SELECT description FROM mapping WHERE family=? AND code=?", (row.FAMILY, row.CODE)).fetchone()
                            if old is not None and old != (description,):
                                raise ValueError("Reference description changed between extracted monthly chunks")
                            db.execute("INSERT OR IGNORE INTO mapping VALUES (?,?,?)", (row.FAMILY, row.CODE, description))
                            db.execute("INSERT INTO features VALUES (?,?,?,?,?,?)", (doc, row.FAMILY, str(row.CODE), description,
                                                                                  int(row.EVENT_COUNT), int(row.DESCRIPTION_VARIANTS)))
                db.execute("INSERT INTO chunks VALUES (?,?,?)", (start, end, _chunk_hash(db, start, end)))
            telemetry["new_chunks"] += 1
            print("NLP monthly requests validated:", end, "/", len(requests))
        if db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] != len(requests):
            raise ValueError("NLP checkpoint has incomplete monthly documents")
        return db, telemetry
    except BaseException:
        db.close()
        raise


def _documents(db, analyzer, selected=None):
    current, counts = None, Counter()
    for doc, description, events in db.execute("SELECT doc,description,n FROM features WHERE description IS NOT NULL ORDER BY doc,family,code"):
        if current is not None and doc != current:
            yield current, counts
            counts = Counter()
        current = doc
        words = Counter(analyzer(description))
        counts.update({word: number*events for word, number in words.items() if selected is None or word in selected})
    if current is not None:
        yield current, counts


def extract_and_fit(read_query, metadata, config, checkpoint_directory):
    cfg = dict(config)
    required = {"medical", "reference", "event_id_column", "max_terms", "chunk_size", "max_source_rows_per_chunk", "run_identity"}
    if set(cfg) != required or not isinstance(cfg["run_identity"], str) or not cfg["run_identity"]:
        raise ValueError("NLP configuration must declare sources, bounds and run identity")
    for key, high in (("max_terms", 4096), ("chunk_size", 10000), ("max_source_rows_per_chunk", 1000000)):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= high:
            raise ValueError("Invalid NLP resource bound: " + key)
    meta, requests, positions = make_requests(metadata)
    identity = dict(configuration=cfg, requests=stable_hash([[int(r.DOC_ID),r.PATIENT_ID,r.AS_OF_DATE.strftime('%Y-%m-%d'),r.SPLIT]
                                                            for r in requests.itertuples(index=False)]),
                    sql_sha256=hashlib.sha256(source_sql(cfg, requests.iloc[:1]).encode()).hexdigest(),
                    schema="train_monthly_tfidf_v1", token_pattern=TOKEN_PATTERN, stop_words_sha256=stable_hash(STOP_WORDS))
    identity["mapping_fingerprint"] = mapping_fingerprint(read_query, cfg)
    directory = Path(checkpoint_directory)
    db, telemetry = _spool(read_query, requests, cfg, directory, identity)
    output_directory = tempfile.TemporaryDirectory(prefix="monthly_nlp_arrays_")
    mappings = []
    try:
        if mapping_fingerprint(read_query, cfg) != identity["mapping_fingerprint"]:
            raise ValueError("NLP reference mapping changed during extraction; do not fit mixed mapping versions")
        analyzer = CountVectorizer(lowercase=True, stop_words=STOP_WORDS, ngram_range=(1,2),
                                   token_pattern=TOKEN_PATTERN).build_analyzer()
        train = requests.SPLIT.eq("train").to_numpy()
        frequency = Counter()
        for doc, counts in _documents(db, analyzer):
            if train[doc]:
                frequency.update(counts)
        terms = sorted(frequency, key=lambda word: (-frequency[word], word))[:cfg["max_terms"]]
        if not terms:
            raise ValueError("No tokenized mapped medical descriptions in TRAIN; no empty feature package is allowed")
        lookup = {term: j for j,term in enumerate(terms)}
        document_frequency = np.zeros(len(terms), dtype=np.int64)
        for doc, counts in _documents(db, analyzer, lookup):
            if train[doc]:
                for term in counts:
                    document_frequency[lookup[term]] += 1
        idf = np.log((1+int(train.sum()))/(1+document_frequency))+1
        tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
        tfidf.idf_ = idf
        doc_values = np.lib.format.open_memmap(Path(output_directory.name)/"documents.npy", mode="w+",
            dtype=np.float32, shape=(len(requests),len(terms)))
        mappings.append(doc_values); doc_values[:] = 0
        batch = []
        def transform_batch(batch):
            rows, columns, values = [], [], []
            for i, (_, counts) in enumerate(batch):
                for word, count in counts.items():
                    rows.append(i); columns.append(lookup[word]); values.append(count)
            counts = sparse.csr_matrix((np.asarray(values, dtype=np.float64), (rows, columns)), shape=(len(batch),len(terms)))
            output = tfidf.transform(counts).toarray().astype(np.float32)
            doc_values[[doc for doc,_ in batch]] = output
        for item in _documents(db, analyzer, lookup):
            batch.append(item)
            if len(batch) == 256:
                transform_batch(batch); batch = []
        if batch:
            transform_batch(batch)
        mask_by_doc = np.zeros(len(requests), dtype=bool)
        for doc, observed in db.execute("SELECT doc,observed FROM docs"):
            mask_by_doc[doc] = bool(observed)
        x = np.lib.format.open_memmap(Path(output_directory.name)/"sequence.npy", mode="w+",
            dtype=np.float32, shape=(len(meta),12,len(terms)))
        mappings.append(x)
        mask = mask_by_doc[positions]
        for start in range(0, len(meta), 256):
            x[start:start+256] = doc_values[positions[start:start+256]]
        x.flush(); doc_values._mmap.close(); mappings.remove(doc_values)
        state = dict(schema="monthly_tfidf_v1", terms=terms, idf=idf.tolist(), norm="l2", sublinear_tf=True,
                     smooth_idf=True, lowercase=True, ngram_range=[1,2], token_pattern=TOKEN_PATTERN,
                     stop_words=STOP_WORDS, training_documents=int(train.sum()),
                     document_frequency=document_frequency.tolist(), training_term_counts=[int(frequency[t]) for t in terms],
                     fit_scope="train_only", extraction_identity=identity,
                     train_patients_sha256=stable_hash(sorted(meta.loc[meta.SPLIT.eq("train"), "PATIENT_ID"].unique().tolist())),
                     train_document_count=int(train.sum()), term_order_sha256=stable_hash(terms))
        receipt = dict(fit_scope="train_only", unique_monthly_documents=len(requests), snapshot_positions=len(meta)*12,
            train_patients_sha256=state["train_patients_sha256"], train_document_count=int(train.sum()),
            reused_positions=int(len(meta)*12-len(requests)), retained_terms=len(terms), training_documents=int(train.sum()),
            source_feature_rows=db.execute("SELECT COUNT(*) FROM features").fetchone()[0],
            unmapped_feature_rows=db.execute("SELECT COUNT(*) FROM features WHERE description IS NULL").fetchone()[0],
            ambiguous_mapping_rows=db.execute("SELECT COUNT(*) FROM features WHERE variants>1").fetchone()[0],
            chunk_content_hashes=[list(r) for r in db.execute("SELECT start,end,sha FROM chunks ORDER BY start")],
            checkpoint=telemetry, orientation="0=newest", mask_kind="eligible medical-event activity; not continuous coverage",
            window="current partial calendar month plus eleven preceding complete calendar months",
            source_availability="LATEST tables; resumed chunks preserve prior reads; no transactional source snapshot",
            class_labels_used=False, patient_subsampling=False)
        return dict(X_sequence=x, mask=mask, feature_names=terms, metadata=meta, fitted_state=state, receipt=receipt,
                    _temporary_directory=output_directory, _mappings=mappings)
    except BaseException:
        for value in mappings:
            if not value._mmap.closed:
                value._mmap.close()
        output_directory.cleanup()
        raise
    finally:
        db.close()


def release_result(result):
    for value in result.pop("_mappings", []):
        if not value._mmap.closed:
            value._mmap.close()
    owner = result.pop("_temporary_directory", None)
    if owner is not None:
        owner.cleanup()
