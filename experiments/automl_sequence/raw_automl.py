"""Target-independent raw AutoML histories adapted from supplied procedure SQL.

Only SELECT queries run. Legacy RESP-conditioned maps, category filtering,
binary inference, caps, learned encodings, RND and model importance are omitted.
Source values stay private; returned receipts contain aggregate evidence only.
The notebook injects ``inputs`` so the embedded module has no package dependency.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import sqlite3
import tempfile
import os
from contextlib import closing

import numpy as np
import pandas as pd

inputs = None

DEFAULT_CONFIG = {
    "source_schema": "DSVC_TAKEDA_FULLMAP_PLAID_PROD.COHORT_1009719",
    "mapping_table": "DSVC_TAKEDA_TA_PRIVATE.DS_ML_PRD.DX_PX_RX_PLAID",
    "date_policy": "month_end_history", "chunk_size": 5000,
    "max_features": None,
}


def _config(config):
    cfg = dict(DEFAULT_CONFIG)
    incoming = config or {}
    if set(incoming) - set(cfg):
        raise ValueError("Unknown raw AutoML source configuration: " + str(set(incoming)-set(cfg)))
    cfg.update(incoming)
    for key, parts in (("source_schema", 2), ("mapping_table", 3)):
        if not isinstance(cfg[key], str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*){"+str(parts-1)+"}", cfg[key]):
            raise ValueError("Invalid explicit source identifier: " + key)
    if type(cfg["chunk_size"]) is not int or not 1 <= cfg["chunk_size"] <= 10000:
        raise ValueError("chunk_size must be between 1 and 10000")
    if cfg["max_features"] is not None and (type(cfg["max_features"]) is not int or cfg["max_features"] < 1):
        raise ValueError("max_features must be an explicit positive integer or None")
    if cfg["date_policy"] not in ("same_day_months", "month_end_history"):
        raise ValueError("Unknown date policy")
    return cfg


def raw_sql_template(source_config=None):
    """Return exact read-only long-form SQL with one keys-only request slot.

    Medical encounter grain preserves the source procedure's patient/date/DX
    grouping, with JSON encoding to avoid null/concatenation collisions. Claims
    counts and provider counts are distinct counts; recent counts use -4 to -1
    calendar months inclusive. No label appears in SQL.
    """
    cfg = _config(source_config)
    schema, mapping = cfg["source_schema"], cfg["mapping_table"]
    tokens = []

    def token(relation, category, entity="ENCOUNTER_KEY", condition="TRUE", rx_paid=True):
        # Source procedure treats paid RX activity as the feature population;
        # only transaction-result counts deliberately retain every result.
        if relation in ("RX_BASE", "RX_MAPPED") and rx_paid:
            condition = "UPPER(TRANSACTION_RESULT)='PAID' AND (" + condition + ")"
        tokens.append("SELECT PATIENT_ID,AS_OF_DATE," + category + " AS CATEGORY,"
                      + entity + " AS ENTITY_KEY,EVENT_DATE,IS_RECENT FROM " + relation
                      + " WHERE " + condition)

    # New raw namespaces keep source family/level explicit; the original private
    # regexp_comb UDF is replaced by the deterministic, hashed Python sanitizer.
    for prefix, relation, levels in (("DX", "DX_MAPPED", 2), ("PX", "PX_MAPPED", 3)):
        token(relation, "'"+prefix+"_DESC_'||MAP_DESC")
        for level in range(1, levels+1):
            token(relation, "'"+prefix+"_LEVEL_"+str(level)+"_'||LEVEL_"+str(level)+"_DESC")
    for family, relation in (("DX", "DX_BASE"), ("PX", "PX_BASE")):
        stem = "MX" if family == "DX" else "PX"
        token(relation, "'"+stem+"_PAYER_MIX_'||INSURANCE_GROUP")
        token(relation, "'"+stem+"_INSURANCE_SEGMENT_'||INSURANCE_SEGMENT")
        token(relation, "'HCPS_"+family+"_SP_PRIMARY_'||TRIM(SPLIT_PART(PRIMARY_SPECIALTY,',',1))")
        token(relation, "'CLAIMS_"+family+"'")
        token(relation, "'HCPS_"+family+"'", "PROVIDER_ID", "PROVIDER_ID IS NOT NULL")
    token("DX_BASE", "'CLAIM_TYP_'||EVENT_SOURCE")
    token("DX_BASE", "'HCP_TYP_'||PROVIDER_TYPE")
    token("DX_BASE", "'REFERRING_ENCOUNTER'", condition="REFERRING_NPI IS NOT NULL")
    token("DX_BASE", "'VISIT_TYP_VST_'||VISIT_TYPE")
    token("DX_BASE", "'BILL_TYP_BILL_TYPE_CODE_'||BILL_TYPE_CODE")
    token("DX_BASE", "'PL_'||PLACE_OF_SERVICE")
    # Legacy procedure counts any non-null indicator, not specifically Y.
    token("DX_MAPPED", "'DX_WITH_CHRONIC_INDICATOR_RECORDED'", condition="CHRONIC_INDICATOR IS NOT NULL")
    token("PX_BASE", "'PROC_TYPE_'||PROCEDURE_CODE_TYPE")
    for level in range(1, 5):
        token("RX_MAPPED", "'RX_LEVEL_"+str(level)+"_'||LEVEL_"+str(level)+"_DESC", "CLAIM_KEY")
    token("RX_BASE", "'RX_PAYER_MIX_'||INSURANCE_GROUP", "CLAIM_KEY")
    token("RX_BASE", "'RX_INSURANCE_SEGMENT_'||INSURANCE_SEGMENT", "CLAIM_KEY")
    token("RX_BASE", "'RX_CHANNEL_'||PHARMACY_CHANNEL", "CLAIM_KEY",
          "UPPER(PHARMACY_CHANNEL)<>'UNKNOWN'")
    token("RX_BASE", "'HCPS_RX_HCP_S1_'||TRIM(SPLIT_PART(PRIMARY_SPECIALTY,',',1))", "CLAIM_KEY")
    token("RX_BASE", "'CLAIMS_RX'", "CLAIM_KEY")
    token("RX_BASE", "'CLAIMS_NRX'", "CLAIM_KEY", "IS_NRX=1")
    token("RX_BASE", "'HCPS_RX'", "PROVIDER_ID", "PROVIDER_ID IS NOT NULL")
    token("RX_BASE", "'CLAIMS_RX_'||TRANSACTION_RESULT", "CLAIM_KEY", rx_paid=False)
    token("RX_BASE", "'CLAIMS_RX_ROUTE_'||ROUTE", "CLAIM_KEY", "UPPER(TRANSACTION_RESULT)='PAID'")
    token("RX_BASE", "'CLAIMS_RX_SOURCE_'||RX_SOURCE", "CLAIM_KEY", "UPPER(TRANSACTION_RESULT)='PAID'")
    token_sql = "\nUNION ALL\n".join(tokens)
    return f"""WITH REQUESTS AS ({{request_sql}}),
MAP AS (
 SELECT DISTINCT UPPER(C_TYPE) AS C_TYPE,TRIM(TO_VARCHAR(CODE)) AS CODE,
 TO_VARCHAR(\"DESC\") AS MAP_DESC,TO_VARCHAR(LEVEL_1_DESC) AS LEVEL_1_DESC,
 TO_VARCHAR(LEVEL_2_DESC) AS LEVEL_2_DESC,TO_VARCHAR(LEVEL_3_DESC) AS LEVEL_3_DESC,
 TO_VARCHAR(LEVEL_4_DESC) AS LEVEL_4_DESC,CHRONIC_INDICATOR
 FROM {mapping} WHERE UPPER(CURR_IND)='Y'
),
MED AS (
 SELECT R.PATIENT_ID,R.AS_OF_DATE,M.SERVICE_DATE::DATE AS EVENT_DATE,
 TO_JSON(ARRAY_CONSTRUCT(TO_VARCHAR(M.PATIENT_ID),TO_VARCHAR(M.SERVICE_DATE::DATE),M.DIAGNOSIS_CODES)) AS ENCOUNTER_KEY,
 IFF(M.SERVICE_DATE::DATE BETWEEN DATEADD(MONTH,-4,R.AS_OF_DATE) AND DATEADD(MONTH,-1,R.AS_OF_DATE),1,0) AS IS_RECENT,
 TO_VARCHAR(M.DIAGNOSIS_CODES) AS DIAGNOSIS_CODES,TRIM(TO_VARCHAR(M.PROCEDURE_CODE)) AS PROCEDURE_CODE,
 TO_VARCHAR(M.PROCEDURE_CODE_TYPE) AS PROCEDURE_CODE_TYPE,TRIM(TO_VARCHAR(M.NDC11)) AS NDC11,
 TO_VARCHAR(M.EVENT_SOURCE) AS EVENT_SOURCE,TO_VARCHAR(M.PLACE_OF_SERVICE) AS PLACE_OF_SERVICE,
 TO_VARCHAR(M.BILL_TYPE_CODE) AS BILL_TYPE_CODE,TO_VARCHAR(M.VISIT_TYPE) AS VISIT_TYPE,
 TO_VARCHAR(M.REFERRING_NPI) AS REFERRING_NPI,TO_VARCHAR(M.RENDERING_NPI) AS PROVIDER_ID,
 TO_VARCHAR(P.INSURANCE_GROUP) AS INSURANCE_GROUP,TO_VARCHAR(P.INSURANCE_SEGMENT) AS INSURANCE_SEGMENT,
 TO_VARCHAR(H.PRIMARY_SPECIALTY) AS PRIMARY_SPECIALTY,TO_VARCHAR(H.PROVIDER_TYPE) AS PROVIDER_TYPE
 FROM REQUESTS R JOIN {schema}.MEDICAL_EVENTS_LATEST M ON TO_VARCHAR(M.PATIENT_ID)=R.PATIENT_ID
 AND M.SERVICE_DATE::DATE BETWEEN DATEADD(MONTH,-12,R.AS_OF_DATE) AND R.AS_OF_DATE
 LEFT JOIN {schema}.PLANS_LATEST P ON M.KH_PLAN_ID=P.KH_PLAN_ID
 LEFT JOIN {schema}.PROVIDERS_LATEST H ON M.RENDERING_NPI=H.NPI
),
DX_BASE AS (SELECT * FROM MED),
DX_CODES AS (SELECT M.*,TRIM(D.VALUE::VARCHAR) AS DX_CODE FROM DX_BASE M,
 LATERAL SPLIT_TO_TABLE(TRIM(M.DIAGNOSIS_CODES,'|'),'|') D),
DX_MAPPED AS (SELECT M.*,P.MAP_DESC,P.LEVEL_1_DESC,P.LEVEL_2_DESC,P.CHRONIC_INDICATOR
 FROM DX_CODES M JOIN MAP P ON P.C_TYPE='DX' AND P.CODE=M.DX_CODE),
PX_BASE AS (SELECT * FROM MED WHERE NULLIF(PROCEDURE_CODE,'') IS NOT NULL),
PX_MAPPED AS (SELECT M.*,P.MAP_DESC,P.LEVEL_1_DESC,P.LEVEL_2_DESC,P.LEVEL_3_DESC
 FROM PX_BASE M JOIN MAP P ON P.C_TYPE='PX' AND P.CODE=M.PROCEDURE_CODE),
RX_PHARM AS (
 SELECT R.PATIENT_ID,R.AS_OF_DATE,F.FILL_DATE::DATE AS EVENT_DATE,
 'RX:'||TO_VARCHAR(F.PHARMACY_EVENT_ID) AS CLAIM_KEY,
 IFF(F.FILL_DATE::DATE BETWEEN DATEADD(MONTH,-4,R.AS_OF_DATE) AND DATEADD(MONTH,-1,R.AS_OF_DATE),1,0) AS IS_RECENT,
 TRIM(TO_VARCHAR(F.NDC11)) AS NDC11,TO_VARCHAR(F.PRESCRIBER_NPI) AS PROVIDER_ID,
 TO_VARCHAR(F.PHARMACY_CHANNEL) AS PHARMACY_CHANNEL,TO_VARCHAR(F.ROUTE) AS ROUTE,
 TO_VARCHAR(F.TRANSACTION_RESULT) AS TRANSACTION_RESULT,IFF(F.FILL_NUMBER=0,1,0) AS IS_NRX,'RX' AS RX_SOURCE,
 TO_VARCHAR(P.INSURANCE_GROUP) AS INSURANCE_GROUP,TO_VARCHAR(P.INSURANCE_SEGMENT) AS INSURANCE_SEGMENT,
 TO_VARCHAR(H.PRIMARY_SPECIALTY) AS PRIMARY_SPECIALTY
 FROM REQUESTS R JOIN {schema}.PHARMACY_EVENTS_LATEST F ON TO_VARCHAR(F.PATIENT_ID)=R.PATIENT_ID
 AND F.FILL_DATE::DATE BETWEEN DATEADD(MONTH,-12,R.AS_OF_DATE) AND R.AS_OF_DATE
 AND UPPER(F.TRANSACTION_STATUS) IN ('FINAL','STANDALONE')
 LEFT JOIN {schema}.PLANS_LATEST P ON F.PRIMARY_KH_PLAN_ID=P.KH_PLAN_ID
 LEFT JOIN {schema}.PROVIDERS_LATEST H ON F.PRESCRIBER_NPI=H.NPI
),
RX_BASE AS (
 SELECT * FROM RX_PHARM WHERE NULLIF(NDC11,'') IS NOT NULL
 UNION ALL
 SELECT PATIENT_ID,AS_OF_DATE,EVENT_DATE,'MX:'||ENCOUNTER_KEY AS CLAIM_KEY,IS_RECENT,NDC11,PROVIDER_ID,
 NULL AS PHARMACY_CHANNEL,NULL AS ROUTE,'PAID' AS TRANSACTION_RESULT,0 AS IS_NRX,'MX' AS RX_SOURCE,
 INSURANCE_GROUP,INSURANCE_SEGMENT,PRIMARY_SPECIALTY FROM MED WHERE NULLIF(NDC11,'') IS NOT NULL
),
RX_MAPPED AS (SELECT M.*,P.LEVEL_1_DESC,P.LEVEL_2_DESC,P.LEVEL_3_DESC,P.LEVEL_4_DESC
 FROM RX_BASE M JOIN MAP P ON P.C_TYPE='RX' AND P.CODE=M.NDC11),
TOKENS AS (
{token_sql}
),
COUNTS AS (
 SELECT PATIENT_ID,AS_OF_DATE,CATEGORY,COUNT(DISTINCT ENTITY_KEY) AS VALUE,
 COUNT(DISTINCT IFF(IS_RECENT=1,ENTITY_KEY,NULL)) AS RECENT_VALUE,MAX(EVENT_DATE) AS MAX_EVENT_DATE
 FROM TOKENS WHERE NULLIF(TRIM(CATEGORY),'') IS NOT NULL AND ENTITY_KEY IS NOT NULL
 GROUP BY PATIENT_ID,AS_OF_DATE,CATEGORY
),
EVENT_COVERAGE AS (
 SELECT PATIENT_ID,AS_OF_DATE,MAX(EVENT_DATE) AS MAX_EVENT_DATE
 FROM (SELECT PATIENT_ID,AS_OF_DATE,EVENT_DATE FROM MED UNION ALL
 SELECT PATIENT_ID,AS_OF_DATE,EVENT_DATE FROM RX_PHARM)
 GROUP BY PATIENT_ID,AS_OF_DATE
),
YOB AS (
 SELECT TO_VARCHAR(PATIENT_ID) AS PATIENT_ID,
 COALESCE(TRY_TO_NUMBER(TO_VARCHAR(PATIENT_YOB)),YEAR(TRY_TO_DATE(TO_VARCHAR(PATIENT_YOB)))) AS BIRTH_YEAR
 FROM {schema}.PATIENT_DEMOGRAPHICS_LATEST
),
AGES AS (
 SELECT R.PATIENT_ID,R.AS_OF_DATE,
 IFF(COUNT(DISTINCT Y.BIRTH_YEAR)=1 AND MIN(Y.BIRTH_YEAR) BETWEEN 1850 AND YEAR(R.AS_OF_DATE),
 YEAR(R.AS_OF_DATE)-MIN(Y.BIRTH_YEAR),NULL) AS AGE,
 IFF(COUNT(DISTINCT Y.BIRTH_YEAR)>1,1,0) AS AGE_CONFLICT
 FROM REQUESTS R LEFT JOIN YOB Y ON Y.PATIENT_ID=R.PATIENT_ID GROUP BY R.PATIENT_ID,R.AS_OF_DATE
)
SELECT R.PATIENT_ID,R.AS_OF_DATE,'COVERAGE' AS ROW_KIND,NULL::VARCHAR AS CATEGORY,
 NULL::NUMBER AS VALUE,NULL::NUMBER AS RECENT_VALUE,C.MAX_EVENT_DATE,A.AGE,A.AGE_CONFLICT
 FROM REQUESTS R LEFT JOIN EVENT_COVERAGE C USING(PATIENT_ID,AS_OF_DATE)
 LEFT JOIN AGES A USING(PATIENT_ID,AS_OF_DATE)
UNION ALL
SELECT PATIENT_ID,AS_OF_DATE,'FEATURE' AS ROW_KIND,CATEGORY,VALUE,RECENT_VALUE,MAX_EVENT_DATE,
 NULL AS AGE,NULL AS AGE_CONFLICT FROM COUNTS WHERE VALUE>0"""


def _feature_name(category, measure):
    clean = re.sub(r"[^A-Z0-9]+", "_", category.upper()).strip('_')[:160] or "CATEGORY"
    suffix = hashlib.sha256(category.encode('utf-8')).hexdigest()[:12]
    return "AML_" + clean + "__" + suffix + ("_COUNT" if measure == "count" else "_RECENT_PCT")


def _frame(value):
    result = value if isinstance(value, pd.DataFrame) else value.toPandas()
    result = result.copy()
    result.columns = [str(c).upper() for c in result.columns]
    if result.columns.duplicated().any():
        raise ValueError("Duplicate raw extraction columns")
    return result


def _read_long(read_query, requests, template, chunk_size):
    """Yield bounded aggregate frames; never collect all query results."""
    for start in range(0, len(requests), chunk_size):
        batch = requests.iloc[start:start+chunk_size]
        values = ["('"+patient.replace("'", "''")+"','"+date.strftime('%Y-%m-%d')+"')"
                  for patient, date in batch.itertuples(index=False, name=None)]
        request_sql = "SELECT column1::VARCHAR AS PATIENT_ID,TO_DATE(column2::VARCHAR) AS AS_OF_DATE FROM VALUES " + ','.join(values)
        result = read_query(template.replace('{request_sql}', request_sql))
        if hasattr(result, 'toLocalIterator') and not isinstance(result,pd.DataFrame):
            columns = list(result.columns)
            # SQL's UNION guarantees nulls in these numeric columns. Whole-query
            # toPandas represents nullable integral Spark columns as float64;
            # preserve that representation even in an all-FEATURE row chunk.
            integral = {'ByteType','ShortType','IntegerType','LongType'}
            float_columns = [field.name for field in getattr(getattr(result,'schema',None),'fields',[])
                             if type(field.dataType).__name__ in integral and field.nullable
                             and field.name.upper() in {'VALUE','RECENT_VALUE','AGE','AGE_CONFLICT'}]
            def make_frame(rows):
                frame=pd.DataFrame.from_records(rows, columns=columns)
                for name in float_columns:
                    frame[name]=frame[name].astype(np.float64)
                return _frame(frame)
            rows = []
            for row in result.toLocalIterator(prefetchPartitions=False):
                rows.append(tuple(row))
                if len(rows) == 4096:
                    yield make_frame(rows)
                    rows.clear()
            if rows:
                yield make_frame(rows)
        else:
            result = _frame(result)
            for row_start in range(0,len(result),4096):
                yield result.iloc[row_start:row_start+4096].copy()
        print("AutoML extraction requests completed:",min(start+chunk_size,len(requests)),"/",len(requests))


def _spool_long(read_query, requests, template, chunk_size, train_patients, db):
    """Validate once while writing an indexed private disk spool.

    raw_json stores exactly the bytes used by inputs.frame_hash before sorting;
    SQLite supplies deterministic global key order without a full pandas sort.
    """
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-16384')
    db.execute('CREATE TABLE raw(p TEXT,d TEXT,k TEXT,c TEXT,raw_json BLOB,v REAL,r REAL,e TEXT,a REAL,ac REAL, PRIMARY KEY(p,d,k,c)) WITHOUT ROWID')
    db.execute('CREATE TABLE train(p TEXT PRIMARY KEY) WITHOUT ROWID')
    db.executemany('INSERT INTO train VALUES (?)',((p,) for p in sorted(train_patients)))
    columns = None
    total = 0
    needed = {'PATIENT_ID','AS_OF_DATE','ROW_KIND','CATEGORY','VALUE','RECENT_VALUE','MAX_EVENT_DATE','AGE','AGE_CONFLICT'}
    for part in _read_long(read_query,requests,template,chunk_size):
        if not needed.issubset(part):
            raise ValueError("Raw extraction response lacks required long-form fields")
        if columns is None:
            columns = list(part.columns)
        elif columns != list(part.columns):
            raise ValueError("Raw extraction response columns changed between batches")
        if part.PATIENT_ID.isna().any():
            raise ValueError("Raw extraction contains null patient keys")
        part['PATIENT_ID'] = part.PATIENT_ID.map(str)
        for col in ('AS_OF_DATE','MAX_EVENT_DATE'):
            part[col] = pd.to_datetime(part[col],errors='raise')
            if part[col].dt.tz is not None or not part[col].dropna().eq(part[col].dropna().dt.normalize()).all():
                raise ValueError("Expected exact date-grain raw source dates")
        if not part.ROW_KIND.isin(['COVERAGE','FEATURE']).all():
            raise ValueError("Unknown raw extraction record kind")
        if part.AS_OF_DATE.isna().any() or (part.MAX_EVENT_DATE.notna() & part.MAX_EVENT_DATE.gt(part.AS_OF_DATE)).any():
            raise ValueError("Invalid or future raw source date")
        if (part.MAX_EVENT_DATE.notna() & part.MAX_EVENT_DATE.lt(part.AS_OF_DATE-pd.DateOffset(months=12))).any():
            raise ValueError("Raw source event evidence predates the 12-month window")
        records = []
        for row in part.itertuples(index=False,name=None):
            item = dict(zip(columns,row))
            feature = item['ROW_KIND'] == 'FEATURE'
            category = item['CATEGORY']
            event = item['MAX_EVENT_DATE']
            if feature:
                if pd.isna(category) or not str(category).strip():
                    raise ValueError("Raw feature category must be explicit")
                try:
                    value,recent = float(item['VALUE']),float(item['RECENT_VALUE'])
                except (TypeError,ValueError,OverflowError) as exc:
                    raise ValueError("Raw distinct counts must be finite nonnegative integers") from exc
                if not np.isfinite([value,recent]).all() or value <= 0 or recent < 0 or value != np.floor(value) or recent != np.floor(recent) or recent > value:
                    raise ValueError("Raw distinct counts must be finite nonnegative integers with valid annual/recent relationship")
                if pd.isna(event):
                    raise ValueError("Feature counts require in-window event evidence for a requested key")
                age=conflict=None
            else:
                value=recent=None
                age = None if pd.isna(item['AGE']) else float(item['AGE'])
                conflict = None if pd.isna(item['AGE_CONFLICT']) else float(item['AGE_CONFLICT'])
                if age is not None and (not np.isfinite(age) or age < 0):
                    raise ValueError("Invalid raw age")
            records.append((item['PATIENT_ID'],item['AS_OF_DATE'].strftime('%Y-%m-%d'),item['ROW_KIND'],str(category) if feature else '',
                inputs.canonical_row_bytes(row),value,recent,None if pd.isna(event) else event.strftime('%Y-%m-%d'),age,conflict))
        try:
            db.executemany('INSERT INTO raw VALUES (?,?,?,?,?,?,?,?,?,?)',records)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Duplicate raw coverage keys or aggregated raw feature rows") from exc
        total += len(part)
        db.commit()
    if columns is None:
        raise ValueError("Raw extraction response lacks required long-form fields")
    # Build once on disk: category materialization below must not scan the full
    # extraction again for each of the up to 255 retained category pairs.
    db.execute('CREATE INDEX raw_by_category ON raw(k,c,p,d)')
    coverage = pd.read_sql_query("SELECT p AS PATIENT_ID,d AS AS_OF_DATE,e AS MAX_EVENT_DATE,a AS AGE,ac AS AGE_CONFLICT FROM raw WHERE k='COVERAGE' ORDER BY p,d",db)
    coverage['AS_OF_DATE']=pd.to_datetime(coverage.AS_OF_DATE)
    coverage['MAX_EVENT_DATE']=pd.to_datetime(coverage.MAX_EVENT_DATE)
    joined=requests.merge(coverage,on=inputs.KEYS,how='outer',indicator=True,validate='one_to_one')
    if not joined._merge.eq('both').all():
        raise ValueError("Raw extraction did not return exact requested coverage keys")
    coverage=joined.drop(columns='_merge').sort_values(inputs.KEYS).reset_index(drop=True)
    coverage['IS_OBSERVED']=coverage.MAX_EVENT_DATE.notna()
    if db.execute("SELECT 1 FROM raw f LEFT JOIN raw c ON f.p=c.p AND f.d=c.d AND c.k='COVERAGE' WHERE f.k='FEATURE' AND (c.p IS NULL OR c.e IS NULL) LIMIT 1").fetchone():
        raise ValueError("Feature counts require in-window event evidence for a requested key")
    support=dict(db.execute("SELECT c,COUNT(DISTINCT raw.p) FROM raw JOIN train ON raw.p=train.p WHERE k='FEATURE' GROUP BY c"))
    h=hashlib.sha256(json.dumps(columns,separators=(',',':')).encode())
    for (payload,) in db.execute('SELECT raw_json FROM raw ORDER BY p,d,k,c'):
        h.update(payload)
    return coverage,support,h.hexdigest(),total


def _build_raw_automl_inputs(read_query, metadata, source_config, db):
    """Extract raw long rows once per unique patient/date; return AUTOML_SETUP.

    Vocabulary is label-blind and restricted to TRAIN patients. If max_features
    is explicitly configured, retain the highest distinct-TRAIN-patient support
    (ties by deterministic name), recording every excluded candidate. This
    resource bound is not model feature selection or proof of optimality.

    IS_OBSERVED means an eligible source event exists in the requested window,
    not a complete observed year. No-event rows remain NaN, never fabricated
    zero history. AGE alone cannot establish source-window observation.
    """
    if inputs is None:
        raise RuntimeError("Inject the shared inputs module before raw extraction")
    if not callable(read_query):
        raise TypeError("read_query must use the authorized existing source connection")
    cfg = _config(source_config)
    meta = inputs.normalize_snapshots(metadata)
    requests, _ = inputs.make_requests(meta, date_policy=cfg['date_policy'])
    template = raw_sql_template(cfg)
    started = time.perf_counter()
    train_patients = set(meta.loc[meta.SPLIT.eq('train'),'PATIENT_ID'])
    coverage,support,long_hash,long_rows = _spool_long(read_query, requests, template,
        cfg['chunk_size'], train_patients, db)
    candidates = []
    for category in sorted(support):
        for measure in ('count','recent_pct'):
            name = _feature_name(category, measure)
            candidates.append(dict(name=name, kind='count' if measure=='count' else 'continuous',
                definition=("Distinct raw claim/encounter/provider identifiers in inclusive preceding 12 calendar months" if measure=='count'
                            else "100*distinct recent/distinct 12-month count, rounded to integer; recent=-4 through -1 months inclusive; emitted only annual>6 and recent>2, otherwise 0"),
                raw_category=category, measure=measure, train_patients=int(support[category])))
    candidates.sort(key=lambda item: (-item['train_patients'], item['name']))
    # Reserve one slot for explicitly defined raw age when resource cap applies.
    limit = cfg['max_features']
    categories = sorted(support, key=lambda category: (-support[category], category))
    keep_categories = set(categories if limit is None else categories[:max(0,(limit-1)//2)])
    selected = [item for item in candidates if item['raw_category'] in keep_categories]
    age = dict(name='AML_AGE_YEARS', kind='continuous', definition='Calendar as-of year minus birth year; conflicting/missing birth year remains missing',
               raw_category='AGE', measure='age', train_patients=int(len(train_patients)))
    specs = sorted(selected+[age], key=lambda item:item['name'])
    names = [s['name'] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("Raw AutoML sanitizer collision")
    values = inputs.disk_array((len(coverage),len(specs)),np.float64,"tak861-panel-")
    try:
        for start in range(0,len(values),inputs.ARRAY_ROW_CHUNK):
            values[start:start+inputs.ARRAY_ROW_CHUNK] = 0.
        observed = coverage.IS_OBSERVED.to_numpy()
        values[~observed] = np.nan
        key_index = pd.MultiIndex.from_frame(coverage[inputs.KEYS])
        selected_by_category = {}
        for j,spec in enumerate(specs):
            if spec['measure']=='age':
                ages = pd.to_numeric(coverage.AGE,errors='raise').to_numpy(dtype=float)
                if np.isinf(ages).any() or np.any(ages[np.isfinite(ages)]<0):
                    raise ValueError("Invalid raw age")
                values[observed,j] = ages[observed]
            else:
                selected_by_category.setdefault(spec['raw_category'],[]).append((j,spec['measure']))
        for category,columns_for_category in selected_by_category.items():
            cursor = db.execute("SELECT p,d,v,r FROM raw WHERE k='FEATURE' AND c=? ORDER BY p,d",(category,))
            while True:
                batch = cursor.fetchmany(inputs.ARRAY_ROW_CHUNK)
                if not batch: break
                part = pd.DataFrame.from_records(batch,columns=inputs.KEYS+['VALUE','RECENT_VALUE'])
                part['AS_OF_DATE']=pd.to_datetime(part.AS_OF_DATE)
                rows = key_index.get_indexer(pd.MultiIndex.from_frame(part[inputs.KEYS]))
                annual,recent=part.VALUE.to_numpy(),part.RECENT_VALUE.to_numpy()
                pct=np.where((annual>6)&(recent>2),np.floor(100.*recent/annual+.5),0.)
                for j,measure in columns_for_category:
                    values[rows,j]=annual if measure=='count' else pct
        values.flush()
        panel_meta=coverage[inputs.KEYS+['IS_OBSERVED','MAX_EVENT_DATE']].copy()
        panel_meta['OBSERVATION_EVIDENCE']=np.where(observed,
            'eligible_source_event_in_12m_window; continuous_coverage_unknown',
            'no_eligible_source_event; window_availability_unknown')
        if values.nbytes <= inputs.PANEL_MEMORY_LIMIT:
            panel=pd.concat([panel_meta[inputs.KEYS+['IS_OBSERVED']],pd.DataFrame(np.asarray(values).copy(),columns=names)],axis=1)
            panel['MAX_EVENT_DATE']=panel_meta.MAX_EVENT_DATE
            panel['OBSERVATION_EVIDENCE']=panel_meta.OBSERVATION_EVIDENCE
            inputs.release_array(values)
        else:
            values=inputs.readonly_array(values)
            panel=inputs.DiskPanel(panel_meta,values,names)
    except BaseException:
        inputs.release_array(values)
        raise
    specs=inputs.validate_feature_specs(specs)
    contract = dict(source_id=cfg['source_schema']+' raw AutoML categories',
        source_version='LATEST tables read for this extraction; long_content_sha256='+long_hash,
        extraction_sha256=hashlib.sha256(template.encode('utf-8')).hexdigest(),
        cutoff_rule='event_date between DATEADD(month,-12,AS_OF_DATE) and AS_OF_DATE inclusive; recent -4 through -1 months inclusive',
        fit_scope='deterministic_raw',feature_semantics_evidence='User-supplied full FEATURE_CREATION_PROC raw event/category branches; label-conditioned steps removed; declared adaptations recorded',
        vocabulary_scope='TRAIN-patient raw category presence/frequency only; RESP never queried or used',
        vocabulary_train_patients_sha256=inputs.train_patient_hash(meta),
        vocabulary_sha256=inputs.stable_hash(names),source_config=cfg,
        observation_policy='eligible event evidence, not continuous observation; no-event window remains missing',
        historical_availability='unresolved: LATEST source/mapping tables do not establish historical record availability',
        raw_adaptations=['target-independent current external mapping only; no RESP-conditioned flags',
            'collision-safe JSON medical encounter key at recovered patient/date/diagnosis grain',
            'source-prefixed RX/MX identifiers avoid cross-source collisions',
            'explicit family/level namespace and deterministic hashed sanitizer replace unavailable regexp_comb',
            'ICD10PCS all_procedures_single enrichment omitted; main procedure mapping retained',
            'AGE uses raw calendar-year difference; no target-inferred binary conversion',
            'no legacy caps, VALUE_P scaling, importance filter or model refitting'])
    panel = inputs.validate_panel(panel,specs)
    manifest = inputs.make_panel_manifest(panel,specs,contract)
    removed = [item for item in candidates if item['name'] not in set(names)]
    receipt = dict(execution_status='raw_extraction_completed_on_configured_sources',
        query_count=int(np.ceil(len(requests)/cfg['chunk_size'])),requested_unique_keys=len(requests),
        returned_long_rows=long_rows,observed_window_keys=int(observed.sum()),unavailable_window_keys=int((~observed).sum()),
        age_conflict_keys=int(pd.to_numeric(coverage.AGE_CONFLICT,errors='raise').fillna(0).sum()),
        original_train_candidate_features=len(candidates)+1,retained_input_features=len(specs),
        vocabulary_policy='label-blind TRAIN patient frequency; category-name ties; keep count/recent pairs together; AGE slot reserved',
        vocabulary_max_features=limit,removed_feature_candidates=removed,
        removed_reason='predeclared input vocabulary resource cap, not outcome-based selection',
        all_candidate_specs=candidates+[age],elapsed_seconds=time.perf_counter()-started,
        source_content_sha256=long_hash,sql_template_sha256=contract['extraction_sha256'],
        source_availability_verified=False,clinical_model_results_produced=False,
        limitations=contract['raw_adaptations']+[contract['historical_availability'],contract['observation_policy']])
    manifest['raw_extraction_receipt'] = receipt
    return dict(feature_specs=specs,contract=contract,cached_panel=panel,cached_manifest=manifest,
                extractor=None,sql_template=None,receipt=receipt)


def build_raw_automl_inputs(read_query, metadata, source_config=None):
    """One extraction pass into a bounded disk spool; no full long-frame collect."""
    with tempfile.TemporaryDirectory(prefix='tak861-raw-spool-') as scratch:
        with closing(sqlite3.connect(os.path.join(scratch,'raw.sqlite'))) as db:
            return _build_raw_automl_inputs(read_query,metadata,source_config,db)
