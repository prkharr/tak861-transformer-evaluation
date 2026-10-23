"""Fresh source-to-count construction for the configured cohort.

The SQL reuses the recovered RX/PX/DX extraction structure with an explicit new
contract: twelve calendar positions ending at each snapshot's END_DT, source-row
RX/PX activity and exploded DX occurrences. It does not claim adjudicated use or
historical record availability. Dictionary filters never use RESP summaries.
The caller supplies an inherited dictionary and patient split membership;
historical tensor values are never reused. Adapted from the local source-builder
prototype, with cohort-specific assumptions removed.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd


DEFAULT_SOURCES = {
    "cohort": None,  # resolved by exact client table metadata
    "pharmacy": "DSVC_TAKEDA_FULLMAP_PLAID_PROD.COHORT_1009719.PHARMACY_EVENTS_LATEST",
    "medical": "DSVC_TAKEDA_FULLMAP_PLAID_PROD.COHORT_1009719.MEDICAL_EVENTS_LATEST",
    "code_reference": "DSVC_TAKEDA_TA_PRIVATE.DS_ML_PRD.DX_PX_RX_PLAID",
    "dx_reference": None,
}
POLICY = {
    "version": "fresh_source_counts_v1",
    "positions": 12,
    "orientation": "TIME_STEP=0 newest; 11 oldest; no export reversal",
    "window": "start of END_DT calendar month minus 11 months through END_DT inclusive",
    "event_date": "FILL_DATE for RX; SERVICE_DATE for DX/PX; date-granular",
    "rx_count": "source-row activity, including all transaction statuses",
    "px_count": "source-row activity; no distinct-encounter claim",
    "dx_count": "each nonblank pipe-delimited diagnosis-code occurrence",
    "claim_dedup": "none; source-row activity is intentional, not adjudicated utilization",
    "mapping_dedup": "distinct normalized family/code/category pairs",
    "mapping_multiplicity": "one code may contribute once to each distinct mapped category",
    "rx_category": "deepest nonblank LEVEL_4_DESC to LEVEL_1_DESC",
    "px_category": "nonblank LEVEL_1_DESC",
    "dx_category": "CCSR_CATEGORY_DESCRIPTION",
    "code_normalization": "RX trim; PX uppercase+trim; DX uppercase+trim+remove periods",
    "feature_names": "family + '__' + trimmed original category; categories are not renamed",
    "vocabulary": "fixed inherited 1028 count feature names; no new category selection",
    "vocabulary_scope": "historical dictionary retained; no new-cohort held-out feature discovery",
    "zero": "no mapped source activity; not proof of observation or padding",
    "record_availability": "not reconstructed; event cutoff is enforced but ingestion-as-of is unverified",
    "dictionary_timing": "upstream reference version captured at build; historical-as-of equivalence unverified",
    "population_labels": "cohort RESP preserved; shared-key changes require producer review",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _fqn(value):
    _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*\.[A-Za-z_][A-Za-z0-9_$]*\.[A-Za-z_][A-Za-z0-9_$]*", value),
             "Expected an explicit database.schema.table identifier.")
    name = value.upper()
    forbidden = ("_DL_POC_", "_TENSOR_MONTHLY", "_PATIENT_SPLIT", "_SNAPSHOTS", "_FEATURE_MAP",
                 "_MODEL_RUN_", "_INPUTS", "_PREPARED")
    _require(not any(part in name for part in forbidden), "Historical pipeline artifacts are not source-builder inputs.")
    return name


def _column(value):
    _require(isinstance(value, str) and bool(value) and '"' not in value and not any(x in value for x in ("\n", "\r", ";")),
             "Invalid column identifier.")
    return '"' + value + '"'


def _literal(value):
    _require(isinstance(value, str), "SQL literal must be text.")
    return "'" + value.replace("'", "''") + "'"


def _pandas(result):
    frame = result.copy() if isinstance(result, pd.DataFrame) else result.toPandas()
    frame.columns = [str(c).upper() for c in frame.columns]
    _require(frame.columns.is_unique, "Source result has duplicate case-insensitive columns.")
    return frame


def _schema(read_query, source):
    result = read_query("SELECT * FROM " + _fqn(source) + " WHERE 1=0")
    if isinstance(result, pd.DataFrame):
        types = {str(k): str(v) for k, v in result.dtypes.items()}
    else:
        types = {str(k): str(v) for k, v in result.dtypes}
    _require(len({c.upper() for c in types}) == len(types), "Ambiguous case-insensitive source columns.")
    return {c.upper(): {"name": c, "type": t} for c, t in types.items()}


def _needed(schema, names, role):
    _require(set(names) <= set(schema), role + " lacks required source columns: " + repr(sorted(set(names) - set(schema))))


def _discover_dx(read_query, common_scope=None):
    databases = ["DSVC_TAKEDA_TA_PRIVATE", "DSVC_TAKEDA_FULLMAP_PLAID_PROD"]
    if common_scope is not None:
        _require(isinstance(common_scope, (tuple, list)) and len(common_scope) == 2,
                 "common_scope must contain database and schema names only.")
        database, schema = common_scope
        _fqn(str(database) + "." + str(schema) + ".PROBE")
        databases.insert(0, str(database).upper())
    candidates, errors = [], []
    for database in dict.fromkeys(databases):
        query = f"""SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME
FROM {database}.INFORMATION_SCHEMA.COLUMNS
WHERE COLUMN_NAME IN ('ICD_10_CM_CODE','CCSR_CATEGORY_DESCRIPTION')
  AND TABLE_SCHEMA <> 'INFORMATION_SCHEMA'"""
        try:
            frame = _pandas(read_query(query))
        except Exception as exc:
            errors.append({"database": database, "error_class": type(exc).__name__})
            continue
        _needed({c: None for c in frame}, ["TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME"], "DX discovery")
        for key, group in frame.groupby(["TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME"]):
            if {"ICD_10_CM_CODE", "CCSR_CATEGORY_DESCRIPTION"} <= set(group.COLUMN_NAME.str.upper()):
                candidates.append(_fqn(".".join(map(str, key))))
    candidates = sorted(set(candidates))
    # The recovered code explicitly used the working common-source scope.
    if common_scope is not None:
        prefix = ".".join(map(str, common_scope)).upper() + "."
        scoped = [x for x in candidates if x.startswith(prefix) and x.rsplit(".", 1)[-1] in {"DX_CCSR_LATEST", "DX_CSSR_LATEST"}]
        if len(scoped) == 1:
            return scoped[0], {"candidates": candidates, "metadata_errors": errors, "rule": "unique exact recovered name in common scope"}
    _require(len(candidates) == 1, "Resolve one DX source with ICD_10_CM_CODE and CCSR_CATEGORY_DESCRIPTION; candidates=" + repr(candidates) + "; metadata error classes=" + repr(errors))
    return candidates[0], {"candidates": candidates, "metadata_errors": errors, "rule": "unique schema-compatible visible reference"}


def resolve_source_contract(read_query, sources=None, common_scope=None, family_column=None,
                            rx_values=("rx",), px_values=("px",)):
    """Inspect raw/reference schemas and resolve an explicit fresh-build contract.

    ``read_query(sql)`` returns a Spark DataFrame or small pandas frame. The
    upstream code reference is used directly, without a target-filtered V63 map.
    A CODE_TYPE field is usable only when the requested literal family values
    really occur; its name is never assumed to imply rx/px semantics.
    """
    incoming = {} if sources is None else dict(sources)
    _require(not (set(incoming) - set(DEFAULT_SOURCES)), "Unknown source binding.")
    resolved = {**DEFAULT_SOURCES, **incoming}
    discovery = {"rule": "explicit DX reference"}
    if resolved["dx_reference"] is None:
        resolved["dx_reference"], discovery = _discover_dx(read_query, common_scope)
    resolved = {k: _fqn(v) for k, v in resolved.items()}
    _require(resolved["code_reference"] != "DSVC_TAKEDA_TA_PRIVATE.DS_ML.TAK861_TX_READY_V63_MAPPING",
             "Use the upstream reference, not the empty/target-filtered historical V63 mapping.")
    schemas = {role: _schema(read_query, name) for role, name in resolved.items()}
    _needed(schemas["cohort"], ["PATIENT_ID", "END_DT", "RESP"], "Cohort")
    _needed(schemas["pharmacy"], ["PATIENT_ID", "FILL_DATE"], "Pharmacy")
    ndc = [c for c in schemas["pharmacy"] if re.sub(r"[^A-Z0-9]", "", c) == "NDC11"]
    _require(len(ndc) == 1, "Pharmacy must have exactly one NDC11/NDC 11 field.")
    _needed(schemas["medical"], ["PATIENT_ID", "SERVICE_DATE", "PROCEDURE_CODE", "DIAGNOSIS_CODES"], "Medical")
    _needed(schemas["code_reference"], ["CODE", "LEVEL_1_DESC", "LEVEL_2_DESC", "LEVEL_3_DESC", "LEVEL_4_DESC"], "Upstream code reference")
    _needed(schemas["dx_reference"], ["ICD_10_CM_CODE", "CCSR_CATEGORY_DESCRIPTION"], "DX reference")
    if family_column is None:
        possible = [x for x in ("C_TYPE", "CODE_TYPE") if x in schemas["code_reference"]]
        _require(len(possible) == 1, "Specify which verified reference field supplies RX/PX families; found " + repr(possible))
        family_column = possible[0]
    _require(family_column.upper() in schemas["code_reference"], "Configured family column is absent.")
    family_column = schemas["code_reference"][family_column.upper()]["name"]
    rx_values, px_values = tuple(sorted(set(str(x).strip().lower() for x in rx_values))), tuple(sorted(set(str(x).strip().lower() for x in px_values)))
    _require(rx_values and px_values and all(rx_values + px_values) and not set(rx_values).intersection(px_values), "RX/PX family literals must be nonempty and disjoint.")
    type_query = f"SELECT DISTINCT LOWER(TRIM(TO_VARCHAR({_column(family_column)}))) AS FAMILY_VALUE FROM {resolved['code_reference']}"
    types = _pandas(read_query(type_query))
    _needed({c: None for c in types}, ["FAMILY_VALUE"], "Family discovery")
    observed = sorted(set(types.FAMILY_VALUE.dropna().astype(str)))
    _require(set(rx_values + px_values) <= set(observed), "Upstream reference does not contain configured RX/PX family values; observed=" + repr(observed))
    contract = {"sources": resolved, "schemas": schemas, "family_column": family_column,
                "rx_values": list(rx_values), "px_values": list(px_values),
                "ndc_column": schemas["pharmacy"][ndc[0]]["name"], "dx_discovery": discovery,
                "observed_family_values": observed, "policy": dict(POLICY)}
    contract["contract_sha256"] = _hash(contract)
    return contract


def _check_contract(contract):
    _require(isinstance(contract, dict), "Source contract must be a dictionary.")
    payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
    _require(contract.get("contract_sha256") == _hash(payload), "Source contract changed after resolution.")
    _require(contract.get("policy") == POLICY, "Unknown or altered source-count policy.")
    for value in contract["sources"].values():
        _fqn(value)


def mapping_ctes(contract):
    _check_contract(contract)
    source, dx = contract["sources"]["code_reference"], contract["sources"]["dx_reference"]
    family = _column(contract["family_column"])
    rx_types = ",".join(map(_literal, contract["rx_values"]))
    px_types = ",".join(map(_literal, contract["px_values"]))
    return f"""rx_map AS (
 SELECT DISTINCT TRIM(TO_VARCHAR(CODE)) AS CODE,
 COALESCE(NULLIF(TRIM(TO_VARCHAR(LEVEL_4_DESC)),''), NULLIF(TRIM(TO_VARCHAR(LEVEL_3_DESC)),''),
          NULLIF(TRIM(TO_VARCHAR(LEVEL_2_DESC)),''), NULLIF(TRIM(TO_VARCHAR(LEVEL_1_DESC)),'')) AS CATEGORY
 FROM {source} WHERE LOWER(TRIM(TO_VARCHAR({family}))) IN ({rx_types})
), px_map AS (
 SELECT DISTINCT UPPER(TRIM(TO_VARCHAR(CODE))) AS CODE,
 NULLIF(TRIM(TO_VARCHAR(LEVEL_1_DESC)),'') AS CATEGORY
 FROM {source} WHERE LOWER(TRIM(TO_VARCHAR({family}))) IN ({px_types})
), dx_map AS (
 SELECT DISTINCT UPPER(REPLACE(TRIM(TO_VARCHAR(ICD_10_CM_CODE)),'.','')) AS CODE,
 NULLIF(TRIM(TO_VARCHAR(CCSR_CATEGORY_DESCRIPTION)),'') AS CATEGORY FROM {dx}
), all_mapping AS (
 SELECT 'RX' AS FAMILY, CODE, CATEGORY FROM rx_map WHERE CODE IS NOT NULL AND CODE<>'' AND CATEGORY IS NOT NULL
 UNION ALL SELECT 'PX', CODE, CATEGORY FROM px_map WHERE CODE IS NOT NULL AND CODE<>'' AND CATEGORY IS NOT NULL
 UNION ALL SELECT 'DX', CODE, CATEGORY FROM dx_map WHERE CODE IS NOT NULL AND CODE<>'' AND CATEGORY IS NOT NULL
)"""


def mapping_audit_query(contract):
    return "WITH " + mapping_ctes(contract) + """
SELECT FAMILY, COUNT(*) AS MAPPING_PAIRS, COUNT(DISTINCT CODE) AS MAPPED_CODES,
 COUNT(DISTINCT CATEGORY) AS CATEGORIES, TO_VARCHAR(HASH_AGG(HASH(FAMILY,CODE,CATEGORY))) AS MAPPING_HASH
FROM all_mapping GROUP BY FAMILY ORDER BY FAMILY"""


def cohort_query(contract):
    _check_contract(contract)
    return "SELECT PATIENT_ID, END_DT, RESP FROM " + contract["sources"]["cohort"]


def monthly_count_query(contract):
    """Push raw scans and category aggregation to Snowflake, never the driver."""
    _check_contract(contract)
    source = contract["sources"]
    ndc = _column(contract["ndc_column"])
    return f"""WITH cohort AS (
 SELECT PATIENT_ID, END_DT::DATE AS END_DT,
 DATE_TRUNC('MONTH',DATEADD('MONTH',-11,END_DT::DATE))::DATE AS WINDOW_START
 FROM {source['cohort']}
), bounds AS (
 SELECT PATIENT_ID, MIN(WINDOW_START) AS MIN_DATE, MAX(END_DT) AS MAX_DATE
 FROM cohort GROUP BY PATIENT_ID
), {mapping_ctes(contract)},
rx_code_day AS (
 SELECT r.PATIENT_ID, r.FILL_DATE::DATE AS EVENT_DATE, TRIM(TO_VARCHAR(r.{ndc})) AS CODE, COUNT(*) AS VALUE
 FROM {source['pharmacy']} r JOIN bounds b ON r.PATIENT_ID=b.PATIENT_ID
 AND r.FILL_DATE >= b.MIN_DATE AND r.FILL_DATE < DATEADD('DAY',1,b.MAX_DATE)
 WHERE r.{ndc} IS NOT NULL GROUP BY 1,2,3
), mx_day AS (
 SELECT m.PATIENT_ID, m.SERVICE_DATE::DATE AS EVENT_DATE, TO_VARCHAR(m.DIAGNOSIS_CODES) AS DIAGNOSIS_CODES,
 UPPER(TRIM(TO_VARCHAR(m.PROCEDURE_CODE))) AS PROCEDURE_CODE, COUNT(*) AS VALUE
 FROM {source['medical']} m JOIN bounds b ON m.PATIENT_ID=b.PATIENT_ID
 AND m.SERVICE_DATE >= b.MIN_DATE AND m.SERVICE_DATE < DATEADD('DAY',1,b.MAX_DATE)
 WHERE m.DIAGNOSIS_CODES IS NOT NULL OR m.PROCEDURE_CODE IS NOT NULL GROUP BY 1,2,3,4
), dx_exploded AS (
 SELECT m.PATIENT_ID, m.EVENT_DATE, m.VALUE,
 UPPER(REPLACE(TRIM(f.VALUE::VARCHAR),'.','')) AS CODE
 FROM mx_day m, LATERAL FLATTEN(INPUT=>SPLIT(m.DIAGNOSIS_CODES,'|')) f
 WHERE TRIM(f.VALUE::VARCHAR)<>''
), event_days AS (
 SELECT r.PATIENT_ID,r.EVENT_DATE,'RX' AS FAMILY,d.CATEGORY,SUM(r.VALUE) AS VALUE
 FROM rx_code_day r JOIN rx_map d ON r.CODE=d.CODE WHERE d.CATEGORY IS NOT NULL GROUP BY 1,2,3,4
 UNION ALL
 SELECT m.PATIENT_ID,m.EVENT_DATE,'PX',d.CATEGORY,SUM(m.VALUE)
 FROM mx_day m JOIN px_map d ON m.PROCEDURE_CODE=d.CODE WHERE d.CATEGORY IS NOT NULL GROUP BY 1,2,3,4
 UNION ALL
 SELECT m.PATIENT_ID,m.EVENT_DATE,'DX',d.CATEGORY,SUM(m.VALUE)
 FROM dx_exploded m JOIN dx_map d ON m.CODE=d.CODE WHERE d.CATEGORY IS NOT NULL GROUP BY 1,2,3,4
)
SELECT s.PATIENT_ID,s.END_DT,DATEDIFF('MONTH',e.EVENT_DATE,s.END_DT)::INTEGER AS TIME_STEP,
 e.FAMILY,e.FAMILY||'__'||e.CATEGORY AS FEATURE_NAME,SUM(e.VALUE) AS VALUE,
 MIN(e.EVENT_DATE) AS FIRST_EVENT_DATE,MAX(e.EVENT_DATE) AS LAST_EVENT_DATE
FROM cohort s JOIN event_days e ON s.PATIENT_ID=e.PATIENT_ID
 AND e.EVENT_DATE >= s.WINDOW_START AND e.EVENT_DATE <= s.END_DT
GROUP BY 1,2,3,4,5"""


def normalize_cohort(frame, expected_population=None):
    frame = _pandas(frame)
    _needed({c: None for c in frame}, ["PATIENT_ID", "END_DT", "RESP"], "Cohort")
    out = frame[["PATIENT_ID", "END_DT", "RESP"]].copy()
    _require(len(out) > 0 and not out.isna().any().any(), "Cohort keys/labels must be nonempty and complete.")
    _require(out.PATIENT_ID.map(lambda x: isinstance(x, str) and bool(x.strip())).all(), "Original patient keys must remain nonblank strings.")
    dates = pd.to_datetime(out.END_DT, errors="raise")
    _require(dates.dt.tz is None and dates.eq(dates.dt.normalize()).all(), "Snapshot END_DT must be an exact date.")
    out["END_DT"] = dates.dt.strftime("%Y-%m-%d")
    _require(out.RESP.isin([0, 1]).all(), "RESP must be exactly 0/1.")
    out["RESP"] = out.RESP.astype("int64")
    _require(not out.duplicated(["PATIENT_ID", "END_DT"]).any(), "Duplicate cohort snapshot keys.")
    out = out.sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    if expected_population is not None:
        observed = {"snapshots": len(out), "patients": int(out.PATIENT_ID.nunique()), "positives": int(out.RESP.sum())}
        _require(observed == expected_population, "Source V63 population differs from the declared population: " + repr(observed))
    return out


def validate_aggregate_rows(rows, metadata, feature_map):
    """Small-frame validation/reference used by tests and optional small extracts.

    The production Spark path applies equivalent bounds and uniqueness checks
    without collecting source events or the full aggregated table to pandas.
    """
    frame = _pandas(rows)
    required = ["PATIENT_ID", "END_DT", "TIME_STEP", "FEATURE_NAME", "VALUE", "FIRST_EVENT_DATE", "LAST_EVENT_DATE"]
    _needed({c: None for c in frame}, required, "Monthly aggregates")
    _require(not frame[required].isna().any().any(), "Null monthly aggregate field.")
    cohort = normalize_cohort(metadata)
    keys = set(zip(cohort.PATIENT_ID, cohort.END_DT))
    dates = pd.to_datetime(frame.END_DT, errors="raise")
    _require(dates.eq(dates.dt.normalize()).all(), "Aggregate END_DT contains intraday time.")
    frame["END_DT"] = dates.dt.strftime("%Y-%m-%d")
    _require(set(zip(frame.PATIENT_ID, frame.END_DT)) <= keys, "Unexpected aggregate snapshot.")
    _require(set(frame.FEATURE_NAME) <= set(feature_map.FEATURE_NAME), "Unknown aggregate feature.")
    steps, values = pd.to_numeric(frame.TIME_STEP), pd.to_numeric(frame.VALUE)
    _require(np.isfinite(steps).all() and steps.isin(range(12)).all(), "Invalid monthly position.")
    _require(np.isfinite(values).all() and values.gt(0).all() and values.eq(np.floor(values)).all(), "Counts must be positive finite integers.")
    _require(np.array_equal(values.to_numpy(dtype=np.float64), values.to_numpy(dtype=np.float32).astype(np.float64)), "Count loses exact float32 precision.")
    _require(not frame.duplicated(["PATIENT_ID", "END_DT", "TIME_STEP", "FEATURE_NAME"]).any(), "Duplicate monthly coordinate.")
    first, last = pd.to_datetime(frame.FIRST_EVENT_DATE), pd.to_datetime(frame.LAST_EVENT_DATE)
    period = dates.dt.to_period("M")
    starts = (period - 11).dt.to_timestamp()
    first_step = (dates.dt.year - first.dt.year) * 12 + dates.dt.month - first.dt.month
    last_step = (dates.dt.year - last.dt.year) * 12 + dates.dt.month - last.dt.month
    _require(first.ge(starts).all() and last.le(dates).all() and first.le(last).all()
             and first_step.eq(steps).all() and last_step.eq(steps).all(), "Event outside its snapshot cutoff/calendar bin.")
    return frame


def same_cohort(left, right):
    """Compare normalized values across pandas object/StringDtype versions."""
    return normalize_cohort(left).to_csv(index=False) == normalize_cohort(right).to_csv(index=False)


def audit_feature_overlap(observed, reference_map, max_missing_feature_fraction=.10):
    if not 0 <= max_missing_feature_fraction < 1:
        raise ValueError("max_missing_feature_fraction must be in [0,1)")
    missing = sorted(set(reference_map.FEATURE_NAME) - set(observed.FEATURE_NAME))
    unknown = sorted(set(observed.FEATURE_NAME) - set(reference_map.FEATURE_NAME))
    print("Count dictionary audit:", len(missing), "inherited features without activity;",
          len(unknown), "observed categories outside dictionary")
    if len(missing) / len(reference_map) > max_missing_feature_fraction:
        raise ValueError("Inherited feature inactivity exceeds declared floor; verify mapping names/source drift before explicitly adjusting max_missing_feature_fraction")
    return dict(missing_feature_fraction=len(missing)/len(reference_map),
                max_missing_feature_fraction=max_missing_feature_fraction)


def build_source_frames(spark, read_query, contract, reference_map, prepared_metadata,
                        max_missing_feature_fraction=.10):
    """Return fresh Spark frames plus a JSON-serializable source manifest.

    This function performs no warehouse writes. The caller materializes the
    arrays into the new run's INPUTS artifact. Keep
    ``cached_aggregates`` alive until all outputs are materialized, then unpersist.
    """
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    _check_contract(contract)
    cohort = normalize_cohort(read_query(cohort_query(contract)))
    if not same_cohort(cohort, prepared_metadata):
        raise ValueError("Cohort changed between audit and count extraction")
    mapping_before = _pandas(read_query(mapping_audit_query(contract))).sort_values("FAMILY").reset_index(drop=True)
    _needed({c: None for c in mapping_before}, ["FAMILY", "MAPPING_PAIRS", "MAPPED_CODES", "CATEGORIES", "MAPPING_HASH"], "Mapping audit")
    _require(set(mapping_before.FAMILY) == {"RX", "DX", "PX"} and mapping_before.MAPPING_PAIRS.gt(0).all(), "A source mapping branch is empty.")
    sql = monthly_count_query(contract)
    aggregates = read_query(sql).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        observed = aggregates.groupBy('FEATURE_NAME','FAMILY').agg(F.sum('VALUE').alias('ACTIVITY')).toPandas()
        fmap = reference_map.copy()
        overlap_audit = audit_feature_overlap(observed, fmap, max_missing_feature_fraction)
        family_counts = {str(k): int(v) for k, v in fmap.FAMILY.value_counts().items()}
        matched = observed[observed.FEATURE_NAME.isin(fmap.FEATURE_NAME)]
        if set(matched.FAMILY) != {'DX', 'PX', 'RX'}:
            raise ValueError("Fresh counts do not match all three inherited dictionary families; verify mappings")
        metadata = prepared_metadata.copy()
        split_config = dict(n_timesteps=12, method='inherited_patient_membership_plus_hash_new_ids',
            feature_order_sha256=hashlib.sha256(json.dumps(fmap.FEATURE_NAME.tolist(), ensure_ascii=False).encode()).hexdigest())
        metadata['SPLIT_CONFIG'] = _json(split_config)
        snapshots = spark.createDataFrame([(r.PATIENT_ID, pd.Timestamp(r.END_DT).date(), int(r.RESP)) for r in cohort.itertuples()],
                                         "PATIENT_ID string, END_DT date, RESP long")
        split = spark.createDataFrame([(r.PATIENT_ID, pd.Timestamp(r.END_DT).date(), int(r.RESP), r.SPLIT, r.SPLIT_CONFIG) for r in metadata.itertuples()],
                                     "PATIENT_ID string, END_DT date, RESP long, SPLIT string, SPLIT_CONFIG string")
        feature_map = spark.createDataFrame([(int(r.FEATURE_INDEX), r.FEATURE_COLUMN, r.FEATURE_NAME, r.FAMILY) for r in fmap.itertuples()],
                                           "FEATURE_INDEX long, FEATURE_COLUMN string, FEATURE_NAME string, FAMILY string")
        keys = ["PATIENT_ID", "END_DT"]
        required = keys + ["TIME_STEP", "FEATURE_NAME", "FAMILY", "VALUE", "FIRST_EVENT_DATE", "LAST_EVENT_DATE"]
        bad = None
        for name in required:
            bad = F.col(name).isNull() if bad is None else bad | F.col(name).isNull()
        start = F.add_months(F.trunc(F.col("END_DT"), "month"), -11)
        first_step = (F.year("END_DT") - F.year("FIRST_EVENT_DATE")) * 12 + F.month("END_DT") - F.month("FIRST_EVENT_DATE")
        last_step = (F.year("END_DT") - F.year("LAST_EVENT_DATE")) * 12 + F.month("END_DT") - F.month("LAST_EVENT_DATE")
        value = F.col("VALUE").cast("double")
        bad = (bad | ~F.col("TIME_STEP").isin(list(range(12))) | (value <= 0) | F.isnan(value)
               | (value != F.floor(value)) | (value.cast("float").cast("double") != value)
               | (F.col("FIRST_EVENT_DATE") < start) | (F.col("LAST_EVENT_DATE") > F.col("END_DT"))
               | (F.col("FIRST_EVENT_DATE") > F.col("LAST_EVENT_DATE"))
               | (first_step != F.col("TIME_STEP")) | (last_step != F.col("TIME_STEP")))
        _require(aggregates.where(bad).limit(1).count() == 0, "Invalid source count, coordinate or event cutoff.")
        _require(aggregates.groupBy(*(keys + ["TIME_STEP", "FEATURE_NAME"])).count().where(F.col("count") != 1).limit(1).count() == 0,
                 "Duplicate monthly source coordinate.")
        _require(aggregates.join(snapshots.select(*keys), keys, "left_anti").limit(1).count() == 0, "Source aggregation contains an unexpected snapshot.")
        mapped = aggregates.join(feature_map.select("FEATURE_NAME", "FEATURE_COLUMN"), "FEATURE_NAME", "inner")
        aliases = fmap.FEATURE_COLUMN.tolist()
        wide = mapped.groupBy(*(keys + ["TIME_STEP"])).pivot("FEATURE_COLUMN", aliases).agg(F.sum("VALUE"))
        positions = spark.range(12).select(F.col("id").cast("int").alias("TIME_STEP"))
        spine = snapshots.crossJoin(positions)
        tensor = spine.join(wide, keys + ["TIME_STEP"], "left").fillna(0, subset=aliases)
        tensor = tensor.select("PATIENT_ID", "END_DT", "RESP", "TIME_STEP", *[F.col(c).cast("float").alias(c) for c in aliases])
        # Catch source/mapping mutation across independent connector queries.
        _require(same_cohort(cohort, read_query(cohort_query(contract))), "Source cohort changed during construction.")
        mapping_after = _pandas(read_query(mapping_audit_query(contract))).sort_values("FAMILY").reset_index(drop=True)
        _require(mapping_before.equals(mapping_after), "Source mappings changed during construction.")
        # All values here describe source dictionaries or aggregate structure, not patient rows.
        audit_records = [{str(k): (v.item() if isinstance(v, np.generic) else str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                          for k, v in row.items()} for row in mapping_before.to_dict("records")]
        manifest = {"schema": 1, "contract": contract, "source_sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                    "mapping_audit": audit_records, "feature_overlap_audit": overlap_audit,
                    "feature_count": len(fmap), "expected_feature_count": 1028,
                    "unmapped_observed_categories": sorted(set(observed.FEATURE_NAME)-set(fmap.FEATURE_NAME)),
                    "out_of_vocabulary_snapshot_activity": float(observed.loc[~observed.FEATURE_NAME.isin(fmap.FEATURE_NAME),'ACTIVITY'].sum()),
                    "total_mapped_snapshot_activity": float(observed.ACTIVITY.sum()),
                    "inherited_features_without_activity": sorted(set(fmap.FEATURE_NAME)-set(observed.FEATURE_NAME)),
                    "family_counts": family_counts, "feature_map_sha256": hashlib.sha256(fmap.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
                    "population": {"snapshots": len(cohort), "patients": int(cohort.PATIENT_ID.nunique()), "positives": int(cohort.RESP.sum())},
                    "split_config": split_config, "historical_inputs_read": ['FEATURE_MAP','PATIENT_SPLIT'],
                    "validation": "source schema/family checks, unique cohort, paired mapping hashes, integer counts, exact snapshot cutoffs/calendar bins, complete new spine"}
        manifest["manifest_sha256"] = _hash(manifest)
        return {"FEATURE_MAP": feature_map, "SNAPSHOTS": snapshots, "PATIENT_SPLIT": split, "TENSOR_MONTHLY": tensor,
                "manifest": manifest, "metadata": metadata, "features": fmap.FEATURE_NAME.tolist(),
                "feature_map": fmap, "cached_aggregates": aggregates}
    except BaseException:
        aggregates.unpersist(blocking=False)
        raise
