"""Embedded by the delivery notebooks; no repository dependency at runtime."""
import base64
import hashlib
import io
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_sf(suffix):
    return (spark.read.format("snowflake").options(**sf_options_dl_poc)
            .option("dbtable", f"{PREFIX}_{suffix}").load())


def table_exists(table):
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", table):
        raise ValueError("Use uppercase letters, numbers and underscores in table names.")
    query = ("SELECT TABLE_NAME FROM DSVC_TAKEDA_TA_PRIVATE.INFORMATION_SCHEMA.TABLES "
             f"WHERE TABLE_SCHEMA = 'DS_ML' AND TABLE_NAME = '{table}'")
    return (spark.read.format("snowflake").options(**sf_options_dl_poc)
            .option("query", query).load().limit(1).count() > 0)


def checked_metadata(frame, include_split=False):
    columns = ["PATIENT_ID", "END_DT", "RESP"]
    if include_split:
        columns += ["SPLIT", "SPLIT_CONFIG"]
    out = frame.loc[:, columns].copy()
    if out.empty or out.isna().any().any():
        raise ValueError("Snapshot metadata must be nonempty and contain no nulls.")
    if not out.PATIENT_ID.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("PATIENT_ID must retain its original nonempty string value.")
    dates = pd.to_datetime(out.END_DT, errors="raise")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("END_DT must be a date without an intraday time/timezone.")
    out["END_DT"] = dates.dt.strftime("%Y-%m-%d")
    if not out.RESP.isin([0, 1]).all():
        raise ValueError("RESP must be exactly 0 or 1 before conversion.")
    out["RESP"] = out.RESP.astype("int64")
    if out.duplicated(["PATIENT_ID", "END_DT"]).any():
        raise ValueError("Duplicate patient/date keys in snapshot metadata.")
    if include_split:
        if set(out.SPLIT) != {"train", "validation", "test"}:
            raise ValueError("Expected the saved train, validation and test assignments.")
        if out.groupby("PATIENT_ID", observed=True).SPLIT.nunique().gt(1).any():
            raise ValueError("Patient overlap between splits.")
        if len(out.SPLIT_CONFIG.unique()) != 1:
            raise ValueError("The split contains inconsistent creation settings.")
        for _, part in out.groupby("SPLIT", observed=True):
            if set(part.RESP) != {0, 1}:
                raise ValueError("Each split must contain both response classes.")
    return out.sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)


def validate_metadata_pair(snapshot_frame, manifest_frame, features):
    source = checked_metadata(snapshot_frame)
    manifest = checked_metadata(manifest_frame, include_split=True)
    if not source.equals(manifest[["PATIENT_ID", "END_DT", "RESP"]]):
        raise ValueError("Frozen split and source snapshots differ in keys or labels.")
    creation = json.loads(manifest.SPLIT_CONFIG.iloc[0])
    # Match the feature-order hash created in the completed split notebook.
    feature_order_hash = hashlib.sha256(
        json.dumps(features, ensure_ascii=False).encode("utf-8")).hexdigest()
    if creation.get("feature_order_sha256") != feature_order_hash or creation.get("n_timesteps") != 12:
        raise ValueError("Feature order or timesteps differ from the frozen split.")
    return manifest, creation


def new_tensor_buffer(metadata, feature_count, seq_len=12):
    temporary = tempfile.TemporaryDirectory(prefix="dl_poc_")
    path = Path(temporary.name) / "counts.float32"
    X = np.memmap(path, dtype="<f4", mode="w+", shape=(len(metadata), seq_len, feature_count))
    return temporary, X


def fill_tensor(X, metadata, sequence_rows):
    """Place rows by canonical keys, independent of Spark partition order."""
    positions = {(r.PATIENT_ID, r.END_DT): i for i, r in enumerate(metadata.itertuples())}
    seen = np.zeros(len(metadata), dtype=bool)
    seq_len, feature_count = X.shape[1:]
    for row in sequence_rows:
        raw_date = row["END_DT"]
        if raw_date is None or row["PATIENT_ID"] is None:
            raise ValueError("Null monthly snapshot key.")
        date = pd.Timestamp(raw_date)
        if date.tzinfo is not None or date != date.normalize():
            raise ValueError("Monthly END_DT is not an exact date.")
        key = (row["PATIENT_ID"], date.strftime("%Y-%m-%d"))
        if key not in positions:
            raise ValueError("Unexpected monthly snapshot key.")
        i = positions[key]
        if seen[i] or row["RESP"] != int(metadata.RESP.iloc[i]):
            raise ValueError("Duplicate monthly snapshot or changed label.")
        sequence = row["SEQUENCE"]
        steps = [month["T"] for month in sequence]
        if len(sequence) != seq_len or any(t is None for t in steps):
            raise ValueError("Expected exactly 12 complete timesteps per snapshot.")
        # Check original values before any integer conversion.
        if sorted(steps) != list(range(seq_len)):
            raise ValueError("Timesteps must be unique integers 0 through 11.")
        sequence = sorted(sequence, key=lambda month: month["T"])
        values = np.asarray([month["V"] for month in sequence], dtype=np.float32)
        if values.shape != (seq_len, feature_count):
            raise ValueError("Monthly feature shape does not match the vocabulary.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Counts must be finite, nonnegative float32 values with no nulls.")
        X[i] = values
        seen[i] = True
    if not seen.all():
        raise ValueError("Monthly data is missing original snapshots, including zero-activity sequences.")
    X.flush()


def input_fingerprints(X, metadata, features):
    tensor_hash = hashlib.sha256()
    tensor_hash.update(canonical_json(list(X.shape)).encode("utf-8"))
    for start in range(0, len(X), 128):
        tensor_hash.update(np.asarray(X[start:start + 128], dtype="<f4").tobytes(order="C"))
    records = [[str(r.PATIENT_ID), str(r.END_DT), int(r.RESP), str(r.SPLIT)]
               for r in metadata.itertuples()]
    return {"model_input_sha256": tensor_hash.hexdigest(),
            "snapshot_manifest_sha256": digest_json(records),
            "feature_names_sha256": digest_json(features),
            "split_config_sha256": digest_json(json.loads(metadata.SPLIT_CONFIG.iloc[0]))}


def load_inputs():
    from pyspark.sql import functions as F
    mapping = read_sf("FEATURE_MAP").orderBy("FEATURE_INDEX").collect()
    if not mapping or [r["FEATURE_INDEX"] for r in mapping] != list(range(len(mapping))):
        raise ValueError("Invalid feature indices.")
    features = [r["FEATURE_NAME"] for r in mapping]
    aliases = [r["FEATURE_COLUMN"] for r in mapping]
    if (aliases != [f"F{i:04d}" for i in range(len(features))]
            or not all(isinstance(f, str) and f for f in features)
            or len(set(features)) != len(features)):
        raise ValueError("Invalid feature names, aliases or order.")
    source = read_sf("SNAPSHOTS").select("PATIENT_ID", "END_DT", "RESP").toPandas()
    frozen = read_sf("PATIENT_SPLIT").select(
        "PATIENT_ID", "END_DT", "RESP", "SPLIT", "SPLIT_CONFIG").toPandas()
    metadata, creation = validate_metadata_pair(source, frozen, features)
    observed = (len(metadata), metadata.PATIENT_ID.nunique(), int(metadata.RESP.sum()), len(features))
    if observed != (23151, 12447, 1345, 1028):
        raise ValueError(f"V63 population changed: snapshots/patients/positives/features = {observed}.")
    monthly = read_sf("TENSOR_MONTHLY")
    if set(monthly.columns) != set(["PATIENT_ID", "END_DT", "RESP", "TIME_STEP"] + aliases):
        raise ValueError("Monthly table columns differ from the frozen feature map.")
    print("Building the model input in a temporary driver file (about 1.06 GiB).", flush=True)
    grouped = (monthly.groupBy("PATIENT_ID", "END_DT", "RESP")
        .agg(F.collect_list(F.struct(
            F.col("TIME_STEP").alias("T"),
            F.array(*[F.when(F.col(name) >= 0, F.col(name).cast("float"))
                      .otherwise(F.lit(None).cast("float")) for name in aliases]).alias("V")
        )).alias("SEQUENCE"))
        .repartition(128))
    temporary, X = new_tensor_buffer(metadata, len(features))
    try:
        fill_tensor(X, metadata, grouped.toLocalIterator(prefetchPartitions=False))
        hashes = input_fingerprints(X, metadata, features)
        X.flags.writeable = False
    except BaseException:
        X._mmap.close()
        temporary.cleanup()
        raise
    y = metadata.RESP.to_numpy(dtype=np.float32)
    indices = {name: np.flatnonzero(metadata.SPLIT.to_numpy() == name)
               for name in ("train", "validation", "test")}
    print(f"Validated tensor shape {X.shape}; patient overlap = 0.", flush=True)
    return {"X": X, "y": y, "metadata": metadata, "features": features,
            "indices": indices, "hashes": hashes, "split_creation": creation,
            "temporary_directory": temporary}


def pack_artifacts(artifacts, chunk_size=50000):
    rows = []
    for name, blob in artifacts.items():
        encoded = base64.b64encode(blob).decode("ascii")
        pieces = [encoded[i:i + chunk_size] for i in range(0, len(encoded), chunk_size)] or [""]
        digest = hashlib.sha256(blob).hexdigest()
        rows.extend((name, i, len(pieces), len(blob), digest, piece)
                    for i, piece in enumerate(pieces))
    return rows


def unpack_artifacts(rows, expected_names):
    groups = {}
    for row in rows:
        name, i, count, size, digest, payload = tuple(row)
        if any(value != int(value) for value in (i, count, size)):
            raise ValueError("Nonintegral artifact chunk metadata.")
        groups.setdefault(name, []).append((int(i), int(count), int(size), digest, payload))
    if set(groups) != set(expected_names):
        raise ValueError("Missing or unexpected saved artifacts.")
    result = {}
    for name, pieces in groups.items():
        pieces.sort(key=lambda p: p[0])
        count, size, digest = pieces[0][1:4]
        if (count < 1 or size < 0 or len(pieces) != count
                or [p[0] for p in pieces] != list(range(count))
                or any(p[1:4] != (count, size, digest) for p in pieces)):
            raise ValueError("Missing, duplicate or inconsistent artifact chunks.")
        blob = base64.b64decode("".join(p[4] for p in pieces), validate=True)
        if len(blob) != size or hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("Artifact length/hash mismatch.")
        result[name] = blob
    return result


ARTIFACT_COLUMNS = ["ARTIFACT_NAME", "CHUNK_INDEX", "CHUNK_COUNT", "BYTE_LENGTH", "SHA256", "PAYLOAD_BASE64"]


def read_artifacts(table, expected_names):
    rows = (spark.read.format("snowflake").options(**sf_options_dl_poc)
            .option("dbtable", table).load().select(*ARTIFACT_COLUMNS).collect())
    return unpack_artifacts(rows, expected_names)


def save_artifacts(table, artifacts):
    # A matching existing result can be verified after an interrupted read-back.
    if table_exists(table):
        if read_artifacts(table, artifacts) != artifacts:
            raise FileExistsError("Destination contains different artifacts; choose a new RUN_ID.")
        print(f"Existing artifacts verified: DSVC_TAKEDA_TA_PRIVATE.DS_ML.{table}")
        return
    schema = ("ARTIFACT_NAME STRING, CHUNK_INDEX INT, CHUNK_COUNT INT, "
              "BYTE_LENGTH LONG, SHA256 STRING, PAYLOAD_BASE64 STRING")
    frame = spark.createDataFrame(pack_artifacts(artifacts), schema=schema)
    (frame.write.format("snowflake").options(**sf_options_dl_poc)
     .option("dbtable", table).option("truncate_columns", "off")
     .mode("errorifexists").save())
    if read_artifacts(table, artifacts) != artifacts:
        raise ValueError("Saved artifact read-back differs from the completed run.")
    print(f"Saved and verified: DSVC_TAKEDA_TA_PRIVATE.DS_ML.{table}")
