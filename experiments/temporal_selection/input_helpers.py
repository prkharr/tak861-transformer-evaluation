"""Embedded by the delivery notebooks; no repository dependency at runtime."""
import base64
import hashlib
import io
import json
import math
import re
import tempfile
import uuid
from contextlib import contextmanager
from itertools import islice
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


@contextmanager
def _blob_reader(blob):
    """Bytes and caller-owned files share the same raw-byte hash contract."""
    if isinstance(blob, Path):
        with blob.open("rb") as stream:
            yield stream
    elif isinstance(blob, (bytes, bytearray, memoryview)):
        # BytesIO copies its input; avoid it for already allocated blobs.
        class Reader:
            offset = 0
            def read(self, size):
                block = memoryview(blob)[self.offset:self.offset + size]
                self.offset += len(block)
                return block
        yield Reader()
    else:
        raise TypeError("Artifact payload must be bytes or pathlib.Path")


def artifact_signature(blob):
    hasher, size = hashlib.sha256(), 0
    with _blob_reader(blob) as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
            size += len(block)
    return size, hasher.hexdigest()


def iter_artifact_rows(artifacts, chunk_size=50000, signatures=None):
    """Same rows as the former packer, without whole-blob/base64/list copies."""
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    for name, blob in artifacts.items():
        size, digest = signatures[name] if signatures is not None else artifact_signature(blob)
        count = max(1, math.ceil((4 * ((size + 2) // 3)) / chunk_size))
        # Encode multiples of three so concatenation equals one base64 encoding.
        raw_size = max(3, (chunk_size // 4) * 3)
        pending, index = "", 0
        with _blob_reader(blob) as stream:
            while True:
                block = stream.read(raw_size)
                if not block:
                    break
                pending += base64.b64encode(block).decode("ascii")
                while len(pending) >= chunk_size:
                    yield (name, index, count, size, digest, pending[:chunk_size])
                    pending = pending[chunk_size:]
                    index += 1
        if pending or size == 0:
            yield (name, index, count, size, digest, pending)


def pack_artifacts(artifacts, chunk_size=50000):
    """Compatibility helper for small callers; production writes use the iterator."""
    return list(iter_artifact_rows(artifacts, chunk_size))


def unpack_artifacts(rows, expected_names, *, disk_names=(), output_dir=None, ordered=False):
    """Decode ordered chunks incrementally; optional large payloads remain on disk.

    The default still accepts unordered small row collections. read_artifacts
    supplies bounded, ordered chunks, so it never takes that sorting path.
    """
    names, disk_names = set(expected_names), set(disk_names)
    if not disk_names <= names or (disk_names and output_dir is None):
        raise ValueError("disk_names must be expected names and require output_dir")
    if not ordered:
        rows = sorted(rows, key=lambda row: (row[0], row[1]))
    result, created, stream = {}, [], None
    current, count, size, digest, next_index, actual_size, pending = None, 0, 0, None, 0, 0, ""

    def finish():
        nonlocal stream
        if next_index != count or pending or actual_size != size or hasher.hexdigest() != digest:
            raise ValueError("Missing, duplicate or inconsistent artifact chunks, or length/hash mismatch.")
        if current in disk_names:
            stream.close()
            result[current] = path
        else:
            result[current] = stream.getvalue()
            stream.close()
        stream = None

    try:
        for row in rows:
            name, i, row_count, row_size, row_digest, payload = tuple(row)
            if any(value != int(value) for value in (i, row_count, row_size)):
                raise ValueError("Nonintegral artifact chunk metadata.")
            i, row_count, row_size = int(i), int(row_count), int(row_size)
            if name != current:
                if current is not None:
                    finish()
                if name not in names or name in result or row_count < 1 or row_size < 0:
                    raise ValueError("Missing, duplicate or unexpected saved artifacts.")
                current, count, size, digest = name, row_count, row_size, row_digest
                next_index, actual_size, pending, hasher = 0, 0, "", hashlib.sha256()
                if name in disk_names:
                    directory = Path(output_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    # Artifact names never become local path components.
                    path = directory / ("artifact_" + uuid.uuid4().hex)
                    stream = path.open("xb")
                    created.append(path)
                else:
                    stream = io.BytesIO()
            if i != next_index or (row_count, row_size, row_digest) != (count, size, digest):
                raise ValueError("Missing, duplicate or inconsistent artifact chunks.")
            pending += payload
            length = (len(pending) // 4) * 4
            if length:
                encoded, pending = pending[:length], pending[length:]
                block = base64.b64decode(encoded, validate=True)
                # Padding is legal only in the final chunk of an artifact.
                if "=" in encoded and next_index != count - 1:
                    raise ValueError("Unexpected base64 padding before final artifact chunk")
                stream.write(block)
                hasher.update(block)
                actual_size += len(block)
            next_index += 1
        if current is not None:
            finish()
        if set(result) != names:
            raise ValueError("Missing or unexpected saved artifacts.")
        return result
    except BaseException:
        if stream is not None:
            stream.close()
        for path in created:
            path.unlink(missing_ok=True)
        raise


ARTIFACT_COLUMNS = ["ARTIFACT_NAME", "CHUNK_INDEX", "CHUNK_COUNT", "BYTE_LENGTH", "SHA256", "PAYLOAD_BASE64"]
ARTIFACT_BATCH_ROWS = 1024


def _checked_table(table):
    if not isinstance(table, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", table):
        raise ValueError("Use uppercase letters, numbers and underscores in table names.")
    return table


def _query_artifacts(query):
    return (spark.read.format("snowflake").options(**sf_options_dl_poc)
            .option("query", query).load())


def _read_artifact_rows(table, expected_names, batch_rows=ARTIFACT_BATCH_ROWS):
    """Bound each connector result, not just the Python iterator's buffer."""
    table = _checked_table(table)
    if not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    metadata = _query_artifacts(
        f"SELECT ARTIFACT_NAME,CHUNK_COUNT,BYTE_LENGTH,SHA256,COUNT(*) AS ROW_COUNT FROM {table} "
        "GROUP BY ARTIFACT_NAME,CHUNK_COUNT,BYTE_LENGTH,SHA256").limit(len(set(expected_names)) + 1).collect()
    descriptions = {}
    for row in metadata:
        name, count, size, digest, actual_count = tuple(row)
        if (name in descriptions or name not in expected_names
                or any(value != int(value) for value in (count, size, actual_count))
                or count < 1 or size < 0 or count != actual_count):
            raise ValueError("Missing, duplicate or inconsistent artifact chunks.")
        descriptions[name] = (int(count), int(size), digest)
    if set(descriptions) != set(expected_names):
        raise ValueError("Missing or unexpected saved artifacts.")
    for name in sorted(descriptions):
        count = descriptions[name][0]
        quoted_name = "'" + str(name).replace("'", "''") + "'"
        for start in range(0, count, batch_rows):
            query = ("SELECT " + ",".join(ARTIFACT_COLUMNS) + f" FROM {table} "
                f"WHERE ARTIFACT_NAME={quoted_name} AND CHUNK_INDEX >= {start} "
                f"AND CHUNK_INDEX < {min(start + batch_rows, count)} ORDER BY CHUNK_INDEX")
            # Connector partition order is not guaranteed by the pushed ORDER BY.
            batch = _query_artifacts(query).limit(batch_rows + 1).collect()
            yield from sorted(batch, key=lambda row: row[1])
            del batch


def read_artifacts(table, expected_names, *, disk_names=(), output_dir=None, batch_rows=ARTIFACT_BATCH_ROWS):
    return unpack_artifacts(_read_artifact_rows(table, expected_names, batch_rows), expected_names,
        disk_names=disk_names, output_dir=output_dir, ordered=True)


def _verify_saved_artifacts(table, signatures, batch_rows=ARTIFACT_BATCH_ROWS):
    with tempfile.TemporaryDirectory(prefix="dl_poc_artifact_verify_") as directory:
        saved = read_artifacts(table, signatures, disk_names=signatures, output_dir=directory,
                               batch_rows=batch_rows)
        if any(artifact_signature(saved[name]) != signature for name, signature in signatures.items()):
            raise ValueError("Saved artifact read-back differs from the completed run.")


def _artifact_ddl(sql):
    # Official connector utility, with a java.util.Map overload for Python.
    # https://github.com/snowflakedb/spark-snowflake/blob/master/src/main/scala/net/snowflake/spark/snowflake/Utils.scala
    spark._jvm.net.snowflake.spark.snowflake.Utils.runQuery(dict(sf_options_dl_poc), sql)


def _has_artifact_jvm():
    """Detect JVM availability before transferring any artifact payload."""
    try:
        return getattr(spark, "_jvm", None) is not None
    except (AttributeError, NotImplementedError):
        # Shared/serverless Spark Connect sessions can raise on this attribute.
        return False


def save_artifacts(table, artifacts, *, chunk_size=50000, batch_rows=ARTIFACT_BATCH_ROWS):
    """Bounded writes with immutable, verified completed artifacts.

    Classic Spark verifies a pending table before atomic rename. Without a JVM,
    write directly using errorifexists then append. A failed direct transfer can
    leave an incomplete final-name table: readers reject its counts/hash and a
    retry fails closed, requiring a new RUN_ID. Neither mode overwrites a table.
    """
    table = _checked_table(table)
    if not artifacts or not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError("Require artifacts and positive batch_rows")
    signatures = {name: artifact_signature(blob) for name, blob in artifacts.items()}
    if table_exists(table):
        try:
            _verify_saved_artifacts(table, signatures, batch_rows)
        except ValueError as exc:
            raise FileExistsError("Destination contains incomplete or different artifacts; choose a new RUN_ID.") from exc
        print(f"Existing artifacts verified: DSVC_TAKEDA_TA_PRIVATE.DS_ML.{table}")
        return
    atomic_rename = _has_artifact_jvm()
    staging = table + "__PENDING_" + uuid.uuid4().hex.upper() if atomic_rename else table
    _checked_table(staging)
    if not atomic_rename:
        print("Artifact publication: JVM unavailable; direct immutable batches. "
              "Interrupted uploads remain invalid and require a new RUN_ID.", flush=True)
    schema = ("ARTIFACT_NAME STRING, CHUNK_INDEX INT, CHUNK_COUNT INT, "
              "BYTE_LENGTH LONG, SHA256 STRING, PAYLOAD_BASE64 STRING")
    rows, first = iter_artifact_rows(artifacts, chunk_size, signatures), True
    while True:
        batch = list(islice(rows, batch_rows))
        if not batch:
            break
        frame = spark.createDataFrame(batch, schema=schema)
        (frame.write.format("snowflake").options(**sf_options_dl_poc)
         .option("dbtable", staging).option("truncate_columns", "off")
         .mode("errorifexists" if first else "append").save())
        first = False
        del frame, batch
    _verify_saved_artifacts(staging, signatures, batch_rows)
    if atomic_rename:
        # Rename preserves the verified object/bytes; do not download it twice.
        # RENAME fails rather than replacing another writer's completed stage.
        _artifact_ddl(f"ALTER TABLE {staging} RENAME TO {table}")
    print(f"Saved and verified: DSVC_TAKEDA_TA_PRIVATE.DS_ML.{table}")
