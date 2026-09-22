import base64
ARTIFACT_COLUMNS = ["ARTIFACT_NAME", "CHUNK_INDEX", "CHUNK_COUNT", "BYTE_LENGTH", "SHA256", "PAYLOAD_BASE64"]

def table_exists(table):
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", table):
        raise ValueError("Use uppercase letters, numbers and underscores in table names.")
    query = ("SELECT TABLE_NAME FROM DSVC_TAKEDA_TA_PRIVATE.INFORMATION_SCHEMA.TABLES "
             f"WHERE TABLE_SCHEMA = 'DS_ML' AND TABLE_NAME = '{table}'")
    return (spark.read.format("snowflake").options(**sf_options_dl_poc)
            .option("query", query).load().limit(1).count() > 0)

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