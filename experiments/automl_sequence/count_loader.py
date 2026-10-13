"""Reuse saved raw monthly counts, never a fitted model or prior predictions.

Adapted from rigorous_transformer.workflow.load_count_inputs. Four read-only
tables: FEATURE_MAP, SNAPSHOTS, PATIENT_SPLIT, TENSOR_MONTHLY. The mask means an
explicit saved calendar position exists, NOT confirmed clinical observation.
"""
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

inputs = None  # injected by standalone notebook installation


def array_sha256(value):
    hasher = hashlib.sha256()
    class Sink:
        def write(self, block):
            hasher.update(block)
            return len(block)
    np.save(Sink(), np.asarray(value), allow_pickle=False)
    return hasher.hexdigest()


def load_count_inputs(read_sf, source_prefix, source_version):
    """read_sf(suffix) returns Spark frames from the configured source prefix."""
    from pyspark.sql import functions as F
    if not source_prefix or not source_version:
        raise ValueError("Declare exact reused count source and version")
    mapping = [r.asDict() for r in read_sf("FEATURE_MAP").orderBy("FEATURE_INDEX").collect()]
    if len(mapping) != 1028:
        raise ValueError("The count reference requires the full 1,028 feature universe")
    features = [r["FEATURE_NAME"] for r in mapping]
    aliases = [r["FEATURE_COLUMN"] for r in mapping]
    if ([r["FEATURE_INDEX"] for r in mapping] != list(range(1028))
            or aliases != [f"F{i:04d}" for i in range(1028)] or len(set(features)) != 1028):
        raise ValueError("Count feature order/index/alias mismatch")
    frozen_raw = read_sf("PATIENT_SPLIT").select("PATIENT_ID", "END_DT", "RESP", "SPLIT", "SPLIT_CONFIG").toPandas()
    if frozen_raw.SPLIT_CONFIG.isna().any() or frozen_raw.SPLIT_CONFIG.nunique() != 1:
        raise ValueError("Count split creation configurations are inconsistent")
    split_creation = json.loads(frozen_raw.SPLIT_CONFIG.iloc[0])
    if split_creation.get("n_timesteps") != 12:
        raise ValueError("Count source split does not declare 12 monthly timesteps")
    frozen = inputs.normalize_snapshots(frozen_raw).sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    source_raw = read_sf("SNAPSHOTS").select("PATIENT_ID", "END_DT", "RESP").toPandas()
    source_raw["SPLIT"] = "train"
    source = inputs.normalize_snapshots(source_raw).sort_values(["PATIENT_ID", "END_DT"]).reset_index(drop=True)
    if not source[["PATIENT_ID", "END_DT", "RESP"]].equals(frozen[["PATIENT_ID", "END_DT", "RESP"]]):
        raise ValueError("Count source and frozen split cohort/labels differ")
    monthly = read_sf("TENSOR_MONTHLY")
    if set(monthly.columns) != set(["PATIENT_ID", "END_DT", "RESP", "TIME_STEP"] + aliases):
        raise ValueError("Count tensor schema differs from feature map")
    grouped = monthly.groupBy("PATIENT_ID", "END_DT", "RESP").agg(F.collect_list(F.struct(
        F.col("TIME_STEP").alias("T"), F.array(*[F.col(name).cast("double") for name in aliases]).alias("V"))).alias("SEQUENCE"))
    temporary = tempfile.TemporaryDirectory(prefix="tak861_automl_count_reference_")
    values = np.memmap(Path(temporary.name) / "counts.f32", dtype="<f4", mode="w+", shape=(len(frozen), 12, 1028))
    positions = {(r.PATIENT_ID, r.END_DT): i for i, r in enumerate(frozen.itertuples())}
    seen = np.zeros(len(frozen), dtype=bool)
    try:
        for row in grouped.toLocalIterator(prefetchPartitions=False):
            date = pd.Timestamp(row["END_DT"])
            if date.tzinfo is not None or date != date.normalize():
                raise ValueError("Count tensor dates must be exact dates")
            key = (str(row["PATIENT_ID"]), date)
            if key not in positions:
                raise ValueError("Count tensor contains unexpected snapshot")
            i = positions[key]
            seq = sorted(row["SEQUENCE"], key=lambda r: r["T"])
            if seen[i] or row["RESP"] != frozen.RESP.iloc[i] or [r["T"] for r in seq] != list(range(12)):
                raise ValueError("Count tensor duplicate, changed label or incomplete calendar positions")
            block = np.asarray([r["V"] for r in seq], dtype=np.float64)
            finite = np.isfinite(block)
            if block.shape != (12, 1028) or np.isinf(block).any() or (block[finite] < 0).any() or (block[finite] != np.floor(block[finite])).any():
                raise ValueError("Invalid raw count values")
            converted = block.astype(np.float32)
            if (converted[finite].astype(np.float64) != block[finite]).any():
                raise ValueError("Count exceeds exact float32 representation")
            values[i] = converted
            seen[i] = True
        if not seen.all():
            raise ValueError("Count source omits snapshots or zero-activity calendar rows")
        values.flush()
        values.flags.writeable = False
        # Structural validity preserves old count semantics; no claims about
        # enrollment, observation coverage or actual historical availability.
        valid = np.ones(values.shape[:2], dtype=bool)
        _, grid = inputs.make_requests(frozen, date_policy="month_end_history")
        current_order_hash = hashlib.sha256(json.dumps(features, ensure_ascii=False).encode()).hexdigest()
        contract = dict(source_id=source_prefix, source_version=source_version,
            content_sha256=array_sha256(values), feature_names=features,
            provenance_status="Recovered source construction: monthly calendar buckets; historical execution/availability unresolved",
            orientation="0=newest", date_policy="month_end_history", n_steps=12,
            asof_grid_sha256=inputs.frame_hash(grid),
            mask_kind="saved_calendar_position_not_clinical_observation",
            mask_sha256=array_sha256(valid),
            metadata_sha256=inputs.frame_hash(frozen),
            historical_feature_order_match=(split_creation.get("feature_order_sha256") == current_order_hash),
            split_creation=split_creation,
            calendar_evidence="Original count initialization: TIME_STEP calendar-month difference; current bucket truncated END_DT, older buckets complete months")
        return dict(X=values, metadata=frozen, features=features, valid=valid, contract=contract,
                    temporary_directory=temporary)
    except BaseException:
        values._mmap.close()
        temporary.cleanup()
        raise


def release_count_inputs(data):
    if data and isinstance(data.get("X"), np.memmap):
        data["X"]._mmap.close()
        data["temporary_directory"].cleanup()
