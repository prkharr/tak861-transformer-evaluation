"""Identity-preserving tensor adapters and frozen patient-level splits.

The input monthly table is the completed tensor_complete output. This module does
not reimplement claims extraction, clinical mappings, cohort rules or observability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from targeting_evaluation import KEYS, _binary, _identities, validate_manifest

PREFIXES = ("RX__", "DX__", "PX__")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class SequenceBundle:
    X: np.ndarray
    y: np.ndarray
    snapshots: pd.DataFrame
    feature_names: tuple[str, ...]
    time_steps: tuple[int, ...] = tuple(range(12))


def _dates(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        _check(parsed.notna().all(), "Invalid or missing snapshot dates.")
        _check(parsed.dt.tz is None, "Snapshot dates must be timezone-free calendar dates.")
        _check(parsed.eq(parsed.dt.normalize()).all(), "Intraday timestamps cannot be silently converted to snapshot dates.")
    except (AttributeError, TypeError) as exc:
        raise ValueError("Snapshot dates must be consistent, timezone-free calendar dates.") from None
    return parsed.dt.strftime("%Y-%m-%d")


def validate_bundle(bundle: SequenceBundle, expected_shape: tuple[int, int, int] | None = None) -> SequenceBundle:
    """Check identities and numeric content in chunks, without a whole-array copy."""
    _check(isinstance(bundle.X, np.ndarray) and bundle.X.ndim == 3, "X must be a numeric three-dimensional NumPy array.")
    n, t, f = bundle.X.shape
    _check(n > 0 and t > 0 and f > 0, "X cannot have empty dimensions.")
    _check(np.issubdtype(bundle.X.dtype, np.number) and not np.iscomplexobj(bundle.X), "X must contain real numeric counts.")
    _check(isinstance(bundle.y, np.ndarray) and bundle.y.shape == (n,), "y must be a one-dimensional array aligned with X.")
    _binary(pd.Series(bundle.y), "Tensor labels")
    _check(len(bundle.snapshots) == n, "Snapshot metadata and X have different row counts.")
    canonical = _identities(bundle.snapshots, "Tensor metadata")
    _check(np.array_equal(canonical[KEYS].to_numpy(), bundle.snapshots[KEYS].to_numpy()),
           "Tensor metadata must already use canonical string keys and dates.")
    if "RESP" in bundle.snapshots:
        _check(np.array_equal(bundle.snapshots.RESP.to_numpy(), bundle.y), "Snapshot RESP and y disagree.")
    _check(len(bundle.feature_names) == f and len(set(bundle.feature_names)) == f,
           "Feature names must be unique and exactly aligned with X's feature axis.")
    _check(all(isinstance(name, str) and name.startswith(PREFIXES) for name in bundle.feature_names),
           "Every predictor must use an established RX__, DX__, or PX__ feature name; labels are not predictors.")
    _check(tuple(bundle.time_steps) == tuple(range(t)), "Timesteps must be 0 through T-1, newest to oldest.")
    if expected_shape is not None:
        _check(tuple(bundle.X.shape) == tuple(expected_shape), "Tensor dimensions do not match the configured population contract.")
    for start in range(0, n, 256):
        chunk = bundle.X[start:start + 256]
        _check(np.isfinite(chunk).all() and (chunk >= 0).all(), "X must contain finite nonnegative raw counts.")
        _check((chunk <= np.finfo(np.float32).max).all(), "X contains values outside float32 range.")
    return bundle


def bundle_from_monthly(frame: pd.DataFrame, feature_names: list[str] | tuple[str, ...] | None = None,
                        expected_steps: int = 12) -> SequenceBundle:
    """Convert completed monthly data; never skip malformed snapshots or re-map codes."""
    required = KEYS + ["RESP", "TIME_STEP"]
    _check(isinstance(frame, pd.DataFrame) and set(required).issubset(frame.columns),
           "Monthly input requires PATIENT_ID, END_DT, RESP, and TIME_STEP.")
    _check(frame.columns.is_unique, "Monthly input contains duplicate column names.")
    _check(isinstance(expected_steps, int) and expected_steps > 0, "expected_steps must be positive.")
    available = {c for c in frame.columns if isinstance(c, str) and c.startswith(PREFIXES)}
    names = tuple(sorted(available)) if feature_names is None else tuple(feature_names)
    _check(bool(names) and len(names) == len(set(names)) and set(names) == available,
           "The supplied feature order must cover all and only the existing prefixed features.")
    metadata = frame[required].copy()
    metadata["END_DT"] = _dates(metadata.END_DT)
    _check(metadata.PATIENT_ID.notna().all() and metadata.PATIENT_ID.map(lambda x: isinstance(x, str)).all(),
           "Patient identifiers must be nonmissing strings; do not infer lost leading zeros.")
    # Canonical key ordering must not depend on pandas categorical metadata.
    metadata["PATIENT_ID"] = metadata.PATIENT_ID.astype(str)
    metadata["RESP"] = _binary(metadata.RESP, "Monthly input")
    _check(metadata.TIME_STEP.isin(range(expected_steps)).all(), "Invalid or missing timestep.")
    _check(not metadata.duplicated(KEYS + ["TIME_STEP"]).any(), "Duplicate snapshot-month rows.")
    grouped = metadata.groupby(KEYS, sort=False, observed=True)
    _check(grouped.RESP.nunique().eq(1).all(), "A snapshot has inconsistent RESP values across months.")
    _check(grouped.size().eq(expected_steps).all(), "Each snapshot must have every timestep exactly once; no snapshots may be skipped.")
    # Stable positional indexing works even when the caller's DataFrame index repeats.
    metadata["_row"] = np.arange(len(frame))
    ordered = metadata.sort_values(KEYS + ["TIME_STEP"]).reset_index(drop=True)
    order = ordered._row.to_numpy()
    # NumPy silently drops imaginary components when casting complex arrays.
    # Check before conversion, including NumPy complex scalars in object columns.
    for name in names:
        column = frame[name].to_numpy(copy=False)
        _check(not np.iscomplexobj(column)
               and not (column.dtype == object and any(isinstance(value, (complex, np.complexfloating)) for value in column)),
               "All feature values must be real numeric raw counts.")
    try:
        values = frame.loc[:, list(names)].to_numpy(dtype=np.float32)[order]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("All feature values must be numeric raw counts.") from None
    snapshots = ordered.iloc[::expected_steps][KEYS].reset_index(drop=True)
    y = ordered.iloc[::expected_steps].RESP.to_numpy(dtype=np.int64)
    X = values.reshape(len(snapshots), expected_steps, len(names))
    return validate_bundle(SequenceBundle(X, y, snapshots, names, tuple(range(expected_steps))))


def validate_population(bundle: SequenceBundle, expected: dict[str, Any]) -> None:
    validate_bundle(bundle, tuple(expected["shape"]))
    _check(bundle.snapshots.PATIENT_ID.nunique() == expected["patients"], "Patient count differs from the configured historical population.")
    _check(int(bundle.y.sum()) == expected["resp1"], "Positive snapshot count differs from the configured historical population.")
    actual = {prefix[:2]: sum(name.startswith(prefix) for name in bundle.feature_names) for prefix in PREFIXES}
    _check(actual == expected["domain_counts"], "RX/DX/PX counts differ from the configured feature vocabulary.")


def bundle_fingerprint(bundle: SequenceBundle) -> str:
    """Bind a model to actual raw array values, ordered keys/labels/features and times."""
    validate_bundle(bundle)
    digest = hashlib.sha256()
    description = {"schema": 1, "shape": list(bundle.X.shape), "feature_names": list(bundle.feature_names),
                   "time_steps": list(bundle.time_steps), "values": "canonical_float32_raw_counts"}
    digest.update(json.dumps(description, sort_keys=True, separators=(",", ":")).encode())
    digest.update(bundle.snapshots[KEYS].to_csv(index=False, lineterminator="\n").encode())
    digest.update(np.asarray(bundle.y, dtype="<i8").tobytes())
    for start in range(0, len(bundle.y), 256):
        digest.update(np.asarray(bundle.X[start:start + 256], dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


def validate_source_review(review: dict[str, Any]) -> None:
    """A required local attestation, not evidence inferred from a tensor's values."""
    _check(review.get("population_version") == "V63", "Review must identify the historical V63 population.")
    _check(str(review.get("claims_vintage")) == "20260825", "Review must identify claims vintage 20260825.")
    _check(review.get("value_representation") == "raw_counts", "Input must contain raw counts; log1p is applied once inside the training pipeline.")
    for flag in ("feature_order_verified", "snapshot_semantics_verified", "predictor_cutoff_verified", "outcome_events_excluded"):
        _check(review.get(flag) is True, f"Upstream review is incomplete: {flag}. Tensor values alone cannot verify event-time leakage.")
    _check(isinstance(review.get("review_reference"), str) and bool(review["review_reference"].strip()),
           "Provide a local review reference documenting the upstream cutoff and snapshot checks.")


def save_bundle(bundle: SequenceBundle, directory: str | Path, source_review: dict[str, Any]) -> Path:
    validate_bundle(bundle)
    validate_source_review(source_review)
    directory = Path(directory)
    _check(not directory.exists(), "Bundle directory already exists; load and verify it instead of overwriting.")
    directory.mkdir(parents=True)
    np.save(directory / "X.npy", bundle.X, allow_pickle=False)
    np.save(directory / "y.npy", bundle.y, allow_pickle=False)
    bundle.snapshots[KEYS].to_parquet(directory / "snapshots.parquet", index=False)
    (directory / "features.json").write_text(json.dumps(list(bundle.feature_names), indent=2), encoding="utf-8")
    metadata = {"schema_version": 1, "shape": list(bundle.X.shape), "time_steps": list(bundle.time_steps),
                "source_review": source_review, "bundle_sha256": bundle_fingerprint(bundle)}
    # Written last: its absence marks an incomplete export, which load_bundle rejects.
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def load_bundle(directory: str | Path) -> SequenceBundle:
    directory = Path(directory)
    _check((directory / "metadata.json").exists(), "No complete bundle metadata found; complete the identity-preserving export first.")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    _check(metadata.get("schema_version") == 1, "Unsupported bundle schema.")
    validate_source_review(metadata["source_review"])
    bundle = SequenceBundle(np.load(directory / "X.npy", mmap_mode="r", allow_pickle=False),
                            np.load(directory / "y.npy", allow_pickle=False),
                            pd.read_parquet(directory / "snapshots.parquet"),
                            tuple(json.loads((directory / "features.json").read_text(encoding="utf-8"))),
                            tuple(metadata["time_steps"]))
    validate_bundle(bundle, tuple(metadata["shape"]))
    _check(bundle_fingerprint(bundle) == metadata["bundle_sha256"], "Bundle data/order changed after export; fingerprint mismatch.")
    return bundle


def split_indices(bundle: SequenceBundle, manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Align a frozen full manifest to X without relying on CSV/Parquet row order."""
    validate_bundle(bundle)
    manifest = validate_manifest(manifest)
    left = bundle.snapshots[KEYS].copy().assign(RESP=bundle.y, _row=np.arange(len(bundle.y)))
    _check(len(left) == len(manifest), "Frozen split does not cover the exact tensor population.")
    joined = left.merge(manifest, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_manifest"))
    _check(joined.SPLIT.notna().all() and joined.RESP.eq(joined.RESP_manifest).all(),
           "Frozen split contains missing/mismatched snapshot keys or labels.")
    return {name: joined.loc[joined.SPLIT.eq(name), "_row"].to_numpy(dtype=np.int64)
            for name in ("TRAIN", "VALIDATION", "TEST")}


def freeze_patient_split(bundle: SequenceBundle, path: str | Path, seed: int = 42,
                         train_fraction: float = .70, validation_fraction: float = .15,
                         test_fraction: float = .15) -> pd.DataFrame:
    """Reuse an existing split; otherwise stratify sorted patients by max(RESP)."""
    validate_bundle(bundle)
    path = Path(path)
    _check(path.suffix.lower() == ".csv", "The frozen split path must end in .csv.")
    if path.exists():
        manifest = pd.read_csv(path, dtype={"PATIENT_ID": "string", "END_DT": "string"}, keep_default_na=False)
        split_indices(bundle, manifest)
        return validate_manifest(manifest)
    proportions = np.array([train_fraction, validation_fraction, test_fraction], dtype=float)
    _check(np.isfinite(proportions).all() and (proportions > 0).all() and np.isclose(proportions.sum(), 1),
           "Split fractions must be positive and sum to 1.")
    cohort = bundle.snapshots[KEYS].copy().assign(RESP=bundle.y)
    cohort["PATIENT_ID"] = cohort.PATIENT_ID.astype(str)
    patients = cohort.groupby("PATIENT_ID", sort=True, observed=True).RESP.max()
    _check(patients.nunique() == 2, "Patient-stratified split requires positive and negative patient groups.")
    try:
        train, temporary = train_test_split(patients.index.to_numpy(), train_size=train_fraction,
                                            stratify=patients.to_numpy(), random_state=seed)
        validation, test = train_test_split(temporary, test_size=test_fraction / (validation_fraction + test_fraction),
                                            stratify=patients.loc[temporary].to_numpy(), random_state=seed)
    except ValueError:
        raise ValueError("Too few patients in one class for the requested stratified patient split; review the cohort, do not split snapshots.") from None
    assignments = {patient: name for name, ids in [("TRAIN", train), ("VALIDATION", validation), ("TEST", test)] for patient in ids}
    manifest = validate_manifest(cohort.assign(SPLIT=cohort.PATIENT_ID.map(assignments)))
    split_indices(bundle, manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an already frozen split against concurrent runs.
    with path.open("x", encoding="utf-8", newline="") as handle:
        manifest.to_csv(handle, index=False, lineterminator="\n")
    return manifest


def split_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    checked = validate_manifest(manifest)
    table = checked.groupby("SPLIT", observed=True).agg(n_snapshots=("RESP", "size"), n_patients=("PATIENT_ID", "nunique"), n_resp1=("RESP", "sum"))
    positive_patients = checked.groupby(["SPLIT", "PATIENT_ID"], observed=True).RESP.max().groupby("SPLIT", observed=True).sum()
    table["n_positive_patients"] = positive_patients
    table["n_resp0"] = table.n_snapshots - table.n_resp1
    table["response_rate"] = table.n_resp1 / table.n_snapshots
    return table.reindex(["TRAIN", "VALIDATION", "TEST"]).reset_index()
