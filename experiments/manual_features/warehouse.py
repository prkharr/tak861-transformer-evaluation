"""Embedded warehouse orchestration; relies on the embedded core/storage helpers."""

def read_table(table):
    return (spark.read.format("snowflake").options(**sf_options_dl_poc)
            .option("dbtable", table).load())


def read_query(query):
    return (spark.read.format("snowflake").options(**sf_options_dl_poc)
            .option("query", query).load())


def configured_features():
    rows = read_table(SOURCE_PREFIX + "_MODEL_TYPE").select("MODEL_TYPE", "FEATURES").collect()
    if len(rows) != 1:
        raise ValueError("Expected exactly one MODEL_TYPE configuration row.")
    features = parse_features(rows[0]["FEATURES"])
    summary = read_table(SOURCE_PREFIX + "_FINAL_MODEL").toPandas()
    if "FEATURES" not in summary.columns:
        raise ValueError("FINAL_MODEL lacks FEATURES.")
    summarized = summary.FEATURES.tolist()
    if any(not isinstance(f, str) or not f for f in summarized):
        raise ValueError("Invalid FINAL_MODEL feature name.")
    comparison = {"configured_model_type": str(rows[0]["MODEL_TYPE"]),
                  "configured_feature_count": len(features), "final_summary_rows": len(summary),
                  "configured_not_in_summary": sorted(set(features) - set(summarized)),
                  "summary_not_in_configuration": sorted(set(summarized) - set(features)),
                  "summary_duplicate_names": sorted(summary.loc[summary.FEATURES.duplicated(), "FEATURES"].unique().tolist()),
                  "rule": "MODEL_TYPE.FEATURES is authoritative, in its stored order; FINAL_MODEL is an audit."}
    columns = set(read_table(SOURCE_PREFIX + "_MODEL_DATA").columns)
    missing = set(["PATIENT_ID", "END_DT", "RESP"] + features).difference(columns)
    if missing:
        raise ValueError("MODEL_DATA is missing required columns: " + repr(sorted(missing)))
    return features, comparison


def fetch_source_features(features):
    fields = ["PATIENT_ID", "END_DT", "RESP"] + features
    selected = ", ".join("M." + quote_identifier(f) for f in fields)
    # Select only the frozen cohort. Duplicated source keys remain visible and fail validation.
    query = (f"SELECT {selected} FROM {DATABASE}.DS_ML.{SOURCE_PREFIX}_MODEL_DATA M "
             f"INNER JOIN (SELECT DISTINCT PATIENT_ID, END_DT FROM {DATABASE}.DS_ML.{PREFIX}_SNAPSHOTS) S "
             "ON M.PATIENT_ID = S.PATIENT_ID AND M.END_DT = S.END_DT")
    return read_query(query).toPandas()


PREPARED_NAMES = {"raw_features.npz", "snapshots.csv", "manifest.json"}
SPLIT_NAMES = {"split.csv", "preprocessor.json", "split_audit.json"}
MODEL_NAMES = {"model.bin", "candidate.json", "history.csv"}
SELECTION_NAMES = {"selection.json", "comparison.csv"}


def population_check(metadata):
    observed = (len(metadata), metadata.PATIENT_ID.nunique(), int(metadata.RESP.sum()))
    if observed != (23151, 12447, 1345):
        raise ValueError(f"Frozen V63 cohort changed: snapshots/patients/positives = {observed}")


def prepared_artifacts(metadata, X, features, comparison):
    population_check(metadata)
    manifest = {"schema_version": 1, "dataset_id": DATASET_ID,
                "feature_source": f"{DATABASE}.DS_ML.{SOURCE_PREFIX}_MODEL_DATA",
                "feature_list_source": f"{DATABASE}.DS_ML.{SOURCE_PREFIX}_MODEL_TYPE.FEATURES",
                "features": features, "feature_sha256": digest_json(features),
                "raw_sha256": raw_hash(X, metadata, features), "shape": list(X.shape),
                "min_end_dt": str(metadata.END_DT.min()), "max_end_dt": str(metadata.END_DT.max()),
                "configuration_audit": comparison,
                "representation": "One snapshot row; one numeric value per selected feature. No monthly replication.",
                "historical_feature_availability_verified": False,
                "target": "Existing RESP retained; business objective is 90-day AT escalation; source label construction not independently verified.",
                "evaluation_scope": "Retrospective comparison. The reference model's preselected list may have used patients in the current holdout; existing TEST has been inspected."}
    return {"raw_features.npz": npz_bytes(X=X), "snapshots.csv": metadata.to_csv(index=False).encode(),
            "manifest.json": json_bytes(manifest)}


def load_prepared():
    blobs = read_artifacts(PREPARED_TABLE, PREPARED_NAMES)
    manifest = json.loads(blobs["manifest.json"])
    if manifest["dataset_id"] != DATASET_ID or manifest["schema_version"] != 1:
        raise ValueError("Prepared dataset version differs.")
    features = parse_features(manifest["features"])
    metadata = normalize_metadata(pd.read_csv(io.BytesIO(blobs["snapshots.csv"]), dtype={"PATIENT_ID": str}, keep_default_na=False))
    population_check(metadata)
    with np.load(io.BytesIO(blobs["raw_features.npz"]), allow_pickle=False) as arrays:
        X = arrays["X"].copy()
    if list(X.shape) != manifest["shape"] or X.shape != (len(metadata), len(features)):
        raise ValueError("Prepared array shape differs from the manifest.")
    if np.isinf(X).any() or raw_hash(X, metadata, features) != manifest["raw_sha256"]:
        raise ValueError("Prepared inputs fail fingerprint validation.")
    return metadata, X, manifest


def load_experiment():
    metadata, X, manifest = load_prepared()
    blobs = read_artifacts(SPLIT_TABLE, SPLIT_NAMES)
    audit = json.loads(blobs["split_audit.json"])
    preprocessor = json.loads(blobs["preprocessor.json"])
    frozen = pd.read_csv(io.BytesIO(blobs["split.csv"]), dtype={"PATIENT_ID": str}, keep_default_na=False)
    metadata, hashes = bind_split(metadata, frozen, audit["reference_summary"])
    if audit["raw_sha256"] != manifest["raw_sha256"] or hashes != audit["reference_hashes"]:
        raise ValueError("Preparation and split audits refer to different inputs.")
    if digest_json(preprocessor) != audit["preprocessor_sha256"]:
        raise ValueError("Preprocessing settings changed.")
    indices = {k: np.flatnonzero(metadata.SPLIT.to_numpy() == k) for k in ("train", "validation", "test")}
    # Refit only on frozen TRAIN to verify saved settings have no other data dependency.
    if fit_preprocessor(X[indices["train"]]) != preprocessor:
        raise ValueError("Saved preprocessing differs from TRAIN-only fitting.")
    return {"raw_X": X, "metadata": metadata, "manifest": manifest, "preprocessor": preprocessor,
            "audit": audit, "indices": indices, "y": metadata.RESP.to_numpy(dtype=np.int64),
            "input_id": digest_json({"raw": manifest["raw_sha256"], "split": hashes,
                                      "preprocessor": audit["preprocessor_sha256"]})}


def experiment_partition(data, split):
    idx = data["indices"][split]
    X, missing = transform_features(data["raw_X"][idx], data["preprocessor"])
    return X, missing, data["y"][idx]


def split_report(metadata, scores, threshold):
    metrics = classification_metrics(metadata.RESP.to_numpy(), scores, threshold)
    deciles, top = rank_tables(metadata, scores)
    metrics.update({"snapshots": len(metadata), "patients": int(metadata.PATIENT_ID.nunique()),
                    "positives": int(metadata.RESP.sum()), "prevalence": float(metadata.RESP.mean()),
                    "top10_lift": float(top.loc[top.fraction == .10, "lift"].iloc[0])})
    return metrics, deciles, top


def candidate_table(name):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Unexpected recipe name.")
    return f"{RUN_PREFIX}_MODEL_{name.upper()}"


def candidate_contract(data, recipe, settings):
    return {"input_id": data["input_id"], "implementation_sha256": IMPLEMENTATION_SHA256,
            "recipe": recipe, "settings": settings, "suite_id": SUITE_ID, "dataset_id": DATASET_ID}


def plot_reports(reports, history=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for split, item in reports.items():
        deciles, top = item["deciles"], item["top"]
        axes[0].plot(deciles.decile, deciles.lift, marker="o", label=split.upper())
        axes[1].plot(top.fraction * 100, top.lift, marker="o", label=split.upper())
        axes[2].plot(top.fraction * 100, top.recall, marker="o", label=split.upper())
    axes[0].set(title="Lift by decile (10 = highest scores)", xlabel="Decile", ylabel="Lift")
    axes[0].invert_xaxis()
    axes[1].set(title="Lift at targeting capacity", xlabel="Top % of snapshots", ylabel="Lift")
    axes[2].set(title="Recall at targeting capacity", xlabel="Top % of snapshots", ylabel="Recall")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    return fig, buffer.getvalue()
