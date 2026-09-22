"""Build four self-contained notebooks; no local helper files needed at runtime."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKSPACE = REPO.parent
OUT = WORKSPACE / "outputs" / "Brian_Selected_Features"
MIRROR = REPO / "notebooks" / "Brian Selected Features"


def read(path):
    return path.read_text(encoding="utf-8")


def storage_source():
    return read(HERE / "storage.py")


COMMON = '\n\n'.join([read(HERE / "core.py"), storage_source(),
    read(REPO / "src" / "metrics.py"), read(HERE / "warehouse.py")])
MODELS = read(HERE / "models.py")
IMPLEMENTATION_HASH = hashlib.sha256((COMMON + MODELS + read(Path(__file__))).encode()).hexdigest()
COMMON += f'\nIMPLEMENTATION_SHA256 = "{IMPLEMENTATION_HASH}"\n'

CONFIG = '''
# Paste your existing private sf_options = {...} connection setup here.
# Reuse the Spark Snowflake connector/approved compute used by the previous notebooks.
# Each notebook is self-contained and can run after a Python restart.
# Dependencies: numpy, pandas, scikit-learn, torch, matplotlib, joblib, plus the Spark Snowflake connector.
# If missing, install these on approved compute before running (no package downloads occur in these cells).
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if "sf_options" not in globals() or not isinstance(sf_options, dict):
    raise RuntimeError("Paste your existing private sf_options connection dictionary at the top of this cell.")
if "spark" not in globals():
    raise RuntimeError("Run on the Spark/Databricks environment that connected to Snowflake previously.")
sf_options_dl_poc = dict(sf_options)
DATABASE = "DSVC_TAKEDA_TA_PRIVATE"
sf_options_dl_poc.update({"sfDatabase": DATABASE, "sfSchema": "DS_ML"})
SOURCE_PREFIX = "TAK861_TX_READY_V63"
PREFIX = SOURCE_PREFIX + "_DL_POC"
EXPERIMENT_PREFIX = PREFIX + "_MANUAL_TRANSFORMER_V1"
DATASET_ID = "F001"  # Change to freeze a different source snapshot; use the same value in all four notebooks.
SUITE_ID = "S001"    # Change for different settings/code/seeds; use the same value in notebooks 03 and 04.
import re
if any(not re.fullmatch(r"[A-Z][A-Z0-9_]{0,15}", x) for x in (DATASET_ID, SUITE_ID)):
    raise ValueError("Dataset/suite IDs must be short uppercase identifiers.")
PREPARED_TABLE = f"{EXPERIMENT_PREFIX}_{DATASET_ID}_INPUTS"
SPLIT_TABLE = f"{EXPERIMENT_PREFIX}_{DATASET_ID}_SPLIT"
RUN_PREFIX = f"{EXPERIMENT_PREFIX}_{DATASET_ID}_{SUITE_ID}"
SELECTION_TABLE = RUN_PREFIX + "_SELECTION"
EVALUATION_TABLE = RUN_PREFIX + "_EVALUATION"
print("Selected snapshot features; warehouse namespace:", EXPERIMENT_PREFIX)
'''


def cell(number, title, body):
    return f"# CELL {number}: {title}\n" + dedent(body).strip() + "\n"


preparation = [
    cell(1, "Private connection and versioned output names", CONFIG),
    cell(2, "Embedded input, integrity, storage and metrics helpers", COMMON),
    cell(3, "Read the authoritative list and check MODEL_DATA columns", '''
        features, configuration_audit = configured_features()
        print("Configured feature count:", len(features))
        display(pd.DataFrame({"feature_index": range(len(features)), "feature_name": features}))
        print(canonical_json(configuration_audit))
        # Use every configured predictor even when FINAL_MODEL omits a summary row.
        # Never use importance values, FLAGS, SEQ or a manually transcribed screenshot to filter this list.
    '''),
    cell(4, "Extract the original cohort at the exact original cutoff dates", '''
        snapshots = read_table(PREFIX + "_SNAPSHOTS").select("PATIENT_ID", "END_DT", "RESP").toPandas()
        source = fetch_source_features(features)
        metadata, X = align_features(snapshots, source, features)
        population_check(metadata)
        del source
        print("Validated snapshot matrix:", X.shape, "| Dates:", metadata.END_DT.min(), "to", metadata.END_DT.max())
        print("No shifting dates, patient-only joins, or monthly replication.")
    '''),
    cell(5, "Review feature availability without fitting preprocessing", '''
        profile = pd.DataFrame({"feature": features, "missing_count": np.isnan(X).sum(axis=0),
                                "missing_fraction": np.isnan(X).mean(axis=0)})
        display(profile)
        print("Missing values remain missing in this saved input. TRAIN-only imputation follows in notebook 02.")
        print("Existing RESP is preserved. Feature event/availability timing has not been independently audited.")
    '''),
    cell(6, "Freeze feature order, values and provenance", '''
        preparation_blobs = prepared_artifacts(metadata, X, features, configuration_audit)
        preparation_blobs["manifest.json"] = json_bytes(dict(json.loads(preparation_blobs["manifest.json"]),
                                      implementation_sha256=IMPLEMENTATION_SHA256))
        print("Input fingerprint:", json.loads(preparation_blobs["manifest.json"])["raw_sha256"])
    '''),
    cell(7, "Save and read back the self-contained input bundle", '''
        save_artifacts(PREPARED_TABLE, preparation_blobs)
        verified_metadata, verified_X, verified_manifest = load_prepared()
        assert verified_manifest["raw_sha256"] == raw_hash(X, metadata, features)
        del verified_X
    '''),
    cell(8, "Preparation complete", '''
        print("Saved", len(metadata), "snapshots and", len(features), "selected features.")
        print("Next: notebook 02, with DATASET_ID =", DATASET_ID)
        print("All later notebooks load this frozen input; they do not reread changing source feature values.")
    '''),
]

splitting = [
    cell(1, "Private connection and the same dataset ID", CONFIG),
    cell(2, "Embedded input, integrity, storage and metrics helpers", COMMON),
    cell(3, "Load the frozen selected-feature dataset", '''
        metadata, X, manifest = load_prepared()
        print("Frozen input shape:", X.shape)
    '''),
    cell(4, "Verify original patient assignments against RUN_001", '''
        frozen = read_table(PREFIX + "_PATIENT_SPLIT").select(
            "PATIENT_ID", "END_DT", "RESP", "SPLIT", "SPLIT_CONFIG").toPandas()
        reference_blobs = read_artifacts(PREFIX + "_MODEL_RUN_001",
            {"checkpoint.pt", "training_summary.json", "training_history.csv", "training_history.png"})
        reference = json.loads(reference_blobs["training_summary.json"])
        metadata, reference_hashes = bind_split(metadata, frozen, reference)
        del reference_blobs
        indices = {name: np.flatnonzero(metadata.SPLIT.to_numpy() == name) for name in ("train", "validation", "test")}
        print("Snapshot keys, labels and patient assignments match original RUN_001 fingerprints.")
    '''),
    cell(5, "Fit missing-value handling and scaling on TRAIN only", '''
        preprocessor = fit_preprocessor(X[indices["train"]])
        training_values, training_missing = transform_features(X[indices["train"]], preprocessor)
        all_missing_names = [f for f, flag in zip(manifest["features"], preprocessor["all_missing_train"]) if flag]
        print("Features entirely missing in TRAIN:", all_missing_names)
        print("These remain in the fixed vocabulary, imputed to zero; missingness is encoded.")
        del training_values, training_missing
    '''),
    cell(6, "Review the preserved population and date ranges", '''
        split_summary = metadata.groupby("SPLIT").agg(snapshots=("RESP", "size"),
           patients=("PATIENT_ID", "nunique"), positives=("RESP", "sum"),
           min_end_dt=("END_DT", "min"), max_end_dt=("END_DT", "max"))
        display(split_summary)
        expected = {"train": (16256, 8712, 941), "validation": (3481, 1867, 202), "test": (3414, 1868, 202)}
        observed = {name: tuple(int(split_summary.loc[name, c]) for c in ("snapshots", "patients", "positives")) for name in expected}
        if observed != expected:
            raise ValueError("Original split population differs: " + repr(observed))
    '''),
    cell(7, "Save the split and fitted preprocessing", '''
        audit = {"raw_sha256": manifest["raw_sha256"], "reference_summary": reference,
                 "reference_hashes": reference_hashes, "preprocessor_sha256": digest_json(preprocessor),
                 "implementation_sha256": IMPLEMENTATION_SHA256,
                 "preprocessing_fit_partition": "train", "patient_overlap": 0}
        save_artifacts(SPLIT_TABLE, {"split.csv": metadata.to_csv(index=False).encode(),
            "preprocessor.json": json_bytes(preprocessor), "split_audit.json": json_bytes(audit)})
    '''),
    cell(8, "Split and preprocessing complete", '''
        verified = load_experiment()
        print("Verified experiment input ID:", verified["input_id"])
        print("Next: notebook 03. Train/validation/test assignment has not been regenerated.")
        del verified
    '''),
]

training = [
    cell(1, "Connection and predeclared selection objective", CONFIG + '''
SEED = 42
MAX_EPOCHS = 30
PATIENCE = 6
# Select by VALIDATION top-10% lift, then VALIDATION average precision, then earliest checkpoint.
# TEST is not scored by this notebook. Training lift is a fit diagnostic, not a selection criterion.
'''),
    cell(2, "Embedded models, input checks and warehouse storage", COMMON + '\n\n' + MODELS),
    cell(3, "Load the unchanged cohort; transform TRAIN and VALIDATION", '''
        data = load_experiment()
        Xt, Mt, yt = experiment_partition(data, "train")
        Xv, Mv, yv = experiment_partition(data, "validation")
        train_meta = data["metadata"].iloc[data["indices"]["train"]].reset_index(drop=True)
        val_meta = data["metadata"].iloc[data["indices"]["validation"]].reset_index(drop=True)
        print("Train tensor:", torch.as_tensor(Xt).shape, "| Validation tensor:", torch.as_tensor(Xv).shape)
    '''),
    cell(4, "Configure one Transformer on the manual-feature tensor", '''
        recipes = default_recipes()
        if len(recipes) != 1 or recipes[0]["kind"] != "transformer":
            raise ValueError("This notebook runs exactly one Transformer.")
        settings = {"seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                    "selection": "validation_top10_lift_then_average_precision"}
        suite_contract = {"input_id": data["input_id"], "implementation_sha256": IMPLEMENTATION_SHA256,
                          "recipes": recipes, "settings": settings, "dataset_id": DATASET_ID, "suite_id": SUITE_ID}
        if table_exists(SELECTION_TABLE):
            saved_selection = json.loads(read_artifacts(SELECTION_TABLE, SELECTION_NAMES)["selection.json"])
            if saved_selection["contract"] != suite_contract:
                raise ValueError("This suite already contains different inputs/settings/code. Choose a new SUITE_ID.")
        display(pd.DataFrame([dict(name=r["name"], kind=r["kind"], features=Xt.shape[1], **r["settings"]) for r in recipes]))
        print("Input tensor: [snapshots, manual features]; internal tokens: [batch, features + 1, width].")
    '''),
    cell(5, "Train the Transformer; save TRAIN and VALIDATION lift", '''
        candidate_rows, candidate_reports = [], {}
        recipe = recipes[0]
        table = candidate_table(recipe["name"])
        contract = candidate_contract(data, recipe, settings)
        if table_exists(table):
            saved = read_artifacts(table, MODEL_NAMES)
            report = json.loads(saved["candidate.json"])
            if report["contract"] != contract:
                raise ValueError("Candidate provenance differs. Choose a new SUITE_ID.")
            result = load_candidate(saved["model.bin"])
            print("Reusing verified candidate:", recipe["name"])
        else:
            result = fit_candidate(recipe, Xt, Mt, yt, Xv, Mv, yv, seed=SEED, max_epochs=MAX_EPOCHS, patience=PATIENCE)
            pt = predict_candidate(result, Xt, Mt)
            pv = predict_candidate(result, Xv, Mv)
            threshold = select_validation_threshold(yv, pv)
            tm, td, tt = split_report(train_meta, pt, threshold)
            vm, vd, vt = split_report(val_meta, pv, threshold)
            report = {"contract": contract, "threshold": threshold, "best_epoch": result["best_epoch"],
                      "train": tm, "validation": vm, "runtime": result["runtime"], "seconds": result["seconds"],
                      "train_deciles": td.to_dict("records"), "validation_deciles": vd.to_dict("records"),
                      "train_top_k": tt.to_dict("records"), "validation_top_k": vt.to_dict("records"),
                      "history": result["history"]}
            save_artifacts(table, {"model.bin": dump_candidate(result), "candidate.json": json_bytes(report),
                                  "history.csv": pd.DataFrame(result["history"]).to_csv(index=False).encode()})
        candidate_reports[recipe["name"]] = report
        candidate_rows.append({"name": recipe["name"], "kind": recipe["kind"],
            "train_top10_lift": report["train"]["top10_lift"], "train_ap": report["train"]["average_precision"],
            "validation_top10_lift": report["validation"]["top10_lift"], "validation_ap": report["validation"]["average_precision"],
            "lift_gap": report["train"]["top10_lift"]-report["validation"]["top10_lift"],
            "best_epoch": report["best_epoch"], "threshold": report["threshold"]})
        display(pd.DataFrame(candidate_rows))
        del result
    '''),
    cell(6, "Freeze the best validation checkpoint and threshold", '''
        comparison = pd.DataFrame(candidate_rows)
        winner = recipes[0]["name"]
        selected_recipe = recipes[0]
        report = candidate_reports[winner]
        selection = {"contract": suite_contract, "winner": winner, "recipe": selected_recipe,
                     "model_table": candidate_table(winner), "threshold": report["threshold"],
                     "validation_metrics": report["validation"], "input_id": data["input_id"],
                     "test_used_for_selection": False}
        save_artifacts(SELECTION_TABLE, {"selection.json": json_bytes(selection),
                                        "comparison.csv": comparison.to_csv(index=False).encode()})
        print("Saved Transformer:", winner, "| Validation top-10% lift:", report["validation"]["top10_lift"])
    '''),
    cell(7, "Plot training/validation lift and loss", '''
        import matplotlib.pyplot as plt
        axes = comparison.set_index("name")[["train_top10_lift", "validation_top10_lift"]].plot.bar(figsize=(12, 4))
        axes.set_ylabel("Top-10% lift")
        axes.set_title("One Transformer: training and validation lift")
        plt.tight_layout()
        plt.show()
        for name, report in candidate_reports.items():
            history = pd.DataFrame(report["history"])
            if len(history) > 1:
                fig, axes = plt.subplots(1, 2, figsize=(11, 3))
                history.plot(x="epoch", y=["train_loss", "validation_loss"], ax=axes[0], title=name + " loss")
                history.plot(x="epoch", y=["train_top10_lift", "validation_top10_lift"], ax=axes[1], title=name + " lift")
                fig.tight_layout()
                plt.show()
    '''),
    cell(8, "Training complete; review training lift and overfitting", '''
        for name, report in candidate_reports.items():
            print(name, "TRAIN deciles (in-sample)")
            display(pd.DataFrame(report["train_deciles"]))
        print("Next: notebook 04 evaluates the saved Transformer on TEST.")
        print("Keep validation selection separate from TEST; this existing TEST set has already been inspected.")
    '''),
]

evaluation = [
    cell(1, "Connection and the same dataset/suite IDs", CONFIG),
    cell(2, "Embedded model loading, metrics and report helpers", COMMON + '\n\n' + MODELS),
    cell(3, "Load the saved Transformer and verify provenance", '''
        data = load_experiment()
        selection_blobs = read_artifacts(SELECTION_TABLE, SELECTION_NAMES)
        selection = json.loads(selection_blobs["selection.json"])
        if selection["input_id"] != data["input_id"] or selection["contract"]["implementation_sha256"] != IMPLEMENTATION_SHA256:
            raise ValueError("Winner uses different data or code. Run the matching notebook versions.")
        if selection["test_used_for_selection"] is not False:
            raise ValueError("Selection must exclude TEST.")
        if selection["model_table"] != candidate_table(selection["winner"]):
            raise ValueError("Unexpected model artifact source.")
        saved = read_artifacts(selection["model_table"], MODEL_NAMES)
        candidate = json.loads(saved["candidate.json"])
        expected_contract = candidate_contract(data, selection["recipe"], selection["contract"]["settings"])
        if candidate["contract"] != expected_contract or candidate["threshold"] != selection["threshold"]:
            raise ValueError("Saved candidate differs from the frozen winner/threshold.")
        model = load_candidate(saved["model.bin"])
        threshold = selection["threshold"]
        print("Transformer:", selection["winner"], "| Frozen validation threshold:", threshold)
    '''),
    cell(4, "Score TRAIN, VALIDATION and TEST using the identical saved model", '''
        reports, probabilities = {}, {}
        for split in ("train", "validation", "test"):
            values, missing, labels = experiment_partition(data, split)
            scores = predict_candidate(model, values, missing)
            part = data["metadata"].iloc[data["indices"][split]].reset_index(drop=True)
            metrics, deciles, top = split_report(part, scores, threshold)
            reports[split] = {"metrics": metrics, "deciles": deciles, "top": top}
            probabilities[split] = scores
        if not np.isclose(reports["validation"]["metrics"]["top10_lift"], selection["validation_metrics"]["top10_lift"]):
            raise ValueError("Reloaded model does not reproduce validation lift.")
    '''),
    cell(5, "Display classification metrics, full deciles and capacity lift", '''
        metrics_frame = pd.DataFrame([dict(split=split, **item["metrics"]) for split, item in reports.items()])
        display(metrics_frame)
        for split, item in reports.items():
            print(split.upper(), "(in-sample fit diagnostic)" if split == "train" else "")
            display(item["deciles"])
            display(item["top"])
        print("Decile 10 contains the highest scores. Ties break by patient/date, never by outcome.")
        print("Lift = selected response rate / response rate of that same evaluation split.")
    '''),
    cell(6, "Plot lift, recall, precision-recall and confusion matrices", '''
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay
        fig, lift_png = plot_reports(reports)
        plt.show()
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        for split, scores in probabilities.items():
            labels = data["y"][data["indices"][split]]
            precision, recall, _ = precision_recall_curve(labels, scores)
            axes[0].plot(recall, precision, label=split.upper())
        axes[0].set(title="Precision-recall by split", xlabel="Recall", ylabel="Precision")
        axes[0].legend()
        test_metrics = reports["test"]["metrics"]
        ConfusionMatrixDisplay(np.array([[test_metrics["tn"], test_metrics["fp"]],
                                        [test_metrics["fn"], test_metrics["tp"]]])).plot(ax=axes[1], colorbar=False)
        axes[1].set_title("TEST at frozen validation threshold")
        fig2.tight_layout()
        chart_buffer = io.BytesIO()
        fig2.savefig(chart_buffer, format="png", dpi=150)
        classification_png = chart_buffer.getvalue()
        plt.show()
    '''),
    cell(7, "Save reproducible metrics, plots and ranked split scores", '''
        report = {"winner": selection["winner"], "input_id": data["input_id"],
                  "threshold": threshold, "features": data["manifest"]["features"],
                  "implementation_sha256": IMPLEMENTATION_SHA256,
                  "metrics": {k: v["metrics"] for k, v in reports.items()},
                  "training_lift_is_in_sample": True,
                  "selection_rule": selection["contract"]["settings"]["selection"],
                  "limitations": ["Existing TEST was already inspected in earlier work.",
                    "Selected feature list comes from the client model; its original selection population is unverified.",
                    "Historical availability of source features and 90-day label construction are not independently verified.",
                    "Snapshot metrics include correlated snapshots within each patient.",
                    "A later untouched cohort is needed for independent confirmation of an improvement."]}
        artifacts = {"evaluation.json": json_bytes(report), "metrics.csv": metrics_frame.to_csv(index=False).encode(),
                     "lift.png": lift_png, "classification.png": classification_png,
                     "training_summary.csv": selection_blobs["comparison.csv"]}
        for split, item in reports.items():
            artifacts[split + "_deciles.csv"] = item["deciles"].to_csv(index=False).encode()
            artifacts[split + "_top_k.csv"] = item["top"].to_csv(index=False).encode()
        # Row-level scores remain in the same private warehouse artifact table; no external export.
        score_rows = data["metadata"][["PATIENT_ID", "END_DT", "RESP", "SPLIT"]].copy()
        score_rows["SCORE"] = np.nan
        for split, scores in probabilities.items():
            score_rows.loc[data["indices"][split], "SCORE"] = scores
        artifacts["predictions.csv"] = score_rows.to_csv(index=False).encode()
        if table_exists(EVALUATION_TABLE):
            previous = read_artifacts(EVALUATION_TABLE, artifacts.keys())
            if previous["evaluation.json"] != artifacts["evaluation.json"]:
                raise ValueError("Existing evaluation differs. Do not overwrite; investigate the changed run.")
            print("Existing matching evaluation retained (rendered image bytes can differ by runtime).")
        else:
            save_artifacts(EVALUATION_TABLE, artifacts)
    '''),
    cell(8, "Report training lift, validation lift and retrospective TEST lift", '''
        for split in ("train", "validation", "test"):
            m = reports[split]["metrics"]
            print(f"{split.upper()}: top-10% lift={m['top10_lift']:.3f}, AP={m['average_precision']:.4f}, AUC={m['roc_auc']:.4f}")
        print("Training lift measures fit to training data; higher training lift alone is not success.")
        print("Compare experiments on validation. TEST results are retrospective, not untouched confirmation.")
        print("Compare the same top-10% lift on the same split; training lift alone does not establish improvement.")
        print("Artifacts saved under:", EVALUATION_TABLE)
    '''),
]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    MIRROR.mkdir(parents=True, exist_ok=True)
    specs = [("01_selected_feature_preparation.ipynb", preparation),
             ("02_preserve_split_and_preprocess.ipynb", splitting),
             ("03_selected_feature_training.ipynb", training),
             ("04_selected_feature_lift_evaluation.ipynb", evaluation)]
    for name, cells in specs:
        assert len(cells) == 8
        for i, source in enumerate(cells):
            compile(source, f"{name}:cell{i+1}", "exec")
        payload = {"nbformat": 4, "nbformat_minor": 5,
                   "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                "language_info": {"name": "python"}, "title": name[:-6]},
                   "cells": [{"cell_type": "code", "id": f"selected-cell-{i+1:02d}", "metadata": {},
                              "execution_count": None, "outputs": [], "source": source} for i, source in enumerate(cells)]}
        target = OUT / name
        target.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + '\n', encoding="utf-8")
        shutil.copy2(target, MIRROR / name)
    for filename in ("START_HERE.md", "VALIDATION.md"):
        source = HERE / filename
        if source.exists():
            shutil.copy2(source, OUT / filename)
            shutil.copy2(source, MIRROR / filename)
    archive = OUT.parent / "Brian_Selected_Features_4_Notebooks.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.suffix in {".ipynb", ".md"}:
                z.write(path, path.name)
    print(archive)


if __name__ == "__main__":
    build()
