"""Generate four self-contained Databricks notebooks for controlled V63 experiments."""
import hashlib
import json
from pathlib import Path
from textwrap import dedent

from early_stages import early_notebook_cells

HERE = Path(__file__).resolve().parent
if HERE.name == "dl_poc_experiment_work":
    REPO = HERE.parent.parent / "tak861-transformer-evaluation"
    OUT = HERE.parent / "Brian_Experiments"
else:
    REPO = HERE.parent.parent
    OUT = REPO / "notebooks" / "Brian Experiments"


def read(path):
    return path.read_text(encoding="utf-8")


def cell(number, title, body):
    return f"# CELL {number} — {title}\n" + dedent(body).strip() + "\n"


def write_notebook(name, cells):
    if len(cells) != 8:
        raise ValueError("Each notebook must have exactly eight code cells.")
    payload = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "title": name[:-6]},
        "nbformat": 4, "nbformat_minor": 5}
    for i, source in enumerate(cells, 1):
        compile(source, f"{name}:cell{i}", "exec")
        payload["cells"].append({"cell_type": "code", "id": f"brian-v2-cell-{i:02d}",
                                 "execution_count": None, "metadata": {}, "outputs": [], "source": source})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


CONNECTION = dedent('''
    # Paste your existing private sf_options = {...} setup above this check.
    # Use the same approved compute that connected to Snowflake for RUN_001.
    # No passwords, keys or tokens are included in these files.
    if "sf_options" not in globals():
        raise RuntimeError("Add the existing private sf_options connection setup at the top of this cell.")
    sf_options_dl_poc = sf_options.copy()
    sf_options_dl_poc.update({"sfDatabase": "DSVC_TAKEDA_TA_PRIVATE", "sfSchema": "DS_ML"})
    PREFIX = "TAK861_TX_READY_V63_DL_POC"
    EXPERIMENT_PREFIX = PREFIX + "_BRIAN_V2"
    REFERENCE_RUN_ID = "RUN_001"  # Original saved model anchors input/split fingerprints.
    TARGET_SPECIFICATION = {
        "business_objective": "Rank patients by likelihood of advanced-therapy escalation in the next 90 days",
        "prediction_horizon_days": 90,
        "target_column": "RESP",
        "source": "Client slide Predicting Escalation Risk, shared from the call around 12:30",
        "label_construction_verified_against_business_definition": False,
        "status": "Business definition recorded; existing RESP is preserved without asserting its construction matches the 90-day horizon",
    }
''').strip()

INPUT_SOURCE = read(HERE / "input_helpers.py")
components = [read(REPO / "src" / filename) for filename in ("model.py", "metrics.py", "reproducibility.py")]
components += [read(HERE / filename) for filename in ("feature_selection.py", "experiment_training.py", "experiment_reporting.py")]
implementation_hash = hashlib.sha256((INPUT_SOURCE + "\n" + "\n".join(components)).encode("utf-8")).hexdigest()
DEPENDENCIES = dedent('''
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        import sklearn
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as error:
        raise RuntimeError("Use an approved runtime with PyTorch, scikit-learn, NumPy, pandas and matplotlib.") from error
    print("PyTorch:", torch.__version__, "| scikit-learn:", sklearn.__version__, "| CUDA:", torch.cuda.is_available())
''') + INPUT_SOURCE + "\n\n" + "\n\n".join(components)
DEPENDENCIES += f'\nIMPLEMENTATION_SHA256 = "{implementation_hash}"\n'

SUITE_SETUP = dedent('''
    SUITE_ID = "BRIAN_001"  # New name for any changed settings, feature rules or training seed.
    import re
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,19}", SUITE_ID):
        raise ValueError("SUITE_ID must use 1-20 uppercase letters, digits or underscores, starting with a letter.")
    SELECTION_TABLE = f"{EXPERIMENT_PREFIX}_SELECTION_{SUITE_ID}"
    PREPARATION_TABLE = EXPERIMENT_PREFIX + "_PREPARATION"
    SPLIT_AUDIT_TABLE = EXPERIMENT_PREFIX + "_SPLIT_AUDIT"
''')

LOAD_INPUTS = dedent('''
    if "data" in globals():
        release_experiment_inputs(data)
    for table in (PREPARATION_TABLE, SPLIT_AUDIT_TABLE):
        if not table_exists(table):
            raise FileNotFoundError("Run and save notebooks 01 and 02 before this stage.")
    preparation = json.loads(read_artifacts(PREPARATION_TABLE, ["preparation_report.json"])["preparation_report.json"])
    split_audit = json.loads(read_artifacts(SPLIT_AUDIT_TABLE, ["split_audit.json"])["split_audit.json"])
    data = load_inputs()
    try:
        verify_preparation_audits(data, preparation, split_audit)
    except BaseException:
        release_experiment_inputs(data)
        raise
    display(data["metadata"].groupby("SPLIT", observed=True).agg(
        patients=("PATIENT_ID", "nunique"), snapshots=("RESP", "size"), positive_snapshots=("RESP", "sum")))
    print("Original tensor and patient assignments match the saved preparation audits.")
''')

training = [
    cell(1, "Connection, experiment suite and pending core-feature list", CONNECTION + "\n" + SUITE_SETUP + dedent('''
        TRAINING_SEED = 42  # Change only with a new SUITE_ID; patient assignments stay fixed.
        FEATURE_SELECTION_SEED = 42  # Keep fixed when comparing training seeds.
        MAX_SELECTED_FEATURES = 150
        MIN_TRAIN_PATIENTS = 10
        CORE_FEATURES = []  # Pending: exact approved FEATURE_NAME strings; no guessed mappings.
        EXPERIMENT_NAMES = [
            "all_baseline", "reduced150", "reduced150_dropout", "reduced150_decay",
            "reduced150_small", "reduced150_logistic",
        ]
        # Optional later: core, core_dropout, core_decay, core_small, core_logistic,
        # reduced150_low_lr. Never add core until its exact feature list is supplied.
        # Engineered/static legacy features outside FEATURE_MAP need separate preparation.
        print("Six controlled TRAIN/VALIDATION experiments; no TEST scoring in this notebook.")
        print("reduced150 selects monthly claims features; it does NOT recreate the client's 150 engineered features.")
        print("Core feature experiment is pending" if not CORE_FEATURES else "Core feature list supplied")
    ''')),
    cell(2, "Self-contained feature selection, models, storage and experiment helpers", DEPENDENCIES),
    cell(3, "Validate the unchanged full input and patient split", LOAD_INPUTS),
    cell(4, "Prepare TRAIN-only feature selections and controlled recipes", '''
        # Each regularization variant compares with reduced150, changing one setting.
        # small changes the architecture as one separately identified experiment.
        # Feature screening uses four 3-month windows. Importance is screening, not attribution.
        recipes = experiment_recipes(SUITE_ID, EXPERIMENT_NAMES, TRAINING_SEED)
        selections = {}
        for mode in sorted({recipe["feature_mode"] for recipe in recipes}):
            selections[mode] = select_features(data, mode=mode,
                max_features=MAX_SELECTED_FEATURES, core_features=CORE_FEATURES,
                min_train_patients=MIN_TRAIN_PATIENTS, seed=FEATURE_SELECTION_SEED)
        for recipe in recipes:
            manifest, audit = selections[recipe["feature_mode"]]
            recipe["model_settings"]["input_dim"] = len(manifest["selected_indices"])
            recipe["feature_manifest_sha256"] = manifest["manifest_sha256"]
            recipe["business_target_specification"] = TARGET_SPECIFICATION
        if table_exists(SELECTION_TABLE):
            previous_suite = read_artifacts(SELECTION_TABLE, SELECTION_ARTIFACT_NAMES)
            previous_selection = json.loads(previous_suite["selection.json"])
            if (json.loads(previous_suite["experiment_plan.json"]) != recipes
                    or previous_selection.get("implementation_sha256") != IMPLEMENTATION_SHA256
                    or previous_selection.get("input_hashes") != data["hashes"]):
                raise ValueError("This suite was already finalized with different settings/inputs/code. Choose a new SUITE_ID.")
        display(pd.DataFrame([{
            "experiment": recipe["name"], "reference": recipe["reference"],
            "model": recipe["model_kind"], "features": recipe["model_settings"]["input_dim"],
            "embedding_width": recipe["model_settings"]["d_model"] if recipe["model_kind"] == "transformer" else None,
            "layers": recipe["model_settings"]["encoder_layers"] if recipe["model_kind"] == "transformer" else None,
            "dropout": recipe["model_settings"]["dropout"] if recipe["model_kind"] == "transformer" else None,
            "weight_decay": recipe["training_settings"]["weight_decay"] if recipe["model_kind"] == "transformer" else None,
        } for recipe in recipes]))
        if "reduced" in selections:
            display(selections["reduced"][1])
        if not CORE_FEATURES:
            print("CORE experiment pending; approved list has not been provided.")
    '''),
    cell(5, "Train and save each completed run; safely reuse matching completed runs", '''
        # Every completed run is saved immediately. A later interrupted run can be retrained;
        # earlier completed matching runs are restored, verified and reused.
        if data["X"]._mmap.closed:
            raise RuntimeError("Rerun cell 3 to reload the temporary tensor.")
        run_summaries, run_records = [], []
        for recipe in recipes:
            manifest, audit = selections[recipe["feature_mode"]]
            run_id = recipe["run_id"]
            model_table = f"{EXPERIMENT_PREFIX}_MODEL_{run_id}"
            signature = experiment_signature(recipe, manifest, data, IMPLEMENTATION_SHA256)
            if table_exists(model_table):
                artifacts = read_artifacts(model_table, TRAINING_ARTIFACT_NAMES)
                summary = checked_saved_training(artifacts, data, recipe, manifest, signature)
                print("Reused verified completed run:", run_id, flush=True)
            else:
                checkpoint_bytes, summary, history = run_experiment(data, manifest,
                    ModelConfig(**recipe["model_settings"]), recipe["training_settings"],
                    run_id, model_kind=recipe["model_kind"])
                summary.update(experiment_name=recipe["name"], reference_experiment=recipe["reference"],
                    experiment_signature=signature, implementation_sha256=IMPLEMENTATION_SHA256,
                    business_target_specification=TARGET_SPECIFICATION)
                artifacts = training_artifacts_for_run(checkpoint_bytes, summary, history, manifest, audit)
                save_artifacts(model_table, artifacts)
                print("Saved completed run:", run_id, flush=True)
            run_summaries.append(summary)
            run_records.append({"run_id": run_id, "model_table": model_table,
                "experiment_signature": signature,
                "checkpoint_sha256": hashlib.sha256(artifacts["checkpoint.pt"]).hexdigest(),
                "feature_manifest_sha256": manifest["manifest_sha256"]})
        print("All requested runs completed and saved. TEST has not been scored.")
    '''),
    cell(6, "Compare only training and validation performance", '''
        comparison = comparison_frame(run_summaries)
        display(comparison)
        comparison_plot = comparison_figure(comparison)
        display(comparison_plot)
        print("AP selects the model. Validation lift supports the targeting interpretation.")
        print("A smaller loss gap alone is not evidence of better generalization.")
        print("The logistic baseline uses 12-month totals and discards month order.")
    '''),
    cell(7, "Freeze the validation-selected run before evaluation", '''
        # Deterministic selection: highest validation AP; exact ties use run_id ascending.
        chosen_run_id = str(comparison.iloc[0].run_id)
        chosen_record = next(record for record in run_records if record["run_id"] == chosen_run_id)
        chosen_summary = next(summary for summary in run_summaries if summary["run_id"] == chosen_run_id)
        selection = {
            "format_version": 1, "suite_id": SUITE_ID, "selection_complete": True,
            "selection_metric": "validation_average_precision", "tie_breaker": "run_id_ascending",
            "selected_run_id": chosen_run_id, "selected_model_table": chosen_record["model_table"],
            "selected_checkpoint_sha256": chosen_record["checkpoint_sha256"],
            "selected_feature_manifest_sha256": chosen_record["feature_manifest_sha256"],
            "selected_validation_ap": chosen_summary["best_validation_average_precision"],
            "validation_threshold": chosen_summary["validation_threshold"],
            "input_hashes": data["hashes"], "implementation_sha256": IMPLEMENTATION_SHA256,
            "business_target_specification": TARGET_SPECIFICATION,
            "candidate_runs": run_records, "test_used_for_selection": False,
            "core_status": "pending_approved_list" if not CORE_FEATURES else (
                "included" if any(recipe["feature_mode"] == "core" for recipe in recipes) else "configured_not_requested"),
            "test_status": "Original RUN_001 test results have previously been inspected; not a fresh confirmation cohort.",
        }
        comparison_buffer = io.BytesIO()
        comparison_plot.savefig(comparison_buffer, format="png", dpi=140)
        selection_artifacts = {
            "selection.json": canonical_json(selection).encode("utf-8"),
            "comparison.csv": comparison.to_csv(index=False).encode("utf-8"),
            "comparison.png": comparison_buffer.getvalue(),
            "experiment_plan.json": canonical_json(recipes).encode("utf-8"),
        }
        save_artifacts(SELECTION_TABLE, selection_artifacts)
        print("Frozen validation-selected run:", chosen_run_id)
        print("Run notebook 04 with SUITE_ID =", repr(SUITE_ID))
    '''),
    cell(8, "Verify the selected run and release temporary inputs", '''
        if read_artifacts(SELECTION_TABLE, SELECTION_ARTIFACT_NAMES) != selection_artifacts:
            raise ValueError("Selection read-back differs; investigate before evaluation.")
        release_experiment_inputs(data)
        print("Experiment suite saved and verified. New TEST reports are not required for each candidate.")
    '''),
]

evaluation = [
    cell(1, "Connection and previously frozen experiment suite", CONNECTION + "\n" + SUITE_SETUP + '''
print("Evaluate only the run already selected in notebook 03 using validation AP.")
print("This reuses the previously inspected TEST cohort; fresh confirmation remains a later step.")
'''),
    cell(2, "Self-contained restoration and reporting helpers", DEPENDENCIES + "\n" + read(HERE / "evaluation_helpers.py")),
    cell(3, "Load the frozen selection and verify all original inputs", dedent('''
        if not table_exists(SELECTION_TABLE):
            raise FileNotFoundError("Complete and save notebook 03 before evaluation.")
        selection_artifacts = read_artifacts(SELECTION_TABLE, SELECTION_ARTIFACT_NAMES)
        selection = json.loads(selection_artifacts["selection.json"])
        if (selection.get("suite_id") != SUITE_ID or selection.get("selection_complete") is not True
                or selection.get("test_used_for_selection") is not False
                or selection.get("implementation_sha256") != IMPLEMENTATION_SHA256):
            raise ValueError("Missing, changed or incompatible frozen validation selection.")
        RUN_ID = selection["selected_run_id"]
        MODEL_TABLE = f"{EXPERIMENT_PREFIX}_MODEL_{RUN_ID}"
        EVALUATION_TABLE = f"{EXPERIMENT_PREFIX}_EVAL_{RUN_ID}"
        if selection.get("selected_model_table") != MODEL_TABLE:
            raise ValueError("Selected model table differs from this suite's namespace.")
        if table_exists(EVALUATION_TABLE):
            raise FileExistsError("This selected run already has a saved TEST report. Reuse it instead of retuning on TEST.")
        if not table_exists(MODEL_TABLE):
            raise FileNotFoundError("Selected completed model is missing.")
        training_artifacts = read_artifacts(MODEL_TABLE, TRAINING_ARTIFACT_NAMES)
        if hashlib.sha256(training_artifacts["checkpoint.pt"]).hexdigest() != selection["selected_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed after validation selection.")
        training_summary = json.loads(training_artifacts["training_summary.json"])
    ''') + LOAD_INPUTS + '''
if selection.get("input_hashes") != data["hashes"]:
    release_experiment_inputs(data)
    raise ValueError("Inputs changed after validation selected the run.")
'''),
    cell(4, "Restore the saved feature selection, model and validation threshold", '''
        try:
            saved_manifest = json.loads(training_artifacts["feature_manifest.json"])
            saved_plan = json.loads(selection_artifacts["experiment_plan.json"])
            selected_recipes = [recipe for recipe in saved_plan if recipe["run_id"] == RUN_ID]
            if len(selected_recipes) != 1:
                raise ValueError("Selected run must occur exactly once in the saved experiment plan.")
            selected_recipe = selected_recipes[0]
            expected_signature = experiment_signature(selected_recipe, saved_manifest, data, IMPLEMENTATION_SHA256)
            training_summary = checked_saved_training(training_artifacts, data, selected_recipe, saved_manifest, expected_signature)
            model, checkpoint, device = restore_experiment(training_artifacts["checkpoint.pt"], data, RUN_ID, "auto")
            if (checkpoint["feature_manifest"]["manifest_sha256"] != selection["selected_feature_manifest_sha256"]
                    or checkpoint["validation_threshold"] != selection["validation_threshold"]
                    or checkpoint["best_validation_average_precision"] != selection["selected_validation_ap"]):
                raise ValueError("Frozen selection, summary and checkpoint disagree.")
        except BaseException:
            release_experiment_inputs(data)
            raise
        print("Model:", checkpoint["model_kind"], "| selected features:", len(saved_manifest["selected_indices"]))
        print("Frozen validation threshold:", checkpoint["validation_threshold"])
        print("No feature reselection, model fitting or threshold tuning occurs here.")
    '''),
    cell(5, "Score the selected model on the original TEST cohort", '''
        test_y, test_scores = predict_experiment(model, checkpoint, data, "test", device)
        test_rows = data["indices"]["test"]
        scored_snapshots = data["metadata"].iloc[test_rows][["PATIENT_ID", "END_DT", "RESP"]].copy().reset_index(drop=True)
        if not np.array_equal(test_y, scored_snapshots.RESP.to_numpy()):
            raise ValueError("Prediction labels lost alignment with snapshot keys.")
        scored_snapshots["P_RESP1"] = test_scores
        model_name = "DL-POC " + checkpoint["model_kind"] + " | " + RUN_ID
        results = evaluate_scores(scored_snapshots, checkpoint["validation_threshold"], model_name=model_name)
        evaluation_metadata = {
            "run_id": RUN_ID, "suite_id": SUITE_ID, "model_kind": checkpoint["model_kind"],
            "checkpoint_sha256": selection["selected_checkpoint_sha256"],
            "feature_manifest_sha256": saved_manifest["manifest_sha256"],
            "selected_features": saved_manifest["selected_features"],
            "input_hashes": data["hashes"], "selection_metric": selection["selection_metric"],
            "threshold_source": "frozen validation maximum-F1 threshold",
            "threshold": checkpoint["validation_threshold"], "evaluation_unit": "patient/date snapshot",
            "test_snapshots": len(scored_snapshots), "test_positive_snapshots": int(scored_snapshots.RESP.sum()),
            "ranking_ties": "label-independent snapshot-key hash", "test_used_for_selection": False,
            "test_status": selection["test_status"],
            "limitations": "Previously inspected original TEST; snapshot metrics; no confidence intervals; class-weighted scores are not calibrated probabilities.",
            "source_review": training_summary.get("source_review"),
            "business_target_specification": selection["business_target_specification"],
        }
        print("Selected-run TEST inference complete; original cohort is not a new confirmation holdout.")
    '''),
    cell(6, "Review TEST targeting and classification reports", '''
        for name in ("global_metrics", "deciles", "topk", "ties"):
            print(name)
            display(results[name])
        for name, figure in results["figures"].items():
            display(figure)
        print("Metrics count snapshots, not unique patients. AP is non-interpolated average precision.")
    '''),
    cell(7, "Save aggregate TEST results in the new namespace", '''
        evaluation_artifacts = {"evaluation_metadata.json": canonical_json(evaluation_metadata).encode("utf-8")}
        for name in ("global_metrics", "deciles", "topk", "ties"):
            evaluation_artifacts[f"{name}.csv"] = results[name].to_csv(index=False).encode("utf-8")
        for name, figure in results["figures"].items():
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            evaluation_artifacts[f"{name}.png"] = buffer.getvalue()
        save_artifacts(EVALUATION_TABLE, evaluation_artifacts)
        # No patient IDs or per-snapshot predictions are exported.
    '''),
    cell(8, "Verify the report and release temporary inputs", '''
        if read_artifacts(EVALUATION_TABLE, evaluation_artifacts) != evaluation_artifacts:
            raise ValueError("Saved TEST report failed verification.")
        release_experiment_inputs(data)
        print("Saved and verified:", EVALUATION_TABLE)
        print("Choose further changes on TRAIN/VALIDATION; use a fresh later-period cohort for final confirmation.")
    '''),
]


def build():
    stages = early_notebook_cells(CONNECTION, INPUT_SOURCE)
    stages["03_transformer_training_DL_POC.ipynb"] = training
    stages["04_transformer_evaluation_DL_POC.ipynb"] = evaluation
    for name, cells in stages.items():
        write_notebook(name, cells)
    print(f"Generated four notebooks / 32 compiling code cells in {OUT}")


if __name__ == "__main__":
    build()
