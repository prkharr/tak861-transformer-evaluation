"""Controlled experiment recipes and aggregate-only reporting, embedded in notebooks."""
import copy
import hashlib
import io
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


TRAINING_ARTIFACT_NAMES = {
    "checkpoint.pt", "training_summary.json", "training_history.csv",
    "training_history.png", "feature_manifest.json", "feature_selection.csv",
    "training_topk.csv", "validation_topk.csv",
}
SELECTION_ARTIFACT_NAMES = {
    "selection.json", "comparison.csv", "comparison.png", "experiment_plan.json",
}
DEFAULT_EXPERIMENTS = [
    "all_baseline", "reduced150", "reduced150_dropout", "reduced150_decay",
    "reduced150_small", "reduced150_logistic",
]


def experiment_recipes(suite_id, names, seed=42):
    if not isinstance(suite_id, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,19}", suite_id):
        raise ValueError("SUITE_ID must be 1-20 uppercase letters, digits or underscores, starting with a letter.")
    if not isinstance(names, (list, tuple)) or not names or len(set(names)) != len(names):
        raise ValueError("Choose a nonempty list of distinct experiment names.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Training seed must be an integer in [0, 2**32).")
    model = dict(seq_len=12, d_model=128, n_heads=4, encoder_layers=2,
                 feedforward_dim=256, dropout=0.2)
    training = dict(seed=seed, epochs=20, patience=5, min_delta=1e-4,
                    batch_size=64, learning_rate=1e-3, weight_decay=1e-4,
                    grad_clip=1.0, device="auto", logistic_C=1.0,
                    logistic_max_iter=2000)
    catalog = {}
    for base, mode in (("all_baseline", "all"), ("reduced150", "reduced"), ("core", "core")):
        catalog[base] = dict(feature_mode=mode, model_kind="transformer",
                             model_settings=copy.deepcopy(model), training_settings=copy.deepcopy(training),
                             reference="all_baseline" if base != "all_baseline" else None)
        if base == "all_baseline":
            continue
        changes = {
            "dropout": ("model_settings", {"dropout": 0.3}),
            "decay": ("training_settings", {"weight_decay": 1e-3}),
            "small": ("model_settings", {"d_model": 64, "encoder_layers": 1, "feedforward_dim": 128}),
            "low_lr": ("training_settings", {"learning_rate": 3e-4}),
            "logistic": (None, {}),
        }
        for suffix, (section, delta) in changes.items():
            item = copy.deepcopy(catalog[base])
            item["reference"] = base
            if section:
                item[section].update(delta)
            else:
                item["model_kind"] = "logistic"
            catalog[f"{base}_{suffix}"] = item
    unknown = sorted(set(names) - set(catalog))
    if unknown:
        raise ValueError(f"Unknown experiments {unknown}; available: {sorted(catalog)}")
    recipes = []
    for name in names:
        recipe = copy.deepcopy(catalog[name])
        recipe.update(name=name, run_id=f"{suite_id}_{name.upper()}")
        recipes.append(recipe)
    return recipes


def experiment_signature(recipe, manifest, data, implementation_sha256):
    return digest_json({"recipe": recipe, "feature_manifest": manifest,
                        "input_hashes": data["hashes"], "implementation_sha256": implementation_sha256})


def experiment_history_figure(history, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    if summary["model_kind"] == "logistic":
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.05, .65, "Regularized logistic regression\nTRAIN-fitted scaling; 12-month count totals\nNo Transformer epochs / dropout", fontsize=12)
        axes[1].text(.05, .65, f"Validation AP: {summary['validation_metrics']['average_precision']:.4f}\n"
                      f"Training AP: {summary['training_metrics']['average_precision']:.4f}", fontsize=12)
    else:
        axes[0].plot(history.epoch, history.training_loss, label="TRAIN, evaluation mode")
        axes[0].plot(history.epoch, history.validation_loss, label="VALIDATION, evaluation mode")
        axes[0].set(xlabel="Epoch", ylabel="TRAIN-weighted BCE", title="Comparable loss (dropout disabled)")
        axes[0].legend(fontsize=8)
        axes[1].plot(history.epoch, history.training_average_precision, label="TRAIN")
        axes[1].plot(history.epoch, history.validation_average_precision, marker="o", label="VALIDATION")
        axes[1].axvline(summary["best_epoch"], color="gray", linestyle="--")
        axes[1].set(xlabel="Epoch", ylabel="Average precision", title="Select checkpoint on validation AP")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(summary["run_id"])
    return fig


def training_artifacts_for_run(blob, summary, history, manifest, feature_audit):
    fig = experiment_history_figure(history, summary)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    return {
        "checkpoint.pt": blob,
        "training_summary.json": canonical_json(summary).encode("utf-8"),
        "training_history.csv": history.to_csv(index=False).encode("utf-8"),
        "training_history.png": buffer.getvalue(),
        "feature_manifest.json": canonical_json(manifest).encode("utf-8"),
        "feature_selection.csv": feature_audit.to_csv(index=False).encode("utf-8"),
        "training_topk.csv": pd.DataFrame(summary["training_topk"]).to_csv(index=False).encode("utf-8"),
        "validation_topk.csv": pd.DataFrame(summary["validation_topk"]).to_csv(index=False).encode("utf-8"),
    }


def checked_saved_training(artifacts, data, recipe, manifest, signature):
    summary = json.loads(artifacts["training_summary.json"])
    saved_manifest = json.loads(artifacts["feature_manifest.json"])
    if saved_manifest != manifest or summary.get("experiment_signature") != signature:
        raise ValueError("Existing run has different settings/features/code/inputs. Choose a new SUITE_ID.")
    if summary.get("test_inference_performed") is not False or summary.get("training_complete") is not True:
        raise ValueError("Training record does not certify TEST exclusion.")
    model, payload, _ = restore_experiment(artifacts["checkpoint.pt"], data, recipe["run_id"], "cpu")
    del model
    if (summary.get("validation_threshold") != payload.get("validation_threshold")
            or payload.get("feature_manifest") != manifest
            or summary.get("input_hashes") != data["hashes"]
            or summary.get("model_kind") != recipe["model_kind"]
            or summary.get("run_id") != recipe["run_id"]
            or payload.get("model_kind") != recipe["model_kind"]
            or payload.get("model_config") != recipe["model_settings"]
            or payload.get("training_settings") != recipe["training_settings"]
            or summary.get("model_config") != payload.get("model_config")
            or summary.get("training_settings") != payload.get("training_settings")
            or summary.get("feature_manifest") != manifest
            or summary.get("feature_manifest_sha256") != manifest["manifest_sha256"]
            or summary.get("best_validation_average_precision") != payload.get("best_validation_average_precision")
            or summary.get("validation_metrics", {}).get("average_precision") != payload.get("best_validation_average_precision")
            or summary.get("best_epoch") != payload.get("best_epoch")):
        raise ValueError("Saved training summary and checkpoint disagree.")
    return summary


def comparison_frame(run_summaries):
    rows = []
    for summary in run_summaries:
        val = summary["validation_metrics"]
        row = {
            "run_id": summary["run_id"], "model_kind": summary["model_kind"],
            "feature_mode": summary["feature_manifest"]["mode"],
            "n_features": len(summary["feature_manifest"]["selected_indices"]),
            "parameter_count": summary["model_parameter_count"],
            "best_epoch": summary["best_epoch"],
            "training_ap": summary["training_metrics"]["average_precision"],
            "validation_ap": val["average_precision"], "validation_roc_auc": val["roc_auc"],
            "validation_precision": val["precision"], "validation_recall": val["recall"],
            "validation_f1": val["f1"], "validation_threshold": summary["validation_threshold"],
        }
        for top in summary["validation_topk"]:
            pct = int(top["top_k_pct"])
            row[f"validation_lift_{pct}"] = top["lift"]
            row[f"validation_recall_{pct}"] = top["recall"]
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty or result.run_id.duplicated().any() or not np.isfinite(result.validation_ap).all():
        raise ValueError("Need unique, completed runs with finite validation AP.")
    return result.sort_values(["validation_ap", "run_id"], ascending=[False, True]).reset_index(drop=True)


def comparison_figure(frame):
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, .55 * len(frame))), constrained_layout=True)
    labels = frame.run_id.to_list()
    y = np.arange(len(frame))
    axes[0].barh(y - .17, frame.training_ap, height=.32, label="TRAIN")
    axes[0].barh(y + .17, frame.validation_ap, height=.32, label="VALIDATION")
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Average precision", title="Validation selects the run")
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncol=2)
    axes[1].barh(y, frame.validation_lift_10, height=.55, color="#007E87")
    axes[1].set(yticks=y, yticklabels=labels, xlabel="Top-10% lift", title="VALIDATION targeting performance")
    axes[1].invert_yaxis()
    axes[1].axvline(1, color="gray", linestyle="--")
    return fig


def verify_preparation_audits(data, preparation, split_audit):
    if not isinstance(preparation, dict) or not isinstance(split_audit, dict):
        raise ValueError("Preparation and split audits must be completed JSON records.")
    if (preparation.get("mode") != "reuse_and_validate_saved_v63_tensor"
            or split_audit.get("mode") != "reuse_original_patient_split"
            or split_audit.get("new_assignments_created") is not False):
        raise ValueError("Expected the preparation and original split audits from notebooks 01 and 02.")
    if preparation.get("tensor_shape") != list(data["X"].shape):
        raise ValueError("Tensor shape differs from the saved preparation audit.")
    for key in ("model_input_sha256", "feature_names_sha256"):
        if preparation.get(key) != data["hashes"][key]:
            raise ValueError(f"Preparation audit differs on {key}.")
    for key in ("snapshot_manifest_sha256", "split_config_sha256", "feature_names_sha256"):
        if split_audit.get(key) != data["hashes"][key]:
            raise ValueError(f"Frozen split audit differs on {key}.")
    source_rows = [[str(r.PATIENT_ID), str(r.END_DT), int(r.RESP)]
                   for r in data["metadata"].itertuples()]
    source_hash = digest_json(source_rows)
    if any(report.get("source_snapshots_sha256") != source_hash for report in (preparation, split_audit)):
        raise ValueError("Preparation/split audit snapshot identity differs.")
    if preparation.get("source_prefix") != split_audit.get("source_prefix"):
        raise ValueError("Preparation and split audit use different sources.")


def release_experiment_inputs(data):
    if data is not None and hasattr(data.get("X"), "_mmap") and not data["X"]._mmap.closed:
        data["X"]._mmap.close()
        data["temporary_directory"].cleanup()
