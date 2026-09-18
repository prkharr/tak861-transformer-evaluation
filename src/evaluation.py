"""One final TEST evaluation from a completed, validation-selected checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from targeting_evaluation import evaluate_model, validate_inputs, write_report
from .data_utils import SequenceBundle
from .metrics import classification_metrics
from .training import load_trained_model, score_split


def _classification_figures(aligned, results, metrics, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    paths = {}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        table = results["deciles"]
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        bars = ax.bar(table.decile, table.response_rate, color="#007E87")
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in table.response_rate], padding=3, fontsize=9)
        ax.axhline(float(aligned.RESP.mean()), color="#777777", linestyle="--", label="Overall TEST response rate")
        ax.set(xticks=range(1, 11), xlabel="Decile (1 = highest propensity)", ylabel="Observed response rate",
               title="Transformer | response rate by decile", ylim=(0, max(table.response_rate.max() * 1.22, .01)))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if ax.get_ylim()[1] > 1:
            ax.set_yticks(np.arange(0, 1.001, .2))
        ax.legend()
        paths["response_rate"] = directory / "response_rate_by_decile.png"
        fig.savefig(paths["response_rate"], dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
        y, scores = aligned.RESP.to_numpy(), aligned.p_transformer.to_numpy()
        if y.sum():
            precision, recall, _ = precision_recall_curve(y, scores)
            axes[0].step(recall, precision, where="post", color="#007E87", label=f"AP = {metrics['average_precision']:.3f}")
            axes[0].axhline(y.mean(), linestyle="--", color="#777777", label="TEST prevalence")
            axes[0].legend()
        else:
            axes[0].text(.5, .5, "Undefined: no positive TEST snapshots", ha="center", transform=axes[0].transAxes)
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            axes[1].plot(fpr, tpr, color="#007E87", label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
            axes[1].plot([0, 1], [0, 1], linestyle="--", color="#777777")
            axes[1].legend(loc="lower right")
        else:
            axes[1].text(.5, .5, "Undefined: only one TEST class", ha="center", transform=axes[1].transAxes)
        axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision–recall (primary)", xlim=(0, 1), ylim=(0, 1.02))
        axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (supporting)", xlim=(0, 1), ylim=(0, 1.02))
        paths["discrimination"] = directory / "discrimination_curves.png"
        fig.savefig(paths["discrimination"], dpi=180)
        plt.close(fig)

        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        fig, ax = plt.subplots(figsize=(5.2, 4.6), layout="constrained")
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=18,
                    color="white" if value > matrix.max() / 2 else "#183B4E")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["RESP=0", "RESP=1"], yticklabels=["RESP=0", "RESP=1"],
               xlabel="Predicted", ylabel="Actual", title=f"TEST confusion matrix\nVALIDATION threshold = {metrics['threshold']:.4f}")
        paths["confusion_matrix"] = directory / "confusion_matrix.png"
        fig.savefig(paths["confusion_matrix"], dpi=180)
        plt.close(fig)
    return paths


def evaluate_checkpoint(bundle: SequenceBundle, manifest: pd.DataFrame, checkpoint_path: str | Path,
                        output_dir: str | Path, device: str = "auto") -> dict:
    """Score TEST only here; retain a fixed checkpoint and validation threshold.

    Nonempty report directories are refused to keep each evaluation reviewable.
    This is an execution guard, not a technical guarantee against later TEST tuning.
    """
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Evaluation report directory is not empty. Review its saved report; do not rerun model selection against TEST.")
    checkpoint_path = Path(checkpoint_path)
    model, payload = load_trained_model(checkpoint_path, device=device)
    del model
    predictions = score_split(bundle, manifest, checkpoint_path, split="TEST", device=device)
    aligned = validate_inputs(manifest, predictions)
    threshold = float(payload["validation_threshold"])
    fixed = classification_metrics(aligned.RESP, aligned.p_transformer, threshold)
    results = evaluate_model(aligned)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    audit = {
        "snapshot_manifest_sha256": payload["snapshot_manifest_sha256"],
        "bundle_sha256": payload["bundle_sha256"], "checkpoint_sha256": checkpoint_hash,
        "patient_disjointness_checked": True, "exact_test_snapshot_match_checked": True,
        "feature_order_and_input_values_match_checkpoint": True,
        "threshold": threshold, "threshold_source": "Maximum F1 on best-checkpoint VALIDATION scores; highest threshold on ties.",
        "checkpoint_selection": "Maximum VALIDATION average precision",
        "test_used_for_training_or_threshold_selection": False,
        "upstream_temporal_leakage_review": "User attestation in the local bundle; not independently proven from tensor values.",
        "lightgbm_status": "Deferred",
    }
    paths = write_report(results, destination, audit=audit)
    metrics_frame = pd.DataFrame([fixed])
    paths["classification_metrics"] = destination / "classification_metrics.csv"
    metrics_frame.to_csv(paths["classification_metrics"], index=False)
    paths["classification_metrics_json"] = destination / "classification_metrics.json"
    paths["classification_metrics_json"].write_text(json.dumps(fixed, indent=2, allow_nan=False), encoding="utf-8")
    paths.update(_classification_figures(aligned, results, fixed, destination))
    extra = ["", "## Fixed operating threshold", "",
             f"Threshold {threshold:.6f} was selected on VALIDATION for the chosen checkpoint and applied unchanged to TEST. "
             "Predicted positive means score >= threshold. This choice is separate from the rank-based top-K metrics.", "",
             "| Metric | TEST value |", "| --- | --- |"]
    for name in ("average_precision", "roc_auc", "precision", "recall", "f1"):
        value = fixed[name]
        extra.append(f"| {name} | {'N/A' if value is None else f'{value:.6f}'} |")
    extra.extend(["", "Threshold-based precision/recall/F1 use the documented zero_division=0 convention. "
                  "Decile capture/recall and lift remain N/A when there are no TEST positives.", "",
                  "![Response rate by decile](response_rate_by_decile.png)", "",
                  "![PR and ROC curves](discrimination_curves.png)", "", "![Confusion matrix](confusion_matrix.png)", "",
                  "The class-weighted model's sigmoid scores support ranking; probability calibration has not been established. "
                  "Interpretation is predictive, not causal."])
    with paths["report"].open("a", encoding="utf-8") as handle:
        handle.write("\n".join(extra) + "\n")
    return {"results": results, "classification_metrics": metrics_frame, "paths": paths, "audit": audit}
