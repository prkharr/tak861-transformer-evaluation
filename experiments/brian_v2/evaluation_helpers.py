"""DL-POC aggregate evaluation, suitable for an inline notebook cell.

No repository imports, file writes, threshold tuning, or patient-level exports.
Rates are fractions; lift is relative to the overall evaluated response rate.
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _real_array(values, name):
    raw = np.asarray(values)
    _require(not np.iscomplexobj(raw) and not (
        raw.dtype == object
        and any(isinstance(value, (complex, np.complexfloating)) for value in raw.flat)
    ), f"{name} must be real, not complex.")
    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must contain numeric values.") from None


def _checked_labels_scores(y, scores):
    labels = _real_array(y, "Labels")
    probabilities = _real_array(scores, "Probabilities")
    _require(labels.ndim == probabilities.ndim == 1
             and len(labels) == len(probabilities) and len(labels) > 0,
             "Labels and probabilities must be aligned, nonempty 1-D arrays.")
    _require(np.isfinite(labels).all() and np.isin(np.asarray(y), [0, 1]).all(),
             "Labels must contain only finite binary 0/1 values.")
    _require(np.isfinite(probabilities).all()
             and ((probabilities >= 0) & (probabilities <= 1)).all(),
             "Probabilities must be finite and within [0, 1].")
    return labels.astype(np.int64), probabilities


def classification_metrics(y, scores, threshold):
    """Use a threshold already selected on VALIDATION; never tune on TEST."""
    labels, probabilities = _checked_labels_scores(y, scores)
    cutoff = _real_array(threshold, "Threshold")
    _require(cutoff.ndim == 0 and np.isfinite(cutoff).all()
             and 0 <= float(cutoff) <= 1,
             "Threshold must be a finite scalar in [0, 1].")
    cutoff = float(cutoff)
    predicted = (probabilities >= cutoff).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(labels, probabilities))
        if labels.sum() else None,
        "roc_auc": float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2 else None,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "threshold": cutoff,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _checked_scored_snapshots(frame, score_column):
    _require(isinstance(frame, pd.DataFrame), "Evaluation input must be a pandas DataFrame.")
    _require(isinstance(score_column, str), "Score column must be a column name.")
    _require(frame.columns.is_unique, "Evaluation input contains duplicate column names.")
    keys = ["PATIENT_ID", "END_DT"]
    _require(score_column not in keys + ["RESP"], "Score column must differ from metadata columns.")
    _require(set(keys + ["RESP", score_column]).issubset(frame.columns),
             "Evaluation input is missing snapshot keys, RESP, or scores.")
    out = frame[keys + ["RESP", score_column]].copy()
    _require(not out[keys].isna().any().any(), "Snapshot keys must not be missing.")
    _require(out.PATIENT_ID.map(lambda value: isinstance(value, str)).all(),
             "PATIENT_ID must contain strings to preserve literal identifiers.")
    out["PATIENT_ID"] = out.PATIENT_ID.astype(str)
    _require(out.PATIENT_ID.str.len().gt(0).all()
             and out.PATIENT_ID.str.strip().eq(out.PATIENT_ID).all(),
             "Patient keys must be nonempty with no surrounding whitespace.")
    dates = out.END_DT.astype(str)
    _require(dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all(),
             "END_DT must contain ISO calendar dates (YYYY-MM-DD), not intraday timestamps.")
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    _require(parsed.notna().all(), "END_DT contains invalid calendar dates.")
    out["END_DT"] = parsed.dt.strftime("%Y-%m-%d")
    _require(not out.duplicated(keys).any(), "Duplicate patient/date snapshots are not allowed.")
    labels, scores = _checked_labels_scores(out.RESP.to_numpy(), out[score_column].to_numpy())
    out["RESP"] = labels
    out[score_column] = scores
    _require(len(out) >= 10, "At least 10 snapshots are required for ten nonempty deciles.")
    return out


def _snapshot_hash(patient, end_date):
    # This public salt gives deterministic, label-independent ties, not anonymity.
    payload = json.dumps(["tak861-targeting-v1", patient, end_date], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _score_order(frame, score_column):
    # Labels are deliberately inaccessible to ordering logic.
    scoring = frame[["PATIENT_ID", "END_DT", score_column]].rename(
        columns={score_column: "_probability"}
    ).copy()
    scoring["_tie"] = [
        _snapshot_hash(patient, date)
        for patient, date in scoring[["PATIENT_ID", "END_DT"]].itertuples(index=False, name=None)
    ]
    scoring["_position"] = np.arange(len(scoring))
    return scoring.sort_values(
        ["_probability", "_tie", "PATIENT_ID", "END_DT"],
        ascending=[False, True, True, True],
    )["_position"].to_numpy()


def _rank_snapshots(checked, score_column):
    ranked = checked.iloc[_score_order(checked, score_column)].reset_index(drop=True)
    n = len(ranked)
    ranks = np.arange(1, n + 1, dtype=np.int64)
    boundaries = (np.arange(1, 11, dtype=np.int64) * n + 9) // 10
    ranked["decile"] = np.searchsorted(boundaries, ranks, side="left") + 1
    return ranked


def _decile_table(ranked, model_name):
    n, positives = len(ranked), int(ranked.RESP.sum())
    overall_rate = positives / n
    table = ranked.groupby("decile").agg(
        n_snapshots=("RESP", "size"), n_resp1=("RESP", "sum")
    ).reindex(range(1, 11)).reset_index()
    table.insert(0, "model", model_name)
    table["population_fraction"] = table.n_snapshots / n
    table["response_rate"] = table.n_resp1 / table.n_snapshots
    table["decile_precision"] = table.response_rate
    table["share_of_all_resp1"] = table.n_resp1 / positives if positives else np.nan
    table["cumulative_n_snapshots"] = table.n_snapshots.cumsum()
    table["cumulative_population_fraction"] = table.cumulative_n_snapshots / n
    table["cumulative_resp1"] = table.n_resp1.cumsum()
    table["cumulative_recall"] = table.cumulative_resp1 / positives if positives else np.nan
    table["cumulative_precision"] = table.cumulative_resp1 / table.cumulative_n_snapshots
    table["decile_lift"] = table.response_rate / overall_rate if positives else np.nan
    table["cumulative_lift"] = table.cumulative_precision / overall_rate if positives else np.nan
    table["overall_test_response_rate"] = overall_rate
    return table


def _topk_table(deciles):
    selected = deciles.loc[deciles.decile.isin([1, 2, 3])].copy()
    selected["top_k_pct"] = selected.decile * 10
    return selected.rename(columns={
        "cumulative_n_snapshots": "n_selected", "cumulative_resp1": "n_resp1_selected",
        "cumulative_recall": "recall", "cumulative_precision": "precision",
        "cumulative_lift": "lift", "cumulative_population_fraction": "actual_population_fraction",
    })[["model", "top_k_pct", "n_selected", "n_resp1_selected", "recall", "precision",
        "lift", "actual_population_fraction"]].rename(
        columns={"n_resp1_selected": "n_resp1"}
    ).reset_index(drop=True)


def _tie_table(ranked, score_column, model_name):
    scores = ranked[score_column].to_numpy()
    rows = []
    for percentage in (10, 20, 30):
        cutoff = (percentage * len(scores) + 99) // 100
        value = scores[cutoff - 1]
        total = int((scores == value).sum())
        included = int((scores[:cutoff] == value).sum())
        rows.append({
            "model": model_name, "top_k_pct": percentage, "cutoff_score": value,
            "n_tied_at_cutoff": total, "n_tied_selected": included,
            "tie_crosses_boundary": included < total,
        })
    return pd.DataFrame(rows)


def _evaluation_figures(results, checked, score_column):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    table, topk = results["deciles"], results["topk"]
    model_name = results["global_metrics"].iloc[0]["model"]
    has_positives = bool(results["global_metrics"].iloc[0]["n_resp1"])
    figures = {}
    color = "#007E87"
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        if has_positives:
            ax.plot(np.r_[0, table.cumulative_population_fraction],
                    np.r_[0, table.cumulative_recall], marker="o", markersize=4,
                    linewidth=2.3, color=color, label=model_name)
            ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Random selection")
            ax.legend(loc="lower right")
        else:
            ax.text(.5, .5, "Recall undefined: no positive TEST snapshots", ha="center", transform=ax.transAxes)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="TEST snapshots selected (actual fraction)",
               ylabel="Cumulative RESP=1 captured / recall", title="Cumulative gains | held-out TEST snapshots")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
        figures["gains"] = fig

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        if has_positives:
            axes[0].bar(table.decile, table.decile_lift, width=.65, color=color)
            axes[1].plot(table.cumulative_population_fraction, table.cumulative_lift,
                         color=color, marker="o", markersize=4, linewidth=2)
        for ax in axes:
            if has_positives:
                ax.axhline(1, color="#777777", linestyle="--", linewidth=1, label="Population baseline = 1")
            else:
                ax.text(.5, .5, "Lift undefined: no positive snapshots", ha="center", transform=ax.transAxes)
            ax.set_ylabel("Lift vs overall TEST response rate")
            ax.grid(axis="y", alpha=.18)
            ax.set_ylim(bottom=0)
        axes[0].set(xticks=range(1, 11), xlabel="Decile (1 = highest propensity)", title="Within-decile lift")
        axes[1].set(xlabel="TEST snapshots selected (actual fraction)", title="Cumulative lift")
        axes[1].xaxis.set_major_formatter(PercentFormatter(1))
        if has_positives:
            axes[1].legend(fontsize=9)
        figures["lift"] = fig

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
        for ax, metric, title in zip(axes, ["recall", "precision", "lift"],
                                     ["Positive snapshots captured", "Precision in selected snapshots", "Cumulative lift"]):
            values = topk[metric].to_numpy(dtype=float)
            bars = ax.bar(np.arange(3), np.nan_to_num(values, nan=0), width=.60, color=color)
            labels = [(f"{value:.2f}×" if metric == "lift" else f"{value:.1%}")
                      if np.isfinite(value) else "N/A" for value in values]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
            ax.set(xticks=range(3), xticklabels=["Top 10%", "Top 20%", "Top 30%"], title=title)
            ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1] * 1.16, .01))
            if metric != "lift":
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                if ax.get_ylim()[1] > 1:
                    ax.set_yticks(np.arange(0, 1.001, .2))
            ax.grid(axis="y", alpha=.18)
        fig.suptitle(f"{model_name} | held-out TEST targeting performance", fontsize=13)
        figures["topk_performance"] = fig

        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        bars = ax.bar(table.decile, table.response_rate, color=color)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in table.response_rate], padding=3, fontsize=9)
        ax.axhline(float(checked.RESP.mean()), color="#777777", linestyle="--",
                   label="Overall TEST response rate")
        ax.set(xticks=range(1, 11), xlabel="Decile (1 = highest propensity)",
               ylabel="Observed response rate", title=f"{model_name} | response rate by decile",
               ylim=(0, max(table.response_rate.max() * 1.22, .01)))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if ax.get_ylim()[1] > 1:
            ax.set_yticks(np.arange(0, 1.001, .2))
        ax.legend()
        figures["response_rate"] = fig

        metrics = results["global_metrics"].iloc[0]
        y, scores = checked.RESP.to_numpy(), checked[score_column].to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), layout="constrained")
        if has_positives:
            precision, recall, _ = precision_recall_curve(y, scores)
            axes[0].step(recall, precision, where="post", color=color,
                         label=f"Average precision = {metrics['average_precision']:.3f}")
            axes[0].axhline(y.mean(), linestyle="--", color="#777777", label="TEST prevalence")
            axes[0].legend()
        else:
            axes[0].text(.5, .5, "Undefined: no positive TEST snapshots", ha="center", transform=axes[0].transAxes)
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            axes[1].plot(fpr, tpr, color=color, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
            axes[1].plot([0, 1], [0, 1], linestyle="--", color="#777777")
            axes[1].legend(loc="lower right")
        else:
            axes[1].text(.5, .5, "Undefined: only one TEST class", ha="center", transform=axes[1].transAxes)
        axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision–recall (primary)",
                    xlim=(0, 1), ylim=(0, 1.02))
        axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (supporting)",
                    xlim=(0, 1), ylim=(0, 1.02))
        fig.suptitle("AP is the non-interpolated average precision, not trapezoidal PR-AUC.", fontsize=10)
        figures["discrimination"] = fig

        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
        fig, ax = plt.subplots(figsize=(5.2, 4.6), layout="constrained")
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (row, column), value in np.ndenumerate(matrix):
            ax.text(column, row, f"{value:,}", ha="center", va="center", fontsize=18,
                    color="white" if value > matrix.max() / 2 else "#183B4E")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["RESP=0", "RESP=1"],
               yticklabels=["RESP=0", "RESP=1"], xlabel="Predicted", ylabel="Actual",
               title=f"TEST confusion matrix\nVALIDATION threshold = {metrics['threshold']:.4f}")
        figures["confusion_matrix"] = fig
    return figures


def evaluate_scores(scored_snapshots, threshold, *, score_column="P_RESP1",
                    model_name="Transformer", make_plots=True):
    """Return aggregate frames and figures for an already verified held-out cohort.

    Input: pandas DataFrame with PATIENT_ID (literal string), END_DT (ISO date or
    Python date), RESP, and score_column. threshold must come from VALIDATION.
    Caller must verify exact TEST keys/labels and frozen model provenance before
    this call. Repeated snapshots contribute separately: metrics are snapshot-
    level, not unique-patient capture. Equal scores use a fixed key hash; decile
    boundaries are ceil(d*N/10). No patient rows are returned or persisted.

    Output keys: global_metrics, deciles, topk, ties (pandas DataFrames), and
    figures (gains, lift, topk_performance, response_rate, discrimination,
    confusion_matrix: matplotlib Figure; empty if disabled).
    Caller may display figures and then close them with plt.close(fig).
    """
    checked = _checked_scored_snapshots(scored_snapshots, score_column)
    metrics = classification_metrics(checked.RESP, checked[score_column], threshold)
    ranked = _rank_snapshots(checked, score_column)
    deciles = _decile_table(ranked, model_name)
    base = {
        "model": model_name, "n_test_snapshots": len(checked),
        "n_test_patients": int(checked.PATIENT_ID.nunique()),
        "n_resp1": int(checked.RESP.sum()), "test_response_rate": float(checked.RESP.mean()),
    }
    results = {
        "global_metrics": pd.DataFrame([{**base, **metrics}]),
        "deciles": deciles,
        "topk": _topk_table(deciles),
        "ties": _tie_table(ranked, score_column, model_name),
    }
    results["figures"] = _evaluation_figures(results, checked, score_column) if make_plots else {}
    return results
