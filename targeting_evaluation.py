"""Snapshot-level held-out TEST targeting evaluation for the current Transformer.

No fitting, threshold tuning, patient-level exports, or clinical mapping is done here.
All rates in returned tables are fractions in [0, 1]; lifts are multiples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEYS = ["PATIENT_ID", "END_DT"]
MODELS = {"Transformer": "p_transformer"}
TOP_K = (10, 20, 30)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identities(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate without including sensitive row values in exceptions."""
    _require(set(KEYS).issubset(frame.columns), f"{name}: missing snapshot key columns.")
    out = frame.copy()
    _require(not out[KEYS].isna().any().any(), f"{name}: missing snapshot keys.")
    _require(out.PATIENT_ID.map(lambda x: isinstance(x, str)).all(),
             f"{name}: PATIENT_ID must be read as strings to preserve leading zeros.")
    _require(out.PATIENT_ID.str.strip().eq(out.PATIENT_ID).all()
             and out.PATIENT_ID.str.len().gt(0).all(), f"{name}: invalid patient key whitespace.")
    # Canonical ordering must depend on literal keys, never category metadata.
    out["PATIENT_ID"] = out.PATIENT_ID.astype(str)
    # Canonical dates prevent silent merging of distinct intraday snapshots.
    dates = out.END_DT.astype(str)
    _require(dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all(),
             f"{name}: END_DT must be an ISO calendar date (YYYY-MM-DD).")
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    _require(parsed.notna().all(), f"{name}: invalid END_DT calendar dates.")
    out["END_DT"] = parsed.dt.strftime("%Y-%m-%d")
    _require(not out.duplicated(KEYS).any(), f"{name}: duplicate snapshot keys.")
    return out


def _binary(series: pd.Series, name: str) -> pd.Series:
    _require(series.notna().all() and series.isin([0, 1]).all(),
             f"{name}: RESP must contain only 0 and 1, with no missing labels.")
    return series.astype(np.int64)


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Check the frozen, full-cohort snapshot manifest for patient disjointness."""
    out = _identities(manifest, "Manifest")
    _require({"RESP", "SPLIT"}.issubset(out.columns), "Manifest: missing RESP or SPLIT.")
    out["RESP"] = _binary(out.RESP, "Manifest")
    _require(out.SPLIT.isin(["TRAIN", "VALIDATION", "TEST"]).all(),
             "Manifest: SPLIT must be TRAIN, VALIDATION, or TEST.")
    _require(set(out.SPLIT) == {"TRAIN", "VALIDATION", "TEST"},
             "Manifest must include all three splits, not only TEST rows.")
    _require(out.groupby("PATIENT_ID", observed=True).SPLIT.nunique().eq(1).all(),
             "Manifest: a patient crosses splits; patient-level leakage detected.")
    return out[KEYS + ["RESP", "SPLIT"]].sort_values(KEYS).reset_index(drop=True)


def manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """Row-order-independent digest of identities, outcomes, and frozen assignments."""
    canonical = validate_manifest(manifest)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_inputs(manifest: pd.DataFrame, transformer: pd.DataFrame) -> pd.DataFrame:
    """Require Transformer scores to cover exactly every frozen TEST snapshot.

    Canonical outcomes come from the manifest. Optional prediction-file outcomes
    are checked for equality, never used to form ranks. No inner join is allowed.
    """
    frozen = validate_manifest(manifest)
    expected = frozen.loc[frozen.SPLIT.eq("TEST"), KEYS + ["RESP"]].copy()
    _require(len(expected) >= 10, "At least 10 TEST snapshots are required for ten nonempty deciles.")
    expected_index = pd.MultiIndex.from_frame(expected[KEYS])
    aligned = expected.copy()
    for name, frame in [("Transformer", transformer)]:
        pred = _identities(frame, name)
        _require("P_RESP1" in pred, f"{name}: missing P_RESP1.")
        actual_index = pd.MultiIndex.from_frame(pred[KEYS])
        _require(len(pred) == len(expected) and expected_index.isin(actual_index).all(),
                 f"{name}: predictions must exactly cover the frozen TEST snapshots; no missing or extra rows.")
        scores = pd.to_numeric(pred.P_RESP1, errors="coerce")
        _require(not np.iscomplexobj(scores) and np.isfinite(scores).all()
                 and scores.between(0, 1).all(),
                 f"{name}: probabilities must be real, finite, and within [0, 1].")
        pred["P_RESP1"] = scores.astype(float)
        if "SPLIT" in pred:
            _require(pred.SPLIT.eq("TEST").all(), f"{name}: score file includes non-TEST rows.")
        if "RESP" in pred:
            pred["RESP"] = _binary(pred.RESP, name)
            checked = expected.merge(pred[KEYS + ["RESP"]], on=KEYS,
                                     how="left", validate="one_to_one", suffixes=("", "_pred"))
            _require(checked.RESP.eq(checked.RESP_pred).all(),
                     f"{name}: prediction-file labels disagree with the frozen manifest.")
        aligned = aligned.merge(pred[KEYS + ["P_RESP1"]].rename(
            columns={"P_RESP1": MODELS[name]}), on=KEYS, how="left", validate="one_to_one")
    return aligned.reset_index(drop=True)


def validate_provenance(metadata: dict[str, Any], manifest: pd.DataFrame) -> None:
    """Verify supplied run declarations; this does not audit training code."""
    digest = manifest_fingerprint(manifest)
    for name in MODELS:
        _require(name in metadata, f"Provenance: missing {name} run metadata.")
        run = metadata[name]
        _require(run.get("snapshot_manifest_sha256") == digest,
                 f"{name}: provenance does not match the supplied frozen manifest.")
        _require(run.get("population_version") == "V63", f"{name}: expected V63 population.")
        _require(str(run.get("claims_vintage")) == "20260825",
                 f"{name}: expected claims vintage 20260825.")
        _require(run.get("model_frozen_before_test") is True,
                 f"{name}: missing model-frozen-before-test declaration.")
        _require(run.get("training_split") == "TRAIN" and run.get("tuning_split") == "VALIDATION"
                 and run.get("scoring_split") == "TEST", f"{name}: invalid declared split roles.")


def _snapshot_hash(patient: str, end_date: str) -> str:
    # A fixed public salt is only for reproducible, label-independent tie-breaking.
    # Hashes are not anonymized data and are never exported in the report.
    key = json.dumps(["tak861-targeting-v1", patient, end_date], separators=(",", ":"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _score_order(frame: pd.DataFrame, score_column: str) -> np.ndarray:
    """This function cannot access labels: select key/score columns immediately."""
    scoring = frame[KEYS + [score_column]].copy()
    _require(not np.iscomplexobj(scoring[score_column]) and np.isfinite(scoring[score_column]).all()
             and scoring[score_column].between(0, 1).all(), "Invalid predicted probabilities.")
    scoring["_tie"] = [_snapshot_hash(p, d) for p, d in scoring[KEYS].itertuples(index=False, name=None)]
    scoring["_position"] = np.arange(len(scoring))
    # Keys provide a deterministic fallback for the practically impossible hash collision.
    return scoring.sort_values([score_column, "_tie"] + KEYS,
                               ascending=[False, True, True, True])["_position"].to_numpy()


def rank_snapshots(aligned: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Decile 1 is highest propensity; b_d = ceil(d*N/10), independent of RESP."""
    checked = _identities(aligned, "Evaluation")
    n = len(checked)
    _require(n >= 10, "At least 10 snapshots are required for ten nonempty deciles.")
    ranked = checked.iloc[_score_order(checked, score_column)].reset_index(drop=True)
    ranks = np.arange(1, n + 1, dtype=np.int64)
    boundaries = (np.arange(1, 11, dtype=np.int64) * n + 9) // 10
    ranked["rank"] = ranks
    ranked["decile"] = np.searchsorted(boundaries, ranks, side="left") + 1
    return ranked


def decile_table(ranked: pd.DataFrame, model_name: str) -> pd.DataFrame:
    y = _binary(ranked.RESP, "Evaluation")
    n, positives = len(y), int(y.sum())
    overall_rate = positives / n
    table = ranked.assign(RESP=y).groupby("decile").agg(
        n_snapshots=("RESP", "size"), n_resp1=("RESP", "sum")).reindex(range(1, 11)).reset_index()
    _require(table.n_snapshots.notna().all(), "All ten deciles must be present.")
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


def topk_table(deciles: pd.DataFrame) -> pd.DataFrame:
    selected = deciles.loc[deciles.decile.isin([1, 2, 3])].copy()
    selected["top_k_pct"] = selected.decile * 10
    return selected.rename(columns={
        "cumulative_n_snapshots": "n_selected", "cumulative_resp1": "n_resp1_selected",
        "cumulative_recall": "recall", "cumulative_precision": "precision",
        "cumulative_lift": "lift", "cumulative_population_fraction": "actual_population_fraction",
    })[["model", "top_k_pct", "n_selected", "n_resp1_selected", "recall", "precision",
        "lift", "actual_population_fraction"]].rename(columns={"n_resp1_selected": "n_resp1"}).reset_index(drop=True)


def _ties(ranked: pd.DataFrame, score_column: str, name: str) -> pd.DataFrame:
    scores = ranked[score_column].to_numpy()
    rows = []
    for k in TOP_K:
        cutoff = (k * len(scores) + 99) // 100
        value = scores[cutoff - 1]
        total = int((scores == value).sum())
        included = int((scores[:cutoff] == value).sum())
        rows.append({"model": name, "top_k_pct": k, "cutoff_score": value,
                     "n_tied_at_cutoff": total, "n_tied_selected": included,
                     "tie_crosses_boundary": included < total})
    return pd.DataFrame(rows)


def evaluate_model(aligned: pd.DataFrame, model_name: str = "Transformer",
                   score_column: str = "p_transformer") -> dict[str, pd.DataFrame]:
    """Evaluate after validate_inputs has checked the complete frozen split."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = _binary(aligned.RESP, "Evaluation")
    ranked = rank_snapshots(aligned, score_column)
    deciles = decile_table(ranked, model_name)
    global_row = {"model": model_name, "n_test_snapshots": len(y),
                  "n_test_patients": aligned.PATIENT_ID.nunique(), "n_resp1": int(y.sum()),
                  "test_response_rate": float(y.mean()),
                  "average_precision": average_precision_score(y, aligned[score_column]) if y.sum() else np.nan,
                  "roc_auc": roc_auc_score(y, aligned[score_column]) if y.nunique() == 2 else np.nan}
    return {"deciles": deciles, "topk": topk_table(deciles),
            "global_metrics": pd.DataFrame([global_row]), "ties": _ties(ranked, score_column, model_name)}


def plot_results(results: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    """Export standalone figures; no patient-level values are plotted or saved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {name: "#007E87" for name in results["deciles"].model.unique()}
    has_positives = bool(results["global_metrics"].iloc[0].n_resp1)
    paths = {}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        for name, table in results["deciles"].groupby("model", sort=False):
            ax.plot(np.r_[0, table.cumulative_population_fraction], np.r_[0, table.cumulative_recall],
                    marker="o", markersize=4, linewidth=2.3, color=colors[name], label=name)
        if has_positives:
            ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Random selection")
        else:
            ax.text(.5, .5, "Recall is undefined: no positive TEST snapshots", ha="center", transform=ax.transAxes)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="TEST snapshots selected (actual fraction)",
               ylabel="Cumulative RESP=1 captured / recall", title="Cumulative gains | held-out TEST snapshots")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
        ax.legend(loc="lower right")
        paths["gains"] = output_dir / "cumulative_gains.png"
        fig.savefig(paths["gains"], dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        for i, (name, table) in enumerate(results["deciles"].groupby("model", sort=False)):
            axes[0].bar(table.decile, table.decile_lift, width=.65,
                        color=colors[name], label=name)
            axes[1].plot(table.cumulative_population_fraction, table.cumulative_lift,
                         color=colors[name], marker="o", markersize=4, linewidth=2, label=name)
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
        axes[1].legend(fontsize=9)
        paths["lift"] = output_dir / "lift.png"
        fig.savefig(paths["lift"], dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
        for ax, metric, title in zip(axes, ["recall", "precision", "lift"],
                                     ["Positive snapshots captured", "Precision in selected snapshots", "Cumulative lift"]):
            for i, name in enumerate(colors):
                table = results["topk"].loc[results["topk"].model.eq(name)].sort_values("top_k_pct")
                bars = ax.bar(np.arange(3), table[metric], width=.60,
                              color=colors[name], label=name)
                labels = [(f"{v:.2f}×" if metric == "lift" else f"{v:.1%}") if np.isfinite(v) else "N/A"
                          for v in table[metric]]
                ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
                if table[metric].isna().all():
                    ax.text(.5, .5, "Undefined: no positive snapshots", ha="center", fontsize=9, transform=ax.transAxes)
            ax.set(xticks=range(3), xticklabels=["Top 10%", "Top 20%", "Top 30%"], title=title)
            ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1] * 1.16, .01))
            if metric != "lift":
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                if ax.get_ylim()[1] > 1:
                    ax.set_yticks(np.arange(0, 1.001, .2))
            ax.grid(axis="y", alpha=.18)
        axes[0].legend(fontsize=9)
        fig.suptitle("Transformer | held-out TEST targeting performance", fontsize=13)
        paths["topk_performance"] = output_dir / "topk_performance.png"
        fig.savefig(paths["topk_performance"], dpi=180)
        plt.close(fig)
    return paths


def _fmt(value: float, style: str = "number") -> str:
    if pd.isna(value):
        return "N/A"
    if style == "percent":
        return f"{value:.2%}"
    if style == "integer":
        return str(int(value))
    return f"{value:.3f}"


def _markdown_table(frame: pd.DataFrame, percent: set[str] | None = None,
                    integer: set[str] | None = None) -> str:
    percent, integer = percent or set(), integer or set()
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(v) if isinstance(v, str) else _fmt(v, "percent" if c in percent else
                  "integer" if c in integer else "number") for c, v in zip(columns, row)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(results: dict[str, pd.DataFrame], output_dir: str | Path,
                 audit: dict[str, Any] | None = None) -> dict[str, Path]:
    """Write only aggregate tables/charts and a measured targeting summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in results.items():
        paths[name] = out / f"{name}.csv"
        table.to_csv(paths[name], index=False)
    paths.update(plot_results(results, out))
    base = results["global_metrics"].iloc[0]
    lines = ["# Transformer held-out TEST targeting evaluation", "",
             f"Evaluated {int(base.n_test_snapshots):,} snapshots from {int(base.n_test_patients):,} patients; "
             f"{int(base.n_resp1):,} positive snapshots; overall TEST response rate {_fmt(base.test_response_rate, 'percent')}.",
             "", "## Top-K targeting results", "",
             "Capture refers to positive snapshots, not unique patients. All selection budgets were prespecified.", ""]
    for row in results["topk"].itertuples(index=False):
        lines.append(f"- Top {row.top_k_pct}% ({row.n_selected:,} snapshots; actual coverage "
                     f"{row.actual_population_fraction:.2%}): {int(row.n_resp1):,} positives captured; "
                     f"Recall@{row.top_k_pct}% = {_fmt(row.recall, 'percent')}; "
                     f"Precision@{row.top_k_pct}% = {_fmt(row.precision, 'percent')}; "
                     f"cumulative lift = {_fmt(row.lift)}×.")
    first = results["topk"].loc[results["topk"].top_k_pct.eq(10)].iloc[0]
    lines.extend(["", f"Top-decile lift: {_fmt(first.lift)}×.", ""])
    if base.n_resp1 == 0:
        lines.append("There are no positive TEST snapshots. Capture/recall and lift cannot be estimated.")
    else:
        lines.append(f"Selecting the highest-scored {_fmt(first.actual_population_fraction, 'percent')} of TEST snapshots "
                     f"captures {_fmt(first.recall, 'percent')} of observed positives, with "
                     f"{_fmt(first.precision, 'percent')} response rate in that selected population.")
    lines.extend(["", "These are descriptive estimates from the held-out cohort. "
                  "At a fixed capacity, recall, precision, and lift are transformations of the same captured-positive count; "
                  "they are not independent evidence. Global ROC-AUC alone does not describe prioritization value. "
                  "No uncertainty intervals have been computed.", "",
                  "LightGBM evaluation and incremental model comparison are deferred until its artifact location is available. "
                  "No claim about Transformer superiority over LightGBM is made.", "",
                  _markdown_table(results["topk"], percent={"recall", "precision", "actual_population_fraction"},
                                  integer={"top_k_pct", "n_selected", "n_resp1"}), "",
                  "## Decile performance", ""])
    cols = ["decile", "n_snapshots", "n_resp1", "response_rate", "share_of_all_resp1", "cumulative_resp1",
            "cumulative_recall", "decile_lift", "cumulative_lift", "cumulative_precision"]
    lines.append(_markdown_table(results["deciles"][cols],
                                 percent={"response_rate", "share_of_all_resp1", "cumulative_recall", "cumulative_precision"},
                                 integer={"decile", "n_snapshots", "n_resp1", "cumulative_resp1"}))
    lines.extend(["", "Response rate is precision within the individual decile. Cumulative precision is precision "
                  "within all selected snapshots through that decile. Capture shares and rates are shown as percentages above; "
                  "the CSV stores fractions in [0, 1].", "",
                  "![Cumulative gains](cumulative_gains.png)", "", "![Lift](lift.png)", "",
                  "![Top-K performance](topk_performance.png)", "", "## Global discrimination (supporting context)", "",
                  _markdown_table(results["global_metrics"][["model", "average_precision", "roc_auc"]]), "",
                  "Average precision is the non-interpolated PR summary; it is not trapezoidal PR-AUC. "
                  "No TEST-tuned classification threshold is used."])
    crossing = int(results["ties"].tie_crosses_boundary.sum())
    lines.extend(["", "## Evaluation notes", "",
                  "Deciles use descending probability and a fixed SHA-256 snapshot-key tie-break. "
                  "Labels are used only after ranks/buckets are formed. Decile 1 is highest propensity. "
                  "Cumulative boundaries are ceil(d × N / 10); actual selected fractions are reported.", "",
                  f"Ties cross {crossing} top-K boundaries; see ties.csv. "
                  "Tie-breaking does not imply discrimination within an equal-score group.", "",
                  "Repeated snapshots contribute repeatedly, so these results are not patient-level outreach capture. "
                  "No unique-patient scoring rule is inferred from TEST outcomes. "
                  "Zero-positive populations have undefined recall/lift (N/A).", "",
                  "The full manifest is checked for patient-disjoint TRAIN/VALIDATION/TEST assignment, and score files must exactly match its TEST keys. "
                  "Model/preprocessing provenance and upstream temporal leakage still require training-pipeline review."])
    if audit is not None:
        paths["audit"] = out / "evaluation_audit.json"
        paths["audit"].write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths["report"] = out / "evaluation_report.md"
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def read_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        # Patient identifiers such as "NA" or "NULL" are literal keys, not missing
        # values. Downstream required-field validation rejects empty fields.
        return pd.read_csv(path, dtype={"PATIENT_ID": "string", "END_DT": "string"},
                           keep_default_na=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input files must be CSV or Parquet.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--provenance", help="Optional training-generated JSON declaration for Transformer")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = read_input(args.manifest)
    aligned = validate_inputs(manifest, read_input(args.transformer))
    if args.provenance:
        validate_provenance(json.loads(Path(args.provenance).read_text(encoding="utf-8")), manifest)
    results = evaluate_model(aligned)
    audit = {"snapshot_manifest_sha256": manifest_fingerprint(manifest),
             "patient_disjointness_checked": True, "exact_test_snapshot_match_checked": True,
             "training_provenance_declarations_checked": bool(args.provenance),
             "training_code_audited": False,
             "population_version_expected": "V63", "claims_vintage_expected": "20260825",
             "decile_boundary": "ceil(d * N / 10)", "ranking_uses_labels": False}
    paths = write_report(results, args.output_dir, audit)
    print(f"Wrote {len(paths)} aggregate evaluation files. No snapshot-level predictions were exported.")


if __name__ == "__main__":
    main()
