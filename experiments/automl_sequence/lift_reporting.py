"""Post-fit, same-population lift comparison with existing V63 scores."""
import hashlib
import json

import numpy as np
import pandas as pd


ARMS = ("COUNT", "STATIC_FUSION", "TEMPORAL_FUSION")
BASELINE = "LIGHTGBM_V63_EXISTING"
FRACTIONS = (.1, .2, .3)


def _metadata(value, split):
    if not isinstance(value, pd.DataFrame):
        raise TypeError("Ordered snapshot metadata must be a DataFrame")
    required = ["PATIENT_ID", "END_DT", "RESP", "SPLIT"]
    if not set(required) <= set(value):
        raise ValueError("Ordered snapshot keys, labels and split are required")
    frame = value[required].copy().reset_index(drop=True)
    if frame.empty or frame.isna().any().any():
        raise ValueError("Snapshot metadata must be nonempty and complete")
    if not frame.PATIENT_ID.map(lambda v: isinstance(v, str) and bool(v) and v == v.strip()).all():
        raise ValueError("Exact nonempty string patient IDs are required")
    frame["END_DT"] = pd.to_datetime(frame.END_DT, errors="raise")
    if frame.END_DT.dt.tz is not None or not frame.END_DT.eq(frame.END_DT.dt.normalize()).all():
        raise ValueError("Snapshot dates must be timezone-free midnight dates")
    if frame.duplicated(["PATIENT_ID", "END_DT"]).any():
        raise ValueError("Duplicate snapshot keys cannot be compared")
    if not frame.RESP.isin([0, 1]).all():
        raise ValueError("Nonmissing binary labels required")
    frame["RESP"] = frame.RESP.astype(np.int8)
    frame["SPLIT"] = frame.SPLIT.astype(str).str.lower()
    if not frame.SPLIT.eq(split).all():
        raise ValueError("Every prediction must belong to the declared evaluation split")
    return frame


def _identity(frame, scores):
    canonical = frame.copy()
    canonical["END_DT"] = canonical.END_DT.dt.strftime("%Y-%m-%d")
    digest = hashlib.sha256(json.dumps(canonical.to_dict("records"), sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode())
    for name in sorted(scores):
        digest.update(name.encode())
        digest.update(np.asarray(scores[name], dtype="<f8").tobytes())
    return digest.hexdigest()


def build_report(metadata, scores_by_model, baseline_result, split, *, comparison,
                 workflow, allow_test=False, selection_locked=False,
                 n_bootstrap=1000, seed=20261013):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unrecognized evaluation split")
    if split == "test" and not (allow_test and selection_locked):
        raise ValueError("TEST reporting requires explicit evaluation and a persisted selection lock")
    if set(scores_by_model) != set(ARMS):
        raise ValueError("All three frozen Transformer arms are required")
    frame = _metadata(metadata, split)
    if baseline_result is None:
        y = frame.RESP.to_numpy()
        scores = {name: np.asarray(scores_by_model[name], dtype=float) for name in ARMS}
        return dict(split=split, primary_metric="lift", primary_fraction=.1,
            snapshots=len(frame), patients=int(frame.PATIENT_ID.nunique()),
            positives=int(y.sum()), prevalence=float(y.mean()),
            metrics={name: comparison.binary_metrics(y, values, top_fractions=FRACTIONS)
                     for name, values in scores.items()}, comparisons={},
            interpretation="TRANSFORMER_ONLY_EXISTING_SCORES_UNAVAILABLE",
            prediction_identity_sha256=_identity(frame, scores),
            limitations=["Existing LightGBM scores unavailable; no LightGBM lift or rank computed"])
    baseline_frame = _metadata(baseline_result["metadata"], split)
    if not frame.equals(baseline_frame):
        raise ValueError("Existing scores must match every snapshot key, label and row order exactly")
    provenance = baseline_result["provenance"]
    if provenance.get("split") != split or provenance.get("same_keys_and_labels") is not True:
        raise ValueError("Existing-score alignment provenance is missing or inconsistent")
    y = frame.RESP.to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Lift comparison and uncertainty require both classes")
    scores = {name: np.asarray(scores_by_model[name], dtype=float) for name in ARMS}
    scores[BASELINE] = np.asarray(baseline_result["scores"], dtype=float)
    metrics = {name: comparison.binary_metrics(y, values, top_fractions=FRACTIONS)
               for name, values in scores.items()}
    pairs = {}
    for name in ARMS:
        lift = workflow.paired_lift_bootstrap(y, scores[name], scores[BASELINE],
                                            frame.PATIENT_ID.to_numpy(), fractions=FRACTIONS,
                                            n_bootstrap=n_bootstrap, seed=seed)
        secondary = comparison.paired_patient_bootstrap(y, scores[name], scores[BASELINE],
                                                       frame.PATIENT_ID.to_numpy(),
                                                       n_bootstrap=n_bootstrap, seed=seed)
        pairs[name] = dict(lift=lift, secondary=secondary)
    scope = {"train": "Training resubstitution; descriptive only",
             "validation": "Development validation; checkpoint selection uses these labels",
             "test": "Retrospective TEST; not an untouched independent holdout"}[split]
    return dict(split=split, primary_metric="lift", primary_fraction=.1,
                snapshots=len(frame), patients=int(frame.PATIENT_ID.nunique()),
                positives=int(y.sum()), prevalence=float(y.mean()),
                metrics=metrics, comparisons=pairs, baseline_provenance=dict(provenance),
                prediction_identity_sha256=_identity(frame, scores),
                bootstrap_resamples=n_bootstrap, bootstrap_seed=seed,
                score_aggregation="Arithmetic mean of frozen per-seed predictions for each Transformer arm",
                interpretation="DESCRIPTIVE_ONLY", evaluation_scope=scope,
                limitations=["Existing LightGBM fitting membership is unverified; observed lift differences do not establish generalization superiority",
                             "Bootstrap intervals condition on fitted predictions and do not resolve fitting provenance or model-selection uncertainty",
                             "Same ordered snapshots, labels, prevalence and exact ceil(fraction*N) capacity for every model"])


def report_table(report):
    rows = []
    for model, metric in report["metrics"].items():
        lifts = {float(item["fraction"]): item["lift"] for item in metric["topk"]}
        rows.append({"Model": model, "Split": report["split"],
                     "Lift@10%": lifts[.1], "Lift@20%": lifts[.2], "Lift@30%": lifts[.3],
                     "AP": metric["average_precision"], "AUC": metric["roc_auc"],
                     "Snapshots": report["snapshots"], "Positives": report["positives"]})
    return pd.DataFrame(rows)


def interval_table(report):
    rows = []
    for model, pair in report["comparisons"].items():
        for fraction, interval in pair["lift"]["lift_by_fraction"].items():
            rows.append(dict(Model=model, Reference=BASELINE, Split=report["split"],
                             Metric=f"Lift@{round(float(fraction)*100)}%",
                             Delta=interval["delta"], Lower=interval["lower"], Upper=interval["upper"],
                             Reliable=pair["lift"]["interval_reliable"]))
        for name in ("average_precision", "roc_auc"):
            interval = pair["secondary"][name]
            rows.append(dict(Model=model, Reference=BASELINE, Split=report["split"], Metric=name,
                             Delta=interval["delta"], Lower=interval["lower"], Upper=interval["upper"],
                             Reliable=pair["secondary"]["interval_reliable"]))
    return pd.DataFrame(rows)


def ranking_table(report):
    table = report_table(report)[["Model", "Lift@10%"]].copy()
    table["Rank"] = table["Lift@10%"].rank(method="min", ascending=False).astype(int)
    return table[["Rank", "Model", "Lift@10%"]].sort_values(
        ["Rank", "Model"], kind="stable").reset_index(drop=True)
