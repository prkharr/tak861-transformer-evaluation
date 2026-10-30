"""Frozen production scores versus registered models on the SAME current population, lift only.

Scope and honesty rules:
- Production scores are loaded from a configured scored table on exact PATIENT_ID + END_DT
  keys of the current population, labelled with the CURRENT RESP. Keys are never repaired:
  missing, non-string or padded IDs and non-midnight or timezone dates in the scored table
  block the comparison. No nearest-date or intersection fallback; missing or duplicate keys
  stop. Nothing is refitted.
- The evaluation split is development VALIDATION (its labels selected checkpoints) or the
  retrospective, already-inspected TEST. Neither is an untouched holdout, so the best status
  is "MATCHED_POPULATION_PROVENANCE_DOCUMENTED": same population and target, with every
  provenance field and flag supplied. Those fields are assertions recorded in the
  configuration, not evidence verified here. Otherwise the status is
  "COMMON_POPULATION_DESCRIPTIVE" with unknown production training overlap.
- When no scored table is configured, the read fails, or the keys cannot be aligned, the
  production row is BLOCKED with a sanitized reason and the exact fields needed; model
  metrics are still reported and the production row is never ranked. No metric is fabricated.
- Only lift@10/20/30 and the paired patient-cluster lift bootstrap are reported.
- `comparison` and `workflow` are injected modules (shared metric and bootstrap code).
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd

PRODUCTION = "PRODUCTION_FROZEN_SCORES"
DOCUMENTED = "MATCHED_POPULATION_PROVENANCE_DOCUMENTED"
DESCRIPTIVE = "COMMON_POPULATION_DESCRIPTIVE"
FRACTIONS = (.1, .2, .3)
KEYS = ["PATIENT_ID", "END_DT"]
SPLITS = ("train", "validation", "test")
SCOPE = {"train": "Training resubstitution; descriptive only",
         "validation": "Development validation; checkpoint selection used these labels; not an untouched holdout",
         "test": "Retrospective TEST with reconstructed labels, already inspected; not an untouched holdout"}
REQUIRED_EVIDENCE_FIELDS = ("production_model_fingerprint", "production_training_window_end",
                            "production_training_patient_exclusion_reference",
                            "target_definition_match_reference", "historical_input_vintage_reference")
REQUIRED_EVIDENCE_FLAGS = ("patient_disjoint_from_production_training", "same_target_definition",
                           "historical_inputs_frozen")
DEFAULT_CONFIG = {"score_table": "", "score_column": "SCORE", "label_column": None,
                  "score_kind": "probability", "max_rows": 2000000, "evidence": {}}


class ProductionScoresUnavailable(Exception):
    def __init__(self, reason, required):
        super().__init__(reason)
        self.reason, self.required = reason, list(required)


def _config(config):
    cfg = dict(DEFAULT_CONFIG)
    incoming = dict(config or {})
    unknown = set(incoming) - set(cfg)
    if unknown:
        raise ValueError("Unknown production comparison configuration: " + str(sorted(unknown)))
    cfg.update(incoming)
    cfg["evidence"] = dict(cfg.get("evidence") or {})
    if cfg["score_kind"] != "probability":
        raise ValueError("Only probability scores in [0,1] are comparable with the shared metrics; no rescaling is applied")
    if type(cfg["max_rows"]) is not int or cfg["max_rows"] < 1:
        raise ValueError("max_rows must be a positive integer")
    for key in ("score_column", "label_column"):
        value = cfg[key]
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", str(value)):
            raise ValueError("Invalid column identifier for " + key)
    if cfg["score_table"] and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){2}", cfg["score_table"]):
        raise ValueError("score_table must be a fully qualified database.schema.table identifier")
    return cfg


def _gate(split, allow_test, selection_locked):
    """TEST is reported only on explicit request with a persisted selection lock; checked before any read."""
    if split not in SPLITS:
        raise ValueError("Unrecognized evaluation split")
    if split == "test" and not (allow_test and selection_locked):
        raise ValueError("TEST reporting requires explicit evaluation and a persisted selection lock")


def _exact_keys(frame):
    """Exact-key contract shared with the frozen score loader: nothing is stripped, cast or normalized."""
    if frame[KEYS].isna().any().any():
        raise ValueError("Missing PATIENT_ID or END_DT keys")
    if not frame.PATIENT_ID.map(lambda v: isinstance(v, str) and bool(v) and v == v.strip()).all():
        raise ValueError("Patient IDs must be exact nonempty strings without surrounding whitespace")
    dates = pd.to_datetime(frame.END_DT, errors="raise")
    if dates.dt.tz is not None:
        raise ValueError("Timezone-dependent snapshot dates are unsupported")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Snapshot dates must be midnight dates")
    frame = frame.copy()
    frame["END_DT"] = dates
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{int(frame.duplicated(KEYS).sum())} duplicate PATIENT_ID/END_DT rows")
    return frame


def _metadata(value, split):
    if not isinstance(value, pd.DataFrame) or not {"PATIENT_ID", "END_DT", "RESP", "SPLIT"} <= set(value):
        raise ValueError("Ordered snapshot metadata with PATIENT_ID, END_DT, RESP and SPLIT is required")
    frame = value[["PATIENT_ID", "END_DT", "RESP", "SPLIT"]].copy().reset_index(drop=True)
    if frame.empty or frame.isna().any().any():
        raise ValueError("Snapshot metadata must be nonempty and complete")
    frame = _exact_keys(frame)
    if not frame.RESP.isin([0, 1]).all():
        raise ValueError("Nonmissing binary labels required")
    frame["RESP"] = frame.RESP.astype(np.int8)
    frame["SPLIT"] = frame.SPLIT.astype(str).str.lower()
    if not frame.SPLIT.eq(split).all():
        raise ValueError("Every prediction must belong to the declared evaluation split")
    return frame


def required_fields(cfg):
    missing = ["score_table"] if not cfg["score_table"] else []
    missing += [f for f in REQUIRED_EVIDENCE_FIELDS if not str(cfg["evidence"].get(f) or "").strip()]
    missing += [f for f in REQUIRED_EVIDENCE_FLAGS if cfg["evidence"].get(f) is not True]
    return missing


def classify_benchmark(cfg, scored_label_mismatches, split):
    """Documented provenance only with complete evidence; the record is an assertion, never a verification."""
    reasons = []
    for field in REQUIRED_EVIDENCE_FIELDS:
        if not str(cfg["evidence"].get(field) or "").strip():
            reasons.append("missing evidence field: " + field)
    for flag in REQUIRED_EVIDENCE_FLAGS:
        if cfg["evidence"].get(flag) is not True:
            reasons.append("not affirmed: " + flag)
    if scored_label_mismatches is None:
        reasons.append("scored labels not supplied; agreement with the current target unknown")
    elif scored_label_mismatches > 0:
        reasons.append(f"{scored_label_mismatches} scored rows carry a different label than the current target")
    documented = not reasons
    scope = SCOPE[split]
    return dict(status=DOCUMENTED if documented else DESCRIPTIVE, reasons=reasons, evaluation_scope=scope,
                statement=(("Same current population and target on " + scope + ". Production training exclusion, "
                            "target match and frozen inputs are asserted by the supplied evidence record and are not "
                            "independently verified here. This is not a future untouched holdout benchmark")
                           if documented else
                           ("Descriptive common-population comparison on " + scope + " with unknown production "
                            "training overlap; not a held-out benchmark and not evidence of generalization")),
                evidence_is_assertion=True, evidence=dict(cfg["evidence"]))


def _read(read_query, sql, max_rows, missing):
    """Source access and collection failures block with the exception class only: no message, no credentials."""
    try:
        value = read_query(sql)
        if not isinstance(value, pd.DataFrame):
            if not hasattr(value, "toPandas"):
                raise TypeError("Reader must return a Spark or pandas DataFrame")
            value = value.limit(max_rows + 1).toPandas()
    except TypeError:
        raise
    except Exception as error:  # deliberately sanitized source failure; logic errors are not caught elsewhere
        raise ProductionScoresUnavailable("Scored table read failed: " + type(error).__name__, missing) from None
    return value


def load_production_scores(read_query, metadata, config, split):
    """Exact-key production scores for the current population; raises ProductionScoresUnavailable when blocked."""
    cfg = _config(config)
    frame = _metadata(metadata, split)
    missing = required_fields(cfg)
    if not cfg["score_table"]:
        raise ProductionScoresUnavailable("No production scored table configured for the current population", missing)
    columns = KEYS + [cfg["score_column"]] + ([cfg["label_column"]] if cfg["label_column"] else [])
    sql = "SELECT " + ", ".join(columns) + " FROM " + cfg["score_table"]
    scored = _read(read_query, sql, cfg["max_rows"], missing).copy()
    names = [str(c).upper() for c in scored.columns]
    if len(set(names)) != len(names):
        raise ProductionScoresUnavailable("Case-insensitive duplicate columns in the scored table", missing)
    scored.columns = names
    if len(scored) > cfg["max_rows"]:
        raise ProductionScoresUnavailable("Scored table exceeds the configured row budget; narrow it to the current population", missing)
    needed = set(c.upper() for c in columns)
    if not needed <= set(scored.columns):
        raise ProductionScoresUnavailable("Scored table lacks required columns: " + ", ".join(sorted(needed - set(scored.columns))), missing)
    score_col, label_col = cfg["score_column"].upper(), (cfg["label_column"].upper() if cfg["label_column"] else None)
    # Keep the scored table's own columns apart from the current labels before merging.
    renamed = {score_col: "PRODUCTION_SCORE"}
    if label_col:
        renamed[label_col] = "SCORED_LABEL"
    scored = scored[KEYS + list(renamed)].rename(columns=renamed)
    try:
        scored = _exact_keys(scored)
    except ValueError as error:
        raise ProductionScoresUnavailable("Scored table keys are not exact: " + str(error), missing) from None
    joined = frame.merge(scored, on=KEYS, how="left", indicator=True, validate="one_to_one")
    absent = int(joined._merge.ne("both").sum())
    if absent:
        raise ProductionScoresUnavailable(f"{absent} of {len(frame)} current snapshots have no production score on the exact key; "
                                          "nearest-date or intersection fallbacks are not permitted", missing)
    scores = pd.to_numeric(joined["PRODUCTION_SCORE"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ProductionScoresUnavailable("Production scores must be finite probabilities in [0,1] for every current snapshot", missing)
    mismatches = None
    if label_col:
        labels = pd.to_numeric(joined["SCORED_LABEL"], errors="coerce")
        mismatches = int(labels.isna().sum() + (labels.notna() & labels.ne(joined.RESP)).sum())
    extra = int(len(scored) - len(frame))
    digest = hashlib.sha256(scores.astype("<f8").tobytes()).hexdigest()
    provenance = dict(baseline_kind="production_frozen_scores", split=split, same_keys_and_labels=True,
                      labels_source="current population labels; scored-table labels are compared only, never used",
                      score_table=cfg["score_table"], score_column=cfg["score_column"],
                      scored_label_mismatches=mismatches, extra_scored_rows_ignored=extra,
                      fit_membership_verified=False, refit_performed=False,
                      scores_sha256=digest, snapshots=len(frame))
    return dict(metadata=frame, scores=scores, provenance=provenance)


def _identity(frame, scores_by_model):
    canonical = frame.copy()
    canonical["END_DT"] = canonical.END_DT.dt.strftime("%Y-%m-%d")
    digest = hashlib.sha256(json.dumps(canonical.to_dict("records"), sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode())
    for name in sorted(scores_by_model):
        digest.update(name.encode())
        digest.update(np.asarray(scores_by_model[name], dtype="<f8").tobytes())
    return digest.hexdigest()


def _vector(values, length, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (length,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Scores must be one finite probability per current snapshot: " + str(name))
    return values


def compare(metadata, model_scores, production, split, *, comparison, workflow, config=None,
            allow_test=False, selection_locked=False, n_bootstrap=1000, seed=20261013):
    """Lift@10/20/30 and paired patient-cluster lift uncertainty of every model against frozen production."""
    cfg = _config(config)
    _gate(split, allow_test, selection_locked)
    frame = _metadata(metadata, split)
    if not model_scores:
        raise ValueError("At least one registered model prediction vector is required")
    if PRODUCTION in {str(name) for name in model_scores}:
        raise ValueError(PRODUCTION + " is reserved for the frozen production scores")
    scores = {str(name): _vector(values, len(frame), name) for name, values in model_scores.items()}
    y = frame.RESP.to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Lift comparison and uncertainty require both classes")
    metrics = {name: comparison.binary_metrics(y, values, top_fractions=FRACTIONS) for name, values in scores.items()}
    report = dict(split=split, primary_metric="lift", primary_fraction=.1, reported_metrics=["lift@10%", "lift@20%", "lift@30%"],
                  reference=PRODUCTION, snapshots=len(frame), patients=int(frame.PATIENT_ID.nunique()),
                  positives=int(y.sum()), prevalence=float(y.mean()), metrics=metrics, comparisons={},
                  bootstrap_resamples=n_bootstrap, bootstrap_seed=seed, evaluation_scope=SCOPE[split],
                  limitations=["Bootstrap intervals condition on fitted predictions and exclude training and selection uncertainty",
                               "Same ordered snapshots, labels, prevalence and exact ceil(fraction*N) capacity for every model",
                               "No production refit, no legacy feature-creation procedure, no label changes were performed here",
                               "Evidence fields are assertions recorded in the configuration, not verified provenance"])
    if production is None:
        report["production"] = dict(status="BLOCKED", required=required_fields(cfg),
                                    note="No aligned frozen production scores; no production lift, difference or rank is reported")
        report["benchmark"] = dict(status="NOT_COMPARABLE", reasons=["production scores unavailable"], evaluation_scope=SCOPE[split],
                                   statement="Production comparison blocked; registered models are reported alone")
        report["prediction_identity_sha256"] = _identity(frame, scores)
        return report
    base_frame = _metadata(production["metadata"], split)
    if not base_frame[KEYS + ["RESP"]].equals(frame[KEYS + ["RESP"]]):
        raise ValueError("Production scores must be aligned to every current snapshot key and label in order")
    provenance = dict(production["provenance"])
    if provenance.get("split") != split or provenance.get("same_keys_and_labels") is not True:
        raise ValueError("Production alignment provenance is missing or inconsistent")
    base = _vector(production["scores"], len(frame), PRODUCTION)
    metrics[PRODUCTION] = comparison.binary_metrics(y, base, top_fractions=FRACTIONS)
    patients = frame.PATIENT_ID.to_numpy()
    for name, values in scores.items():
        lift = workflow.paired_lift_bootstrap(y, values, base, patients, fractions=FRACTIONS, n_bootstrap=n_bootstrap, seed=seed)
        report["comparisons"][name] = dict(lift=lift, direction="model_minus_production")
    report["production"] = dict(status="ALIGNED", provenance=provenance)
    report["benchmark"] = classify_benchmark(cfg, provenance.get("scored_label_mismatches"), split)
    report["prediction_identity_sha256"] = _identity(frame, dict(scores, **{PRODUCTION: base}))
    return report


def run(read_query, metadata, config, model_scores, split, *, comparison, workflow,
        allow_test=False, selection_locked=False, n_bootstrap=1000, seed=20261013):
    """Gate, load (or block) production scores, then compare. Never raises for an unavailable production table."""
    _gate(split, allow_test, selection_locked)
    try:
        production = load_production_scores(read_query, metadata, config, split)
        blocked = None
    except ProductionScoresUnavailable as error:
        production, blocked = None, error
    report = compare(metadata, model_scores, production, split, comparison=comparison, workflow=workflow,
                     config=config, allow_test=allow_test, selection_locked=selection_locked,
                     n_bootstrap=n_bootstrap, seed=seed)
    if blocked is not None:
        report["production"].update(reason=blocked.reason, required=blocked.required)
    return report


def _lifts(metric):
    return {float(item["fraction"]): item["lift"] for item in metric["topk"]}


def report_table(report):
    rows = []
    for model, metric in report["metrics"].items():
        lifts = _lifts(metric)
        rows.append({"Model": model, "Split": report["split"], "Lift@10%": lifts[.1], "Lift@20%": lifts[.2],
                     "Lift@30%": lifts[.3], "Snapshots": report["snapshots"], "Positives": report["positives"]})
    if report["production"]["status"] != "ALIGNED":
        rows.append({"Model": PRODUCTION, "Split": report["split"], "Lift@10%": None, "Lift@20%": None, "Lift@30%": None,
                     "Snapshots": report["snapshots"], "Positives": report["positives"]})
    return pd.DataFrame(rows)


def interval_table(report):
    rows = []
    for model, pair in report["comparisons"].items():
        for fraction, interval in pair["lift"]["lift_by_fraction"].items():
            rows.append(dict(Model=model, Reference=PRODUCTION, Split=report["split"],
                             Metric=f"Lift@{round(float(fraction) * 100)}%", Delta=interval["delta"],
                             Lower=interval["lower"], Upper=interval["upper"], Reliable=pair["lift"]["interval_reliable"]))
    return pd.DataFrame(rows, columns=["Model", "Reference", "Split", "Metric", "Delta", "Lower", "Upper", "Reliable"])


def ranking_table(report):
    table = report_table(report)[["Model", "Lift@10%"]].copy()
    table["Status"] = np.where(table["Lift@10%"].notna(), report["benchmark"]["status"], "BLOCKED")
    scored = table[table["Lift@10%"].notna()].copy()
    scored["Rank"] = scored["Lift@10%"].rank(method="min", ascending=False).astype(int)
    blocked = table[table["Lift@10%"].isna()].copy()
    blocked["Rank"] = None
    out = pd.concat([scored.sort_values(["Rank", "Model"], kind="stable"), blocked], ignore_index=True)
    return out[["Rank", "Model", "Lift@10%", "Status"]]
