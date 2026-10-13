"""Post-lock VALIDATION permutation reliance, never feature/model selection.

Grouped perturbations describe fitted predictive dependence, not clinical
importance or causality. Correlation can hide or redistribute reliance. This is
not projection-weight magnitude and does not impose agreement with proxy rank.
"""
import numpy as np
import pandas as pd


PERMUTATION_SEEDS = (271, 811, 1297)


def _groups(arm, prep):
    counts = len(prep["count"]["selected_names"]) if "count" in prep else 0
    automl = len(prep["automl"]["selected_names"])
    families = []
    if arm in {"COUNT", "STATIC_FUSION", "TEMPORAL_FUSION"}:
        families.append(("count", "sequence", np.arange(counts)))
    if arm == "STATIC_FUSION":
        families.append(("automl", "static", np.arange(automl)))
    elif arm == "TEMPORAL_FUSION":
        families.append(("automl", "sequence", np.arange(counts, counts + automl)))
    elif arm in {"SUMMARY", "SEQUENCE"}:
        families.append(("automl", "sequence", np.arange(automl)))
    return [(family, destination, name, columns) for family, destination, columns in families
            for name, columns in zip(("top", "middle", "bottom"), np.array_split(columns, 3))]


def run(data, prep, protocol, results, workflow, modeling, comparison):
    """Evaluate all registered seeds on one common validation subset.

    Run only after the upstream predictor lock. Caller persists this aggregate
    result with that lock. No TEST row or label is inspected here.
    """
    vi = np.asarray(data["indices"]["validation"])
    if vi.ndim != 1 or not len(vi):
        raise ValueError("Nonempty validation indices required")
    meta = data["metadata"].iloc[vi].reset_index(drop=True)
    if not meta.SPLIT.str.lower().eq("validation").all():
        raise ValueError("Reliance diagnostics are validation-only")
    latest = comparison.latest_patient_indices(meta.PATIENT_ID.to_numpy(), meta.END_DT.to_numpy())
    # Choose latest first; never substitute an earlier complete snapshot.
    complete = np.ones(len(vi), bool)
    for key in ("sequence_mask", "count_mask", "summary_mask"):
        if key in data:
            mask = np.asarray(data[key])[vi]
            if mask.dtype != bool or mask.ndim != 2:
                raise ValueError("Availability masks must be boolean N x T")
            complete &= mask.all(axis=1)
    selected = latest[complete[latest]]
    report = dict(schema_version=1, split="validation", prediction_unit="latest_snapshot_per_patient",
                  patient_subset="latest validation snapshot with all count/AutoML position-availability flags true",
                  validation_snapshots=len(vi), latest_patients=len(latest), patients=len(selected),
                  excluded_incomplete_latest=int(len(latest)-len(selected)),
                  permutation_seeds=list(PERMUTATION_SEEDS), rows=[], status="pending",
                  model_selection_performed=False, test_accessed=False,
                  limitations=["Post-lock descriptive diagnostic; do not use for feature or model selection in this run",
                               "Complete recorded-position subset may differ from the operational population",
                               "Position availability is not proof of enrollment or complete clinical observation",
                               "Within-date grouped shuffling can disrupt inter-feature dependencies",
                               "A small or negative lift/AP drop does not establish feature irrelevance",
                               "TRAIN proxy ranking is not expected to equal fitted Transformer reliance"])
    if not len(selected):
        return dict(report, status="skipped", reason="no_complete_latest_validation_snapshots")
    labels = np.asarray(data["y"])[vi][selected]
    if set(np.unique(labels)) != {0, 1}:
        return dict(report, status="skipped", reason="diagnostic_subset_requires_both_classes")
    dates = pd.to_datetime(meta.END_DT.iloc[selected]).to_numpy(dtype="datetime64[D]")
    strata = [np.flatnonzero(dates == date) for date in np.unique(dates)]
    shuffleable = [rows for rows in strata if len(rows) >= 2]
    report.update(date_strata=len(strata), shuffleable_strata=len(shuffleable),
                  shuffleable_patients=sum(len(rows) for rows in shuffleable))
    if not shuffleable:
        return dict(report, status="skipped", reason="no_date_stratum_has_two_patients")
    for arm, config in protocol["arms"].items():
        if config["family"] != "transformer":
            raise ValueError("This registered reliance diagnostic expects Transformer arms")
        runs = results[arm]
        if len(runs) != len(protocol["seeds"]) or [r["summary"]["seed"] for r in runs] != protocol["seeds"]:
            raise ValueError("All registered seeds required for reliance ensemble")
        x, valid, static = workflow.arm_arrays(data, prep, config, "validation")
        x, valid = np.asarray(x)[selected], np.asarray(valid)[selected]
        static = None if static is None else np.asarray(static)[selected]
        def ensemble(values, static_values):
            return np.mean([modeling.predict_encoder(fit, values, valid=valid, static=static_values)
                            for fit in runs], axis=0)
        primary_fraction = float(protocol.get("primary_lift_fraction", .10))
        base_scores = ensemble(x, static)
        baseline = comparison.binary_metrics(labels, base_scores)["average_precision"]
        baseline_lift = modeling.lift_at_fraction(labels, base_scores, primary_fraction)
        for family, destination, rank_group, columns in _groups(arm, prep):
            if len(columns) == 0:
                report["rows"].append(dict(arm=arm, feature_family=family, rank_group=rank_group,
                                           status="skipped_empty_rank_group", n_features=0, patients=len(selected)))
                continue
            for repeat, seed in enumerate(PERMUTATION_SEEDS, 1):
                rng = np.random.default_rng(seed)
                order = np.arange(len(selected))
                for rows in shuffleable:
                    order[rows] = rng.permutation(rows)
                changed = int(np.sum(order != np.arange(len(selected))))
                perturbed_x, perturbed_static = x, static
                if destination == "static":
                    if static is None:
                        raise ValueError("Static reliance group requires a static input branch")
                    perturbed_static = static.copy()
                    perturbed_static[:, columns] = static[order][:, columns]
                else:
                    perturbed_x = x.copy()
                    perturbed_x[:, :, columns] = x[order][:, :, columns]
                permuted_scores = ensemble(perturbed_x, perturbed_static)
                ap = comparison.binary_metrics(labels, permuted_scores)["average_precision"]
                lift = modeling.lift_at_fraction(labels, permuted_scores, primary_fraction)
                report["rows"].append(dict(arm=arm, feature_family=family, rank_group=rank_group,
                                           repeat=repeat, permutation_seed=seed, status="completed",
                                           n_features=len(columns), patients=len(selected),
                                           shuffleable_strata=len(shuffleable), shuffled_patients=changed,
                                           primary_lift_fraction=primary_fraction,
                                           baseline_lift=float(baseline_lift), permuted_lift=float(lift),
                                           lift_drop=float(baseline_lift-lift),
                                           baseline_ap=float(baseline), permuted_ap=float(ap),
                                           ap_drop=float(baseline-ap), direction="baseline_minus_permuted",
                                           interpretation="positive drop indicates predictive reliance under this perturbation"))
    report["status"] = "completed"
    return report
