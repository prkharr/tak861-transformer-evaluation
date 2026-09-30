"""Self-contained TRAIN-only feature selection; embedded verbatim in the notebook.

Input: one numeric feature row per distinct TRAIN patient, selected upstream by
latest snapshot date without using RESP. No held-out population is accepted here.
Output tables retain rejected columns and explicitly separate eligibility,
selection and method-specific scores. Fold frequencies describe stability, not
probabilities that a feature is causal or clinically important.
"""
import hashlib
import inspect
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SELECTION_DEFAULTS = {
    "seed": 42, "n_splits": 3, "min_nonzero_patients": 10,
    "min_observed_patients": 20, "max_missing_fraction": 0.95,
    "filter_max_features": 100, "redundancy_abs_spearman": 0.95,
    "wrapper_shortlist": 100, "wrapper_max_features": 50,
    "wrapper_rfe_step": 0.25, "embedded_max_features": 100,
    "elasticnet_C": 0.1, "elasticnet_l1_ratio": 0.5,
    "linear_max_iter": 2000, "tree_estimators": 128,
    "tree_max_depth": 6, "tree_min_samples_leaf": 15, "n_jobs": 2,
    "log1p_nonnegative": True,
}


def _selection_config(config):
    incoming = {} if config is None else dict(config)
    unknown = set(incoming) - set(SELECTION_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown selection settings: {sorted(unknown)}")
    cfg = {**SELECTION_DEFAULTS, **incoming}
    for key in ("n_splits", "min_nonzero_patients", "min_observed_patients",
                "filter_max_features", "wrapper_shortlist", "wrapper_max_features",
                "embedded_max_features", "linear_max_iter", "tree_estimators",
                "tree_max_depth", "tree_min_samples_leaf", "n_jobs"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["n_splits"] < 2:
        raise ValueError("At least two TRAIN folds are required.")
    for key in ("max_missing_fraction", "redundancy_abs_spearman",
                "wrapper_rfe_step", "elasticnet_l1_ratio"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1].")
    if cfg["elasticnet_C"] <= 0:
        raise ValueError("elasticnet_C must be positive.")
    if not isinstance(cfg["log1p_nonnegative"], bool):
        raise ValueError("log1p_nonnegative must be boolean.")
    return cfg


def _selection_inputs(X, y, cfg):
    if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
        raise ValueError("X must be a nonempty DataFrame with unique feature names.")
    if not all(isinstance(c, str) and c for c in X.columns):
        raise ValueError("Feature names must be nonempty strings.")
    forbidden = {"RESP", "PATIENT_ID", "END_DT", "SPLIT", "TIME_STEP"}
    if forbidden.intersection(c.upper() for c in X.columns):
        raise ValueError("Remove identifiers, dates, split and target from predictors.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Predictors must be numeric; do not silently encode identifiers.")
    values = X.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Infinite feature values are not supported.")
    labels = np.asarray(y)
    if labels.shape != (len(X),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain one binary label per feature row.")
    counts = np.bincount(labels.astype(int), minlength=2)
    if counts.min() < cfg["n_splits"]:
        raise ValueError("Each class must have at least n_splits distinct TRAIN patients.")
    return values, labels.astype(np.int64)


def _fit_quality(values, names, cfg):
    """All thresholds, duplicates, imputation and scaling use this fit partition."""
    observed = np.isfinite(values)
    nonzero = (observed & (values != 0)).sum(axis=0)
    n_observed = observed.sum(axis=0)
    missing = 1 - n_observed / len(values)
    audit = pd.DataFrame({"FEATURE_NAME": names,
        "OBSERVED_TRAIN_PATIENTS": n_observed,
        "NONZERO_TRAIN_PATIENTS": nonzero,
        "MISSING_FRACTION": missing, "ELIGIBLE": False,
        "QUALITY_REASON": "eligible", "DUPLICATE_OF": ""})
    kept = []
    hashes = {}
    for j, name in enumerate(names):
        present = values[observed[:, j], j]
        if not len(present):
            reason = "all_missing"
        elif n_observed[j] < cfg["min_observed_patients"]:
            reason = "insufficient_observations"
        elif missing[j] > cfg["max_missing_fraction"]:
            reason = "high_missingness"
        elif np.ptp(present) == 0:
            reason = "constant_observed_values"
        elif nonzero[j] < cfg["min_nonzero_patients"]:
            reason = "insufficient_nonzero_support"
        else:
            # Canonicalize -0 and NaN before hashing; compare to avoid collisions.
            canonical = values[:, j].copy()
            canonical[canonical == 0] = 0.0
            canonical[np.isnan(canonical)] = np.nan
            digest = hashlib.sha256(canonical.tobytes()).hexdigest()
            duplicate = next((k for k in hashes.get(digest, [])
                if np.array_equal(values[:, j], values[:, k], equal_nan=True)), None)
            if duplicate is not None:
                reason = "exact_duplicate"
                audit.loc[j, "DUPLICATE_OF"] = names[duplicate]
            else:
                hashes.setdefault(digest, []).append(j)
                kept.append(j)
                audit.loc[j, "ELIGIBLE"] = True
                reason = "eligible"
        audit.loc[j, "QUALITY_REASON"] = reason
    if not kept:
        return audit, np.array([], dtype=int), None
    indices = np.asarray(kept, dtype=int)
    raw = values[:, indices]
    medians = np.nanmedian(raw, axis=0)
    imputed = np.where(np.isnan(raw), medians, raw)
    nonnegative = np.nanmin(raw, axis=0) >= 0
    discrete = np.array([np.allclose(col[np.isfinite(col)],
        np.rint(col[np.isfinite(col)]), rtol=0, atol=1e-9) for col in raw.T])
    transformed = imputed.copy()
    log_columns = nonnegative & cfg["log1p_nonnegative"]
    transformed[:, log_columns] = np.log1p(transformed[:, log_columns])
    scaler = StandardScaler().fit(transformed)
    state = {"medians": medians, "log_columns": log_columns, "scaler": scaler,
             "raw": imputed, "Z": scaler.transform(transformed), "discrete": discrete}
    return audit, indices, state


def _quality_transform(values, indices, state):
    data = values[:, indices]
    data = np.where(np.isnan(data), state["medians"], data).copy()
    # A negative holdout value cannot be silently passed through a fitted log rule.
    if np.any(data[:, state["log_columns"]] < 0):
        raise ValueError("Negative value conflicts with a TRAIN-fitted nonnegative feature contract.")
    data[:, state["log_columns"]] = np.log1p(data[:, state["log_columns"]])
    return state["scaler"].transform(data)


def _mi_ranking(raw, y, discrete, names, cfg):
    mi = mutual_info_classif(raw, y, discrete_features=discrete,
                            random_state=cfg["seed"])
    mi = np.nan_to_num(mi, nan=0.0)
    order = sorted(range(len(names)), key=lambda j: (-mi[j], names[j]))
    return mi, np.asarray(order, dtype=int)


def _linear(cfg, penalty="l2"):
    # sklearn >=1.8 expresses penalties through l1_ratio; earlier versions
    # require an explicit penalty argument. Keep the delivered notebook portable.
    penalty_parameter = inspect.signature(LogisticRegression).parameters.get("penalty")
    modern_penalty = penalty_parameter is None or penalty_parameter.default == "deprecated"
    if penalty == "elasticnet":
        arguments = {} if modern_penalty else {"penalty": "elasticnet"}
        return LogisticRegression(**arguments, solver="saga",
            C=cfg["elasticnet_C"], l1_ratio=cfg["elasticnet_l1_ratio"],
            max_iter=cfg["linear_max_iter"], class_weight="balanced",
            random_state=cfg["seed"])
    arguments = {"l1_ratio": 0.0} if modern_penalty else {"penalty": "l2"}
    return LogisticRegression(**arguments, solver="lbfgs", C=1.0,
        max_iter=cfg["linear_max_iter"], class_weight="balanced",
        random_state=cfg["seed"])


def _fit_capture(model, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def _blank_selection(audit, strategy):
    table = audit.copy()
    table["STRATEGY"] = strategy
    table["SELECTED"] = False
    table["RANK"] = np.nan
    table["SELECTION_REASON"] = table["QUALITY_REASON"]
    table.loc[table.ELIGIBLE, "SELECTION_REASON"] = "below_selection_cutoff"
    return table


def _fit_strategies(values, y, names, cfg):
    audit, indices, state = _fit_quality(values, names, cfg)
    observed = np.isfinite(values)
    nonzero = observed & (values != 0)
    for label, label_name in ((1, "POSITIVE"), (0, "NEGATIVE")):
        audit[f"NONZERO_{label_name}_TRAIN_PATIENTS"] = nonzero[y == label].sum(axis=0)
        audit[f"MISSING_{label_name}_TRAIN_PATIENTS"] = (~observed[y == label]).sum(axis=0)
    tables = {s: _blank_selection(audit, s) for s in ("filter", "wrapper", "embedded")}
    status = {s: True for s in tables}
    if not len(indices):
        return tables, indices, state, status
    kept_names = [names[i] for i in indices]
    raw, Z = state["raw"], state["Z"]
    mi, order = _mi_ranking(raw, y, state["discrete"], kept_names, cfg)

    # Filter: MI ranks individual associations; Spearman removes near-duplicates.
    table = tables["filter"]
    for col in ("MUTUAL_INFORMATION", "SPEARMAN_RESP", "PHI_PRESENCE_RESP",
                "CHI2_PRESENCE", "CHI2_P_EXPLORATORY", "CHI2_MIN_EXPECTED_CELL"):
        table[col] = np.nan
    table["REDUNDANT_WITH"] = ""
    table["MI_TREATS_AS_DISCRETE"] = False
    table.loc[indices, "MUTUAL_INFORMATION"] = mi
    table.loc[indices, "MI_TREATS_AS_DISCRETE"] = state["discrete"]
    correlation = pd.DataFrame(raw).corr(method="spearman").to_numpy()
    selected = []
    for rank, j in enumerate(order, 1):
        i = indices[j]
        table.loc[i, "RANK"] = rank
        observed_rows = observed[:, i]
        observed_values, observed_labels = values[observed_rows, i], y[observed_rows]
        # Association diagnostics use observed pairs, never median-filled labels
        # of absence. MI/selection separately uses the fitted numeric imputation.
        if len(np.unique(observed_labels)) == 2:
            table.loc[i, "SPEARMAN_RESP"] = float(spearmanr(observed_values, observed_labels).statistic)
        presence = observed_values != 0
        if len(np.unique(presence)) == 2 and len(np.unique(observed_labels)) == 2:
            table.loc[i, "PHI_PRESENCE_RESP"] = np.corrcoef(presence.astype(float), observed_labels)[0, 1]
            contingency = np.array([[np.sum((presence == a) & (observed_labels == b))
                                      for b in (0, 1)] for a in (False, True)])
            chi, p, _, expected_cells = chi2_contingency(contingency, correction=False)
            table.loc[i, "CHI2_PRESENCE"] = chi
            table.loc[i, "CHI2_P_EXPLORATORY"] = p
            table.loc[i, "CHI2_MIN_EXPECTED_CELL"] = expected_cells.min()
        redundant = next((k for k in selected
            if abs(correlation[j, k]) >= cfg["redundancy_abs_spearman"]), None)
        if redundant is not None:
            table.loc[i, "SELECTION_REASON"] = "spearman_redundancy"
            table.loc[i, "REDUNDANT_WITH"] = kept_names[redundant]
        elif mi[j] <= 0:
            table.loc[i, "SELECTION_REASON"] = "zero_estimated_mutual_information"
        elif len(selected) < cfg["filter_max_features"]:
            selected.append(j)
            table.loc[i, ["SELECTED", "SELECTION_REASON"]] = [True, "selected"]

    # Wrapper: bounded RFE. It is conditional on an inner-fit MI shortlist.
    table = tables["wrapper"]
    table["SHORTLIST_MI"] = np.nan
    table["RFE_RANK"] = np.nan
    table["FINAL_ABS_COEFFICIENT"] = np.nan
    table.loc[indices, "SELECTION_REASON"] = "outside_mi_shortlist"
    shortlist = order[:cfg["wrapper_shortlist"]]
    shortlist_indices = indices[shortlist]
    table.loc[shortlist_indices, "SHORTLIST_MI"] = mi[shortlist]
    n_select = min(cfg["wrapper_max_features"], len(shortlist))
    if len(shortlist) == 1:
        estimator, converged = _fit_capture(_linear(cfg), Z[:, shortlist], y)
        rfe_rank = np.ones(1, dtype=int)
        support = np.ones(1, dtype=bool)
    else:
        rfe, converged = _fit_capture(RFE(_linear(cfg), n_features_to_select=n_select,
                step=cfg["wrapper_rfe_step"]), Z[:, shortlist], y)
        estimator, rfe_rank, support = rfe.estimator_, rfe.ranking_, rfe.support_
    status["wrapper"] = converged
    table.loc[shortlist_indices, "RFE_RANK"] = rfe_rank
    table.loc[shortlist_indices, "SELECTION_REASON"] = "eliminated_by_rfe"
    selected_indices = shortlist_indices[support]
    table.loc[selected_indices, "SELECTED"] = True
    table.loc[selected_indices, "SELECTION_REASON"] = "selected"
    table.loc[selected_indices, "FINAL_ABS_COEFFICIENT"] = np.abs(estimator.coef_[0])
    wrapper_order = sorted(shortlist_indices, key=lambda i: (
        table.loc[i, "RFE_RANK"],
        -np.nan_to_num(table.loc[i, "FINAL_ABS_COEFFICIENT"]), names[i]))
    table.loc[wrapper_order, "RANK"] = np.arange(1, len(wrapper_order) + 1)

    # Embedded: explicit equally weighted rank aggregation of two fitted models.
    table = tables["embedded"]
    elastic, converged = _fit_capture(_linear(cfg, "elasticnet"), Z, y)
    status["embedded"] = converged
    trees = ExtraTreesClassifier(n_estimators=cfg["tree_estimators"],
        max_depth=cfg["tree_max_depth"], min_samples_leaf=cfg["tree_min_samples_leaf"],
        max_features="sqrt", class_weight="balanced", n_jobs=cfg["n_jobs"],
        random_state=cfg["seed"])
    trees.fit(Z, y)
    coefficient = elastic.coef_[0]
    tree_importance = trees.feature_importances_
    coefficient_rank = pd.Series(np.abs(coefficient)).rank(ascending=False, method="average").to_numpy()
    tree_rank = pd.Series(tree_importance).rank(ascending=False, method="average").to_numpy()
    mean_rank = (coefficient_rank + tree_rank) / 2
    for col in ("ELASTICNET_COEFFICIENT", "ELASTICNET_ABS_COEFFICIENT",
                "TREE_IMPURITY_IMPORTANCE", "ELASTICNET_RANK", "TREE_RANK", "MEAN_COMPONENT_RANK"):
        table[col] = np.nan
    table["ELASTICNET_NONZERO"] = False
    table.loc[indices, "ELASTICNET_COEFFICIENT"] = coefficient
    table.loc[indices, "ELASTICNET_ABS_COEFFICIENT"] = np.abs(coefficient)
    table.loc[indices, "ELASTICNET_NONZERO"] = np.abs(coefficient) > 1e-8
    table.loc[indices, "TREE_IMPURITY_IMPORTANCE"] = tree_importance
    table.loc[indices, "ELASTICNET_RANK"] = coefficient_rank
    table.loc[indices, "TREE_RANK"] = tree_rank
    table.loc[indices, "MEAN_COMPONENT_RANK"] = mean_rank
    embedded_order = sorted(range(len(indices)), key=lambda j: (mean_rank[j], kept_names[j]))
    n_selected = 0
    for rank, j in enumerate(embedded_order, 1):
        i = indices[j]
        table.loc[i, "RANK"] = rank
        if abs(coefficient[j]) <= 1e-8 and tree_importance[j] <= 0:
            table.loc[i, "SELECTION_REASON"] = "zero_importance_in_both_models"
        elif n_selected < cfg["embedded_max_features"]:
            table.loc[i, ["SELECTED", "SELECTION_REASON"]] = [True, "selected"]
            n_selected += 1
    return tables, indices, state, status


def _screening_metrics(y, probability):
    prevalence = float(np.mean(y))
    # Fractional inclusion of an entire tied boundary group avoids arbitrary
    # inflated/deflated top-decile results for constant or heavily tied scores.
    target = max(1, int(np.ceil(0.1 * len(y))))
    boundary = np.sort(probability)[-target]
    above, tied = probability > boundary, probability == boundary
    fraction = (target - above.sum()) / tied.sum()
    top_precision = float((y[above].sum() + fraction * y[tied].sum()) / target)
    return {"AVERAGE_PRECISION": float(average_precision_score(y, probability)),
            "POSITIVE_PREVALENCE": prevalence, "TOP10_PRECISION": top_precision,
            "TOP10_LIFT": top_precision / prevalence}


def select_all(X, y, config=None):
    """Return three feature tables, outer TRAIN-fold diagnostics, and settings.

    The upstream loader must enforce one latest snapshot per TRAIN patient.
    TRAIN cross-validation re-fits *all* preprocessing and selectors separately
    inside every fold. These fixed designs are evaluated with the same L2
    logistic classifier; results are screening diagnostics, not Transformer
    performance or an unbiased estimate after tuning against these results.
    Final lists are then refit on all supplied TRAIN patients. No test is read.
    """
    cfg = _selection_config(config)
    values, labels = _selection_inputs(X, y, cfg)
    names = list(X.columns)
    strategies = ("filter", "wrapper", "embedded")
    selected_counts = {s: np.zeros(len(names), dtype=int) for s in strategies}
    eligible_counts = np.zeros(len(names), dtype=int)
    diagnostics = []
    cv = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    for fold, (fit_rows, score_rows) in enumerate(cv.split(values, labels), 1):
        tables, indices, state, convergence = _fit_strategies(
            values[fit_rows], labels[fit_rows], names, cfg)
        eligible_counts[indices] += 1
        fit_Z = state["Z"] if state is not None else None
        score_Z = _quality_transform(values[score_rows], indices, state) if state is not None else None
        for strategy in (*strategies, "all_eligible"):
            if strategy == "all_eligible":
                mask = np.zeros(len(names), dtype=bool)
                mask[indices] = True
            else:
                mask = tables[strategy].SELECTED.to_numpy(dtype=bool)
                selected_counts[strategy] += mask
            columns = np.flatnonzero(mask[indices])
            scoring_converged = True
            if len(columns):
                model, scoring_converged = _fit_capture(_linear(cfg), fit_Z[:, columns], labels[fit_rows])
                probability = model.predict_proba(score_Z[:, columns])[:, 1]
            else:
                probability = np.repeat(labels[fit_rows].mean(), len(score_rows))
            diagnostics.append({"STRATEGY": strategy, "FOLD": fold,
                "FIT_PATIENTS": len(fit_rows), "SCORE_PATIENTS": len(score_rows),
                "SELECTED_FEATURES": int(mask.sum()),
                "SELECTOR_CONVERGED": convergence.get(strategy, True),
                "SCORER_CONVERGED": scoring_converged,
                "EVALUATION": "TRAIN_outer_fold_fixed_L2_logistic",
                **_screening_metrics(labels[score_rows], probability)})
    final_tables, _, _, final_convergence = _fit_strategies(values, labels, names, cfg)
    for strategy in strategies:
        table = final_tables[strategy]
        table["FOLD_SELECTION_COUNT"] = selected_counts[strategy]
        table["FOLD_SELECTION_FREQUENCY"] = selected_counts[strategy] / cfg["n_splits"]
        table["FOLD_ELIGIBILITY_FREQUENCY"] = eligible_counts / cfg["n_splits"]
        table["FINAL_SELECTOR_CONVERGED"] = final_convergence[strategy]
        table["FIT_POPULATION"] = "latest_snapshot_per_TRAIN_patient"
        final_tables[strategy] = table.sort_values(["SELECTED", "RANK", "FEATURE_NAME"],
            ascending=[False, True, True], na_position="last").reset_index(drop=True)
    return {**final_tables, "diagnostics": pd.DataFrame(diagnostics), "config": cfg}
