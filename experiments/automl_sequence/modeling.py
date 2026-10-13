"""TRAIN-only preparation and bounded fresh fits for the two AutoML experiments.

Summary uses one token; history uses twelve genuinely historical tokens. Zero is
a measurement, never a padding rule. The caller must supply date-validated masks,
patient-disjoint splits and a frozen source feature order. This module never
accepts TEST data while fitting. No historical fitted state is reused.
"""
import copy
import hashlib
import io
import json
import os
import platform
import importlib.metadata
import random
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def lift_at_fraction(y, scores, fraction=.10):
    """Snapshot lift at exact ceil(fraction*N), with label-blind row-order ties.

    Scores may be arbitrary finite ranking values. This is a snapshot-capacity
    estimand, not unique-patient interventions or a patient-weighted loss.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    if (y.ndim != 1 or len(y) == 0 or scores.shape != y.shape
            or not set(np.unique(y)) <= {0, 1} or y.sum() == 0
            or not np.isfinite(scores).all() or not np.isfinite(fraction)
            or not 0 < fraction <= 1):
        raise ValueError("Lift requires aligned binary labels with positives, finite scores and fraction in (0,1]")
    k = max(1, int(np.ceil(float(fraction) * len(y))))
    order = np.lexsort((np.arange(len(y)), -scores.astype(float)))
    return float((y[order[:k]].sum() / k) / y.mean())


def _array(x, valid=None, *, finite=False, allow_empty=False):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3 or min(x.shape) < 1 or not np.issubdtype(x.dtype, np.number):
        raise ValueError("Expected numeric N x T x F inputs")
    if valid is None:
        valid = np.ones(x.shape[:2], dtype=bool)
    valid = np.asarray(valid)
    if valid.dtype != bool or valid.shape != x.shape[:2]:
        raise ValueError("Availability mask must be boolean N x T; True means available")
    if not allow_empty and not valid.any(axis=1).all():
        raise ValueError("A snapshot has no available timestep; resolve or explicitly exclude it upstream")
    observed = x[valid]
    if np.iscomplexobj(x) or np.isinf(observed).any() or (finite and not np.isfinite(observed).all()):
        raise ValueError("Observed inputs must be real and finite (NaN allowed only before imputation)")
    return x, valid


def sample_weights(groups, mode="patient"):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0 or any(x is None for x in groups):
        raise ValueError("Expected nonempty patient IDs")
    if mode == "snapshot":
        return np.ones(len(groups), dtype=np.float32)
    if mode != "patient":
        raise ValueError("weighting must be patient or snapshot")
    _, inverse, counts = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    w = 1. / counts[inverse]
    return (w / w.mean()).astype(np.float32)


def class_weight(y, weights, balance):
    y, weights = np.asarray(y), np.asarray(weights)
    if (set(np.unique(y)) != {0, 1} or y.shape != weights.shape
            or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Class weighting requires both binary classes and positive aligned weights")
    ratio = float(weights[y == 0].sum() / weights[y == 1].sum())
    if balance not in {"none", "sqrt", "full"}:
        raise ValueError("balance must be none, sqrt or full")
    return {"none": 1., "sqrt": ratio ** .5, "full": ratio}[balance]


def _kinds(feature_names, feature_kinds):
    if feature_kinds is None:
        return ["numeric"] * len(feature_names)
    kinds = ([feature_kinds.get(name, "numeric") for name in feature_names]
             if isinstance(feature_kinds, dict) else list(feature_kinds))
    if len(kinds) != len(feature_names) or not set(kinds) <= {"numeric", "continuous", "count", "binary"}:
        raise ValueError("Kinds must be numeric/continuous/count/binary; categorical values need an explicit encoding contract")
    return kinds


def _fit_transform(summary, sequence, valid, feature_names, feature_kinds):
    """Fit one common transformation from TRAIN observations of both arms."""
    values = np.concatenate([summary[:, 0, :], sequence[valid]], axis=0)
    names, kinds = list(feature_names), _kinds(feature_names, feature_kinds)
    quality, medians, centers, scales, actual_kinds, eligible, binary_contracts = [], [], [], [], [], [], []
    seen = {}
    for j, name in enumerate(names):
        a = np.asarray(values[:, j], dtype=np.float64)
        observed = a[np.isfinite(a)]
        unique, freq = np.unique(observed, return_counts=True)
        kind = kinds[j]
        binary_contract = kind == "binary"
        if kind == "numeric":
            kind = "binary" if len(unique) and set(unique) <= {0., 1.} else "continuous"
        if kind == "binary" and not set(unique) <= {0., 1.}:
            raise ValueError("Non-binary measurements in declared binary feature: " + name)
        if kind == "count" and (np.any(observed < 0) or np.any(observed != np.floor(observed))):
            raise ValueError("Noninteger or negative value in declared count feature: " + name)
        transformed = np.log1p(a) if kind == "count" else a
        median = float(np.nanmedian(transformed)) if len(observed) else 0.
        imputed = np.where(np.isnan(transformed), median, transformed)
        center = 0. if kind == "binary" else float(imputed.mean())
        sd = float(imputed.std())
        # Preserve rare binary values; never percentile-clip sparse indicators.
        scale = 1. if kind == "binary" else (max(sd, 1.) if kind == "count" else max(sd, 1e-6))
        reason = "all_missing" if not len(observed) else ("constant" if len(unique) == 1 else "retained")
        duplicate_of = None
        canonical = np.nan_to_num(a, nan=0.)
        digest = hashlib.sha256(canonical.tobytes() + np.isnan(a).tobytes()).hexdigest()
        if reason == "retained" and digest in seen:
            reason, duplicate_of = "exact_duplicate", names[seen[digest]]
        elif reason == "retained":
            seen[digest] = j
            eligible.append(j)
        quality.append(dict(feature=name, source_index=j, kind=kind, reason=reason,
                            duplicate_of=duplicate_of, missing_fraction=float(np.isnan(a).mean()),
                            observed_values=int(len(observed)), unique_values=int(len(unique)),
                            dominant_fraction=float(freq.max() / freq.sum()) if len(freq) else None,
                            near_constant=bool(len(freq) and freq.max() / freq.sum() >= .995),
                            variance=float(np.var(observed)) if len(observed) else None,
                            low_variance_flag=bool(len(observed) and np.var(observed) < 1e-6),
                            min=float(observed.min()) if len(observed) else None,
                            max=float(observed.max()) if len(observed) else None,
                            p01=float(np.quantile(observed, .01)) if len(observed) else None,
                            p99=float(np.quantile(observed, .99)) if len(observed) else None))
        medians.append(median); centers.append(center); scales.append(scale); actual_kinds.append(kind)
        binary_contracts.append(binary_contract)
    if not eligible:
        raise ValueError("No nonconstant nonduplicate feature survives TRAIN quality checks")
    return dict(feature_names=names, kinds=actual_kinds, median=medians, center=centers,
                scale=scales, eligible_indices=eligible, quality=quality, binary_contract=binary_contracts,
                clipping="none", fitted_on="TRAIN summary and valid historical rows only",
                near_constant_policy="flag only; preserve rare clinical signals")


def _transform_all(x, state, valid=None):
    x, valid = _array(x, valid, allow_empty=True)
    if x.shape[-1] != len(state["feature_names"]):
        raise ValueError("Raw input width differs from fitted feature order")
    # One float64 feature at a time avoids a full N x T x F float64 copy.
    # Output remains identical float32 data in the frozen source-column order.
    a = np.empty(x.shape, dtype=np.float32)
    unavailable = ~valid
    for j, kind in enumerate(state["kinds"]):
        column = np.asarray(x[:, :, j], dtype=np.float64).copy()
        # Masked payload is unavailable, not a measurement; clear first.
        column[unavailable] = 0.
        if kind == "count":
            vals = column[valid]
            finite = vals[np.isfinite(vals)]
            if (finite < 0).any() or (finite != np.floor(finite)).any():
                raise ValueError("New data violates declared count semantics")
            np.log1p(column, out=column)
        elif kind == "binary" and state.get("binary_contract", [False] * len(state["kinds"]))[j]:
            vals = column[valid]
            if not set(np.unique(vals[np.isfinite(vals)])) <= {0., 1.}:
                raise ValueError("New data violates binary semantics")
        column[np.isnan(column)] = state["median"][j]
        column -= state["center"][j]
        column /= state["scale"][j]
        column[unavailable] = 0.
        if not np.isfinite(column).all():
            raise ValueError("Nonfinite transformed values")
        a[:, :, j] = column
        if not np.isfinite(a[:, :, j]).all():
            raise ValueError("Transformed values overflow float32")
    return a


def transform_inputs(x, preparation, valid=None):
    """Transform without fitting; output feature order is selected_names."""
    a = _transform_all(x, preparation["transform"], valid)
    return a[:, :, preparation["selected_indices"]]


def _proxy(summary_transformed, sequence_transformed, valid):
    """Four descriptors per original feature; temporal-only signal can rank."""
    mask = valid[:, :, None]
    count = np.maximum(valid.sum(1)[:, None], 1)
    t = np.arange(sequence_transformed.shape[1], dtype=float)[None, :, None]
    tmean = np.where(mask, t, 0.).sum(1) / count
    centered_t = t - tmean[:, None, :]
    denom = np.where(mask, centered_t**2, 0.).sum(1)
    output = np.empty((len(sequence_transformed), sequence_transformed.shape[-1], 4), dtype=np.float64)
    # Feature blocking bounds temporary tensors without changing reductions
    # along the temporal axis, arithmetic precision or descriptor ordering.
    for start in range(0, sequence_transformed.shape[-1], 64):
        end = min(start+64, sequence_transformed.shape[-1])
        x = sequence_transformed[:, :, start:end].astype(np.float64)
        mean = np.where(mask, x, 0.).sum(1) / count
        variance = np.where(mask, (x-mean[:, None, :])**2, 0.).sum(1) / count
        slope = np.where(mask, centered_t * (x - mean[:, None, :]), 0.).sum(1) / np.maximum(denom, 1.)
        output[:, start:end, 0] = summary_transformed[:, 0, start:end]
        output[:, start:end, 1] = mean
        output[:, start:end, 2] = np.sqrt(np.maximum(variance, 0))
        output[:, start:end, 3] = slope
    return output


def _rank(proxy, y, indices, names, weights, fraction=.10):
    scores = []
    for j in indices:
        descriptors, lifts = [], []
        for d in range(proxy.shape[-1]):
            a = proxy[:, j, d]
            if len(np.unique(a)) <= 1:
                ap_pos = ap_neg = float(np.average(y, weights=weights))
                lifts.append(1.)
            else:
                ap_pos = float(average_precision_score(y, a, sample_weight=weights))
                ap_neg = float(average_precision_score(y, -a, sample_weight=weights))
                lifts.append(max(lift_at_fraction(y, a, fraction), lift_at_fraction(y, -a, fraction)))
            descriptors.append(max(ap_pos, ap_neg))
        scores.append(dict(index=int(j), feature=names[j], rank_score=max(lifts),
                           descriptor_lift=dict(zip(["summary", "historical_mean", "historical_sd", "historical_slope"], lifts)),
                           descriptor_ap=dict(zip(["summary", "historical_mean", "historical_sd", "historical_slope"], descriptors))))
    return sorted(scores, key=lambda r: (-r["rank_score"], r["index"]))


def prepare_features(summary_train, sequence_train, y_train, patient_ids_train,
                     feature_names, valid_train=None, feature_kinds=None, config=None):
    """Common feature subset, selected strictly inside TRAIN by grouped proxy CV.

    The proxy uses summary, historical mean, standard deviation and time slope.
    Groups of four descriptors rank/select original features, preserving features
    that vary only historically. It does not claim a Transformer-optimal subset.
    Fold-local preprocessing and ranking prevent fitting-fold target leakage.
    """
    cfg = dict(n_splits=3, seed=42, candidate_sizes=[16, 32, 64, 128, 256, 512, 1028],
               proxy_C=.1, proxy_max_iter=1000, weighting="patient", primary_lift_fraction=.10)
    incoming = config or {}
    if set(incoming) - set(cfg):
        raise ValueError("Unknown preparation configuration: " + str(set(incoming) - set(cfg)))
    cfg.update(incoming)
    if not np.isfinite(cfg["primary_lift_fraction"]) or not 0 < cfg["primary_lift_fraction"] <= 1:
        raise ValueError("primary_lift_fraction must be in (0,1]")
    sequence, valid = _array(sequence_train, valid_train, allow_empty=True)
    summary, smask = _array(summary_train, valid[:, :1], allow_empty=True)
    y, groups = np.asarray(y_train), np.asarray(patient_ids_train)
    names = list(feature_names)
    if (summary.shape[1] != 1 or sequence.shape[1] != 12 or summary.shape[0] != sequence.shape[0]
            or summary.shape[-1] != sequence.shape[-1] or len(names) != summary.shape[-1]
            or len(set(names)) != len(names) or y.shape != (len(summary),) or groups.shape != y.shape
            or set(np.unique(y)) != {0, 1}):
        raise ValueError("Aligned summary N x 1 x F, monthly N x 12 x F, labels and patient IDs required")
    if any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("Feature names must be nonempty unique strings")
    if {"PATIENT_ID", "END_DT", "RESP", "SPLIT", "LABEL", "TARGET"} & {n.upper() for n in names}:
        raise ValueError("Identifier or target is present in feature names")
    if (type(cfg["n_splits"]) is not int or cfg["n_splits"] < 3
            or any(type(k) is not int or k < 1 for k in cfg["candidate_sizes"])):
        raise ValueError("At least three grouped folds and positive integer K candidates required")
    started = time.perf_counter()
    state = _fit_transform(summary, sequence, valid, names, feature_kinds)
    eligible = state["eligible_indices"]
    candidates = sorted({min(k, len(eligible)) for k in cfg["candidate_sizes"]} | {len(eligible)})
    weights = sample_weights(groups, cfg["weighting"])
    full_summary = _transform_all(summary, state, smask)
    full_proxy = _proxy(full_summary, _transform_all(sequence, state, valid), valid)
    ranking = _rank(full_proxy, y, eligible, names, weights, cfg["primary_lift_fraction"])
    splitter = StratifiedGroupKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    cv_rows, fold_rankings = [], []
    for fold, (ftrain, fval) in enumerate(splitter.split(summary, y, groups)):
        if set(groups[ftrain]) & set(groups[fval]) or any(set(np.unique(y[idx])) != {0, 1} for idx in (ftrain, fval)):
            raise ValueError("Grouped CV requires patient separation and both classes in every fold")
        fold_state = _fit_transform(summary[ftrain], sequence[ftrain], valid[ftrain], names, feature_kinds)
        a = _proxy(_transform_all(summary[ftrain], fold_state, smask[ftrain]),
                   _transform_all(sequence[ftrain], fold_state, valid[ftrain]), valid[ftrain])
        b = _proxy(_transform_all(summary[fval], fold_state, smask[fval]),
                   _transform_all(sequence[fval], fold_state, valid[fval]), valid[fval])
        w = sample_weights(groups[ftrain], cfg["weighting"])
        ranks = _rank(a, y[ftrain], fold_state["eligible_indices"], names, w, cfg["primary_lift_fraction"])
        order = [r["index"] for r in ranks]
        fold_rankings.append(order)
        for k in candidates:
            cols = order[:min(k, len(order))]
            clf = LogisticRegression(C=cfg["proxy_C"], solver="lbfgs", max_iter=cfg["proxy_max_iter"],
                                     random_state=cfg["seed"], tol=1e-4)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clf.fit(a[:, cols, :].reshape(len(a), -1), y[ftrain], sample_weight=w)
            score = clf.predict_proba(b[:, cols, :].reshape(len(b), -1))[:, 1]
            cv_rows.append(dict(fold=fold, candidate_k=k, fitted_k=len(cols),
                                lift=lift_at_fraction(y[fval], score, cfg["primary_lift_fraction"]),
                                average_precision=float(average_precision_score(y[fval], score)),
                                warnings=[str(w.message) for w in caught]))
    cv_summary = []
    for k in candidates:
        ap = np.asarray([r["average_precision"] for r in cv_rows if r["candidate_k"] == k])
        lifts = np.asarray([r["lift"] for r in cv_rows if r["candidate_k"] == k])
        cv_summary.append(dict(k=k, mean_ap=float(ap.mean()), se_ap=float(ap.std(ddof=1) / np.sqrt(len(ap))),
                               mean_lift=float(lifts.mean()), se_lift=float(lifts.std(ddof=1) / np.sqrt(len(lifts)))))
    best = max(cv_summary, key=lambda r: (r["mean_lift"], -r["k"]))
    boundary = best["mean_lift"] - best["se_lift"]
    selected_k = min(r["k"] for r in cv_summary if r["mean_lift"] >= boundary)
    selected_indices = [r["index"] for r in ranking[:selected_k]]
    for rank, row in enumerate(ranking, 1):
        ranks = [order.index(row["index"]) + 1 for order in fold_rankings if row["index"] in order]
        row.update(rank=rank, selected=row["index"] in selected_indices,
                   fold_rank_mean=float(np.mean(ranks)) if ranks else None,
                   fold_rank_sd=float(np.std(ranks)) if ranks else None,
                   fold_eligible_count=len(ranks))
    selected = set(selected_indices)
    for row in state["quality"]:
        row["selected"] = row["source_index"] in selected
        row["final_reason"] = ("proxy_rank_below_selected_prefix" if row["reason"] == "retained"
                               and not row["selected"] else row["reason"])
    # Correlation is diagnostic only: deleting correlated rare codes is not automatic.
    matrix = full_summary[:, 0, eligible].astype(np.float64)
    # float32 reductions can report nonzero SD for an exactly constant column.
    active = np.flatnonzero(np.ptp(matrix, axis=0) > 0)
    correlations = []
    condition = None
    if len(active) > 1:
        corr = np.corrcoef(matrix[:, active], rowvar=False)
        ii, jj = np.where(np.triu(np.abs(corr) >= .95, k=1))
        correlations = [dict(feature_a=names[eligible[active[i]]], feature_b=names[eligible[active[j]]],
                             pearson=float(corr[i, j])) for i, j in zip(ii, jj)]
        eigenvalues = np.linalg.eigvalsh(corr)
        condition = (None if eigenvalues[0] <= 1e-12 else float(eigenvalues[-1] / eigenvalues[0]))
    result = dict(schema_version=1, transform=state, selected_indices=selected_indices,
                  selected_names=[names[j] for j in selected_indices], quality=state["quality"],
                  ranking=ranking, cv_rows=cv_rows, cv_summary=cv_summary, config=cfg,
                  selection_rule="smallest K within one SE of best TRAIN-grouped-CV proxy lift at primary capacity",
                  primary_lift_fraction=cfg["primary_lift_fraction"],
                  rank_metric="max bidirectional snapshot lift across TRAIN summary/mean/SD/slope; AP secondary diagnostic",
                  lift_tie_policy="frozen input row order; no label-based tie breaking",
                  proxy_limitation="Grouped summary/mean/SD/slope proxy; not a claim of Transformer-optimal K",
                  selection_used_test=False, selection_used_external_validation=False,
                  train_snapshots=len(y), train_patients=len(np.unique(groups)),
                  correlation_pairs_ge_095=correlations,
                  correlation_condition_number=condition,
                  singular_correlation=bool(condition is None and len(active) > 1),
                  preprocessing_note="No clipping; binary preserved; log1p only declared counts; other numeric standardized",
                  wall_seconds=time.perf_counter() - started)
    result["preparation_sha256"] = stable_hash(result)
    return result


DEFAULT_ENCODER = dict(d_model=64, heads=4, ff=128, depth=1, dropout=.2, pooling="mean",
                       weight_decay=1e-3, learning_rate=3e-4, batch_size=256,
                       epochs=30, patience=5, gradient_clip=1., scheduler="plateau",
                       weighting="patient", balance="sqrt", threads=8, device="cpu",
                       optimizer_betas=[.9, .999], optimizer_eps=1e-8, primary_lift_fraction=.10)


def _config(config):
    incoming = config or {}
    if set(incoming) - set(DEFAULT_ENCODER):
        raise ValueError("Unknown encoder settings: " + str(set(incoming) - set(DEFAULT_ENCODER)))
    cfg = {**copy.deepcopy(DEFAULT_ENCODER), **copy.deepcopy(incoming)}
    if not np.isfinite(cfg["primary_lift_fraction"]) or not 0 < cfg["primary_lift_fraction"] <= 1:
        raise ValueError("primary_lift_fraction must be in (0,1]")
    for name in ("d_model", "heads", "ff", "depth", "batch_size", "epochs", "patience", "threads"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if cfg["d_model"] < 2 or cfg["d_model"] % cfg["heads"] or not 0 <= cfg["dropout"] < 1:
        raise ValueError("Invalid encoder dimension, heads or dropout")
    if cfg["pooling"] not in {"mean", "attention"} or cfg["scheduler"] not in {"none", "plateau"}:
        raise ValueError("Unknown pooling/scheduler")
    for name in ("weight_decay", "learning_rate", "gradient_clip", "optimizer_eps"):
        if not np.isfinite(cfg[name]) or cfg[name] < 0 or (name != "weight_decay" and cfg[name] == 0):
            raise ValueError("Invalid " + name)
    if len(cfg["optimizer_betas"]) != 2 or not all(0 <= b < 1 for b in cfg["optimizer_betas"]):
        raise ValueError("Invalid optimizer betas")
    if cfg["weighting"] not in {"patient", "snapshot"} or cfg["balance"] not in {"none", "sqrt", "full"}:
        raise ValueError("Unknown weighting or balance")
    return cfg


def seed_run(seed):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def make_encoder(n_features, n_steps, config, static_dim=0):
    import torch
    from torch import nn
    cfg = _config(config)
    d = cfg["d_model"]
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(n_features, d)
            self.input_dropout = nn.Dropout(cfg["dropout"])
            layer = nn.TransformerEncoderLayer(d, cfg["heads"], cfg["ff"], cfg["dropout"],
                                               activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg["depth"], enable_nested_tensor=False)
            for block in self.encoder.layers:
                for p in block.parameters():
                    if p.ndim > 1:
                        nn.init.xavier_uniform_(p)
            self.norm = nn.LayerNorm(d)
            self.pool_score = nn.Linear(d, 1) if cfg["pooling"] == "attention" else None
            self.head = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(d, 1))
            positions = torch.arange(n_steps, dtype=torch.float32)[:, None]
            freq = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.) / d))
            pe = torch.zeros(n_steps, d)
            pe[:, 0::2] = torch.sin(positions * freq)
            pe[:, 1::2] = torch.cos(positions * freq[:pe[:, 1::2].shape[1]])
            self.register_buffer("position", pe)
            self.static_dim = int(static_dim)
            self.static_branch = None
            if static_dim:
                # Preserve the sequence-only initial prediction. Unlike a zeroed
                # multi-layer branch, this branch has nonzero first-step gradients.
                self.static_branch = nn.Sequential(nn.Linear(static_dim, d), nn.GELU(),
                                                   nn.Dropout(cfg["dropout"]))
                nn.init.zeros_(self.static_branch[0].weight)
                nn.init.zeros_(self.static_branch[0].bias)

        def forward(self, x, valid, static=None):
            if valid.dtype != torch.bool or not bool(valid.any(1).all()):
                raise ValueError("Every input needs at least one explicitly available timestep")
            x = x.masked_fill(~valid[:, :, None], 0.)
            z = self.input_dropout(self.projection(x)) + self.position[:x.shape[1]]
            z = self.norm(self.encoder(z, src_key_padding_mask=~valid))
            if self.pool_score is None:
                pooled = z.masked_fill(~valid[:, :, None], 0.).sum(1) / valid.sum(1, keepdim=True)
            else:
                weights = self.pool_score(z).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
                pooled = (z * weights[:, :, None]).sum(1)
            if self.static_branch is not None:
                if static is None or static.shape != (len(x), self.static_dim):
                    raise ValueError("Aligned static input required by the fitted fusion architecture")
                pooled = pooled + self.static_branch(static)
            elif static is not None:
                raise ValueError("Static inputs supplied to sequence-only architecture")
            return self.head(pooled).squeeze(-1)
    return Encoder()


def quick_metrics(y, scores, primary_lift_fraction=.10):
    y, scores = np.asarray(y), np.asarray(scores)
    if (y.shape != scores.shape or set(np.unique(y)) != {0, 1}
            or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Aligned binary labels and finite probability scores required")
    return dict(primary_lift=lift_at_fraction(y, scores, primary_lift_fraction),
                primary_lift_fraction=primary_lift_fraction,
                average_precision=float(average_precision_score(y, scores)),
                roc_auc=float(roc_auc_score(y, scores)),
                log_loss=float(log_loss(y, np.clip(scores, 1e-7, 1-1e-7))),
                brier_score=float(np.mean((y - scores) ** 2)))


def _static_array(value, n):
    if value is None:
        return None
    a = np.asarray(value)
    if a.ndim == 3 and a.shape[1] == 1:
        a = a[:, 0, :]
    if a.ndim != 2 or len(a) != n or a.shape[1] < 1 or not np.isfinite(a).all():
        raise ValueError("Static features must be finite aligned N x F or N x 1 x F")
    return np.asarray(a, dtype=np.float32)


def _predict(model, x, valid, device="cpu", batch_size=512, static=None):
    import torch
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xx = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size], dtype=np.float32)).to(device)
            mask = torch.from_numpy(valid[start:start + batch_size]).to(device)
            static_batch = (None if static is None else
                            torch.from_numpy(np.ascontiguousarray(static[start:start + batch_size])).to(device))
            output.append(model(xx, mask, static_batch).sigmoid().cpu().numpy())
    return np.concatenate(output)


def train_encoder(X_train, y_train, groups_train, X_val, y_val, valid_train=None,
                  valid_val=None, config=None, seed=42, groups_val=None,
                  static_train=None, static_val=None):
    """Fresh TRAIN-only fit; validation selects stopping, never gradients."""
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    a, ma = _array(X_train, valid_train, finite=True)
    b, mb = _array(X_val, valid_val, finite=True)
    y, vy, groups = np.asarray(y_train), np.asarray(y_val), np.asarray(groups_train)
    if (a.shape[1:] != b.shape[1:] or y.shape != (len(a),) or vy.shape != (len(b),)
            or groups.shape != y.shape or any(set(np.unique(t)) != {0, 1} for t in (y, vy))):
        raise ValueError("Training/validation shapes or labels invalid")
    if groups_val is not None and (np.shape(groups_val) != vy.shape or set(groups) & set(groups_val)):
        raise ValueError("Patient overlap or malformed validation IDs")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    cfg = _config(config)
    sa, sb = _static_array(static_train, len(a)), _static_array(static_val, len(b))
    if (sa is None) != (sb is None) or (sa is not None and sa.shape[1] != sb.shape[1]):
        raise ValueError("Both training and validation need the same static feature dimension")
    static_dim = 0 if sa is None else sa.shape[1]
    started = time.perf_counter()
    seed_run(seed); torch.set_num_threads(cfg["threads"])
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    model = make_encoder(a.shape[-1], a.shape[1], cfg, static_dim).to(cfg["device"])
    weights = sample_weights(groups, cfg["weighting"])
    positive_weight = class_weight(y, weights, cfg["balance"])
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim > 1], "weight_decay": cfg["weight_decay"]},
        {"params": [p for p in model.parameters() if p.ndim <= 1], "weight_decay": 0.}],
        lr=cfg["learning_rate"], betas=tuple(cfg["optimizer_betas"]), eps=cfg["optimizer_eps"])
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=2)
                 if cfg["scheduler"] == "plateau" else None)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(positive_weight, device=cfg["device"]))
    static_dataset = np.zeros((len(a), 0), dtype=np.float32) if sa is None else sa
    loader = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(ma),
                                     torch.tensor(y, dtype=torch.float32), torch.from_numpy(weights),
                                     torch.from_numpy(np.ascontiguousarray(static_dataset))),
                        batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    best_lift, best_epoch, stale, best_state, history = -np.inf, None, 0, None, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); total = 0.
        for xx, mask, yy, ww, ss in loader:
            xx, mask, yy, ww, ss = [t.to(cfg["device"]) for t in (xx, mask, yy, ww, ss)]
            opt.zero_grad(set_to_none=True)
            loss = (loss_fn(model(xx, mask, ss if static_dim else None), yy) * ww).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            opt.step(); total += float(loss.detach()) * len(xx)
        train_scores = _predict(model, a, ma, cfg["device"], static=sa)
        val_scores = _predict(model, b, mb, cfg["device"], static=sb)
        tm, vm = quick_metrics(y, train_scores, cfg["primary_lift_fraction"]), quick_metrics(vy, val_scores, cfg["primary_lift_fraction"])
        history.append(dict(epoch=epoch, optimization_loss=total/len(a), train_ap=tm["average_precision"],
                            train_lift=tm["primary_lift"], validation_lift=vm["primary_lift"],
                            validation_ap=vm["average_precision"], train_log_loss=tm["log_loss"],
                            validation_log_loss=vm["log_loss"], learning_rate=opt.param_groups[0]["lr"]))
        if vm["primary_lift"] > best_lift:
            best_lift, best_epoch, stale = vm["primary_lift"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(vm["primary_lift"])
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best_state, strict=True)
    train_scores = _predict(model, a, ma, cfg["device"], static=sa)
    val_scores = _predict(model, b, mb, cfg["device"], static=sb)
    stream = io.BytesIO(); torch.save(best_state, stream); blob = stream.getvalue()
    summary = dict(seed=seed, config=cfg, config_sha256=stable_hash(cfg), input_shape=list(a.shape[1:]),
                   runtime_versions={"python": platform.python_version(), **{package: importlib.metadata.version(package)
                       for package in ("numpy", "pandas", "scikit-learn", "torch")}},
                   model_sha256=hashlib.sha256(blob).hexdigest(), artifact_format="torch_weights_only_state_dict",
                   train_metrics=quick_metrics(y, train_scores, cfg["primary_lift_fraction"]),
                   validation_metrics=quick_metrics(vy, val_scores, cfg["primary_lift_fraction"]),
                   generalization_gap_lift=lift_at_fraction(y, train_scores, cfg["primary_lift_fraction"])-lift_at_fraction(vy, val_scores, cfg["primary_lift_fraction"]),
                   generalization_gap_ap=float(average_precision_score(y, train_scores)-average_precision_score(vy, val_scores)),
                   parameter_count=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch,
                   epochs_observed=len(history), positive_weight=positive_weight,
                   patient_disjointness_verified=groups_val is not None, wall_seconds=time.perf_counter()-started,
                   selection_metric="validation_lift_at_primary_capacity", primary_lift_fraction=cfg["primary_lift_fraction"],
                   checkpoint_tie_rule="earliest epoch with maximum validation lift; AP is not a tiebreaker",
                   lift_population="patient_snapshots; exact ceil capacity; frozen row-order score ties", test_used=False,
                   static_dim=static_dim,
                   architecture="pooled_sequence_plus_GELU_static_branch" if static_dim else "sequence_encoder",
                   projection_column_norm=model.projection.weight.detach().cpu().norm(dim=0).tolist(),
                   projection_norm_note="parameter magnitude, not predictive feature importance")
    return dict(summary=summary, model_blob=blob, config=cfg, history=history,
                train_scores=train_scores, val_scores=val_scores, validation_scores=val_scores)


def predict_encoder(run, X, valid=None, device="cpu", static=None):
    import torch
    s = run["summary"]
    if hashlib.sha256(run["model_blob"]).hexdigest() != s["model_sha256"] or stable_hash(s["config"]) != s["config_sha256"]:
        raise ValueError("Saved model or configuration hash mismatch")
    x, mask = _array(X, valid, finite=True)
    if list(x.shape[1:]) != s["input_shape"]:
        raise ValueError("Prediction shape differs from trained input")
    static_values = _static_array(static, len(x))
    expected_static = s.get("static_dim", 0)
    if (0 if static_values is None else static_values.shape[1]) != expected_static:
        raise ValueError("Static feature dimension differs from fitted model")
    model = make_encoder(x.shape[-1], x.shape[1], s["config"], expected_static).to(device)
    model.load_state_dict(torch.load(io.BytesIO(run["model_blob"]), map_location=device, weights_only=True), strict=True)
    return _predict(model, x, mask, device, static=static_values)


def distribution_diagnostics(train_values, validation_values, feature_names,
                             valid_train=None, valid_validation=None):
    """TRAIN/VALIDATION descriptive checks only; never used to fit transforms."""
    a, ma = _array(train_values, valid_train, finite=True, allow_empty=True)
    b, mb = _array(validation_values, valid_validation, finite=True, allow_empty=True)
    if a.shape[-1] != b.shape[-1] or len(feature_names) != a.shape[-1]:
        raise ValueError("Diagnostic feature names/width mismatch")
    aa, bb = a[ma].astype(np.float64), b[mb].astype(np.float64)
    if not len(aa) or not len(bb):
        return [dict(feature=name, train_observed_rows=len(aa), validation_observed_rows=len(bb),
                     train_mean=float(aa[:, j].mean()) if len(aa) else None,
                     validation_mean=float(bb[:, j].mean()) if len(bb) else None,
                     train_sd=float(aa[:, j].std()) if len(aa) else None,
                     validation_sd=float(bb[:, j].std()) if len(bb) else None,
                     standardized_mean_difference=None, status="one_split_has_no_observed_auxiliary_values")
                for j, name in enumerate(feature_names)]
    return [dict(feature=name, train_mean=float(aa[:, j].mean()), validation_mean=float(bb[:, j].mean()),
                 train_observed_rows=len(aa), validation_observed_rows=len(bb), status="available_observations_only",
                 train_sd=float(aa[:, j].std()), validation_sd=float(bb[:, j].std()),
                 standardized_mean_difference=float((bb[:, j].mean()-aa[:, j].mean()) / max(float(aa[:, j].std()), 1e-6)))
            for j, name in enumerate(feature_names)]
