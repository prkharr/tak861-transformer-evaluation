"""Auditable, bounded model search. Inputs are private raw count tensors.

This module never receives TEST rows during fitting. Each run is a complete
TRAIN/VALIDATION experiment; seeds are repetitions, not selection candidates.
"""
import copy
import hashlib
import io
import json
import random
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def seed_run(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def sample_weights(groups, mode="snapshot"):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0 or any(v is None for v in groups):
        raise ValueError("Expected nonempty patient groups")
    if mode == "snapshot":
        return np.ones(len(groups), dtype=np.float32)
    if mode != "patient":
        raise ValueError("Unknown weighting unit")
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[inv]
    return (w / w.mean()).astype(np.float32)


def class_weight(y, weights, balance):
    y, weights = np.asarray(y), np.asarray(weights)
    if (y.ndim != 1 or weights.shape != y.shape or set(np.unique(y)) != {0, 1}
            or not np.isfinite(weights).all() or (weights <= 0).any()
            or balance not in {"none", "sqrt", "full"}):
        raise ValueError("Class weights require aligned binary labels, positive weights, and a known balance mode")
    ratio = float(weights[y == 0].sum() / weights[y == 1].sum())
    return {"none": 1., "sqrt": ratio ** .5, "full": ratio}[balance]


def raw_representation(x, representation):
    """No zero-derived padding. Missing measurements remain missing."""
    x = np.asarray(x)
    if (x.ndim != 3 or x.shape[1] != 12 or min(x.shape) < 1 or
            not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x)):
        raise ValueError("Expected N x 12 x F nonnegative counts; NaN means missing")
    # Validate before float32 conversion so fractions cannot round into integers.
    # A row batch avoids several full-size boolean copies of the 1 GiB tensor.
    for start in range(0, len(x), 256):
        block = x[start:start+256]
        if np.isinf(block).any() or (block < 0).any():
            raise ValueError("Expected finite nonnegative counts; NaN means missing")
        finite = block[np.isfinite(block)]
        if (finite != np.floor(finite)).any():
            raise ValueError("Fractional input is not a raw integer count")
    x = np.asarray(x, dtype=np.float32)
    if representation == "monthly":
        return x.copy()
    if representation == "without_step0":
        out = x.copy()
        out[:, 0, :] = 0
        return out
    if representation == "quarterly":
        # np.sum deliberately propagates missing constituent months.
        return x.reshape(len(x), 4, 3, x.shape[2]).sum(axis=2)
    if representation == "flattened":
        return x.reshape(len(x), -1).copy()
    if representation == "annual":
        return x.sum(axis=1)
    if representation == "windows":
        return np.concatenate([x[:, :n].sum(axis=1) for n in (1, 3, 6, 12)], axis=1)
    raise ValueError("Unknown representation")


def fit_numeric_transform(raw, scale=False, log=True):
    raw = np.asarray(raw)
    if raw.ndim not in (2, 3) or min(raw.shape) < 1 or np.isinf(raw).any() or (log and (raw < 0).any()):
        raise ValueError("Invalid fitting measurements")
    a = np.log1p(raw) if log else raw.copy()
    flat = a.reshape(-1, a.shape[-1])
    # Train-only medians; all-missing fields remain 0 and are explicitly counted.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(flat, axis=0)
    all_missing = np.isnan(median)
    median[all_missing] = 0
    fitted = np.where(np.isnan(flat), median, flat)
    center = fitted.mean(axis=0) if scale else np.zeros(flat.shape[1])
    # A unit floor avoids exploding rare standardized counts.
    train_sd = fitted.std(axis=0)
    divisor = np.maximum(train_sd, 1.) if scale else np.ones(flat.shape[1])
    return dict(median=median.tolist(), center=center.tolist(), divisor=divisor.tolist(),
                log=bool(log), scale=bool(scale), missing_cells=int(np.isnan(raw).sum()),
                all_missing_fields=int(all_missing.sum()), fitted_on="TRAIN only",
                training_transformed_sd=(train_sd / divisor).tolist())


def numeric_transform(raw, state):
    raw = np.asarray(raw)
    for key in ("median", "center", "divisor"):
        vector = np.asarray(state[key])
        if vector.shape != (raw.shape[-1],) or not np.isfinite(vector).all():
            raise ValueError("Preprocessing state width/values do not match model input")
    if (np.asarray(state["divisor"]) <= 0).any():
        raise ValueError("Preprocessing divisor must be positive")
    a = np.log1p(raw) if state["log"] else np.asarray(raw).copy()
    a = np.where(np.isnan(a), np.asarray(state["median"]), a)
    result = (a - np.asarray(state["center"])) / np.asarray(state["divisor"])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite transformed model input")
    return result.astype(np.float32)


def _resolved_config(config, input_dim):
    """Record actual defaults so the serialized experiment is reproducible."""
    cfg = copy.deepcopy(config)
    if "columns" in cfg:
        cfg["columns"] = np.asarray(cfg["columns"]).tolist()
    if cfg.get("family") not in {"transformer", "lightgbm"}:
        raise ValueError("Unsupported candidate family")
    if cfg.get("representation") not in {"monthly", "quarterly", "without_step0", "flattened", "annual", "windows"}:
        raise ValueError("Unknown representation")
    defaults = dict(scale=False, weighting="snapshot", balance="none", columns=list(range(input_dim)))
    if cfg["family"] == "transformer":
        defaults.update(d_model=64, heads=4, ff=128, dropout=.2, depth=1, pooling="mean",
                        projection_l1=0., explicit_l2=0., weight_decay=1e-3,
                        learning_rate=3e-4, batch_size=256, scheduler="plateau",
                        optimizer_betas=[.9, .999], optimizer_eps=1e-8, gradient_clip=1.)
    else:
        defaults.update(n_estimators=600, learning_rate=.03, num_leaves=15, max_depth=-1,
                        min_child_samples=50, colsample_bytree=.8, subsample=.8,
                        reg_lambda=1., reg_alpha=0., early_stopping_rounds=30)
    unknown = set(cfg) - set(defaults) - {"family", "representation", "subset", "drop_step0"}
    if unknown:
        raise ValueError("Unknown candidate settings: " + ", ".join(sorted(unknown)))
    cfg = {**defaults, **cfg}
    if "drop_step0" in cfg and not isinstance(cfg["drop_step0"], bool):
        raise ValueError("drop_step0 must be boolean")
    if cfg["weighting"] not in {"snapshot", "patient"} or cfg["balance"] not in {"none", "sqrt", "full"}:
        raise ValueError("Unknown weighting/balance mode")
    if not isinstance(cfg["scale"], bool):
        raise ValueError("scale must be boolean")
    if len(cfg["columns"]) != input_dim or len(set(cfg["columns"])) != input_dim:
        raise ValueError("Configured original columns must uniquely match the selected tensor width")
    if not np.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if cfg["family"] == "transformer":
        if cfg["representation"] not in {"monthly", "quarterly", "without_step0"}:
            raise ValueError("Sequence architecture needs sequence representation")
        for key in ("d_model", "heads", "ff", "depth", "batch_size"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(key + " must be a positive integer")
        if cfg["d_model"] < 2 or cfg["d_model"] % cfg["heads"]:
            raise ValueError("Invalid embedding/head dimensions")
        if cfg["pooling"] not in {"mean", "attention"} or cfg["scheduler"] not in {"plateau", "none"}:
            raise ValueError("Unknown pooling/scheduler")
        if not 0 <= cfg["dropout"] < 1:
            raise ValueError("Dropout must be in [0,1)")
        for key in ("projection_l1", "explicit_l2", "weight_decay"):
            if not np.isfinite(cfg[key]) or cfg[key] < 0:
                raise ValueError(key + " must be finite and nonnegative")
        if cfg["weight_decay"] and cfg["explicit_l2"]:
            raise ValueError("Choose explicit L2 or AdamW decay, never both; set weight_decay=0 explicitly")
        if (len(cfg["optimizer_betas"]) != 2 or not all(0 <= b < 1 for b in cfg["optimizer_betas"])
                or not np.isfinite(cfg["optimizer_eps"]) or cfg["optimizer_eps"] <= 0
                or not np.isfinite(cfg["gradient_clip"]) or cfg["gradient_clip"] <= 0):
            raise ValueError("Invalid optimizer/clipping configuration")
    else:
        if cfg["representation"] not in {"flattened", "annual", "windows"}:
            raise ValueError("LightGBM needs a tabular representation")
        for key in ("n_estimators", "num_leaves", "min_child_samples", "early_stopping_rounds"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(key + " must be a positive integer")
        if cfg["num_leaves"] < 2 or not all(0 < cfg[k] <= 1 for k in ("colsample_bytree", "subsample")):
            raise ValueError("Invalid tree leaf/subsampling configuration")
        if type(cfg["max_depth"]) is not int or cfg["max_depth"] == 0 or cfg["max_depth"] < -1:
            raise ValueError("max_depth must be -1 or a positive integer")
        if any(not np.isfinite(cfg[k]) or cfg[k] < 0 for k in ("reg_alpha", "reg_lambda")):
            raise ValueError("Tree penalties must be finite and nonnegative")
    return cfg


def _configured_representation(x, config):
    # The declared ablation must also work for direct model restoration, without
    # requiring an undocumented preprocessing operation by its caller.
    if config.get("drop_step0", False):
        x = raw_representation(x, "without_step0")
    return raw_representation(x, config["representation"])


def make_encoder(input_dim, config):
    import torch
    from torch import nn
    d = int(config.get("d_model", 64))
    heads = int(config.get("heads", 4))
    if heads < 1 or d % heads or d < 2:
        raise ValueError("Invalid embedding/head dimensions")
    if config.get("pooling", "mean") not in {"mean", "attention"}:
        raise ValueError("Unknown pooling strategy")

    class CountEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(input_dim, d)
            drop = float(config.get("dropout", .2))
            self.input_dropout = nn.Dropout(drop)
            layer = nn.TransformerEncoderLayer(d, heads, int(config.get("ff", 128)),
                       drop, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, int(config.get("depth", 1)),
                                                 enable_nested_tensor=False)
            for layer in self.encoder.layers:
                for p in layer.parameters():
                    if p.ndim > 1:
                        nn.init.xavier_uniform_(p)
            self.norm = nn.LayerNorm(d)
            self.pooling = config.get("pooling", "mean")
            self.pool_score = nn.Linear(d, 1) if self.pooling == "attention" else None
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 1))
            pos = torch.arange(12, dtype=torch.float32)[:, None]
            frequency = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.) / d))
            pe = torch.zeros(12, d)
            pe[:, 0::2] = torch.sin(pos * frequency)
            pe[:, 1::2] = torch.cos(pos * frequency[:pe[:, 1::2].shape[1]])
            self.register_buffer("position", pe)

        def forward(self, x):
            z = self.input_dropout(self.projection(x)) + self.position[:x.shape[1]]
            z = self.norm(self.encoder(z))
            pooled = (z * self.pool_score(z).softmax(1)).sum(1) if self.pool_score is not None else z.mean(1)
            return self.head(pooled).squeeze(-1)

    return CountEncoder()


def penalties(model, config):
    """L1 = lambda1*mean(abs(projection W)); L2 = lambda2*mean(matrix W**2).

    Biases and layer-normalization vectors are excluded from L2; sinusoidal
    positions are fixed buffers. These terms are recomputed each optimizer step.
    Normalization by parameter count is intentional and logged; coefficients are
    not numerically equivalent to the unnormalized AdamW decay coefficient.
    """
    if any(not np.isfinite(config.get(k, 0.)) or config.get(k, 0.) < 0
           for k in ("projection_l1", "explicit_l2", "weight_decay")):
        raise ValueError("Regularization coefficients must be finite and nonnegative")
    l1 = float(config.get("projection_l1", 0.)) * model.projection.weight.abs().mean()
    l2_coef = float(config.get("explicit_l2", 0.))
    if l2_coef and config.get("weight_decay", 0.):
        raise ValueError("Choose explicit L2 or AdamW decay, never both")
    matrices = [p for p in model.parameters() if p.ndim > 1]
    l2 = l2_coef * sum(p.square().sum() for p in matrices) / sum(p.numel() for p in matrices)
    return l1, l2


def quick_metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if (y.ndim != 1 or scores.shape != y.shape or set(np.unique(y)) != {0, 1}
            or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Metrics need finite scores and both classes")
    return dict(average_precision=float(average_precision_score(y, scores)),
                roc_auc=float(roc_auc_score(y, scores)),
                log_loss=float(log_loss(y, np.clip(scores, 1e-7, 1-1e-7))))


def predict_encoder(model, values, device="cpu", batch_size=512):
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            scores = model(torch.from_numpy(values[start:start+batch_size]).to(device)).sigmoid()
            result.append(scores.cpu().numpy())
    return np.concatenate(result)


def train_candidate(x_train, y_train, groups_train, x_validation, y_validation,
                    config, seed, *, epochs=30, patience=5, threads=8, device="cpu", groups_validation=None):
    """Fit TRAIN/VALIDATION only; callers must enforce frozen split membership.

    There is deliberately no TEST argument. Optional validation patient IDs allow
    this function to independently verify patient disjointness as well.
    """
    import torch
    x_train, x_validation = np.asarray(x_train), np.asarray(x_validation)
    y_train, y_validation, groups_train = np.asarray(y_train), np.asarray(y_validation), np.asarray(groups_train)
    if (x_train.ndim != 3 or x_validation.ndim != 3 or x_train.shape[1:] != x_validation.shape[1:]
            or y_train.ndim != 1 or y_validation.ndim != 1 or groups_train.ndim != 1
            or len(x_train) != len(y_train) or len(groups_train) != len(y_train)
            or len(x_validation) != len(y_validation)):
        raise ValueError("TRAIN alignment mismatch")
    config = _resolved_config(config, x_train.shape[-1])
    if groups_validation is not None:
        groups_validation = np.asarray(groups_validation)
        if groups_validation.shape != y_validation.shape or set(groups_train) & set(groups_validation):
            raise ValueError("TRAIN/VALIDATION patient overlap or validation group mismatch")
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_validation)) != {0, 1}:
        raise ValueError("TRAIN and VALIDATION need both labels")
    if any(type(n) is not int or n < 1 for n in (epochs, patience, threads)) or type(seed) is not int or seed < 0:
        raise ValueError("Invalid training limits")
    started = time.perf_counter()
    seed_run(seed)
    torch.set_num_threads(threads)
    family = config["family"]
    representation = config["representation"]
    raw_train = _configured_representation(x_train, config)
    raw_val = _configured_representation(x_validation, config)
    w = sample_weights(groups_train, config.get("weighting", "snapshot"))
    positive_weight = class_weight(y_train, w, config.get("balance", "none"))
    transform = fit_numeric_transform(raw_train, scale=config.get("scale", False),
                                     log=family != "lightgbm")
    a = numeric_transform(raw_train, transform)
    b = numeric_transform(raw_val, transform)
    del raw_train, raw_val
    history, best_epoch = [], None
    if family == "transformer":
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        if a.ndim != 3:
            raise ValueError("Sequence architecture needs sequence representation")
        model = make_encoder(a.shape[-1], config).to(device)
        wd = float(config.get("weight_decay", 1e-3))
        if wd and config.get("explicit_l2", 0):
            raise ValueError("Do not combine explicit L2 with AdamW decay")
        # Biases and normalization are not decayed; matrices share one convention.
        param_groups = [{"params": [p for p in model.parameters() if p.ndim > 1], "weight_decay": wd},
                        {"params": [p for p in model.parameters() if p.ndim <= 1], "weight_decay": 0.}]
        optimizer = torch.optim.AdamW(param_groups, lr=config["learning_rate"], betas=tuple(config["optimizer_betas"]), eps=config["optimizer_eps"])
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
                     if config["scheduler"] == "plateau" else None)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(positive_weight, device=device))
        loader = DataLoader(TensorDataset(torch.from_numpy(a), torch.as_tensor(y_train, dtype=torch.float32),
                                         torch.from_numpy(w)), batch_size=config.get("batch_size", 256),
                            shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        best_ap, stale, best_state = -np.inf, 0, None
        for epoch in range(1, epochs+1):
            model.train()
            total, n, penalty_total, l1_total, l2_total = 0., 0, 0., 0., 0.
            for xx, yy, ww in loader:
                xx, yy, ww = xx.to(device), yy.to(device), ww.to(device)
                optimizer.zero_grad(set_to_none=True)
                l1, l2 = penalties(model, config)
                # Weights are normalized over the complete training population.
                loss = (loss_fn(model(xx), yy) * ww).mean() + l1 + l2
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite optimization loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(xx)
                penalty_total += float((l1+l2).detach()) * len(xx)
                l1_total += float(l1.detach()) * len(xx)
                l2_total += float(l2.detach()) * len(xx)
                n += len(xx)
            val_scores = predict_encoder(model, b, device)
            train_scores = predict_encoder(model, a, device)
            vm, tm = quick_metrics(y_validation, val_scores), quick_metrics(y_train, train_scores)
            history.append(dict(epoch=epoch, optimization_loss=total/n, regularization_loss=penalty_total/n,
                                projection_l1_loss=l1_total/n, explicit_l2_loss=l2_total/n,
                                train_ap=tm["average_precision"], validation_ap=vm["average_precision"],
                                train_log_loss=tm["log_loss"], validation_log_loss=vm["log_loss"],
                                learning_rate=optimizer.param_groups[0]["lr"]))
            if vm["average_precision"] > best_ap:
                best_ap, best_epoch, stale = vm["average_precision"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step(vm["average_precision"])
            if stale >= patience:
                break
        model.load_state_dict(best_state, strict=True)
        train_scores, val_scores = predict_encoder(model, a, device), predict_encoder(model, b, device)
        buffer = io.BytesIO()
        torch.save(best_state, buffer)
        model_blob = buffer.getvalue()
        parameter_count = sum(p.numel() for p in model.parameters())
        importance = model.projection.weight.detach().cpu().norm(dim=0).tolist()
        artifact_format = "torch_state_dict_weights_only"
    elif family == "lightgbm":
        import lightgbm as lgb
        if a.ndim != 2:
            raise ValueError("LightGBM needs a tabular representation")
        model = lgb.LGBMClassifier(n_estimators=config.get("n_estimators", 600),
                    learning_rate=config.get("learning_rate", .03), num_leaves=config.get("num_leaves", 15),
                    max_depth=config.get("max_depth", -1), min_child_samples=config.get("min_child_samples", 50),
                    colsample_bytree=config.get("colsample_bytree", .8), subsample=config.get("subsample", .8),
                    subsample_freq=1, reg_lambda=config.get("reg_lambda", 1.),
                    reg_alpha=config.get("reg_alpha", 0.), scale_pos_weight=positive_weight,
                    random_state=seed, n_jobs=threads, deterministic=True, force_col_wise=True, verbosity=-1)
        metric_history = {}
        def ap_metric(y, p):
            return "average_precision", float(average_precision_score(y, p)), True
        model.set_params(metric="None")
        model.fit(a, y_train, sample_weight=w, eval_set=[(b, y_validation)], eval_metric=ap_metric,
                  callbacks=[lgb.early_stopping(config["early_stopping_rounds"], first_metric_only=True, verbose=False),
                             lgb.record_evaluation(metric_history)])
        best_epoch = int(model.best_iteration_)
        train_scores = model.booster_.predict(a, num_iteration=best_epoch)
        val_scores = model.booster_.predict(b, num_iteration=best_epoch)
        history = [dict(epoch=i+1, validation_ap=float(v))
                   for i, v in enumerate(metric_history["valid_0"]["average_precision"])]
        model_blob = model.booster_.model_to_string(num_iteration=best_epoch).encode()
        parameter_count = int(sum(t["num_leaves"] for t in model.booster_.dump_model()["tree_info"]))
        importance = model.booster_.feature_importance(importance_type="gain").tolist()
        artifact_format = "lightgbm_text"
    else:
        raise ValueError("Unsupported candidate family")
    importance_blocks = 1 if family == "transformer" else len(importance) // x_train.shape[-1]
    original_importance = np.asarray(importance).reshape(importance_blocks, x_train.shape[-1]).sum(axis=0).tolist()
    summary = dict(config=config, seed=int(seed), training_complete=True, best_epoch=best_epoch,
                   epochs_observed=len(history), validation_evaluations=len(history),
                   train_metrics=quick_metrics(y_train, train_scores),
                   validation_metrics=quick_metrics(y_validation, val_scores),
                   generalization_gap_ap=float(average_precision_score(y_train, train_scores) -
                                               average_precision_score(y_validation, val_scores)),
                   positive_weight=positive_weight, parameter_count=parameter_count,
                   complexity_unit="parameters" if family == "transformer" else "tree_leaves",
                   wall_seconds=time.perf_counter()-started, artifact_format=artifact_format,
                   model_sha256=hashlib.sha256(model_blob).hexdigest(), transform=transform,
                   transform_sha256=stable_hash(transform), config_sha256=stable_hash(config),
                   raw_input_shape=list(x_train.shape[1:]),
                   patient_disjointness_verified=groups_validation is not None,
                   regularization_definition=("L1=lambda1*mean(abs(input_projection)); explicit_L2=lambda2*mean(all_matrix_weights_squared); AdamW decay excludes biases/norm" if family == "transformer" else "LightGBM native reg_alpha/reg_lambda"),
                   importance=importance, aggregated_original_importance=original_importance,
                   importance_blocks_per_original_feature=importance_blocks,
                   importance_definition=("input_projection_column_L2_norm; model-internal diagnostic, not held-out reliance" if family == "transformer" else "LightGBM gain; summed across temporal summary blocks per original feature; not held-out reliance"),
                   test_inference_performed=False,
                   selection_scope="Repeated frozen VALIDATION; results are development estimates")
    return dict(summary=summary, history=history, model_blob=model_blob,
                train_scores=train_scores, validation_scores=val_scores)


def make_saved_predictor(result, *, device="cpu"):
    """Restore once for frozen diagnostics; never fit or update preprocessing.

    Call only after a persisted decision lock when using TEST. Reuse the
    returned predictor for bounded validation perturbations to avoid repeated
    checkpoint loading. Copy metadata so later caller mutations cannot change
    the restored input contract.
    """
    import torch
    s = result["summary"]
    blob = result["model_blob"]
    if hashlib.sha256(blob).hexdigest() != s["model_sha256"]:
        raise ValueError("Model artifact hash mismatch")
    cfg = copy.deepcopy(s["config"])
    if stable_hash(cfg) != s["config_sha256"] or stable_hash(s["transform"]) != s["transform_sha256"]:
        raise ValueError("Config/preprocessing artifact hash mismatch")
    transform = copy.deepcopy(s["transform"])
    raw_shape = tuple(s["raw_input_shape"])
    if cfg["family"] == "transformer":
        model = make_encoder(raw_shape[-1], cfg).to(device)
        state = torch.load(io.BytesIO(blob), map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
    elif cfg["family"] == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_str=blob.decode())
    else:
        raise ValueError("Unsupported saved model family")

    def predict(x):
        if tuple(np.shape(x)[1:]) != raw_shape:
            raise ValueError("Saved model raw input shape mismatch")
        a = numeric_transform(_configured_representation(x, cfg), transform)
        if cfg["family"] == "transformer":
            return predict_encoder(model, a, device)
        return model.predict(a)
    return predict


def predict_saved(result, x, *, device="cpu"):
    """One-shot restored prediction; keep TEST behind the persisted lock."""
    return make_saved_predictor(result, device=device)(x)


def choose_finalists(results, expected_candidates, seeds):
    """Mean seed AP wins; no picking one lucky seed. No uncertainty-based equivalence claim."""
    if (not expected_candidates or len(set(expected_candidates)) != len(expected_candidates)
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError("Declare nonempty unique candidates and seeds")
    if set(results) != set(expected_candidates):
        raise ValueError("Search incomplete: do not select from a partially finished grid")
    rows = []
    for name in expected_candidates:
        runs = results[name]
        if sorted(r["summary"]["seed"] for r in runs) != sorted(seeds):
            raise ValueError("Missing/duplicate seeds")
        if not all(r["summary"]["training_complete"] for r in runs):
            raise ValueError("Incomplete model run")
        cfg = runs[0]["summary"]["config"]
        if any(r["summary"]["config"] != cfg for r in runs):
            raise ValueError("Seed repetitions have different configurations")
        values = [r["summary"]["validation_metrics"]["average_precision"] for r in runs]
        if not np.isfinite(values).all() or not all(0 <= value <= 1 for value in values):
            raise ValueError("Invalid validation AP; cannot select a winner")
        rows.append(dict(candidate=name, family=cfg["family"], mean_validation_ap=float(np.mean(values)),
                         sd_validation_ap=float(np.std(values, ddof=1)) if len(values)>1 else None,
                         n_features=len(cfg["columns"]),
                         mean_complexity=float(np.mean([r["summary"]["parameter_count"] for r in runs])),
                         fits=len(runs), validation_evaluations=sum(r["summary"]["validation_evaluations"] for r in runs)))
    winners = {}
    for family in sorted(set(r["family"] for r in rows)):
        family_rows = [r for r in rows if r["family"] == family]
        # Parsimony breaks exact score ties; non-significance is not equivalence.
        winner = sorted(family_rows, key=lambda r: (-r["mean_validation_ap"], r["n_features"],
                                                   r["mean_complexity"], r["candidate"]))[0]
        winners[family] = winner["candidate"]
    return winners, rows
