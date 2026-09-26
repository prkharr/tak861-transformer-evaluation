"""Selected-feature experiments, embedded after the input/model/metric helpers.

Training reads TRAIN and VALIDATION rows only. TEST scoring is a separate call.
The sequence model consumes log1p monthly counts; the logistic comparator uses
TRAIN-standardized log1p total counts and intentionally discards month order.
"""
import copy
import hashlib
import io
import json
import warnings
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ExperimentSequenceDataset(Dataset):
    def __init__(self, data, split, selected_indices):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split.")
        self.data = data
        self.rows = np.asarray(data["indices"][split], dtype=np.int64)
        self.columns = np.asarray(selected_indices, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        # Index the row first: mixed NumPy advanced indexing would swap axes.
        values = np.array(self.data["X"][row][:, self.columns], dtype=np.float32, copy=True)
        return torch.from_numpy(values), torch.tensor(float(self.data["y"][row]), dtype=torch.float32)


def _experiment_loader(data, split, columns, batch_size, seed, shuffle=False):
    return DataLoader(ExperimentSequenceDataset(data, split, columns),
                      batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def _experiment_log1p(values):
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative raw counts before log1p.")
    return torch.log1p(values)


class NumericLogisticModel(nn.Module):
    """Safe inference from numeric parameters; contains no pickled estimator."""
    def __init__(self, coefficient, intercept, scaler_mean, scaler_scale, seq_len):
        super().__init__()
        tensors = [torch.as_tensor(value, dtype=torch.float64).clone()
                   for value in (coefficient, intercept, scaler_mean, scaler_scale)]
        coefficient, intercept, scaler_mean, scaler_scale = tensors
        if (coefficient.ndim != 1 or not coefficient.numel()
                or scaler_mean.shape != coefficient.shape or scaler_scale.shape != coefficient.shape
                or intercept.shape != (1,) or (scaler_scale <= 0).any()
                or not all(torch.isfinite(value).all() for value in tensors)):
            raise ValueError("Invalid numeric logistic model/scaler parameters.")
        self.seq_len, self.input_dim = int(seq_len), len(coefficient)
        self.register_buffer("coefficient", coefficient)
        self.register_buffer("intercept", intercept)
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def forward(self, raw_monthly):
        if raw_monthly.ndim != 3 or tuple(raw_monthly.shape[1:]) != (self.seq_len, self.input_dim):
            raise ValueError("Logistic input must retain the selected monthly tensor layout.")
        if not torch.isfinite(raw_monthly).all() or (raw_monthly < 0).any():
            raise ValueError("Expected finite nonnegative raw monthly counts.")
        totals = raw_monthly.to(dtype=torch.float64).sum(dim=1)
        standardized = (_experiment_log1p(totals) - self.scaler_mean) / self.scaler_scale
        return standardized @ self.coefficient + self.intercept[0]


def _experiment_forward(model, raw, kind):
    if kind == "transformer":
        return model(_experiment_log1p(raw))
    if kind == "logistic":
        return model(raw)
    raise ValueError("Unknown model kind.")


def _experiment_predict_loader(model, loader, device, kind, criterion=None):
    model.eval()
    labels, scores, loss_sum, count = [], [], 0.0, 0
    with torch.inference_mode():
        for raw, y in loader:
            raw, y = raw.to(device), y.to(device)
            logits = _experiment_forward(model, raw, kind)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite model predictions.")
            if criterion is not None:
                loss = criterion(logits, y.to(logits.dtype))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite diagnostic loss.")
                loss_sum += float(loss.item()) * len(y)
            count += len(y)
            labels.append(y.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    if not count:
        raise ValueError("Cannot score an empty split.")
    return (None if criterion is None else loss_sum / count,
            np.concatenate(labels), np.concatenate(scores))


def _experiment_topk(data, split, labels, scores):
    """Same label-independent score tie ordering as the TEST report helper."""
    rows = data["indices"][split]
    frame = data["metadata"].iloc[rows][["PATIENT_ID", "END_DT"]].copy().reset_index(drop=True)
    if len(frame) != len(labels) or len(frame) < 10:
        raise ValueError("At least ten aligned snapshots are required for top-k reporting.")
    frame["END_DT"] = pd.to_datetime(frame.END_DT).dt.strftime("%Y-%m-%d")
    frame["_score"] = np.asarray(scores)
    frame["_position"] = np.arange(len(frame))
    frame["_tie"] = [hashlib.sha256(json.dumps(
        ["tak861-targeting-v1", str(patient), str(date)], separators=(",", ":")
    ).encode("utf-8")).hexdigest() for patient, date in frame[["PATIENT_ID", "END_DT"]].itertuples(index=False, name=None)]
    order = frame.sort_values(["_score", "_tie", "PATIENT_ID", "END_DT"],
                              ascending=[False, True, True, True])._position.to_numpy()
    ranked_y = np.asarray(labels)[order]
    positives, n = int(ranked_y.sum()), len(ranked_y)
    if not 0 < positives < n:
        raise ValueError("Top-k reporting requires both classes.")
    reports = []
    for percentage in (10, 20, 30):
        selected = (percentage * n + 99) // 100
        captured = int(ranked_y[:selected].sum())
        precision = captured / selected
        reports.append({"top_k_pct": percentage, "n_selected": selected,
                        "n_resp1": captured, "precision": precision,
                        "recall": captured / positives, "lift": precision / (positives / n),
                        "actual_population_fraction": selected / n})
    return reports


def _experiment_settings(settings, kind):
    settings = dict(settings)
    for name in ("batch_size",):
        if not isinstance(settings.get(name), int) or isinstance(settings[name], bool) or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    # seed_everything performs complete seed validation before any fitting.
    if kind == "transformer":
        for name in ("epochs", "patience"):
            if not isinstance(settings.get(name), int) or isinstance(settings[name], bool) or settings[name] <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("learning_rate", "grad_clip"):
            if not np.isfinite(settings[name]) or settings[name] <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("weight_decay", "min_delta"):
            if not np.isfinite(settings[name]) or settings[name] < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")
    elif kind == "logistic":
        settings.setdefault("logistic_C", 1.0)
        settings.setdefault("logistic_max_iter", 2000)
        if not np.isfinite(settings["logistic_C"]) or settings["logistic_C"] <= 0:
            raise ValueError("logistic_C must be positive and finite.")
        if (not isinstance(settings["logistic_max_iter"], int)
                or isinstance(settings["logistic_max_iter"], bool) or settings["logistic_max_iter"] <= 0):
            raise ValueError("logistic_max_iter must be a positive integer.")
    else:
        raise ValueError("model_kind must be transformer or logistic.")
    return settings


def _experiment_class_counts(data):
    counts = {}
    for split in ("train", "validation"):
        labels = np.asarray(data["y"][data["indices"][split]])
        if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
            raise ValueError("Training/validation labels must be binary.")
        positives = int(labels.sum())
        if not 0 < positives < len(labels):
            raise ValueError(f"{split} requires both classes.")
        if len(labels) < 10:
            raise ValueError(f"{split} requires at least ten snapshots for top-k metrics.")
        counts[split] = {"snapshots": len(labels), "positives": positives,
                         "negatives": len(labels) - positives}
    return counts


def _logistic_train_matrix(data, columns, batch_size):
    """Build only TRAIN aggregate rows, never a copy of the full monthly tensor."""
    rows = data["indices"]["train"]
    out = np.empty((len(rows), len(columns)), dtype=np.float64)
    loader = _experiment_loader(data, "train", columns, batch_size, 0)
    offset = 0
    for raw, _ in loader:
        if not torch.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("Logistic training requires finite nonnegative counts.")
        values = np.log1p(raw.numpy().sum(axis=1, dtype=np.float64))
        out[offset:offset + len(values)] = values
        offset += len(values)
    return out


def run_experiment(data, manifest, model_config, settings, run_id, model_kind="transformer"):
    """Train one fixed feature/model experiment; select checkpoint/threshold on VAL."""
    manifest = copy.deepcopy(validate_feature_manifest(manifest, data))
    columns = manifest["selected_indices"]
    config = asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else dict(model_config)
    if (config.get("input_dim") != len(columns)
            or config.get("seq_len") != data["X"].shape[1]):
        raise ValueError("Model dimensions must match the selected features and full time axis.")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("A nonempty run_id is required.")
    settings = _experiment_settings(settings, model_kind)
    seed_everything(settings["seed"])
    counts = _experiment_class_counts(data)
    pos_weight = counts["train"]["negatives"] / counts["train"]["positives"]
    device = resolve_device(settings.get("device", "auto")) if model_kind == "transformer" else torch.device("cpu")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    loaders = {split: _experiment_loader(data, split, columns, settings["batch_size"], settings["seed"])
               for split in ("train", "validation")}
    history, best_ap, best_epoch, n_iterations = [], -np.inf, None, None
    if model_kind == "transformer":
        model = ClaimsTransformer(ModelConfig(**config)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        online_loader = _experiment_loader(data, "train", columns, settings["batch_size"], settings["seed"], True)
        patience_reference, without_progress, best_state = -np.inf, 0, None
        print(f"{run_id}: {len(columns)} features; device={device}; TRAIN positive weight={pos_weight:.4f}", flush=True)
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            online_sum, n = 0.0, 0
            for raw, y in online_loader:
                raw, y = raw.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(_experiment_forward(model, raw, model_kind), y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite optimization loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                online_sum += float(loss.item()) * len(y)
                n += len(y)
            train_loss, train_y, train_scores = _experiment_predict_loader(model, loaders["train"], device, model_kind, criterion)
            val_loss, val_y, val_scores = _experiment_predict_loader(model, loaders["validation"], device, model_kind, criterion)
            val_ap = float(average_precision_score(val_y, val_scores))
            history.append({"epoch": epoch, "optimization_loss": online_sum / n,
                            "training_loss": train_loss, "validation_loss": val_loss,
                            "training_average_precision": float(average_precision_score(train_y, train_scores)),
                            "training_roc_auc": float(roc_auc_score(train_y, train_scores)),
                            "validation_average_precision": val_ap,
                            "validation_roc_auc": float(roc_auc_score(val_y, val_scores))})
            if val_ap > best_ap:
                best_ap, best_epoch = val_ap, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if val_ap > patience_reference + settings["min_delta"]:
                patience_reference, without_progress = val_ap, 0
            else:
                without_progress += 1
            print(f"Epoch {epoch:02d}: eval TRAIN loss={train_loss:.5f}; VAL loss={val_loss:.5f}; VAL AP={val_ap:.5f}", flush=True)
            if without_progress >= settings["patience"]:
                print(f"Early stopping; restoring epoch {best_epoch}.", flush=True)
                break
        model.load_state_dict(best_state, strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        transform = "selected_monthly_log1p"
        preprocessing = "Selected raw monthly counts -> log1p once per batch; no fitted scaler."
    else:
        aggregate = _logistic_train_matrix(data, columns, settings["batch_size"])
        scaler = StandardScaler().fit(aggregate)
        standardized = scaler.transform(aggregate)
        estimator = LogisticRegression(C=settings["logistic_C"], class_weight="balanced", solver="lbfgs",
                                       max_iter=settings["logistic_max_iter"], random_state=settings["seed"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(standardized, data["y"][data["indices"]["train"]])
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                raise RuntimeError("Logistic regression did not converge; no checkpoint was accepted. Increase logistic_max_iter or review inputs.")
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
        if list(estimator.classes_) != [0, 1]:
            raise ValueError("Unexpected logistic class order.")
        n_iterations = int(np.max(estimator.n_iter_))
        model = NumericLogisticModel(estimator.coef_[0], estimator.intercept_, scaler.mean_, scaler.scale_, config["seq_len"])
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        parameter_count = len(columns) + 1
        transform = "selected_monthly_sum_log1p_train_standardize"
        preprocessing = "Selected monthly raw counts -> sum across months -> log1p -> StandardScaler fitted on TRAIN only; L2 logistic regression, balanced class weights. Month order is discarded."
        del aggregate, standardized, estimator, scaler
    # These final diagnostics describe the restored best model, not the last epoch.
    train_loss, train_y, train_scores = _experiment_predict_loader(model, loaders["train"], device, model_kind, criterion)
    val_loss, val_y, val_scores = _experiment_predict_loader(model, loaders["validation"], device, model_kind, criterion)
    threshold = float(select_validation_threshold(val_y, val_scores))
    best_ap = float(average_precision_score(val_y, val_scores))
    if model_kind == "logistic":
        history.append({"epoch": None, "optimization_loss": None,
                        "training_loss": train_loss, "validation_loss": val_loss,
                        "training_average_precision": float(average_precision_score(train_y, train_scores)),
                        "training_roc_auc": float(roc_auc_score(train_y, train_scores)),
                        "validation_average_precision": best_ap,
                        "validation_roc_auc": float(roc_auc_score(val_y, val_scores))})
    summary = {"run_id": run_id, "training_complete": True, "model_kind": model_kind,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "best_epoch": best_epoch, "epochs_completed": len(history) if model_kind == "transformer" else 0,
               "optimizer_iterations": n_iterations,
               "best_validation_average_precision": best_ap, "validation_threshold": threshold,
               "training_metrics": {**classification_metrics(train_y, train_scores, threshold), "loss": train_loss},
               "validation_metrics": {**classification_metrics(val_y, val_scores, threshold), "loss": val_loss},
               "training_topk": _experiment_topk(data, "train", train_y, train_scores),
               "validation_topk": _experiment_topk(data, "validation", val_y, val_scores),
               "train_pos_weight": float(pos_weight), "class_counts": counts,
               "model_parameter_count": parameter_count, "model_config": config,
               "training_settings": settings, "feature_manifest": manifest,
               "feature_manifest_sha256": manifest["manifest_sha256"],
               "selected_feature_count": len(columns), "input_hashes": copy.deepcopy(data["hashes"]),
               "resolved_device": str(device), "preprocessing": preprocessing,
               "loss_definition": "Snapshot-average binary cross-entropy weighted by TRAIN negatives/positives; TRAIN and VALIDATION diagnostic losses both use eval mode.",
               "test_inference_performed": False,
               "source_review": "Structural checks only; clinical cutoff, outcome-event exclusion and historical claims availability are not certified by these notebooks.",
               "runtime": runtime_metadata()}
    payload = {"format_version": 2, "training_complete": True, "run_id": run_id,
               "model_kind": model_kind, "model_state_dict": best_state, "model_config": config,
               "training_settings": settings, "input_hashes": copy.deepcopy(data["hashes"]),
               "feature_names": list(data["features"]), "feature_manifest": manifest,
               "time_steps": list(range(config["seq_len"])), "tensor_shape": [int(value) for value in data["X"].shape],
               "transform": transform, "validation_threshold": threshold,
               "selection_metric": "validation_average_precision", "best_epoch": best_epoch,
               "best_validation_average_precision": best_ap}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue(), summary, pd.DataFrame(history)


def _verify_experiment_payload(payload, data, run_id):
    if (not isinstance(payload, dict) or payload.get("format_version") != 2
            or payload.get("training_complete") is not True or payload.get("run_id") != run_id):
        raise ValueError("Checkpoint is incomplete, unsupported, or belongs to another run.")
    if payload.get("input_hashes") != data["hashes"]:
        raise ValueError("Full tensor content, keys, labels, feature order or frozen split changed after training.")
    if (payload.get("feature_names") != list(data["features"])
            or payload.get("time_steps") != list(range(data["X"].shape[1]))
            or payload.get("tensor_shape") != list(data["X"].shape)):
        raise ValueError("Checkpoint full tensor layout differs from current inputs.")
    manifest = validate_feature_manifest(payload.get("feature_manifest"), data)
    config = payload.get("model_config")
    if not isinstance(config, dict) or config.get("input_dim") != len(manifest["selected_indices"]) or config.get("seq_len") != data["X"].shape[1]:
        raise ValueError("Saved model dimensions disagree with the selected feature manifest.")
    threshold = payload.get("validation_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Missing valid frozen validation threshold.")
    expected_transform = {"transformer": "selected_monthly_log1p",
                          "logistic": "selected_monthly_sum_log1p_train_standardize"}
    if payload.get("model_kind") not in expected_transform or payload.get("transform") != expected_transform[payload["model_kind"]]:
        raise ValueError("Unknown or inconsistent preprocessing/model kind.")
    return manifest


def restore_experiment(blob, data, run_id, device="auto"):
    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    manifest = _verify_experiment_payload(payload, data, run_id)
    kind, state = payload["model_kind"], payload["model_state_dict"]
    if kind == "transformer":
        model = ClaimsTransformer(ModelConfig(**payload["model_config"]))
    else:
        expected = {"coefficient", "intercept", "scaler_mean", "scaler_scale"}
        if not isinstance(state, dict) or set(state) != expected or any(not torch.is_tensor(value) for value in state.values()):
            raise ValueError("Logistic checkpoint must contain only the expected numeric tensors.")
        model = NumericLogisticModel(state["coefficient"], state["intercept"], state["scaler_mean"], state["scaler_scale"], payload["model_config"]["seq_len"])
        if model.input_dim != len(manifest["selected_indices"]):
            raise ValueError("Logistic coefficients disagree with the frozen feature selection.")
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model weights.")
    resolved = resolve_device(device)
    model.to(resolved).eval()
    return model, payload, resolved


def predict_experiment(model, payload, data, split, device):
    """Explicit inference entry point; TEST is allowed here, never in training."""
    manifest = _verify_experiment_payload(payload, data, payload.get("run_id"))
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown inference split.")
    loader = _experiment_loader(data, split, manifest["selected_indices"],
                                payload["training_settings"]["batch_size"], payload["training_settings"]["seed"])
    _, labels, scores = _experiment_predict_loader(model, loader, device, payload["model_kind"])
    return labels, scores
