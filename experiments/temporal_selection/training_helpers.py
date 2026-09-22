"""Notebook-local training; consumes the verified memory-mapped tensor."""
from dataclasses import asdict
from datetime import datetime, timezone

from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, data, split):
        self.data = data
        self.indices = data["indices"][split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        x = torch.from_numpy(np.array(self.data["X"][row], dtype=np.float32, copy=True))
        return x, torch.tensor(float(self.data["y"][row]), dtype=torch.float32)


def make_loader(data, split, batch_size, seed, shuffle=False):
    return DataLoader(SequenceDataset(data, split), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, drop_last=False,
                      generator=torch.Generator().manual_seed(seed))


def checked_log1p(x):
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError("Expected finite nonnegative raw counts before log1p.")
    return torch.log1p(x)


def predict_loader(model, loader, device, criterion=None):
    model.eval()
    labels, scores, total_loss, count = [], [], 0.0, 0
    with torch.inference_mode():
        for x, y in loader:
            x = checked_log1p(x.to(device))
            y = y.to(device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite model predictions.")
            if criterion is not None:
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite validation loss.")
                total_loss += float(loss.item()) * len(y)
            count += len(y)
            labels.append(y.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return total_loss / count, np.concatenate(labels), np.concatenate(scores)


def train_run(data, model_config, settings, run_id):
    if tuple(data["X"].shape[1:]) != (model_config.seq_len, model_config.input_dim):
        raise ValueError("Architecture and tensor dimensions disagree.")
    for name in ("epochs", "patience", "batch_size"):
        if not isinstance(settings[name], int) or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("learning_rate", "grad_clip"):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    for name in ("weight_decay", "min_delta"):
        if not np.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f"{name} must be nonnegative and finite.")
    seed_everything(settings["seed"])
    device = resolve_device(settings["device"])
    class_counts = {}
    for name in ("train", "validation"):
        y = data["y"][data["indices"][name]]
        positives = int(y.sum())
        negatives = len(y) - positives
        if not positives or not negatives:
            raise ValueError(f"{name} requires both classes.")
        class_counts[name] = {"snapshots": len(y), "positives": positives, "negatives": negatives}
    pos_weight = class_counts["train"]["negatives"] / class_counts["train"]["positives"]
    model = ClaimsTransformer(model_config).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    train_loader = make_loader(data, "train", settings["batch_size"], settings["seed"], True)
    validation_loader = make_loader(data, "validation", settings["batch_size"], settings["seed"])
    diagnostic_loader = make_loader(data, "train", settings["batch_size"], settings["seed"])
    best_ap, patience_reference = -np.inf, -np.inf
    without_progress, best_epoch, best_state = 0, None, None
    history = []
    print(f"Training on {device}; TRAIN positive weight = {pos_weight:.4f}.", flush=True)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for x, y in train_loader:
            x, y = checked_log1p(x.to(device)), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            count += len(y)
        val_loss, val_y, val_scores = predict_loader(model, validation_loader, device, criterion)
        val_ap = float(average_precision_score(val_y, val_scores))
        train_loss, train_y, train_scores = predict_loader(model, diagnostic_loader, device, criterion)
        train_lift = top10_lift(train_y, train_scores, data["metadata"].iloc[data["indices"]["train"]])
        val_lift = top10_lift(val_y, val_scores, data["metadata"].iloc[data["indices"]["validation"]])
        history.append({"epoch": epoch, "optimization_loss": loss_sum / count,
                        "training_loss": train_loss, "training_top10_lift": train_lift,
                        "validation_top10_lift": val_lift,
                        "training_average_precision": float(average_precision_score(train_y, train_scores)),
                        "validation_loss": val_loss, "validation_average_precision": val_ap,
                        "validation_roc_auc": float(roc_auc_score(val_y, val_scores))})
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_ap > patience_reference + settings["min_delta"]:
            patience_reference, without_progress = val_ap, 0
        else:
            without_progress += 1
        print(f"Epoch {epoch:02d}: train loss={loss_sum/count:.5f}; "
              f"validation loss={val_loss:.5f}; validation AP={val_ap:.5f}; "
              f"TRAIN lift={train_lift:.3f}; VALIDATION lift={val_lift:.3f}", flush=True)
        if without_progress >= settings["patience"]:
            print(f"Early stopping; restoring epoch {best_epoch}.", flush=True)
            break
    model.load_state_dict(best_state, strict=True)
    _, val_y, val_scores = predict_loader(model, validation_loader, device, criterion)
    threshold = select_validation_threshold(val_y, val_scores)
    summary = {"run_id": run_id, "training_complete": True,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "best_epoch": int(best_epoch), "epochs_completed": len(history),
               "best_validation_average_precision": best_ap,
               "validation_threshold": threshold,
               "validation_metrics": classification_metrics(val_y, val_scores, threshold),
               "train_pos_weight": float(pos_weight), "class_counts": class_counts,
               "model_parameter_count": sum(p.numel() for p in model.parameters()),
               "model_config": asdict(model_config), "training_settings": dict(settings),
               "input_hashes": data["hashes"], "resolved_device": str(device),
               "preprocessing": "per-batch log1p of raw counts; no fitted preprocessing",
               "test_inference_performed": False,
               "source_review": "Structural checks only; clinical cutoff, outcome-event exclusion and historical claims availability are not certified by these notebooks.",
               "runtime": runtime_metadata()}
    payload = {"format_version": 1, "training_complete": True, "run_id": run_id,
               "model_state_dict": best_state, "model_config": asdict(model_config),
               "training_settings": dict(settings), "input_hashes": data["hashes"],
               "feature_names": list(data["features"]), "time_steps": list(range(model_config.seq_len)),
               "tensor_shape": [int(v) for v in data["X"].shape],
               "transform": "log1p", "validation_threshold": float(threshold),
               "selection_metric": "validation_average_precision", "best_epoch": int(best_epoch),
               "best_validation_average_precision": best_ap}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue(), summary, pd.DataFrame(history)


def load_verified_model(blob, data, run_id, device="auto"):
    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format_version") != 1
            or payload.get("training_complete") is not True or payload.get("run_id") != run_id):
        raise ValueError("Checkpoint is incomplete, unsupported, or belongs to another run.")
    if payload.get("input_hashes") != data["hashes"]:
        raise ValueError("Tensor content, keys, labels, feature order or frozen split changed after training.")
    if (payload.get("feature_names") != data["features"]
            or payload.get("time_steps") != list(range(data["X"].shape[1]))
            or payload.get("tensor_shape") != list(data["X"].shape)):
        raise ValueError("Checkpoint tensor layout differs from current inputs.")
    threshold = payload.get("validation_threshold")
    if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Missing valid frozen validation threshold.")
    if payload.get("transform") != "log1p":
        raise ValueError("Unknown preprocessing; evaluation stopped.")
    model = ClaimsTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    resolved = resolve_device(device)
    model.to(resolved).eval()
    return model, payload, resolved
