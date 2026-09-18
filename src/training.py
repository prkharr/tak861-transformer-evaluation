"""Train on TRAIN/VALIDATION, then separately score the frozen checkpoint.

No TEST predictions or performance metrics are calculated during training. The
full bundle and manifest hashes bind the checkpoint to the exact frozen inputs;
these are integrity checks, not model-selection statistics.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from targeting_evaluation import manifest_fingerprint
from .data_utils import SequenceBundle, bundle_fingerprint, split_indices, validate_bundle
from .metrics import classification_metrics, select_validation_threshold
from .model import ClaimsTransformer, ModelConfig
from .reproducibility import resolve_device, runtime_metadata, seed_everything, seed_worker


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    patience: int = 5
    min_delta: float = 1e-4
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    transform: str = "log1p"
    device: str = "auto"
    num_workers: int = 0
    print_progress: bool = True

    def __post_init__(self):
        for name in ("epochs", "patience", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        if not isinstance(self.print_progress, bool):
            raise ValueError("print_progress must be a boolean.")
        for name in ("min_delta", "weight_decay"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("learning_rate", "grad_clip"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.transform not in {"log1p", "none"}:
            raise ValueError("transform must be 'log1p' or 'none'.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")


class _SequenceRows(Dataset):
    def __init__(self, bundle: SequenceBundle, indices: np.ndarray):
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        # Copy one row, not the whole tensor; this also supports read-only mmap.
        x = torch.from_numpy(np.array(self.bundle.X[row], dtype=np.float32, copy=True))
        return x, torch.tensor(float(self.bundle.y[row]), dtype=torch.float32)


def _loader(bundle, indices, config, shuffle=False, device=None):
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        _SequenceRows(bundle, indices), batch_size=config.batch_size,
        shuffle=shuffle, num_workers=config.num_workers, generator=generator,
        worker_init_fn=seed_worker, pin_memory=device is not None and device.type == "cuda",
        drop_last=False,
    )


def _transform_batch(x, transform):
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite float32 input; check count magnitudes and source tensor.")
    if transform == "log1p":
        if (x < 0).any():
            raise ValueError("log1p count preprocessing requires nonnegative inputs.")
        return torch.log1p(x)
    return x


def _evaluate(model, loader, criterion, device, transform):
    model.eval()
    loss_sum, count = 0.0, 0
    labels, probabilities = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = _transform_batch(x.to(device), transform)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite validation loss; training stopped.")
            loss_sum += float(loss.item()) * len(y)
            count += len(y)
            labels.append(y.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return loss_sum / count, np.concatenate(labels), np.concatenate(probabilities)


def _save_checkpoint(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_transformer(
    bundle: SequenceBundle,
    manifest: pd.DataFrame,
    output_dir: Union[str, Path],
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainConfig] = None,
) -> dict:
    """Train a new run; fail rather than overwrite an existing run directory.

    Early stopping monitors VALIDATION average precision (noninterpolated AP).
    The highest observed AP is checkpointed; min_delta governs patience only.
    Any equal-AP checkpoint tie retains the earliest epoch. The restored model's
    VALIDATION predictions choose a single F1 threshold, never TEST predictions.
    """
    validate_bundle(bundle)
    indices = split_indices(bundle, manifest)
    config = train_config or TrainConfig()
    architecture = model_config or ModelConfig(input_dim=bundle.X.shape[2], seq_len=bundle.X.shape[1])
    if (architecture.seq_len, architecture.input_dim) != tuple(bundle.X.shape[1:]):
        raise ValueError("Model dimensions do not match the tensor's time/feature axes.")
    class_counts = {}
    for split in ("TRAIN", "VALIDATION"):
        y = np.asarray(bundle.y)[indices[split]]
        positive = int(y.sum())
        negative = len(y) - positive
        if not positive or not negative:
            raise ValueError(f"{split} must contain both response classes.")
        class_counts[split] = {"n_snapshots": len(y), "n_resp0": negative, "n_resp1": positive}
    pos_weight = class_counts["TRAIN"]["n_resp0"] / class_counts["TRAIN"]["n_resp1"]
    data_hash = bundle_fingerprint(bundle)
    manifest_hash = manifest_fingerprint(manifest)
    device = resolve_device(config.device)
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Run directory must be new or empty; choose a fresh run name.")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best_transformer.pt"
    history_path = destination / "training_history.csv"
    metadata_path = destination / "training_metadata.json"

    seed_everything(config.seed)
    model = ClaimsTransformer(architecture).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = _loader(bundle, indices["TRAIN"], config, shuffle=True, device=device)
    validation_loader = _loader(bundle, indices["VALIDATION"], config, device=device)
    payload = {
        "format_version": 1, "training_complete": False,
        "model_config": asdict(architecture), "train_config": asdict(config),
        "bundle_sha256": data_hash, "snapshot_manifest_sha256": manifest_hash,
        "feature_names": [str(name) for name in bundle.feature_names],
        "time_steps": [int(step) for step in bundle.time_steps],
        "tensor_shape": [int(n) for n in bundle.X.shape],
        "transform": config.transform, "train_pos_weight": float(pos_weight),
        "training_split": "TRAIN", "tuning_split": "VALIDATION",
        "selection_metric": "validation_average_precision",
        "validation_threshold": None,
    }
    history = []
    best_ap, patience_reference = -np.inf, -np.inf
    epochs_without_progress = 0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum, n_train = 0.0, 0
        for x, y in train_loader:
            x = _transform_batch(x.to(device), config.transform)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; training stopped.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            n_train += len(y)
        validation_loss, validation_y, validation_scores = _evaluate(
            model, validation_loader, criterion, device, config.transform,
        )
        validation_ap = float(average_precision_score(validation_y, validation_scores))
        history.append({
            "epoch": epoch, "training_loss": train_loss_sum / n_train,
            "validation_loss": validation_loss,
            "validation_average_precision": validation_ap,
            "validation_roc_auc": float(roc_auc_score(validation_y, validation_scores)),
        })
        if validation_ap > best_ap:
            best_ap = validation_ap
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            payload.update(model_state_dict=best_state, best_epoch=epoch, best_validation_average_precision=best_ap)
            _save_checkpoint(payload, checkpoint_path)
        if validation_ap > patience_reference + config.min_delta:
            patience_reference = validation_ap
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1
        pd.DataFrame(history).to_csv(history_path, index=False)
        if config.print_progress:
            row = history[-1]
            print(
                f"Epoch {epoch:02d}/{config.epochs}: "
                f"train loss={row['training_loss']:.5f}; "
                f"validation loss={validation_loss:.5f}; "
                f"validation AP={validation_ap:.5f}; "
                f"validation ROC-AUC={row['validation_roc_auc']:.5f}",
                flush=True,
            )
        if epochs_without_progress >= config.patience:
            if config.print_progress:
                print(f"Early stopping; restoring epoch {payload['best_epoch']}.", flush=True)
            break

    model.load_state_dict(best_state, strict=True)
    _, validation_y, validation_scores = _evaluate(model, validation_loader, criterion, device, config.transform)
    threshold = select_validation_threshold(validation_y, validation_scores)
    payload.update(validation_threshold=threshold, training_complete=True)
    _save_checkpoint(payload, checkpoint_path)
    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": asdict(architecture), "train_config": asdict(config),
        "tensor_shape": payload["tensor_shape"], "feature_count": architecture.input_dim,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "class_counts": class_counts, "train_pos_weight": float(pos_weight),
        "best_epoch": payload["best_epoch"], "epochs_completed": len(history),
        "best_validation_average_precision": best_ap,
        "validation_threshold": threshold,
        "threshold_selection": "Maximum VALIDATION F1; exact ties choose the highest threshold.",
        "validation_metrics": classification_metrics(validation_y, validation_scores, threshold),
        "snapshot_manifest_sha256": manifest_hash, "bundle_sha256": data_hash,
        "training_split": "TRAIN", "tuning_split": "VALIDATION",
        "test_inference_performed": False, "resolved_device": str(device),
        "preprocessing": "Per-batch log1p with no fitted parameters." if config.transform == "log1p" else "No feature transformation.",
        "probability_caveat": "Positive-class-weighted BCE sigmoid scores are ranking scores; calibration is not established.",
        "runtime": runtime_metadata(),
    }
    metadata_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "checkpoint_path": checkpoint_path, "history_path": history_path,
        "metadata_path": metadata_path, "summary": summary,
    }


def load_trained_model(checkpoint_path: Union[str, Path], device: str = "auto"):
    """Load only a completed state-dict checkpoint with restricted unpickling."""
    resolved = resolve_device(device)
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported Transformer checkpoint format.")
    if payload.get("training_complete") is not True:
        raise ValueError("Training did not finish; checkpoint is not ready for scoring.")
    threshold = payload.get("validation_threshold")
    if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Checkpoint lacks a valid fixed VALIDATION threshold.")
    if payload.get("transform") not in {"log1p", "none"}:
        raise ValueError("Checkpoint has an unknown feature transformation.")
    model = ClaimsTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved).eval()
    return model, payload


def score_split(
    bundle: SequenceBundle,
    manifest: pd.DataFrame,
    checkpoint_path: Union[str, Path],
    split: str = "TEST",
    device: str = "auto",
) -> pd.DataFrame:
    """Score one unchanged split without changing weights or choosing a threshold."""
    if split not in {"TRAIN", "VALIDATION", "TEST"}:
        raise ValueError("split must be TRAIN, VALIDATION, or TEST.")
    validate_bundle(bundle)
    indices = split_indices(bundle, manifest)
    model, payload = load_trained_model(checkpoint_path, device=device)
    if manifest_fingerprint(manifest) != payload["snapshot_manifest_sha256"]:
        raise ValueError("Frozen manifest differs from the training manifest.")
    if bundle_fingerprint(bundle) != payload["bundle_sha256"]:
        raise ValueError("Tensor/labels/row metadata differ from the training bundle.")
    if list(bundle.feature_names) != payload["feature_names"] or list(bundle.time_steps) != payload["time_steps"]:
        raise ValueError("Feature order or time order differs from the checkpoint.")
    if list(bundle.X.shape) != payload["tensor_shape"]:
        raise ValueError("Tensor shape differs from the checkpoint.")
    if tuple(bundle.X.shape[1:]) != (model.config.seq_len, model.config.input_dim):
        raise ValueError("Checkpoint architecture does not match the tensor.")
    config = TrainConfig(**payload["train_config"])
    resolved = next(model.parameters()).device
    loader = _loader(bundle, indices[split], config, device=resolved)
    probabilities = []
    with torch.inference_mode():
        for x, _ in loader:
            x = _transform_batch(x.to(resolved), payload["transform"])
            probabilities.append(torch.sigmoid(model(x)).cpu().numpy())
    scores = np.concatenate(probabilities)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction scores; evaluation stopped.")
    rows = indices[split]
    output = bundle.snapshots.iloc[rows][["PATIENT_ID", "END_DT"]].copy().reset_index(drop=True)
    output["P_RESP1"] = scores
    output["RESP"] = np.asarray(bundle.y)[rows].astype(np.int64)
    output["SPLIT"] = split
    return output
