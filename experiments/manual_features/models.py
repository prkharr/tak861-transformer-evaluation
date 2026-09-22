"""Snapshot-feature models. Fit APIs deliberately accept TRAIN and VALIDATION only."""
import copy
import io
import os
import random
import platform
import time
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
import sklearn


def default_recipes():
    """One predeclared Transformer; no model-family or architecture search."""
    return [{"name": "transformer_dropout", "kind": "transformer", "settings": {
        "width": 64, "heads": 4, "layers": 2, "feedforward": 128,
        "dropout": .35, "weight_decay": .0001, "learning_rate": .0005, "batch_size": 128}}]


def seed_selected(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def selection_key(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 1 or y.shape != p.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Selection requires aligned binary labels with both classes.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid selection probabilities.")
    # Input rows have canonical patient/date order. Stable sorting breaks ties by that order.
    order = np.argsort(-p, kind="stable")
    k = max(1, int(np.ceil(.10 * len(y))))
    lift = float(y[order[:k]].mean() / y.mean())
    return lift, float(average_precision_score(y, p))


class FeatureTokenTransformer(nn.Module):
    def __init__(self, features, settings):
        super().__init__()
        width = settings["width"]
        self.features = features
        # Each token has a learned feature identity plus its standardized numeric value.
        self.value_weight = nn.Parameter(torch.randn(features, width) * .02)
        self.feature_bias = nn.Parameter(torch.randn(features, width) * .02)
        self.missing_embedding = nn.Parameter(torch.randn(features, width) * .02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=settings["heads"],
                    dim_feedforward=settings["feedforward"], dropout=settings["dropout"],
                    batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=settings["layers"], enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, values, missing):
        if values.ndim != 2 or values.shape[1] != self.features or values.shape != missing.shape:
            raise ValueError("Expected one value and missingness flag per selected feature.")
        tokens = values.unsqueeze(-1) * self.value_weight + self.feature_bias
        tokens = tokens + missing.unsqueeze(-1) * self.missing_embedding
        sequence = torch.cat([self.cls_token.expand(len(values), -1, -1), tokens], dim=1)
        return self.output(self.encoder(sequence)[:, 0]).squeeze(-1)


def network_scores(network, X, missing, device, batch_size=512):
    network.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            values = torch.as_tensor(X[start:start+batch_size], dtype=torch.float32, device=device)
            mask = torch.as_tensor(missing[start:start+batch_size], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(network(values, mask)).cpu().numpy())
    return np.concatenate(scores).astype(float)


def weighted_probability_loss(y, p, positive_weight):
    p = np.clip(p, 1e-7, 1-1e-7)
    return float(np.mean(-positive_weight * y * np.log(p) - (1-y) * np.log1p(-p)))


def fit_candidate(recipe, X_train, missing_train, y_train, X_val, missing_val, y_val,
                  seed=42, max_epochs=30, patience=6):
    for X, mask, y in ((X_train, missing_train, y_train), (X_val, missing_val, y_val)):
        if X.ndim != 2 or X.shape != mask.shape or len(X) != len(y) or set(np.unique(y)) != {0, 1}:
            raise ValueError("Invalid partition dimensions or binary labels.")
        if not np.isfinite(X).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError("Invalid preprocessed values or missingness flags.")
    if X_train.shape[1] != X_val.shape[1] or max_epochs < 1 or patience < 1:
        raise ValueError("Invalid model dimensions or training limits.")
    seed_selected(seed)
    started = time.time()
    positive_weight = float((len(y_train)-y_train.sum()) / y_train.sum())
    result = {"recipe": copy.deepcopy(recipe), "n_features": X_train.shape[1], "seed": seed,
              "history": [], "positive_weight": positive_weight,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "sklearn": sklearn.__version__, "torch": str(torch.__version__)}}
    settings = recipe["settings"]
    if recipe["kind"] != "transformer":
        raise ValueError("Only the Transformer is supported.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result["runtime"]["device"] = str(device)
    network = FeatureTokenTransformer(X_train.shape[1], settings).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                   torch.tensor(missing_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True,
                   num_workers=0, generator=torch.Generator().manual_seed(seed))
    best_key, best_state, stale = (-np.inf, -np.inf), None, 0
    for epoch in range(1, max_epochs+1):
        network.train()
        loss_sum = 0.0
        for values, mask, labels in loader:
            values, mask, labels = values.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(values, mask), labels)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite.")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels)
        train_scores = network_scores(network, X_train, missing_train, device)
        val_scores = network_scores(network, X_val, missing_val, device)
        train_key, val_key = selection_key(y_train, train_scores), selection_key(y_val, val_scores)
        result["history"].append({"epoch": epoch, "train_batch_loss": loss_sum/len(y_train),
             "train_loss": weighted_probability_loss(y_train, train_scores, positive_weight),
             "validation_loss": weighted_probability_loss(y_val, val_scores, positive_weight),
             "train_top10_lift": train_key[0], "train_ap": train_key[1],
             "validation_top10_lift": val_key[0], "validation_ap": val_key[1]})
        print(f"{recipe['name']} epoch {epoch}: TRAIN lift={train_key[0]:.3f}; VALIDATION lift={val_key[0]:.3f}, AP={val_key[1]:.4f}", flush=True)
        if val_key > best_key:
            best_key = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
            result["best_epoch"] = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    result["state_dict"] = best_state
    del network, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result["seconds"] = time.time()-started
    return result


def predict_candidate(result, X, missing):
    if X.ndim != 2 or X.shape != missing.shape or X.shape[1] != result["n_features"]:
        raise ValueError("Prediction feature dimensions changed.")
    if not np.isfinite(X).all() or not np.isin(missing, [0, 1]).all():
        raise ValueError("Invalid prediction inputs.")
    if result["recipe"]["kind"] == "transformer":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = FeatureTokenTransformer(result["n_features"], result["recipe"]["settings"]).to(device)
        network.load_state_dict(result["state_dict"])
        p = network_scores(network, X, missing, device)
        del network
    else:
        raise ValueError("Only the Transformer is supported.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid predicted probabilities.")
    return p


def dump_candidate(result):
    buffer = io.BytesIO()
    joblib.dump(result, buffer, compress=3)
    return buffer.getvalue()


def load_candidate(blob):
    # Only load this suite's own checksum-verified, provenance-checked warehouse artifacts.
    # Joblib is pickle based: never use this function for an external/untrusted model file.
    return joblib.load(io.BytesIO(blob))
