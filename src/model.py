"""Small sequence classifier; zero-activity months remain real timesteps."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    seq_len: int = 12
    d_model: int = 128
    n_heads: int = 4
    encoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.2

    def __post_init__(self):
        sizes = (self.input_dim, self.seq_len, self.d_model, self.n_heads,
                 self.encoder_layers, self.feedforward_dim)
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in sizes):
            raise ValueError("All model dimensions must be positive integers.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


class ClaimsTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.input_dim, config.d_model)
        self.position = nn.Parameter(torch.empty(1, config.seq_len, config.d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.feedforward_dim, dropout=config.dropout,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.encoder_layers, enable_nested_tensor=False,
        )
        # TransformerEncoder clones the initial layer; initialize matrix weights
        # independently so the layers do not begin with identical weights.
        for encoder_layer in self.encoder.layers:
            for parameter in encoder_layer.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, 64), nn.ReLU(),
            nn.Dropout(config.dropout), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.config.seq_len, self.config.input_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected[0]}, {expected[1]}] input.")
        if not x.is_floating_point():
            raise TypeError("Transformer input must be floating point.")
        # A zero month is observed absence of activity, not padding. A causal
        # mask is unnecessary because every included month precedes the cutoff.
        hidden = self.projection(x) + self.position
        pooled = self.norm(self.encoder(hidden)).mean(dim=1)
        return self.head(pooled).squeeze(-1)
