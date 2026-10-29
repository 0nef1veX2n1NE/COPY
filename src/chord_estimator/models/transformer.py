from __future__ import annotations

import math
import torch
from torch import nn

from ..config import ModelConfig


def sinusoidal_positions(length: int, width: int, device, dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / width))
    result = torch.zeros(length, width, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * scale)
    result[:, 1::2] = torch.cos(position * scale[: result[:, 1::2].shape[1]])
    return result.to(dtype=dtype)


class ChordTransformer(nn.Module):
    """Non-causal (bidirectional-context) Transformer frame classifier."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.cnn_frontend:
            self.frontend = nn.Sequential(
                nn.Conv1d(config.input_bins, config.d_model, 5, padding=2), nn.GELU(),
                nn.Conv1d(config.d_model, config.d_model, 3, padding=1), nn.GELU(),
            )
            self.input_projection = None
        else:
            self.frontend = None
            self.input_projection = nn.Linear(config.input_bins, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers, norm=nn.LayerNorm(config.d_model))
        self.chord_head = nn.Linear(config.d_model, config.num_classes)
        self.boundary_head = nn.Linear(config.d_model, 1) if config.boundary_head else None

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, frequency]")
        if self.frontend is not None:
            hidden = self.frontend(features.transpose(1, 2)).transpose(1, 2)
        else:
            hidden = self.input_projection(features)
        hidden = hidden + sinusoidal_positions(hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype).unsqueeze(0)
        padding_mask = ~valid_mask if valid_mask is not None else None
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        output = {"chord_logits": self.chord_head(hidden)}
        if self.boundary_head is not None:
            output["boundary_logits"] = self.boundary_head(hidden).squeeze(-1)
        return output

