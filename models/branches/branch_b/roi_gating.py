from __future__ import annotations

import torch
import torch.nn as nn

from utils.common import ConvBNReLU


class SoftROIGating(nn.Module):
    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        self.map_encoder = nn.Sequential(
            ConvBNReLU(3, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, objectness_map: torch.Tensor, uncertainty_map: torch.Tensor, boundary_prior: torch.Tensor) -> torch.Tensor:
        gate_logits = self.map_encoder(torch.cat([objectness_map, uncertainty_map, boundary_prior], dim=1))
        return torch.sigmoid(gate_logits)
