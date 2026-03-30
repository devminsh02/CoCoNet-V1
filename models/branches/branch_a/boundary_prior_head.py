from __future__ import annotations

import torch
import torch.nn as nn

from utils.common import ConvBNReLU


class BoundaryPriorHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvBNReLU(channels, channels), nn.Conv2d(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
