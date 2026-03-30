from __future__ import annotations

import torch
import torch.nn as nn

from utils.common import MLP


class AffinityHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge_mlp = MLP(in_dim=channels * 2 + 2, hidden_dim=channels, out_dim=1)

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        x_src = tokens[:, src, :]
        x_dst = tokens[:, dst, :]
        delta = coords[:, dst, :] - coords[:, src, :]
        return self.edge_mlp(torch.cat([x_src, x_dst, delta], dim=-1))
