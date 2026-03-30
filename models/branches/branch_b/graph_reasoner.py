from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from utils.common import MLP


class GraphMessagePassingLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.msg_mlp = MLP(in_dim=channels * 2 + 2, hidden_dim=channels, out_dim=channels)
        self.upd_mlp = MLP(in_dim=channels * 2, hidden_dim=channels, out_dim=channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, edge_valid_mask: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = x.shape
        src_index = edge_index[0]
        dst_index = edge_index[1]
        out = []
        for b in range(bsz):
            xb = x[b]
            cb = coords[b]
            valid_edges = edge_valid_mask[b]
            if not bool(valid_edges.any()):
                out.append(xb)
                continue
            src = src_index[valid_edges]
            dst = dst_index[valid_edges]
            with torch.amp.autocast(device_type=xb.device.type, enabled=False):
                xb_fp32 = xb.float()
                cb_fp32 = cb.float()
                x_src = xb_fp32[src]
                x_dst = xb_fp32[dst]
                delta = cb_fp32[dst] - cb_fp32[src]
                msg = self.msg_mlp(torch.cat([x_src, x_dst, delta], dim=-1))
                agg = torch.zeros((num_tokens, channels), device=xb.device, dtype=torch.float32)
                agg.index_add_(0, dst, msg)
                upd = self.upd_mlp(torch.cat([xb_fp32, agg], dim=-1))
                out_b = self.norm(xb_fp32 + upd)
            out.append(out_b.to(dtype=xb.dtype))
        return torch.stack(out, dim=0)


class GraphReasoner(nn.Module):
    def __init__(self, channels: int = 256, k_neighbors: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.layers = nn.ModuleList([GraphMessagePassingLayer(channels) for _ in range(num_layers)])

    def _build_knn_graph(self, coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            coords_fp32 = coords.float()
            dist = torch.cdist(coords_fp32, coords_fp32, p=2)
            n = dist.size(0)
            dist = dist + torch.eye(n, device=coords.device, dtype=dist.dtype) * 1e6
            k = min(self.k_neighbors, max(1, n - 1))
            nn_idx = torch.topk(dist, k=k, largest=False).indices
            src = torch.arange(n, device=coords.device).unsqueeze(1).repeat(1, k).reshape(-1)
            dst = nn_idx.reshape(-1)
            return torch.stack([src, dst], dim=0)

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor, token_valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        edge_index = self._build_knn_graph(coords[0])
        src = edge_index[0]
        dst = edge_index[1]
        edge_valid_mask = token_valid_mask[:, src] & token_valid_mask[:, dst]
        x = tokens
        for layer in self.layers:
            x = layer(x, coords, edge_index, edge_valid_mask)
        return {'tokens': x, 'edge_index': edge_index, 'edge_valid_mask': edge_valid_mask}
