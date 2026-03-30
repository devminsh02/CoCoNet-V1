from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffinityLoss(nn.Module):
    """
    Mask-only surrogate affinity supervision for Branch B.

    Because the current pipeline does not load pair-level contour annotations,
    this loss derives soft edge labels from the boundary target itself.

    The target for an edge is built from:
      1) soft boundary support at the source/destination token locations
      2) soft boundary support sampled along the line segment between the tokens

    This is intentionally conservative: it enables full-option experiments
    without requiring instance/edge GT reintegration.
    """

    def __init__(
        self,
        line_samples: int = 7,
        boundary_pool_kernel: int = 7,
        target_threshold: float = 0.08,
        pos_weight: float = 2.0,
        regression_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.line_samples = max(3, int(line_samples))
        k = max(1, int(boundary_pool_kernel))
        self.boundary_pool_kernel = k if k % 2 == 1 else k + 1
        self.target_threshold = float(target_threshold)
        self.pos_weight = float(pos_weight)
        self.regression_weight = float(regression_weight)

    def _coords_to_grid(self, coords_yx: torch.Tensor) -> torch.Tensor:
        coords_yx = coords_yx.float().clamp(0.0, 1.0)
        return torch.stack([coords_yx[..., 1] * 2.0 - 1.0, coords_yx[..., 0] * 2.0 - 1.0], dim=-1)

    def _sample_support(self, support_map: torch.Tensor, coords_yx: torch.Tensor) -> torch.Tensor:
        grid = self._coords_to_grid(coords_yx).unsqueeze(2)
        sampled = F.grid_sample(support_map.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return sampled.squeeze(1).squeeze(-1)

    def _sample_line_support(self, support_map: torch.Tensor, src_coords: torch.Tensor, dst_coords: torch.Tensor) -> torch.Tensor:
        steps = torch.linspace(0.0, 1.0, steps=self.line_samples, device=support_map.device, dtype=support_map.dtype)
        steps = steps.view(1, 1, self.line_samples, 1)
        line_coords = src_coords.unsqueeze(2) * (1.0 - steps) + dst_coords.unsqueeze(2) * steps
        grid = self._coords_to_grid(line_coords)
        sampled = F.grid_sample(support_map.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return sampled.squeeze(1).mean(dim=-1)

    def _build_soft_targets(
        self,
        boundary_target: torch.Tensor,
        fragment_coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        pad = self.boundary_pool_kernel // 2
        soft_boundary = F.max_pool2d(boundary_target.float(), kernel_size=self.boundary_pool_kernel, stride=1, padding=pad)
        token_support = self._sample_support(soft_boundary, fragment_coords)
        src = edge_index[0]
        dst = edge_index[1]
        src_coords = fragment_coords[:, src, :]
        dst_coords = fragment_coords[:, dst, :]
        src_support = token_support[:, src]
        dst_support = token_support[:, dst]
        endpoint_support = torch.sqrt(torch.clamp(src_support * dst_support, min=0.0))
        line_support = self._sample_line_support(soft_boundary, src_coords, dst_coords)
        edge_target = 0.5 * line_support + 0.5 * endpoint_support
        edge_target = torch.where(edge_target >= self.target_threshold, edge_target, torch.zeros_like(edge_target))
        return edge_target.clamp(0.0, 1.0)

    def forward(
        self,
        affinity_logits: torch.Tensor | None,
        boundary_target: torch.Tensor,
        fragment_coords: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_valid_mask: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if affinity_logits is None or fragment_coords is None or edge_index is None:
            return boundary_target.new_tensor(0.0)

        logits = affinity_logits.float()
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        if logits.dim() != 2:
            raise ValueError(f'Expected affinity logits with shape [B, E] or [B, E, 1], got {tuple(affinity_logits.shape)}')

        target = self._build_soft_targets(boundary_target=boundary_target, fragment_coords=fragment_coords, edge_index=edge_index).detach()
        valid = torch.ones_like(target, dtype=torch.bool)
        if edge_valid_mask is not None:
            valid = valid & edge_valid_mask.bool()
        if token_valid_mask is not None:
            src = edge_index[0]
            dst = edge_index[1]
            valid = valid & token_valid_mask[:, src].bool() & token_valid_mask[:, dst].bool()
        if not bool(valid.any()):
            return boundary_target.new_tensor(0.0)

        weights = (1.0 + self.pos_weight * target).detach()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        bce = (bce * weights * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

        prob = torch.sigmoid(logits)
        reg = F.smooth_l1_loss(prob[valid], target[valid], reduction='mean')
        return bce + self.regression_weight * reg
