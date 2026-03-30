from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologySurrogateLoss(nn.Module):
    """
    Closure-aware topology surrogate.

    The current project does not carry explicit topology labels, so this loss
    supervises the closure map with the boundary target and additionally keeps
    the refined boundary and closure responses consistent inside a dilated
    boundary band.
    """

    def __init__(self, band_kernel: int = 7, consistency_weight: float = 0.35) -> None:
        super().__init__()
        k = max(1, int(band_kernel))
        self.band_kernel = k if k % 2 == 1 else k + 1
        self.consistency_weight = float(consistency_weight)

    def _make_band(self, boundary_target: torch.Tensor) -> torch.Tensor:
        pad = self.band_kernel // 2
        return F.max_pool2d(boundary_target.float(), kernel_size=self.band_kernel, stride=1, padding=pad)

    def _weighted_bce(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
        return (loss * weight).sum() / weight.sum().clamp_min(1e-6)

    def _weighted_soft_dice(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits.float())
        target = target.float()
        weight = weight.float()
        intersection = (prob * target * weight).sum(dim=(2, 3))
        union = (prob * weight).sum(dim=(2, 3)) + (target * weight).sum(dim=(2, 3))
        return 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()

    def forward(
        self,
        boundary_logits: torch.Tensor | None,
        boundary_target: torch.Tensor,
        closure_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if closure_logits is None:
            return boundary_target.new_tensor(0.0)

        target = boundary_target.float()
        band = self._make_band(target)
        closure_weight = 0.5 + 1.5 * band
        loss = self._weighted_bce(closure_logits, target, closure_weight)
        loss = loss + self._weighted_soft_dice(closure_logits, target, closure_weight)

        if boundary_logits is not None:
            boundary_prob = torch.sigmoid(boundary_logits.float())
            closure_prob = torch.sigmoid(closure_logits.float())
            consistency_weight = 0.25 + band
            consistency = (torch.abs(boundary_prob - closure_prob) * consistency_weight).sum() / consistency_weight.sum().clamp_min(1e-6)
            loss = loss + self.consistency_weight * consistency
        return loss
