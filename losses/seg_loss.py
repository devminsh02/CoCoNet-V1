from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.iou_weight = float(iou_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = 1.0 - (intersection + 1.0) / (union + 1.0)
        return self.bce_weight * bce + self.iou_weight * iou.mean()
