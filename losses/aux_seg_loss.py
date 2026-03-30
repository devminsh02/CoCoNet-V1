from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from losses.seg_loss import BCEDiceLoss


class AuxSegLoss(nn.Module):
    def __init__(self, coarse_weight: float = 0.5, fine_weight: float = 0.5, objectness_weight: float = 0.25, boundary_prior_weight: float = 0.15) -> None:
        super().__init__()
        self.coarse_weight = float(coarse_weight)
        self.fine_weight = float(fine_weight)
        self.objectness_weight = float(objectness_weight)
        self.boundary_prior_weight = float(boundary_prior_weight)
        self.loss_fn = BCEDiceLoss()

    def forward(self, outputs: Dict[str, Dict[str, torch.Tensor]], target_mask: torch.Tensor, target_boundary: torch.Tensor) -> torch.Tensor:
        pred = outputs['pred']
        aux = outputs['aux']
        loss = target_mask.new_tensor(0.0)
        if pred.get('coarse_logits') is not None:
            loss = loss + self.coarse_weight * self.loss_fn(pred['coarse_logits'], target_mask)
        if pred.get('fine_logits') is not None:
            loss = loss + self.fine_weight * self.loss_fn(pred['fine_logits'], target_mask)
        if aux.get('objectness_logits') is not None:
            loss = loss + self.objectness_weight * self.loss_fn(aux['objectness_logits'], target_mask)
        if aux.get('boundary_prior_logits') is not None:
            loss = loss + self.boundary_prior_weight * self.loss_fn(aux['boundary_prior_logits'], target_boundary)
        return loss
