from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from losses.affinity_loss import AffinityLoss
from losses.aux_seg_loss import AuxSegLoss
from losses.boundary_loss import BoundaryLoss
from losses.seg_loss import BCEDiceLoss
from losses.topology_loss import TopologySurrogateLoss


class LossManager(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        loss_cfg = cfg['loss']
        self.seg_enabled = bool(loss_cfg['seg']['enabled'])
        self.aux_enabled = bool(loss_cfg['aux_seg']['enabled'])
        self.boundary_enabled = bool(loss_cfg['boundary']['enabled'])
        self.affinity_enabled = bool(loss_cfg['affinity']['enabled'])
        self.topology_enabled = bool(loss_cfg['topology']['enabled'])

        self.boundary_weight = float(loss_cfg['boundary'].get('weight', 1.0))
        self.boundary_candidate_weight = float(loss_cfg['boundary'].get('candidate_weight', 0.0))
        self.affinity_weight = float(loss_cfg['affinity'].get('weight', 0.0))
        self.topology_weight = float(loss_cfg['topology'].get('weight', 0.0))

        self.seg_loss = BCEDiceLoss(
            bce_weight=loss_cfg['seg'].get('bce_weight', 1.0),
            iou_weight=loss_cfg['seg'].get('iou_weight', 1.0),
        )
        self.aux_loss = AuxSegLoss(
            coarse_weight=loss_cfg['aux_seg'].get('coarse_weight', 0.5),
            fine_weight=loss_cfg['aux_seg'].get('fine_weight', 0.5),
            objectness_weight=loss_cfg['aux_seg'].get('objectness_weight', 0.25),
            boundary_prior_weight=loss_cfg['aux_seg'].get('boundary_prior_weight', 0.15),
        )
        self.boundary_loss = BoundaryLoss()
        self.affinity_loss = AffinityLoss(
            line_samples=loss_cfg['affinity'].get('line_samples', 7),
            boundary_pool_kernel=loss_cfg['affinity'].get('boundary_pool_kernel', 7),
            target_threshold=loss_cfg['affinity'].get('target_threshold', 0.08),
            pos_weight=loss_cfg['affinity'].get('pos_weight', 2.0),
            regression_weight=loss_cfg['affinity'].get('regression_weight', 0.25),
        )
        self.topology_loss = TopologySurrogateLoss(
            band_kernel=loss_cfg['topology'].get('band_kernel', 7),
            consistency_weight=loss_cfg['topology'].get('consistency_weight', 0.35),
        )

    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = targets['mask'].new_tensor(0.0)

        if self.seg_enabled:
            losses['seg'] = self.seg_loss(outputs['pred']['final_logits'], targets['mask'])
            total = total + losses['seg']

        if self.aux_enabled:
            losses['aux_seg'] = self.aux_loss(outputs, targets['mask'], targets['boundary'])
            total = total + losses['aux_seg']

        if self.boundary_enabled and outputs['pred'].get('boundary_logits') is not None:
            losses['boundary'] = self.boundary_weight * self.boundary_loss(outputs['pred']['boundary_logits'], targets['boundary'])
            total = total + losses['boundary']
            if self.boundary_candidate_weight > 0 and outputs['aux'].get('boundary_candidate_logits') is not None:
                losses['boundary_candidate'] = self.boundary_candidate_weight * self.boundary_loss(outputs['aux']['boundary_candidate_logits'], targets['boundary'])
                total = total + losses['boundary_candidate']

        if self.affinity_enabled and outputs['pred'].get('affinity_logits') is not None:
            losses['affinity'] = self.affinity_weight * self.affinity_loss(
                affinity_logits=outputs['pred'].get('affinity_logits'),
                boundary_target=targets['boundary'],
                fragment_coords=outputs['meta'].get('fragment_coords'),
                edge_index=outputs['meta'].get('edge_index'),
                edge_valid_mask=outputs['meta'].get('edge_valid_mask'),
                token_valid_mask=outputs['meta'].get('token_valid_mask'),
            )
            total = total + losses['affinity']
        else:
            losses['affinity'] = targets['mask'].new_tensor(0.0)

        if self.topology_enabled and outputs['pred'].get('closure_logits') is not None:
            losses['topology'] = self.topology_weight * self.topology_loss(
                boundary_logits=outputs['pred'].get('boundary_logits'),
                boundary_target=targets['boundary'],
                closure_logits=outputs['pred'].get('closure_logits'),
            )
            total = total + losses['topology']
        else:
            losses['topology'] = targets['mask'].new_tensor(0.0)

        losses['total'] = total
        return losses
