from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from utils.common import ConvBNReLU, upsample_like


class SimpleFPNNeck(nn.Module):
    def __init__(self, in_channels: Dict[str, int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral_c2 = nn.Conv2d(in_channels['c2'], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels['c3'], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels['c4'], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels['c5'], out_channels, 1)
        self.smooth_p2 = ConvBNReLU(out_channels, out_channels, 3)
        self.smooth_p3 = ConvBNReLU(out_channels, out_channels, 3)
        self.smooth_p4 = ConvBNReLU(out_channels, out_channels, 3)
        self.smooth_p5 = ConvBNReLU(out_channels, out_channels, 3)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p5 = self.lateral_c5(feats['c5'])
        p4 = self.lateral_c4(feats['c4']) + upsample_like(p5, feats['c4'])
        p3 = self.lateral_c3(feats['c3']) + upsample_like(p4, feats['c3'])
        p2 = self.lateral_c2(feats['c2']) + upsample_like(p3, feats['c2'])
        return {
            'p2': self.smooth_p2(p2),
            'p3': self.smooth_p3(p3),
            'p4': self.smooth_p4(p4),
            'p5': self.smooth_p5(p5),
        }
