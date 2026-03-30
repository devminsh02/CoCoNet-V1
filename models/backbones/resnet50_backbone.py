from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        width = planes
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _FallbackResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, blocks=3, stride=2)

    def _make_layer(self, block: Type[Bottleneck], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers: List[nn.Module] = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_stages: int = 0) -> None:
        super().__init__()
        model = self._build_model(pretrained)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.freeze_stages(freeze_stages)

    def _build_model(self, pretrained: bool) -> nn.Module:
        try:
            from torchvision.models import ResNet50_Weights, resnet50
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            return resnet50(weights=weights)
        except Exception as exc:
            warnings.warn(f'Falling back to internal ResNet50 (no pretrained weights): {exc}')
            return _FallbackResNet50()

    def freeze_stages(self, freeze_stages: int) -> None:
        if freeze_stages <= 0:
            return
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for stage in stages[:freeze_stages]:
            for p in stage.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}
