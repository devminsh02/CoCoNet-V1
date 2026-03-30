from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
        )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AverageMeter:
    name: str
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def timestamp_string() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def set_seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def upsample_like(src: torch.Tensor, ref: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    return F.interpolate(src, size=ref.shape[-2:], mode=mode, align_corners=False)


def upsample_to(src: torch.Tensor, size: tuple[int, int], mode: str = 'bilinear') -> torch.Tensor:
    return F.interpolate(src, size=size, mode=mode, align_corners=False)


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def sigmoid_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, min=eps, max=1.0 - eps)
    entropy = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
    return entropy


def freeze_batchnorm_modules(module: nn.Module, freeze_affine: bool = False) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
            if freeze_affine:
                if m.weight is not None:
                    m.weight.requires_grad = False
                if m.bias is not None:
                    m.bias.requires_grad = False
