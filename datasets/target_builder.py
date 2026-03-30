from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class TargetBuilderConfig:
    boundary_width: int = 3


class TargetBuilder:
    def __init__(self, boundary_width: int = 3) -> None:
        self.cfg = TargetBuilderConfig(boundary_width=boundary_width)

    def build_boundary_target(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 127).astype(np.uint8) * 255
        k = max(1, int(self.cfg.boundary_width))
        kernel = np.ones((k, k), np.uint8)
        dil = cv2.dilate(mask_u8, kernel, iterations=1)
        ero = cv2.erode(mask_u8, kernel, iterations=1)
        return ((dil - ero) > 0).astype(np.uint8) * 255

    def build(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'mask': (mask > 127).astype(np.uint8) * 255,
            'boundary': self.build_boundary_target(mask),
        }
