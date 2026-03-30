from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SampleRecord:
    image_path: str
    mask_path: str
    dataset_name: str
    sample_id: str
