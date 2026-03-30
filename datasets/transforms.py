from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _pil_gray_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr[None, ...]).contiguous()


def _resize_pair(image: Image.Image, mask: Image.Image, boundary: Image.Image, size: Tuple[int, int]):
    return (
        image.resize(size, Image.BILINEAR),
        mask.resize(size, Image.NEAREST),
        boundary.resize(size, Image.NEAREST),
    )


def _pad_if_needed(image: Image.Image, mask: Image.Image, boundary: Image.Image, size: int):
    pad_w = max(size - image.size[0], 0)
    pad_h = max(size - image.size[1], 0)
    if pad_w == 0 and pad_h == 0:
        return image, mask, boundary
    padding = (0, 0, pad_w, pad_h)
    return (
        ImageOps.expand(image, border=padding, fill=0),
        ImageOps.expand(mask, border=padding, fill=0),
        ImageOps.expand(boundary, border=padding, fill=0),
    )


class TrainPairTransform:
    def __init__(self, input_size: int, hflip: bool = True, random_rescale: bool = True, random_crop: bool = True, color_jitter: bool = True) -> None:
        self.input_size = int(input_size)
        self.hflip = hflip
        self.random_rescale = random_rescale
        self.random_crop = random_crop
        self.color_jitter = color_jitter

    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        if not self.color_jitter:
            return image
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.8:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.8:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
        return image

    def __call__(self, image: Image.Image, mask: Image.Image, boundary: Image.Image) -> Dict[str, torch.Tensor]:
        image = image.convert('RGB')
        mask = mask.convert('L')
        boundary = boundary.convert('L')
        if self.random_rescale:
            scale = random.uniform(0.75, 1.25)
            new_w = max(32, int(round(image.size[0] * scale)))
            new_h = max(32, int(round(image.size[1] * scale)))
            image, mask, boundary = _resize_pair(image, mask, boundary, (new_w, new_h))
        image, mask, boundary = _pad_if_needed(image, mask, boundary, self.input_size)
        if self.random_crop:
            max_left = max(image.size[0] - self.input_size, 0)
            max_top = max(image.size[1] - self.input_size, 0)
            left = random.randint(0, max_left) if max_left > 0 else 0
            top = random.randint(0, max_top) if max_top > 0 else 0
            box = (left, top, left + self.input_size, top + self.input_size)
            image, mask, boundary = image.crop(box), mask.crop(box), boundary.crop(box)
        if image.size != (self.input_size, self.input_size):
            image, mask, boundary = _resize_pair(image, mask, boundary, (self.input_size, self.input_size))
        if self.hflip and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
            boundary = ImageOps.mirror(boundary)
        image = self._apply_color_jitter(image)
        return {'image': _pil_rgb_to_tensor(image), 'mask': _pil_gray_to_tensor(mask), 'boundary': _pil_gray_to_tensor(boundary)}


class EvalImageTransform:
    def __init__(self, input_size: int) -> None:
        self.input_size = int(input_size)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert('RGB').resize((self.input_size, self.input_size), Image.BILINEAR)
        return _pil_rgb_to_tensor(image)
