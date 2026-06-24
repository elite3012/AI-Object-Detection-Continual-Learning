from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageTransform:
    image_size: int
    train: bool = False
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD
    crop_scale: tuple[float, float] = (0.82, 1.0)
    hflip_probability: float = 0.5
    rotation_degrees: float = 10.0
    color_jitter: float = 0.12

    def __call__(self, image: Image.Image) -> torch.Tensor:
        clean = ImageOps.exif_transpose(image).convert("RGB")
        if self.train:
            clean = self._random_square_crop(clean)
            if random.random() < self.hflip_probability:
                clean = ImageOps.mirror(clean)
            clean = clean.rotate(
                random.uniform(-self.rotation_degrees, self.rotation_degrees),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            clean = self._jitter_color(clean)
        else:
            clean = self._center_square_crop(clean)
        clean = clean.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return self._to_normalized_tensor(clean)

    def _center_square_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))

    def _random_square_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        shortest = min(width, height)
        min_scale, max_scale = self.crop_scale
        crop_side = max(1, int(shortest * random.uniform(min_scale, max_scale)))
        left = 0 if width == crop_side else random.randint(0, width - crop_side)
        top = 0 if height == crop_side else random.randint(0, height - crop_side)
        return image.crop((left, top, left + crop_side, top + crop_side))

    def _jitter_color(self, image: Image.Image) -> Image.Image:
        amount = self.color_jitter
        for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            factor = random.uniform(1 - amount, 1 + amount)
            image = enhancer(image).enhance(factor)
        return image

    def _to_normalized_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std
