from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.datasets.base import BaseDataset


class ImageDataset(BaseDataset):
    """Base class for image datasets"""

    def __init__(
        self,
        cache_dir: Path,
        split: str = "train",
        transform=None,
        image_processor=None,
    ):
        self.image_processor = image_processor
        super().__init__(cache_dir, split, transform)

    def _ensure_rgb(self, image: Any) -> np.ndarray:
        """Convert image to RGB format."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3:
                if image.shape[-1] == 4:
                    return image[..., :3]
                elif image.shape[-1] == 3:
                    return image
                elif image.shape[-1] == 1:
                    return np.repeat(image, 3, axis=-1)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        raise ValueError(f"Unsupported image format: {type(image)}")

    def __getitem__(self, idx):
        """Common image processing logic"""
        image = self.data[idx]
        image = self._ensure_rgb(image)

        if self.transform:
            image = self.transform(image)

        if self.image_processor:
            image = self.image_processor(image)

        return image
