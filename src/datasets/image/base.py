from pathlib import Path

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

    def _ensure_rgb(self, image):
        """Your existing _ensure_rgb implementation"""
        pass

    def __getitem__(self, idx):
        """Common image processing logic"""
        image = self.data[idx]
        image = self._ensure_rgb(image)

        if self.transform:
            image = self.transform(image)

        if self.image_processor:
            image = self.image_processor(image)

        return image
