from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base class for all datasets with common functionality"""

    def __init__(
        self,
        cache_dir: Path,
        split: str = "train",
        transform=None,
    ):
        self.cache_dir = cache_dir
        self.split = split
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self) -> Any:
        """Load or create cached dataset"""
        raise NotImplementedError

    def _get_cache_path(self) -> Path:
        """Get path for dataset cache"""
        return self.cache_dir / f"{self.__class__.__name__}_{self.split}.pt"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError
