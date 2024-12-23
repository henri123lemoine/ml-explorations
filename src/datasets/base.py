from pathlib import Path
from typing import Any

import torch
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
        """Template method pattern"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            return self._load_from_cache(cache_path)

        data = self._create_data()
        self._save_to_cache(data, cache_path)
        return data

    def _load_from_cache(self, path: Path) -> Any:
        return torch.load(path)

    def _save_to_cache(self, data: Any, path: Path) -> None:
        torch.save(data, path)

    def _create_data(self) -> Any:
        raise NotImplementedError

    def _get_cache_path(self) -> Path:
        """Get path for dataset cache"""
        return self.cache_dir / f"{self.__class__.__name__}_{self.split}.pt"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError
