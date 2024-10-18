import pickle
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from src.settings import CACHE_PATH


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str):
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    def save_complete_model(
        self, file_name: str | None = None, dir_path: Path = CACHE_PATH, ext: str = "pth"
    ):
        if file_name is None:
            file_name = self.__class__.__name__
        file_path = dir_path / f"{file_name}_cls.{ext}"
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_complete_model(
        cls, file_name: str | None = None, dir_path: Path = CACHE_PATH, ext: str = "pth"
    ):
        if file_name is None:
            file_name = cls.__name__
        file_path = dir_path / f"{file_name}_cls.{ext}"
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def fit(self, X: Any, y: Any):
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("Subclasses must implement predict method")

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement evaluate method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")
