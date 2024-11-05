import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.settings import CACHE_PATH

logger = logging.getLogger(__name__)


class ModelInterface[IN: Any, OUT: Any, DATA: Any](ABC):
    """
    Framework-agnostic interface for all models with generic type support.

    Type Parameters:
        IN: The type of input the model accepts
        OUT: The type of output the model produces
        DATA: The type of data used for training
    """

    # Class variable to track model registry
    registry: ClassVar[dict[str, type["ModelInterface"]]] = {}

    def __init_subclass__(cls) -> None:
        """Register all model subclasses automatically"""
        super().__init_subclass__()
        cls.registry[cls.__name__] = cls

    @abstractmethod
    def predict(self, x: IN) -> OUT:
        """Make a prediction for the given input."""
        pass

    @abstractmethod
    def fit(self, train_data: DATA, val_data: DATA | None = None) -> None:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def evaluate(self, data: DATA) -> dict[str, float]:
        """Evaluate model performance."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model state to the given path."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ModelInterface[IN, OUT, DATA]":
        """Load model state from the given path."""
        pass


class TorchBaseModel[IN: Tensor | dict[str, Tensor], OUT: Tensor | float, DATA](
    ModelInterface[IN, OUT, DATA], nn.Module
):
    """
    Base class for PyTorch models.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, x: IN) -> OUT:
        """Forward pass of the model."""
        pass

    def predict(self, x: IN) -> OUT:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "TorchBaseModel[IN, OUT, DATA]":
        model = cls()
        model.load_state_dict(torch.load(path))
        model.to(model.device)
        return model

    def save_complete_model(
        self, file_name: str | None = None, dir_path: Path = CACHE_PATH, ext: str = "pth"
    ) -> None:
        if file_name is None:
            file_name = self.__class__.__name__
        file_path = dir_path / f"{file_name}_cls.{ext}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_complete_model(
        cls, file_name: str | None = None, dir_path: Path = CACHE_PATH, ext: str = "pth"
    ) -> "TorchBaseModel[IN, OUT, DATA]":
        if file_name is None:
            file_name = cls.__name__
        file_path = dir_path / f"{file_name}_cls.{ext}"
        with open(file_path, "rb") as f:
            model = pickle.load(f)
            model.to(model.device)
            return model


class TorchModel(TorchBaseModel[Tensor, Tensor, DataLoader]):
    """Standard PyTorch model implementation with common functionality."""

    pass


class NumpyModel(ModelInterface[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]):
    """Base class for numpy-based models."""

    pass


class SklearnModel(ModelInterface[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]):
    """Base class for scikit-learn compatible models."""

    pass


class MLXBaseModel[IN, OUT, DATA](ModelInterface[IN, OUT, DATA], nn.Module):
    """Base class for MLX models. MLX is an array framework for machine learning on Apple silicon"""

    def __init__(self) -> None:
        super().__init__()
        # MLX automatically handles device placement
        self.device = "gpu" if str(mx.default_device()) == "gpu" else "cpu"

    @abstractmethod
    def __call__(self, x: IN) -> OUT:
        """Forward pass of the model."""
        pass

    def predict(self, x: IN) -> OUT:
        """Make a prediction using the model."""
        return self(x)

    def save(self, path: Path) -> None:
        """Save model state to the given path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        mx.savez(str(path), **self.parameters())

    @classmethod
    def load(cls, path: Path) -> "MLXBaseModel[IN, OUT, DATA]":
        """Load model state from the given path."""
        model = cls()
        weights = mx.load(str(path))
        model.update(weights)
        return model


class MLXModel[IN, OUT, DATA](MLXBaseModel[IN, OUT, DATA]):
    """Concrete MLX model implementation."""

    pass
