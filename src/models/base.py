import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.settings import CACHE_PATH

logger = logging.getLogger(__name__)

InputType = TypeVar("InputType", Tensor, np.ndarray, dict[str, Tensor])
OutputType = TypeVar("OutputType", Tensor, np.ndarray, float)


class BaseModel(nn.Module, Generic[InputType, OutputType], ABC):
    """
    Base class for all models with generic type support.

    Type Parameters:
        InputType: The type of input the model accepts (Tensor, np.ndarray, or dict of tensors)
        OutputType: The type of output the model produces
    """

    # Class variable to track model registry
    registry: ClassVar[dict[str, type["BaseModel"]]] = {}

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init_subclass__(cls) -> None:
        """Register all model subclasses automatically"""
        super().__init_subclass__()
        cls.registry[cls.__name__] = cls

    @abstractmethod
    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def predict(self, x: InputType) -> OutputType:
        """Make a prediction for the given input."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model state to the given path."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel[InputType, OutputType]":
        """Load model state from the given path."""
        pass


class Model(BaseModel[Tensor, Tensor]):
    """
    Standard model implementation with common functionality.

    This class provides default implementations for saving/loading models
    and defines the interface for training and evaluation.
    """

    def __init__(self):
        super().__init__()

    def save(self, path: Path) -> None:
        """Save model state dict to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "Model":
        """Load model from state dict."""
        model = cls()
        model.load_state_dict(torch.load(path))
        model.to(model.device)
        return model

    def save_complete_model(
        self, file_name: str | None = None, dir_path: Path = CACHE_PATH, ext: str = "pth"
    ) -> None:
        """Save complete model (including non-state_dict attributes)."""
        if file_name is None:
            file_name = self.__class__.__name__
        file_path = dir_path / f"{file_name}_cls.{ext}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            logger.error(f"Failed to save model to {file_path}: {e}")
            raise

    @classmethod
    def load_complete_model(
        cls, file_name: str | None = None, dir_path: Path = CACHE_PATH, ext: str = "pth"
    ) -> "Model":
        """Load complete model (including non-state_dict attributes)."""
        if file_name is None:
            file_name = cls.__name__
        file_path = dir_path / f"{file_name}_cls.{ext}"
        try:
            with open(file_path, "rb") as f:
                model = pickle.load(f)
            model.to(model.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {file_path}: {e}")
            raise

    def fit(self, X: Any, y: Any) -> None:
        """Train the model on the given data."""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, x: Tensor) -> Tensor:
        """Make predictions using the model."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def evaluate(self, x: Tensor, y: Tensor) -> dict[str, float]:
        """Evaluate model performance."""
        raise NotImplementedError("Subclasses must implement evaluate method")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        raise NotImplementedError("Subclasses must implement forward method")
