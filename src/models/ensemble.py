from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.base import ModelInterface, TorchModel


class EnsembleMethod(str, Enum):
    MEAN = "mean"
    VOTE = "vote"
    WEIGHTED = "weighted"


class Ensemble[IN, OUT, DATA](ModelInterface[IN, OUT, DATA]):
    """
    Generic ensemble model that combines predictions from multiple models.
    Models can be of different types but must handle the same input/output types.
    """

    def __init__(
        self,
        models: Sequence[ModelInterface[IN, OUT, DATA]],
        method: EnsembleMethod = EnsembleMethod.MEAN,
        weights: Sequence[float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model must be provided")
        if weights is not None and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")

        self.models = models
        self.method = method
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, x: IN) -> OUT:
        predictions = [model.predict(x) for model in self.models]

        match self.method:
            case EnsembleMethod.MEAN:
                return self._mean_combine(predictions)
            case EnsembleMethod.VOTE:
                return self._vote_combine(predictions)
            case EnsembleMethod.WEIGHTED:
                return self._weighted_combine(predictions)

    def _mean_combine(self, predictions: Sequence[OUT]) -> OUT:
        if isinstance(predictions[0], torch.Tensor):
            return torch.mean(torch.stack(predictions), dim=0)
        return np.mean(predictions, axis=0)

    def _vote_combine(self, predictions: Sequence[OUT]) -> OUT:
        if isinstance(predictions[0], torch.Tensor):
            votes = torch.stack([pred.argmax(dim=-1) for pred in predictions])
            return torch.mode(votes, dim=0).values
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=np.stack([np.argmax(pred, axis=-1) for pred in predictions]),
        )

    def _weighted_combine(self, predictions: Sequence[OUT]) -> OUT:
        if isinstance(predictions[0], torch.Tensor):
            return torch.sum(
                torch.stack(predictions) * torch.tensor(self.weights).view(-1, 1, 1), dim=0
            )
        return np.sum([p * w for p, w in zip(predictions, self.weights)], axis=0)

    def evaluate(self, data: DATA) -> dict[str, float]:
        """
        Evaluate ensemble performance.
        Currently only supports classification metrics.
        """
        if isinstance(data, DataLoader):
            return self._evaluate_torch(data)
        return self._evaluate_numpy(data)

    def _evaluate_torch(self, data: DataLoader) -> dict[str, float]:
        correct = 0
        total = 0
        for inputs, labels in data:
            outputs = self.predict(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        return {"accuracy": correct / total}

    def _evaluate_numpy(self, data: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        X, y = data
        predictions = np.argmax(self.predict(X), axis=1)
        return {"accuracy": np.mean(predictions == y)}

    def calibrate_weights(self, val_data: DATA) -> None:
        """Calibrate ensemble weights based on validation performance."""
        scores = [model.evaluate(val_data)["accuracy"] for model in self.models]
        total = sum(scores)
        self.weights = [score / total for score in scores]

    def save(self, path: Path) -> None:
        """Save ensemble weights and method."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": self.weights, "method": self.method}, path)

    @classmethod
    def load(
        cls, path: Path, models: Sequence[ModelInterface[IN, OUT, DATA]]
    ) -> "Ensemble[IN, OUT, DATA]":
        """Load ensemble configuration. Models must be provided separately."""
        state = torch.load(path)
        return cls(models=models, method=state["method"], weights=state["weights"])


if __name__ == "__main__":
    model1 = TorchModel()
    model2 = TorchModel()
    model3 = TorchModel()

    ensemble = Ensemble(
        models=[model1, model2, model3], method=EnsembleMethod.WEIGHTED, weights=[0.4, 0.3, 0.3]
    )
