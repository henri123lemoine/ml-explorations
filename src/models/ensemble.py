from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from src.models.base import ModelInterface


class EnsembleMethod(str, Enum):
    MEAN = "mean"
    VOTE = "vote"
    WEIGHTED = "weighted"
    STACKING = "stacking"


class Ensemble[IN: Any, OUT: Any, DATA: Any]:
    """
    Generic ensemble model that combines predictions from multiple models.

    Type Parameters:
        IN: The type of input the models accept
        OUT: The type of output the models produce
        DATA: The type of data used for evaluation
    """

    def __init__(
        self,
        models: Sequence[ModelInterface[IN, OUT, DATA]],
        method: EnsembleMethod | str = EnsembleMethod.WEIGHTED,
        weights: Sequence[float] | None = None,
        meta_learner: Any | None = None,
        temperature: float = 1.0,
    ) -> None:
        if not models:
            raise ValueError("At least one model must be provided")
        if weights is not None and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")

        self.models = models
        self.method = EnsembleMethod(method)
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.meta_learner = meta_learner or RandomForestClassifier(n_estimators=100)
        self.temperature = temperature
        self._is_fitted = False

    def predict(self, x: IN) -> OUT:
        """Make predictions using the ensemble."""
        predictions = [model.predict(x) for model in self.models]

        # Apply temperature scaling to soften/sharpen the predictions
        if isinstance(predictions[0], torch.Tensor):
            predictions = [p / self.temperature for p in predictions]
        else:
            predictions = [p / self.temperature for p in predictions]

        match self.method:
            case EnsembleMethod.MEAN:
                return self._mean_combine(predictions)
            case EnsembleMethod.VOTE:
                return self._vote_combine(predictions)
            case EnsembleMethod.WEIGHTED:
                return self._weighted_combine(predictions)
            case EnsembleMethod.STACKING:
                if not self._is_fitted:
                    raise RuntimeError("Meta-learner must be fitted before using stacking")
                return self._stacking_combine(predictions)
            case _:
                raise ValueError(f"Unknown ensemble method: {self.method}")

    def _mean_combine(self, predictions: Sequence[OUT]) -> OUT:
        """Combine predictions by taking their mean."""
        if isinstance(predictions[0], torch.Tensor):
            return torch.mean(torch.stack(predictions), dim=0)
        return np.mean(predictions, axis=0)

    def _vote_combine(self, predictions: Sequence[OUT]) -> OUT:
        """Combine predictions by majority voting."""
        if isinstance(predictions[0], torch.Tensor):
            # Get class predictions
            class_preds = torch.stack([pred.argmax(dim=-1) for pred in predictions])
            # Count votes for each class
            num_classes = predictions[0].size(-1)
            votes = torch.zeros((predictions[0].size(0), num_classes), device=predictions[0].device)

            for pred in class_preds:
                votes.scatter_add_(
                    1, pred.unsqueeze(1), torch.ones_like(pred.unsqueeze(1), dtype=torch.float)
                )

            # Convert votes to probabilities
            return votes / len(predictions)
        else:
            # NumPy implementation
            class_preds = np.stack([np.argmax(pred, axis=-1) for pred in predictions])
            num_classes = predictions[0].shape[-1]
            votes = np.zeros((predictions[0].shape[0], num_classes))

            for pred in class_preds:
                np.add.at(votes, (np.arange(len(pred)), pred), 1)

            return votes / len(predictions)

    def _weighted_combine(self, predictions: Sequence[OUT]) -> OUT:
        """Combine predictions using learned weights."""
        if isinstance(predictions[0], torch.Tensor):
            weights_tensor = torch.tensor(
                self.weights, device=predictions[0].device, dtype=predictions[0].dtype
            )
            return torch.sum(torch.stack(predictions) * weights_tensor.view(-1, 1, 1), dim=0)
        return np.sum([p * w for p, w in zip(predictions, self.weights)], axis=0)

    def _stacking_combine(self, predictions: Sequence[OUT]) -> OUT:
        """Combine predictions using a meta-learner."""
        if isinstance(predictions[0], torch.Tensor):
            # Convert to numpy for sklearn meta-learner
            stacked_preds = torch.stack(predictions).cpu().numpy()
            # Reshape to (n_samples, n_models * n_classes)
            reshaped = stacked_preds.transpose(1, 0, 2).reshape(stacked_preds.shape[1], -1)
            # Get meta-learner predictions and convert back to torch
            meta_preds = self.meta_learner.predict_proba(reshaped)
            return torch.from_numpy(meta_preds).to(predictions[0].device)
        else:
            stacked_preds = np.stack(predictions)
            reshaped = stacked_preds.transpose(1, 0, 2).reshape(stacked_preds.shape[1], -1)
            return self.meta_learner.predict_proba(reshaped)

    def calibrate_weights(
        self,
        val_data: DATA,
        method: str = "softmax",
        temperature: float = 1.0,
    ) -> None:
        """
        Calibrate ensemble weights based on validation performance.

        Args:
            val_data: Validation dataset
            method: Method to compute weights ('softmax' or 'normalize')
            temperature: Temperature for softmax scaling
        """
        scores = [model.evaluate(val_data)["accuracy"] for model in self.models]

        if method == "softmax":
            # Apply softmax with temperature scaling
            scores = np.array(scores) / temperature
            exp_scores = np.exp(scores - np.max(scores))  # For numerical stability
            self.weights = exp_scores / exp_scores.sum()
        else:
            # Simple normalization
            total = sum(scores)
            self.weights = [score / total for score in scores]

    def fit_stacking(
        self,
        train_data: DATA,
        val_data: DATA | None = None,
        cv_folds: int = 5,
    ) -> None:
        """
        Train the meta-learner for stacking.

        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
            cv_folds: Number of cross-validation folds if val_data is None
        """
        if isinstance(train_data, DataLoader):
            X, y = self._prepare_stacking_data(train_data)
            if val_data:
                X_val, y_val = self._prepare_stacking_data(val_data)
                self.meta_learner.fit(X, y)
                self._is_fitted = True
                return

            self.meta_learner.fit(X, y)
            self._is_fitted = True
        else:
            raise NotImplementedError("Stacking currently only supports DataLoader")

    def _prepare_stacking_data(
        self,
        data: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for meta-learner training."""
        all_predictions = []
        all_labels = []

        for batch, labels in data:
            predictions = [model.predict(batch) for model in self.models]
            if isinstance(predictions[0], torch.Tensor):
                predictions = [p.cpu().numpy() for p in predictions]
                labels = labels.cpu().numpy()

            stacked = np.stack(predictions)
            all_predictions.append(stacked)
            all_labels.append(labels)

        X = np.concatenate([p.transpose(1, 0, 2).reshape(p.shape[1], -1) for p in all_predictions])
        y = np.concatenate(all_labels)
        return X, y

    def save(self, path: Path) -> None:
        """Save ensemble configuration and meta-learner if used."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "weights": self.weights,
            "method": self.method,
            "temperature": self.temperature,
        }
        if self.method == EnsembleMethod.STACKING and self._is_fitted:
            state["meta_learner"] = self.meta_learner
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: Path,
        models: Sequence[ModelInterface[IN, OUT, DATA]],
    ) -> "Ensemble[IN, OUT, DATA]":
        """Load ensemble configuration and meta-learner if available."""
        state = torch.load(path)
        ensemble = cls(
            models=models,
            method=state["method"],
            weights=state["weights"],
            temperature=state.get("temperature", 1.0),
        )
        if "meta_learner" in state:
            ensemble.meta_learner = state["meta_learner"]
            ensemble._is_fitted = True
        return ensemble


def evaluate_ensemble(ensemble: Ensemble, data_loader: DataLoader) -> dict[str, float]:
    """Evaluate ensemble performance on a dataset."""
    correct = 0
    total = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(next(iter(ensemble.models)).device)
        labels = labels.to(next(iter(ensemble.models)).device)

        outputs = ensemble.predict(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {"accuracy": correct / total}
