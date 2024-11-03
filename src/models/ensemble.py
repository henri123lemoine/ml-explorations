from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

from src.config import DatasetConfig, PretrainedConfig
from src.datasets.data_processing import create_dataloaders
from src.models.base import Model
from src.models.image.base import PretrainedImageClassifier
from src.settings import MODELS_PATH
from src.train import validate_model


class ModelWrapper:
    """Wrapper to standardize model inputs/outputs"""

    def __init__(self, model: Model, processor: Any = None):
        self.model = model
        self.processor = processor or getattr(model, "processor", None)

    def process_input(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | dict:
        """Standardize input format for the model"""
        if self.processor is not None:
            # Handle pretrained models that need processing
            return self.processor(images=x, return_tensors="pt")
        elif isinstance(x, np.ndarray):
            # Handle raw numpy input for traditional models
            return torch.tensor(x, dtype=torch.float32)
        return x

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Forward pass with appropriate input processing"""
        processed_x = self.process_input(x)

        if isinstance(self.model, PretrainedImageClassifier):
            return self.model.model(**processed_x).logits
        return self.model(processed_x)


class EnsembleModel(Model):
    """Ensemble model that combines multiple models' predictions."""

    def __init__(
        self,
        models: list[Model | PretrainedImageClassifier],
        processors: list | None = None,
        method: str = "weighted_vote",
        weights: list[float] | None = None,
        meta_learner: RandomForestClassifier | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.wrapped_models = [
            ModelWrapper(model, proc)
            for model, proc in zip(models, processors or [None] * len(models))
        ]
        self.method = method
        self.device = device
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.meta_learner = meta_learner

    def predict(self, x):
        """Predict class for input x"""
        self.eval()
        with torch.no_grad():
            # Basic preprocessing for non-ViT models
            if not isinstance(x, torch.Tensor):
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
                x = transform(x).unsqueeze(0)

            outputs = self._get_model_predictions(x)

            if self.method == "weighted_vote":
                return self.weighted_vote(outputs).item()
            return self.vote(outputs).item()

    def _get_model_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from all models."""
        predictions = []

        for wrapper in self.wrapped_models:
            wrapper.model.eval()
            with torch.no_grad():
                outputs = wrapper(x)
                predictions.append(outputs)

        return torch.stack(predictions)

    def vote(self, predictions: torch.Tensor) -> torch.Tensor:
        """Simple majority voting."""
        # Get class predictions from logits
        pred_classes = torch.argmax(predictions, dim=2)
        # Count votes for each class
        vote_counts = torch.zeros((predictions.size(1), predictions.size(2)), device=self.device)

        for i in range(predictions.size(0)):  # For each model
            vote_counts.scatter_add_(
                1,
                pred_classes[i],
                torch.ones_like(pred_classes[i], dtype=torch.float),
            )

        return torch.argmax(vote_counts, dim=1)

    def weighted_vote(self, predictions: torch.Tensor) -> torch.Tensor:
        """Weighted voting using model weights."""
        if self.weights is None:
            raise ValueError("Weights must be provided for weighted voting")

        # Convert weights to tensor
        weights = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)

        # Apply weights to predictions
        weighted_preds = predictions * weights

        # Sum weighted predictions
        ensemble_preds = torch.sum(weighted_preds, dim=0)

        return torch.argmax(ensemble_preds, dim=1)

    def meta_predict(self, predictions: torch.Tensor) -> torch.Tensor:
        """Use meta-learner to combine predictions."""
        if self.meta_learner is None:
            raise ValueError("Meta-learner must be provided for meta-learning approach")

        # Reshape predictions for meta-learner
        batch_size = predictions.size(1)
        flattened = predictions.transpose(0, 1).reshape(batch_size, -1).cpu().numpy()

        # Get meta-learner predictions
        meta_preds = self.meta_learner.predict(flattened)
        return torch.tensor(meta_preds, device=self.device)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through ensemble."""
        predictions = self._get_model_predictions(x)

        if self.method == "vote":
            return self.vote(predictions)
        elif self.method == "weighted_vote":
            return self.weighted_vote(predictions)
        elif self.method == "meta":
            return self.meta_predict(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def calibrate_weights(
        self,
        val_loader: DataLoader,
        method: str = "accuracy",
    ) -> None:
        """
        Calibrate model weights based on validation performance.

        Args:
            val_loader: Validation data loader
            method: Method to compute weights ('accuracy' or 'f1')
        """
        print("Calibrating ensemble weights...")
        model_metrics = []

        # Get validation metrics for each model
        for wrapped_model in self.wrapped_models:
            metrics = validate_model(wrapped_model.model, val_loader, self.device)
            model_metrics.append(metrics[method] if method in metrics else metrics["accuracy"])

        # Normalize metrics to get weights
        total_metric = sum(model_metrics)
        self.weights = [metric / total_metric for metric in model_metrics]

        print("Calibrated weights:", self.weights)

    def fit_meta_learner(
        self,
        train_loader: DataLoader,
        val_ratio: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Train a meta-learner on the predictions of base models.

        Args:
            train_loader: Training data loader
            val_ratio: Ratio of training data to use for validation
            random_state: Random seed for reproducibility
        """
        print("Training meta-learner...")
        all_predictions = []
        all_labels = []

        # Collect predictions from all models
        for batch, labels in train_loader:
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            labels = labels.to(self.device)

            predictions = self._get_model_predictions(batch)
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Prepare data for meta-learner
        X = np.concatenate([p.transpose(1, 0, 2).reshape(p.shape[1], -1) for p in all_predictions])
        y = np.concatenate(all_labels)

        # Initialize and train meta-learner
        self.meta_learner = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        )
        self.meta_learner.fit(X, y)
        print("Meta-learner training completed")


if __name__ == "__main__":
    model_config = PretrainedConfig(
        model_name="google/vit-base-patch16-224",
        model_class=ViTForImageClassification,
        processor_class=ViTImageProcessor,
        num_labels=2,
    )
    vit_model = PretrainedImageClassifier(model_config)
    checkpoint1 = torch.load(MODELS_PATH / "vit" / "best_model.pth")
    vit_model.load_state_dict(checkpoint1["model_state_dict"])

    other_vit_model = PretrainedImageClassifier(model_config)
    checkpoint2 = torch.load(MODELS_PATH / "best_vit_model.pth")
    other_vit_model.load_state_dict(checkpoint2["model_state_dict"])

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    processors = [processor, processor]

    ensemble = EnsembleModel(
        models=[vit_model, other_vit_model],
        processors=processors,
        method="weighted_vote",
    )

    config = DatasetConfig(max_images=1000)

    first_processor = next((p for p in processors if p is not None), None)
    if first_processor is None:
        raise ValueError("At least one processor must be provided")

    train_loader, val_loader = create_dataloaders(processor=first_processor, config=config)

    if ensemble.method == "meta":
        ensemble.fit_meta_learner(train_loader)
    elif ensemble.method == "weighted_vote":
        ensemble.calibrate_weights(val_loader)
