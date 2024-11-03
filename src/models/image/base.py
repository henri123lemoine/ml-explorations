import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.base import Model
from src.models.image.config import PretrainedConfig


class PretrainedImageClassifier(Model):
    """Generic classifier that can work with any pretrained model"""

    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__()
        self.config = config

        # Initialize model and processor
        self.processor = config.processor_class.from_pretrained(config.model_name)

        ## NOTE ##
        # ImageNet:
        ## 444	bicycle-built-for-two, tandem bicycle, tandem
        ## 671	mountain bike, all-terrain bike, off-roader
        ## 670	motor scooter, scooter (to contrast with?)

        # First load the model with original classification head
        original_model = config.model_class.from_pretrained(config.model_name)

        # Get bicycle-related weights from the original classifier
        original_classifier = getattr(original_model, config.classifier_attr)
        if hasattr(original_classifier, "weight"):
            # For binary classification: [bicycle, background]
            bicycle_weights = original_classifier.weight[[671, 444]]  # mountain bike and tandem
            bicycle_weight = bicycle_weights.mean(dim=0, keepdim=True)  # average them

            # For background, average everything except bicycles and similar vehicles
            exclude_indices = {670, 671, 444}  # exclude scooter and bicycle classes
            background_indices = [
                i for i in range(original_classifier.weight.size(0)) if i not in exclude_indices
            ]
            background_weight = original_classifier.weight[background_indices].mean(
                dim=0, keepdim=True
            )

            # Put bicycle first to match dataset labels
            initial_weights = torch.cat([bicycle_weight, background_weight], dim=0)

            if hasattr(original_classifier, "bias"):
                bicycle_biases = original_classifier.bias[[671, 444]]
                bicycle_bias = bicycle_biases.mean().unsqueeze(0)
                background_bias = original_classifier.bias[background_indices].mean().unsqueeze(0)
                initial_bias = torch.cat([bicycle_bias, background_bias], dim=0)

        # Now create our binary classification model
        self.model = config.model_class.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True,
        )

        # Initialize the new classifier with bicycle-related weights
        new_classifier = getattr(self.model, config.classifier_attr)
        if hasattr(new_classifier, "weight") and "initial_weights" in locals():
            print("Initializing classifier with pretrained bicycle weights")  # Debug print
            with torch.no_grad():
                new_classifier.weight.copy_(initial_weights)
                if hasattr(new_classifier, "bias") and "initial_bias" in locals():
                    new_classifier.bias.copy_(initial_bias)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, pixel_values=None):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = [x]
            processed = self.processor(images=x, return_tensors="pt")
            return self.model(**processed).logits
        return self.model(x).logits

    def predict(self, x: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
            return predicted.item() if len(outputs) == 1 else predicted.numpy()

    def save(self, path: Path):
        """Save the model, processor, and config"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model and processor
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

        # Save the config
        torch.save(
            {
                "config": self.config,
                "optimizer_state": self.optimizer.state_dict(),
            },
            os.path.join(path, "training_state.pt"),
        )

    @classmethod
    def load(cls, path: Path):
        """Load a saved model"""
        # Load the saved state
        state = torch.load(os.path.join(path, "training_state.pt"))
        config = state["config"]

        # Create a new instance
        instance = cls(config)

        # Load the model and processor
        instance.model = config.model_class.from_pretrained(path)
        instance.processor = config.processor_class.from_pretrained(path)

        # Load optimizer state
        instance.optimizer.load_state_dict(state["optimizer_state"])

        return instance
