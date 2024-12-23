from typing import Type

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

from src.metrics import accuracy
from src.models.base import TorchModel
from src.models.utils.activations import ReLU
from src.models.utils.initializers import Initializer, Xavier
from src.models.utils.optimizers import Adam
from src.models.utils.schedulers import CosineAnnealing


class MLPLayer(nn.Module):
    """A single MLP layer with configurable initialization"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initializer: Type[Initializer] = Xavier,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

        # Initialize weights using the specified initializer
        initializer_instance = initializer()
        weight_init = torch.tensor(
            initializer_instance.initialize(in_features, out_features)
        ).float()
        self.linear.weight.data = weight_init
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return self.activation(x) if self.activation is not None else x


class MLP(TorchModel):
    """PyTorch implementation of a Multi-Layer Perceptron"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        num_classes: int = 10,
        activation_function: Type[nn.Module] = ReLU,
        optimizer: Type[torch.optim.Optimizer] = Adam,
        scheduler: Type[torch.optim.lr_scheduler._LRScheduler] = CosineAnnealing,
        initializer: Type[Initializer] = Xavier,
        lr: float = 0.001,
        **kwargs,
    ):
        super().__init__()

        # Build layer dimensions
        layer_sizes = [input_dim] + hidden_layers + [num_classes]

        # Create activation function instance
        activation = activation_function()
        if isinstance(activation, nn.Module):
            self.activation = activation
        else:
            # Convert custom activation to PyTorch module
            self.activation = type(
                "CustomActivation",
                (nn.Module,),
                {
                    "forward": lambda self, x: activation.function(x),
                    "backward": lambda self, x: activation.derivative(x),
                },
            )()

        # Build network layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            is_last_layer = i == len(layer_sizes) - 2
            layers.append(
                MLPLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    initializer=initializer,
                    activation=None if is_last_layer else self.activation,
                )
            )

        self.layers = nn.ModuleList(layers)
        self.criterion = nn.CrossEntropyLoss()

        # Setup optimizer and scheduler
        self.optimizer = optimizer(self.parameters(), lr=lr, **kwargs)
        self.scheduler = scheduler(self.optimizer, **kwargs) if scheduler else None

        # Move model to correct device
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network"""
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x: Tensor) -> Tensor:
        """Make predictions for the input"""
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, float]:
        """Single training step"""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()

        acc = accuracy(output.argmax(dim=1).cpu().numpy(), y.cpu().numpy())
        return loss, acc

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, float]:
        """Single validation step"""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            output = self.forward(x)
            loss = self.criterion(output, y)
            acc = accuracy(output.argmax(dim=1).cpu().numpy(), y.cpu().numpy())

        return loss, acc

    def fit(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
        epochs: int = 100,
        print_loss: bool = False,
        log_interval: int = 10,
    ) -> tuple[list[float], list[float]]:
        """Train the model"""
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            self.train()
            epoch_train_acc = []

            for batch_idx, batch in enumerate(train_data):
                _, train_acc = self.training_step(batch)
                epoch_train_acc.append(train_acc)

                if val_data is not None:
                    val_batch = next(iter(val_data))
                    _, val_acc = self.validation_step(val_batch)
                    val_accuracies.append(val_acc)

                train_accuracies.append(train_acc)

                if batch_idx % log_interval == 0 and print_loss:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Training Accuracy: {train_acc:.4f}, "
                        f"Validation Accuracy: {val_acc:.4f}"
                    )

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        return train_accuracies, val_accuracies

    def evaluate(self, data: DataLoader) -> dict[str, float]:
        """Evaluate the model"""
        self.eval()
        total_loss = 0
        total_acc = 0
        n_batches = 0

        with torch.no_grad():
            for batch in data:
                loss, acc = self.validation_step(batch)
                total_loss += loss.item()
                total_acc += acc
                n_batches += 1

        return {"loss": total_loss / n_batches, "accuracy": total_acc / n_batches}
