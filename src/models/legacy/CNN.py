import logging
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.metrics import accuracy
from src.models.base import TorchModel

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """A convolutional block with batch norm and activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
        use_maxpool: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if use_maxpool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.activation(self.bn(self.conv(x))))


class CNN(TorchModel):
    """PyTorch CNN implementation with modern practices"""

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 32,
        num_classes: int = 10,
        conv_channels: List[int] = [32, 64],
        fc_features: List[int] = [128],
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        dropout_rate: float = 0.2,
        use_batchnorm: bool = True,
        lr: float = 0.001,
        **kwargs,
    ):
        super().__init__()

        # Build convolutional layers
        conv_layers = []
        in_channels = num_channels

        for out_channels in conv_channels:
            conv_layers.append(ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate size after convolutions
        self.image_size_after_conv = image_size // (2 ** len(conv_channels))
        conv_output_size = conv_channels[-1] * self.image_size_after_conv**2

        # Build fully connected layers
        fc_layers = []
        in_features = conv_output_size

        for out_features in fc_features:
            fc_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(out_features) if use_batchnorm else nn.Identity(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_features = out_features

        # Add final classification layer
        fc_layers.append(nn.Linear(in_features, num_classes))

        self.fc_layers = nn.Sequential(nn.Flatten(), *fc_layers)

        # Loss and optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.parameters(), lr=lr, **kwargs)
        self.scheduler = scheduler(self.optimizer, **kwargs) if scheduler else None

        # Initialize weights
        self.apply(self._init_weights)

        # Move to device
        self.to(self.device)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize model weights using modern practices"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network"""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def predict(self, x: Tensor) -> Tensor:
        """Make predictions for the input"""
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, float]:
        """Single training step"""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()

        # Gradient clipping (optional but can help stability)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        acc = accuracy(output.argmax(dim=1).cpu().numpy(), y.cpu().numpy())
        return loss, acc

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, float]:
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
        val_data: Optional[DataLoader] = None,
        epochs: int = 100,
        print_loss: bool = False,
        log_interval: int = 10,
    ) -> Tuple[List[float], List[float]]:
        """Train the model"""
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            self.train()
            epoch_train_acc = []
            epoch_val_acc = []

            for batch_idx, batch in enumerate(train_data):
                _, train_acc = self.training_step(batch)
                epoch_train_acc.append(train_acc)

                # Validation
                if val_data is not None:
                    try:
                        val_batch = next(iter(val_data))
                    except StopIteration:
                        val_data_iter = iter(val_data)
                        val_batch = next(val_data_iter)

                    _, val_acc = self.validation_step(val_batch)
                    epoch_val_acc.append(val_acc)

                train_accuracies.append(train_acc)
                if val_data is not None:
                    val_accuracies.append(val_acc)

                if print_loss and (
                    batch_idx % log_interval == 0 or batch_idx == len(train_data) - 1
                ):
                    val_msg = f", Validation Accuracy: {val_acc:.4f}" if val_data else ""
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Training Accuracy: {train_acc:.4f}{val_msg}"
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
