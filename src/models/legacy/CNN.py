import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.metrics import accuracy
from src.models.base import Model

logger = logging.getLogger(__name__)


class CNN(Model):
    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 32,
        num_classes: int = 10,
        optimizer=optim.Adam,
        loss_function=nn.CrossEntropyLoss,
        lr: float = 0.001,
        **kwargs,
    ):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate the size of the image after the convolutional layers
        self.image_size_after_conv = image_size // 4  # Two max pooling layers

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.image_size_after_conv**2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss_function = loss_function()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def train_loop(
        self,
        trainloader,
        testloader,
        epochs: int = 100,
        print_loss: bool = False,
        log_interval: int = 10,
    ):
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            for batch_idx, (X_batch_train, y_batch_train) in enumerate(trainloader):
                X_batch_train = torch.tensor(X_batch_train, dtype=torch.float32)
                y_batch_train = torch.tensor(y_batch_train, dtype=torch.long)

                self.optimizer.zero_grad()
                y_pred = self.forward(X_batch_train)
                loss = self.loss_function(y_pred, y_batch_train)
                loss.backward()
                self.optimizer.step()

                train_accuracy = accuracy(torch.argmax(y_pred, dim=1), y_batch_train)
                train_accuracies.append(train_accuracy)

                with torch.no_grad():
                    self.eval()  # Set the model to evaluation mode
                    X_batch_test, y_batch_test = next(iter(testloader))
                    X_batch_test = torch.tensor(X_batch_test, dtype=torch.float32)
                    y_batch_test = torch.tensor(y_batch_test, dtype=torch.long)

                    y_pred_test = self.forward(X_batch_test)
                    test_accuracy = accuracy(torch.argmax(y_pred_test, dim=1), y_batch_test)
                    test_accuracies.append(test_accuracy)

                if batch_idx % log_interval == 0 or batch_idx == len(trainloader) - 1:
                    if print_loss:
                        message = (
                            f"Epoch {epoch}, Batch {batch_idx}, "
                            f"Training Accuracy: {train_accuracy:.4f}, "
                            f"Test Accuracy: {test_accuracy:.4f}"
                        )
                        print(message)

        return train_accuracies, test_accuracies
