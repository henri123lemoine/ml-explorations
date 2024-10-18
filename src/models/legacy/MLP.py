import logging
from typing import Type

import numpy as np

from src.models.base import Model
from src.models.utils.activations import Activation, ReLU
from src.models.utils.initializers import Initializer, Xavier
from src.models.utils.losses import CrossEntropy, Loss
from src.models.utils.optimizers import GD, Optimizer
from src.models.utils.regularizers import None_, Regularizer
from src.models.utils.schedulers import CosineAnnealing, LRScheduler
from src.utils.metrics import accuracy

logger = logging.getLogger(__name__)

np.random.seed(0)


class MLP(Model):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        num_classes: int = 10,
        activation_function: Type[Activation] = ReLU,
        optimizer: Type[Optimizer] = GD,
        scheduler: Type[LRScheduler] = CosineAnnealing,
        regularizer: Type[Regularizer] = None_,
        loss_function: Type[Loss] = CrossEntropy,
        initializer: Type[Initializer] = Xavier,
        **kwargs,
    ):
        self.activation_function = activation_function()
        self.optimizer = optimizer(scheduler=scheduler(**kwargs), **kwargs)
        self.initializer = initializer()
        self.regularizer = regularizer()
        self.loss_function = loss_function()

        self.layers = [input_dim] + hidden_layers + [num_classes]
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = [
            self.initializer.initialize(self.layers[i], self.layers[i + 1])
            for i in range(len(self.layers) - 1)
        ]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.sw = [np.zeros_like(w) for w in self.weights]

    def forward(self, x, train_mode=True):
        self.a = [x]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], w) + b
            self.a.append(self.activation_function.function(z))
        return self.a[-1]

    def backward(self, x, y, loss_func):
        m = x.shape[0]
        loss_derivative = loss_func.derivative(self.a[-1], y)
        dz = loss_derivative
        for i in reversed(range(len(self.a) - 1)):
            dw = np.dot(self.a[i].T, dz) / m
            updates = self.optimizer.update(self.weights[i], dw, self.vw[i], self.sw[i])
            self.weights[i] = updates[0]
            if len(updates) > 1:
                self.vw[i] = updates[1]
            if len(updates) > 2:
                self.sw[i] = updates[2]
            dz = np.dot(dz, self.weights[i].T) * self.activation_function.derivative(self.a[i])

    def train(self, trainloader, testloader, epochs=100, print_loss=False, log_interval=10):
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            test_iterator = iter(testloader)

            for batch_idx, (X_batch_train, y_batch_train) in enumerate(trainloader):
                y_batch_one_hot = np.eye(self.layers[-1])[y_batch_train]
                yh = self.forward(X_batch_train)
                self.backward(X_batch_train, y_batch_one_hot, self.loss_function)
                train_accuracy = accuracy(np.argmax(yh, axis=1), y_batch_train)
                train_accuracies.append(train_accuracy)

                try:
                    X_batch_test, y_batch_test = next(test_iterator)
                except StopIteration:
                    test_iterator = iter(testloader)
                    X_batch_test, y_batch_test = next(test_iterator)

                yh_test = self.forward(X_batch_test, train_mode=False)

                test_accuracy = accuracy(np.argmax(yh_test, axis=1), y_batch_test)
                test_accuracies.append(test_accuracy)

                if batch_idx % log_interval == 0 or batch_idx == len(trainloader) - 1:
                    if print_loss:
                        message = (
                            f"Epoch {epoch}, Batch {batch_idx}, "
                            f"Training Accuracy: {train_accuracy:.4f}, "
                            f"Test Accuracy: {test_accuracy:.4f}"
                        )
                        logger.info(message)

        return train_accuracies, test_accuracies
