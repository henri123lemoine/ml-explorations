import logging
from abc import ABC, abstractmethod
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.data_processing import (
    add_bias_term,
    one_hot_decode,
    one_hot_encode,
    softmax,
)
from src.utils.metrics import accuracy
from src.utils.optimization import gradient_descent, stochastic_gradient_descent

logger = logging.getLogger(__name__)

np.random.seed(0)


class Model:
    # this should have a way to fit the model to some data
    def fit(self, X, Y):
        del X, Y
        raise NotImplementedError("Subclass must implement fit method")

    # and a way to predict the output of the model given some data
    def predict(self, X):
        del X
        raise NotImplementedError("Subclass must implement predict method")


class LinearRegression(Model):
    def __init__(self):
        self.weights = None

    # regardless of how we train the model, the prediction method will look the same
    def predict(self, X):
        if self.weights is None:
            raise Exception("Model not yet trained")
        X = add_bias_term(X)
        return X @ self.weights


class LinearRegressionAnalytic(LinearRegression):
    def fit(self, X, Y):
        X_b = add_bias_term(X)
        self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ Y


class LinearRegressionGD(LinearRegression):
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, Y, test_func=None, test_interval=100, test_start=0):
        return gradient_descent(
            self,
            X,
            Y,
            lambda weights, X: np.dot(X, weights),
            self.learning_rate,
            self.epochs,
            test_func,
            test_interval,
            test_start,
        )


class LinearRegressionSGD(LinearRegression):
    def __init__(self, learning_rate=0.1, epochs=100, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, Y, test_func=None, test_interval=100, test_start=0):
        return stochastic_gradient_descent(
            self,
            X,
            Y,
            lambda weights, X: X @ weights,
            self.learning_rate,
            self.epochs,
            self.batch_size,
            test_func,
            test_interval,
            test_start,
        )


class LogisticRegression(Model):
    def __init__(self):
        self.weights = None
        self.classes = None

    # returns the predicted class for each data point
    def predict(self, X):
        if self.classes is None:
            raise Exception("Model not yet trained")
        return one_hot_decode(self.predict_probabilities(X), self.classes)

    # returns the probability of each class for each data point
    def predict_probabilities(self, X):
        if self.weights is None:
            raise Exception("Model not yet trained")
        X = add_bias_term(X)
        scores = X @ self.weights
        probabilities = softmax(scores)
        return probabilities


class LogisticRegressionGD(LogisticRegression):
    def __init__(self, learning_rate=0.01, epochs=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, Y, test_func=None, test_interval=100, test_start=0):
        self.classes = np.unique(Y)
        Y = one_hot_encode(Y, self.classes)

        return gradient_descent(
            self,
            X,
            Y,
            lambda weights, X: softmax(X @ weights),
            self.learning_rate,
            self.epochs,
            test_func,
            test_interval,
            test_start,
        )


class LogisticRegressionSGD(LogisticRegression):
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=20):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, Y, test_func=None, test_interval=100, test_start=0):
        self.classes = np.unique(Y)
        Y = one_hot_encode(Y, self.classes)

        return stochastic_gradient_descent(
            self,
            X,
            Y,
            lambda weights, X: softmax(X @ weights),
            self.learning_rate,
            self.epochs,
            self.batch_size,
            test_func,
            test_interval,
            test_start,
        )


# ------------------------------ ACTIVATIONS ------------------------------


class Activation(ABC):
    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class Sigmoid(Activation):
    def function(self, x):
        # return 1 / (1 + np.exp(-x)) # this naïve one is prone to overflow

        # https://shaktiwadekar.medium.com/how-to-avoid-numerical-overflow-in-sigmoid-function-numerically-stable-sigmoid-function-5298b14720f6
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def derivative(self, x):
        sigmoid_x = self.function(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh(Activation):
    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        tanh_x = self.function(x)
        return 1 - tanh_x**2


class Linear(Activation):
    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def function(self, x):
        return np.maximum(self.alpha * x, x)

    def derivative(self, x):  # done in the numpy kindof branchless looking style
        decision = (x > 0).astype(float)  # 1 if x > 0, 0 otherwise
        return decision + self.alpha * (1 - decision)  # 1 if x > 0, alpha otherwise


# ------------------------------ LOSS FUNCTIONS ------------------------------


class Loss(ABC):
    @abstractmethod
    def compute(self, y_pred, y_true):
        pass

    @abstractmethod
    def derivative(self, y_pred, y_true):
        pass


class CrossEntropy(Loss):
    def compute(self, y_pred, y_true):
        # random initial weights might give negative predictions
        # (depending on the activation function) which will cause NaNs
        # in log-loss. We can just clamp the predictions to avoid this.

        return -np.sum(y_true * np.log(np.maximum(y_pred, 1e-8)))

    def derivative(self, y_pred, y_true):
        return y_pred - y_true


# ------------------------------ INITIALIZERS ------------------------------


class Initializer(ABC):
    @abstractmethod
    def initialize(self, size_in, size_out):
        pass


class Zeros(Initializer):
    def initialize(self, size_in, size_out):
        return np.zeros((size_in, size_out))


class Uniform(Initializer):
    def initialize(self, size_in, size_out):
        return np.random.uniform(-1, 1, (size_in, size_out))


class Gaussian(Initializer):
    def initialize(self, size_in, size_out):
        return np.random.randn(size_in, size_out)


class Xavier(Initializer):
    def initialize(self, size_in, size_out):
        return np.random.randn(size_in, size_out) * np.sqrt(2.0 / (size_in + size_out))


class Kaiming(Initializer):
    def initialize(self, size_in, size_out):
        return np.random.randn(size_in, size_out) * np.sqrt(2.0 / size_in)


# ------------------------------ SCHEDULERS ------------------------------


class LRScheduler(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_lr(self, lr, global_step):
        pass


class Constant_(LRScheduler):
    def __init__(self, **kwargs):
        pass

    def get_lr(self, lr, global_step):
        return lr


class ExponentialDecay(LRScheduler):
    def __init__(self, decay_rate, decay_steps, **kwargs):
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def get_lr(self, lr, global_step):
        return lr * (self.decay_rate ** (global_step / self.decay_steps))


class StepDecay(LRScheduler):
    def __init__(self, drop_rate, epochs_drop, **kwargs):
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop

    def get_lr(self, lr, global_step):
        return lr * (self.drop_rate ** (global_step // self.epochs_drop))


class TimeBasedDecay(LRScheduler):
    def __init__(self, decay_rate, **kwargs):
        self.decay_rate = decay_rate

    def get_lr(self, lr, global_step):
        return lr / (1 + self.decay_rate * global_step)


class StepWarmup(LRScheduler):
    def __init__(self, warmup_steps, init_lr, **kwargs):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr

    def get_lr(self, lr, global_step):
        if global_step < self.warmup_steps:
            return self.init_lr + (lr - self.init_lr) * (global_step / self.warmup_steps)
        return lr


class CosineAnnealing(LRScheduler):
    def __init__(self, T_max, eta_min=0, **kwargs):
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, lr, global_step):
        return (
            self.eta_min + (lr - self.eta_min) * (1 + np.cos(np.pi * global_step / self.T_max)) / 2
        )


# ------------------------------ OPTIMIZERS ------------------------------


class Optimizer(ABC):
    def __init__(self, lr=0.01, epsilon=1e-7, scheduler=None, **kwargs):
        self.lr = lr
        self.epsilon = epsilon
        self.scheduler = scheduler
        self.global_step = 0

    def update(self, w, dw, vw=None, sw=None):
        adjusted_lr = (
            self.scheduler.get_lr(self.lr, self.global_step)
            if self.scheduler is not None
            else self.lr
        )
        self.global_step += 1
        return self._update(w, dw, vw, sw, adjusted_lr)

    @abstractmethod
    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        pass

    def decay(self, factor):
        self.lr *= factor


class GD(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        return w - adjusted_lr * dw, None, None


class Momentum(Optimizer):
    def __init__(self, beta1=0.9, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1

    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        if vw is None:
            vw = np.zeros_like(dw)
        vw = self.beta1 * vw + (1 - self.beta1) * dw
        return w - adjusted_lr * vw, vw, None


class RMSProp(Optimizer):
    def __init__(self, beta2=0.999, **kwargs):
        super().__init__(**kwargs)
        self.beta2 = beta2

    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        if sw is None:
            sw = np.zeros_like(dw)
        sw = self.beta2 * sw + (1 - self.beta2) * dw**2
        return w - adjusted_lr * dw / (np.sqrt(sw) + self.epsilon), None, sw


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2

    def _update(self, w, dw, vw=None, sw=None, adjusted_lr=None):
        if vw is None:
            vw = np.zeros_like(dw)
        if sw is None:
            sw = np.zeros_like(dw)
        vw = self.beta1 * vw + (1 - self.beta1) * dw
        sw = self.beta2 * sw + (1 - self.beta2) * dw**2
        corrected_global_step = self.global_step + 1
        vw_corr = vw / (1 - self.beta1**corrected_global_step)
        sw_corr = sw / (1 - self.beta2**corrected_global_step)
        return w - adjusted_lr * vw_corr / (np.sqrt(sw_corr) + self.epsilon), vw, sw


# ----------------------------- REGULARIZERS ------------------------------


class Regularizer(ABC):
    @abstractmethod
    def apply(self, w):
        return 0


class None_(Regularizer):
    def apply(self, w):
        return 0


class L1(Regularizer):
    def __init__(self, lw=0.001):
        self.lw = lw

    def apply(self, w):
        return self.lw * np.sign(w)


class L2(Regularizer):
    def __init__(self, lw=0.9):
        self.lw = lw

    def apply(self, w):
        return w - self.lw * w


class L1L2(Regularizer):
    def __init__(self, lambda1=0.01, lambda2=0.01):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def apply(self, w):
        return w - (self.lambda1 * np.sign(w) + self.lambda2 * w)


# -------------------------------- MODELS ---------------------------------


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


class CNN(nn.Module):
    def __init__(
        self,
        num_channels=3,
        image_size=32,
        num_classes=10,
        optimizer=optim.Adam,
        loss_function=nn.CrossEntropyLoss,
        lr=0.001,
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

    def train_loop(self, trainloader, testloader, epochs=100, print_loss=False, log_interval=10):
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
