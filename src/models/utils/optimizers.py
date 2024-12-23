from abc import ABC, abstractmethod

import numpy as np

from src.datasets.data_processing import add_bias_term


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


### DEPRECATED ###


def gradient_descent(
    model,
    X,
    Y,
    predict_func,
    learning_rate,
    epochs,
    test_func,
    test_interval,
    test_start,
):
    test_results = []
    X = add_bias_term(X)
    num_samples, num_features = X.shape

    # Initialize weights using normal
    if len(Y.shape) == 1:
        model.weights = np.random.normal(0, 0.0005, num_features)
    else:
        model.weights = np.random.normal(0, 0.0005, (num_features, Y.shape[1]))

    # Gradient Descent
    for epoch in range(epochs):
        predictions = predict_func(model.weights, X)
        # Compute gradients
        dw = (1 / num_samples) * np.dot(X.T, (predictions - Y))
        # Update parameters
        model.weights -= learning_rate * dw

        if test_func is not None and epoch % test_interval == 0 and epoch >= test_start:
            test_results.append(test_func(model))

    return test_results


def stochastic_gradient_descent(
    model,
    X,
    Y,
    predict_func,
    learning_rate,
    epochs,
    batch_size,
    test_func,
    test_interval,
    test_start,
):
    test_results = []
    X = add_bias_term(X)
    num_samples, num_features = X.shape

    # Calculate the adjusted number of epochs
    adjusted_epochs = epochs * (num_samples // batch_size)

    # Initialize weights
    if len(Y.shape) == 1:
        model.weights = np.random.normal(0, 0.0005, num_features)
    else:
        model.weights = np.random.normal(0, 0.0005, (num_features, Y.shape[1]))

    # Stochastic Gradient Descent
    for epoch in range(adjusted_epochs):
        # Shuffle data
        shuffle = np.random.permutation(len(Y))
        X_shuffled = X[shuffle]
        Y_shuffled = Y[shuffle]

        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            Y_batch = Y_shuffled[i : i + batch_size]

            predictions = predict_func(model.weights, X_batch)
            # Compute gradients
            dw = (1 / batch_size) * X_batch.T @ (predictions - Y_batch)
            # Update parameters
            model.weights -= learning_rate * dw

        if test_func is not None and epoch % test_interval == 0 and epoch >= test_start:
            test_results.append(test_func(model))

    return test_results
