import logging

import numpy as np

from src.datasets.data_processing import add_bias_term
from src.models.base import Model
from src.models.utils.optimizers import gradient_descent, stochastic_gradient_descent

logger = logging.getLogger(__name__)

np.random.seed(0)


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
