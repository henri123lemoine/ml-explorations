import numpy as np

from src.datasets.data_processing import (
    add_bias_term,
    one_hot_decode,
    one_hot_encode,
    softmax,
)
from src.models.base import Model
from src.models.utils.optimizers import gradient_descent, stochastic_gradient_descent


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
