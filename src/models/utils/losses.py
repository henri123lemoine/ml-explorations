from abc import ABC, abstractmethod

import numpy as np

np.random.seed(0)


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
