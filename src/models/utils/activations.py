import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


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
