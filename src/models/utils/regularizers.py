from abc import ABC, abstractmethod

import numpy as np


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
