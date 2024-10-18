import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

np.random.seed(0)


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
