from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(cls, path):
        pass

    @abstractmethod
    def generate(self, inputs: dict[str, Any], streaming: bool = False) -> Any:
        pass

    @abstractmethod
    def train(self, training_data: Any, training_config: dict[str, Any]):
        pass

    def fit(self, X, y):
        raise NotImplementedError("This model does not support the fit method")

    def predict(self, X):
        raise NotImplementedError("This model does not support the predict method")

    def evaluate(self, X, y):
        raise NotImplementedError("This model does not support the evaluate method")
