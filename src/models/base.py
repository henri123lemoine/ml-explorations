from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def prepare_inputs(self, inputs: Any) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, inputs: dict[str, Any], streaming: bool = False):
        pass

    @abstractmethod
    def train(self, training_data: Any, training_config: dict[str, Any]):
        pass
