from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def prepare_inputs(self, inputs: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], streaming: bool = False):
        pass

    @abstractmethod
    def train(self, training_data: Any, training_config: Dict[str, Any]):
        pass
