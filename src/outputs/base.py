from abc import ABC, abstractmethod
from typing import Any, Generator


class BaseOutput(ABC):
    @abstractmethod
    def process(self, model_output: Any) -> Any:
        pass


class StreamOutput(BaseOutput):
    @abstractmethod
    def stream(self, model_output: Generator) -> Generator:
        pass
