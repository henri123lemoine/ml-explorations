from abc import ABC, abstractmethod
from typing import Any


class BaseInput(ABC):
    @abstractmethod
    def process(self, raw_input: Any) -> Any:
        pass
