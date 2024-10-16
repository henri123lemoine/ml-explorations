from abc import ABC, abstractmethod
from typing import Any, Generator


class BaseModel(ABC):
    @abstractmethod
    def load(self, model_path: str, **kwargs):
        pass

    @abstractmethod
    def prepare_inputs(self, inputs: Any) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, inputs: dict[str, Any], streaming: bool = False) -> Any:
        pass

    @abstractmethod
    def train(self, training_data: Any, training_config: dict[str, Any]):
        pass

    @abstractmethod
    def save(self, save_path: str):
        pass


class BaseInput(ABC):
    @abstractmethod
    def process(self, raw_input: Any) -> dict[str, Any]:
        pass


class BaseOutput(ABC):
    @abstractmethod
    def process(self, model_output: Any) -> Any:
        pass

    @abstractmethod
    def stream(self, model_output: Generator) -> Generator[Any, None, None]:
        raise NotImplementedError("Streaming not implemented for this output processor")


class StreamOutput(BaseOutput):
    @abstractmethod
    def stream(self, model_output: Generator) -> Generator[Any, None, None]:
        pass

    def process(self, model_output: Any) -> Any:
        # Default implementation for non-streaming output
        return "".join(self.stream(model_output))


class SimpleTextInput(BaseInput):
    def process(self, raw_input: str) -> dict:
        return {
            "prompt": raw_input,
            "system_message": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        }


class SimpleTextOutput(BaseOutput):
    def process(self, model_output: str) -> str:
        return model_output


class SimpleStreamOutput(StreamOutput):
    def stream(self, model_output):
        for chunk in model_output:
            yield chunk

    def process(self, model_output: str) -> str:
        # This method is required by the BaseOutput abstract class
        return "".join(self.stream(model_output))
