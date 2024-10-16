from src.inputs.base import BaseInput
from src.models.qwen import QwenModel
from src.outputs.base import BaseOutput, StreamOutput
from src.pipelines.base import Pipeline


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


def test_non_streaming():
    model = QwenModel()
    model.load()

    input_processor = SimpleTextInput()
    output_processor = SimpleTextOutput()

    pipeline = Pipeline(model, input_processor, output_processor)

    result = pipeline.run("Tell me a short joke")
    print("Non-streaming result:")
    print(result)
    print()


def test_streaming():
    model = QwenModel()
    model.load()

    input_processor = SimpleTextInput()
    output_processor = SimpleStreamOutput()

    pipeline = Pipeline(model, input_processor, output_processor)

    print("Streaming result:")
    for chunk in pipeline.run("Tell me a short joke", streaming=True):
        print(chunk, end="", flush=True)
    print("\n")


def main():
    test_non_streaming()
    test_streaming()


if __name__ == "__main__":
    main()
