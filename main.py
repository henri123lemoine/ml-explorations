from src.models.base import SimpleStreamOutput, SimpleTextInput
from src.models.qwen import QwenModel
from src.pipelines.base import Pipeline


def main():
    model = QwenModel()
    model.load()

    input_processor = SimpleTextInput()
    output_processor = SimpleStreamOutput()

    pipeline = Pipeline(model, input_processor, output_processor)

    print("Streaming result:")
    for chunk in pipeline.run("Tell me a short joke", streaming=True):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()
