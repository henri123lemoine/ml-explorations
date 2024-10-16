from src.models.base import BaseInput, BaseModel, BaseOutput


class Pipeline:
    def __init__(self, model: BaseModel, input_processor: BaseInput, output_processor: BaseOutput):
        self.model = model
        self.input_processor = input_processor
        self.output_processor = output_processor

    def run(self, raw_input, streaming: bool = False):
        processed_input = self.input_processor.process(raw_input)
        model_inputs = self.model.prepare_inputs(processed_input)
        model_output = self.model.generate(model_inputs, streaming=streaming)
        if streaming:
            return self.output_processor.stream(model_output)
        else:
            return self.output_processor.process(model_output)
