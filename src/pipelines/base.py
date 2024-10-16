from typing import List

from src.inputs.base import BaseInput
from src.models.base import BaseModel
from src.outputs.base import BaseOutput


class Pipeline:
    def __init__(self, model: BaseModel, input_processor: BaseInput, output_processor: BaseOutput):
        self.model = model
        self.input_processor = input_processor
        self.output_processor = output_processor

    def run(self, raw_input):
        processed_input = self.input_processor.process(raw_input)
        model_output = self.model.generate(processed_input)
        return self.output_processor.process(model_output)


class CompositePipeline:
    def __init__(self, pipelines: List[Pipeline]):
        self.pipelines = pipelines

    def run(self, initial_input):
        result = initial_input
        for pipeline in self.pipelines:
            result = pipeline.run(result)
        return result
