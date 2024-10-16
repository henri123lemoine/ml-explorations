# https://huggingface.co/rhymes-ai/Aria

# src/models/aria.py
from typing import Any, Generator

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from src.models.base import BaseModel
from src.utils import (
    generate_output,
    load_image_from_url,
    move_inputs_to_device,
    prepare_inputs,
)


class AriaModel(BaseModel):
    def __init__(self):
        self.model = None
        self.processor = None

    def load(self, model_path: str = "rhymes-ai/Aria", **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        messages = inputs.get("messages", [])
        image_url = inputs.get("image_url")

        image = load_image_from_url(image_url) if image_url else None

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        model_inputs = prepare_inputs(self.processor, text, image)
        return move_inputs_to_device(model_inputs, self.model)

    def generate(self, inputs: dict[str, Any], streaming: bool = False) -> Any:
        if streaming:
            return self._generate_stream(inputs)
        else:
            return self._generate_non_stream(inputs)

    def _generate_non_stream(self, inputs: dict[str, Any]) -> str:
        return generate_output(self.model, inputs, self.processor)

    def _generate_stream(self, inputs: dict[str, Any]) -> Generator[str, None, None]:
        # Implement streaming logic here
        raise NotImplementedError("Streaming not yet implemented for Aria model")

    def train(self, training_data: Any, training_config: dict[str, Any]):
        raise NotImplementedError("Training not implemented for Aria model yet")

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
