from typing import Any, Generator

import requests
import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor, logging

from src.models.base import Model

logging.set_verbosity_error()


class AriaModel(Model):
    def __init__(self):
        super().__init__()
        self.model: nn.Module | None = None
        self.processor = None

    def load(self, path: str = "rhymes-ai/Aria", **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    def generate(self, inputs: dict[str, Any], streaming: bool = False) -> Any:
        if streaming:
            return self._generate_stream(inputs)
        else:
            return self._generate_non_stream(inputs)

    def fit(self, X: Any, y: Any):
        raise NotImplementedError("Training not implemented for Aria model yet")

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("Predict method not implemented for Aria model")

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        raise NotImplementedError("Evaluate method not implemented for Aria model")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _generate_non_stream(self, inputs: dict[str, Any]) -> str:
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.generate(
                **inputs,
                max_new_tokens=500,
                tokenizer=self.processor.tokenizer,
            )
            output_ids = output[0][inputs["input_ids"].shape[1] :]
            result = self.processor.decode(output_ids, skip_special_tokens=True)
        return result

    def _generate_stream(self, inputs: dict[str, Any]) -> Generator[str, None, None]:
        raise NotImplementedError("Streaming not yet implemented for Aria model")


def load_image_from_url(url: str) -> Image.Image:
    return Image.open(requests.get(url, stream=True).raw)


def prepare_inputs(raw_input: dict[str, Any], processor, model) -> dict[str, Any]:
    messages = raw_input.get("messages", [])
    image_url = raw_input.get("image_url")

    image = load_image_from_url(image_url) if image_url else None

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = (
        processor(text=[text], images=image, return_tensors="pt")
        if image
        else processor(text=[text], return_tensors="pt")
    )

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return {k: v.to(model.device) for k, v in inputs.items()}


if __name__ == "__main__":
    model = AriaModel()
    model.load()

    raw_input = {
        "messages": [
            {"role": "system", "content": "You are Aria, a helpful AI assistant."},
            {"role": "user", "content": "Tell me about this image."},
        ],
        "image_url": "https://example.com/image.jpg",  # Replace with an actual image URL
    }

    model_inputs = prepare_inputs(raw_input, model.processor, model.model)
    output = model.generate(model_inputs)
    print(output)
