from threading import Thread
from typing import Any, Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    logging,
)

from src.models.base import BaseModel

# Set logging level to suppress info messages
logging.set_verbosity_error()


class QwenModel(BaseModel):
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def prepare_inputs(self, inputs: dict[str, str]) -> dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": inputs["system_message"]},
            {"role": "user", "content": inputs["prompt"]},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer([text], return_tensors="pt")

    def generate(self, inputs: dict[str, Any], streaming: bool = False):
        if streaming:
            return self._generate_stream(inputs)
        else:
            return self._generate_non_stream(inputs)

    def _generate_non_stream(self, inputs: dict[str, torch.Tensor]) -> str:
        inputs = inputs.to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _generate_stream(self, inputs: dict[str, torch.Tensor]) -> Generator[str, None, None]:
        inputs = inputs.to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def train(self, training_data: Any, training_config: dict[str, Any]):
        raise NotImplementedError("Training not implemented for Qwen model yet")
