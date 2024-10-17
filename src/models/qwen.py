from threading import Thread
from typing import Any, Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
    logging,
)

from src.models.base import Model

# Set logging level to suppress info messages
logging.set_verbosity_error()


class QwenModel(Model):
    """
    TODO: Explore
    - https://qwenlm.github.io/blog/qwen2.5/
    - https://huggingface.co/Qwen
        https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f
        https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e
        https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d
    - https://github.com/QwenLM/Qwen2.5
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer: PreTrainedTokenizerBase | None = None

    def load(self, path: str = None, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            path or self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path or self.model_name)

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def generate(self, inputs: dict[str, Any], streaming: bool = False):
        if streaming:
            return self._generate_stream(inputs)
        else:
            return self._generate_non_stream(inputs)

    def train(self, training_data: Any, training_config: dict[str, Any]):
        raise NotImplementedError("Training not implemented for Qwen model yet")

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


if __name__ == "__main__":
    model = QwenModel()
    model.load()

    prompt = "Tell me a short joke"
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = model.tokenizer([text], return_tensors="pt")
    model_output = model.generate(model_inputs, streaming=True)

    print("Streaming result:")
    for chunk in model_output:
        print(chunk, end="", flush=True)
