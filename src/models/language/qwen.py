from pathlib import Path
from threading import Thread
from typing import Generator

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
    logging,
)

from src.models.base import TorchModel

logging.set_verbosity_error()


class QwenModel(TorchModel):
    """
    TODO: Explore
    - https://qwenlm.github.io/blog/qwen2.5/
    - https://huggingface.co/Qwen
        https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f
        https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e
        https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d
    - https://github.com/QwenLM/Qwen2.5
    - "unsloth/Qwen2.5-0.5B-bnb-4bit"
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct") -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=self.device
        ).to(self.device)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-0.5B-Instruct"
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.to(self.device))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: Path) -> "QwenModel":
        model = cls()
        model.model = AutoModelForCausalLM.from_pretrained(path).to(model.device)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        return model

    def generate(
        self, inputs: dict[str, Tensor], streaming: bool = False
    ) -> str | Generator[str, None, None]:
        # Ensure inputs are on correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._generate_stream(inputs) if streaming else self._generate_non_stream(inputs)

    def _generate_non_stream(self, inputs: dict[str, Tensor]) -> str:
        output = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _generate_stream(self, inputs: dict[str, Tensor]) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = Thread(
            target=self.model.generate, kwargs=dict(inputs, streamer=streamer, max_new_tokens=512)
        )
        thread.start()

        yield from streamer
        thread.join()

    def fit(self, train_data: DataLoader, val_data: DataLoader | None = None) -> None:
        raise NotImplementedError("Training not implemented for Qwen model")

    def evaluate(self, data: DataLoader) -> dict[str, float]:
        raise NotImplementedError("Evaluation not implemented for Qwen model")


if __name__ == "__main__":
    model = QwenModel("Qwen/Qwen2-7B-Instruct")
    prompt = "Tell me a short joke"
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Move inputs to correct device immediately after tokenization
    model_inputs = {
        k: v.to(model.device) for k, v in model.tokenizer([text], return_tensors="pt").items()
    }

    print("Streaming result:")
    for chunk in model.generate(model_inputs, streaming=True):
        print(chunk, end="", flush=True)
    print()
