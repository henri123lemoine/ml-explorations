# `mlx-community/whisper-large-v3-mlx-4bit`
import torch
from torch import nn
from transformers import AutoModelForCausalLM, logging

from src.models.base import Model

logging.set_verbosity_error()


class WhisperModel(Model):
    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-mlx-4bit"):
        super().__init__()
        self.model_name = model_name
        self.model: nn.Module = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )


if __name__ == "__main__":
    model = WhisperModel()
    print(model.model_name)
    print(model.model)
