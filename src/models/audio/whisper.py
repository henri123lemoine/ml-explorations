from pathlib import Path

import mlx.core as mx
import mlx_whisper
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import logging as transformers_logging

from src.models.base import MLXModel, TorchModel

transformers_logging.set_verbosity_error()


class Whisper(TorchModel):
    """OpenAI's Whisper model for speech recognition."""

    def __init__(self, model_name: str = "openai/whisper-tiny.en") -> None:
        super().__init__()
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        ).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.to(self.device, dtype=self.torch_dtype))

    def transcribe(self, audio: Tensor) -> str:
        """
        Transcribe audio using Whisper.

        Args:
            audio: Raw audio tensor of shape (samples,) or (batch, samples)
                  Expected sample rate: 16000Hz
        """
        # Convert to numpy for the processor
        if isinstance(audio, Tensor):
            audio = audio.numpy()

        # Process the audio
        inputs = self.processor(
            audio,
            return_tensors="pt",
            sampling_rate=16000,
        )

        # Move to device and correct dtype
        inputs = {
            k: v.to(self.device, dtype=self.torch_dtype if k == "input_features" else v.dtype)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def load(cls, path: Path) -> "Whisper":
        model = cls()
        model.model = WhisperForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=model.torch_dtype,
            device_map=model.device,
        ).to(model.device)
        model.processor = WhisperProcessor.from_pretrained(path)
        return model

    def fit(self, train_data: DataLoader, val_data: DataLoader | None = None) -> None:
        raise NotImplementedError("Training not implemented for Whisper model")

    def evaluate(self, data: DataLoader) -> dict[str, float]:
        raise NotImplementedError("Evaluation not implemented for Whisper model")


class WhisperMLX(MLXModel[mx.array, mx.array, DataLoader]):
    """MLX-optimized Whisper model for Apple Silicon.

    Models:
    - mlx-community/whisper-tiny-mlx-4bit
    - mlx-community/whisper-large-v3-mlx-4bit
    """

    def __init__(self, model_name: str = "mlx-community/whisper-tiny-mlx-4bit") -> None:
        super().__init__()
        self.model_name = model_name
        # The model will be loaded on first use

    def transcribe(self, audio_path: str | Path) -> dict[str, str]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing transcription results
        """
        return mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=self.model_name)

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError("Direct model call notimplemented")

    def predict(self, x: mx.array) -> mx.array:
        raise NotImplementedError("Direct prediction not implemented")

    def fit(self, train_data: DataLoader, val_data: DataLoader | None = None) -> None:
        raise NotImplementedError("Training not implemented")

    def evaluate(self, data: DataLoader) -> dict[str, float]:
        raise NotImplementedError("Evaluation not implemented")


if __name__ == "__main__":
    import numpy as np
    import soundfile as sf
    import torch

    audio_path = "/Users/henrilemoine/Downloads/samples_gb0.wav"

    # method 1
    model = Whisper()
    audio_data, sr = sf.read(audio_path)

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    text_1 = model.transcribe(audio_data)

    # method 2
    model = WhisperMLX()
    text_2 = model.transcribe(audio_path)["text"]

    # output
    print(f"Transcription: {text_1}\n")
    print(f"Transcription: {text_2}")
