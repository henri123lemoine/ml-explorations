from pathlib import Path

import mlx_whisper.audio as audio
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import logging as transformers_logging

from src.models.base import TorchModel

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


if __name__ == "__main__":
    import numpy as np
    import soundfile as sf
    import torch
    from torchaudio.transforms import Resample

    model = Whisper()

    audio, sr = sf.read("/Users/henrilemoine/Downloads/samples_gb0.wav")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    # Convert to tensor for resampling
    audio = torch.from_numpy(audio.astype(np.float32))
    # Resample if needed
    if sr != 16000:
        resampler = Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)

    text = model.transcribe(audio)
    print(f"Transcription: {text}")
    print(f"Transcription: {text}")
