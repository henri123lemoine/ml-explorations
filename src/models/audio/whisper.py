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
        )
        model.processor = WhisperProcessor.from_pretrained(path)
        return model


class WhisperMLX(MLXModel[mx.array, mx.array, DataLoader]):
    """MLX-optimized Whisper model for Apple Silicon.

    Models:
    - mlx-community/whisper-tiny-mlx-4bit
    - mlx-community/whisper-large-v3-mlx-4bit
    """

    def __init__(self, model_name: str = "mlx-community/whisper-tiny-mlx-4bit") -> None:
        super().__init__()
        self.model_name = model_name

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing transcription results
        """
        return mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=self.model_name)["text"]


if __name__ == "__main__":
    import time
    from difflib import SequenceMatcher

    import numpy as np
    import soundfile as sf

    # The correct transcription text for comparison
    correct_transcription = (
        "Good morning. This Tuesday is election day. After months of spirited debate and vigorous campaigning, "
        "the time has come for Americans to make important decisions about our nation's future and encourage all "
        "Americans to go to the polls and vote. Election season brings out the spirit of competition between our political parties. "
        "And that competition is an essential part of a healthy democracy. But as the campaigns come to a close, Republicans, "
        "Democrats, and independents can find common ground on at least one point. Our system of representative democracy is one "
        "of America's greatest strengths. The United States was founded on the belief that all men are created equal. Every election day, "
        "millions of Americans of all races, religions, and backgrounds step into voting booths throughout the nation, whether they are rich or poor, "
        "old or young. Each of them has an equal share in choosing the path that our country will take. And every ballot they cast is a reminder that "
        "our founding principles are alive and well. Voting is one of the great privileges of American citizenship. And it has always required brave defenders. "
        "As you head to the polls next week, remember the sacrifices that have been made by generations of Americans in uniform to preserve our way of life. "
        "From Bunker Hill to Baghdad, the men and women of American armed forces have been devoted guardians of our democracy. All of us owe them and their families "
        "a special debt of gratitude on election day. Americans should also remember the important example that our elections set throughout the world. Young democracies "
        "from Georgia and Ukraine to Afghanistan and Iraq can look to the United States for proof that self-government can endure, and nations that still live under "
        "tyranny and oppression can find hope and inspiration in our commitment to liberty. For more than two centuries, Americans have demonstrated the ability of free "
        "people to choose their own leaders. Our nation has flourished because of its commitment to trusting the wisdom of our citizenry. In this year's election, we will "
        "see this tradition continue. And we will be reminded once again that we are blessed to live in a free nation guided by the will of the people. Thank you for listening."
    )

    # Load the audio file
    audio_path = "/Users/henrilemoine/Downloads/samples_gb0.wav"
    audio_data, sr = sf.read(audio_path)

    # Ensure mono and float32 format
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Method 1: Using Whisper
    model_whisper = Whisper()
    start_time = time.time()
    text_1 = model_whisper.transcribe(audio_data)
    time_whisper = time.time() - start_time

    # Method 2: Using WhisperMLX
    model_whisper_mlx = WhisperMLX()
    start_time = time.time()
    text_2 = model_whisper_mlx.transcribe(audio_path)
    time_whisper_mlx = time.time() - start_time

    # Define a function to compute similarity for accuracy measurement
    def compute_similarity(transcription, reference):
        return SequenceMatcher(None, transcription, reference).ratio()

    # Calculate similarities
    similarity_1 = compute_similarity(text_1, correct_transcription)
    similarity_2 = compute_similarity(text_2, correct_transcription)

    # Print results
    print("\n--- Transcription Results ---\n")
    print("Method 1: Whisper")
    print(f"Transcription:\n{text_1}")
    print(f"Time taken: {time_whisper:.2f} seconds")
    print(f"Similarity to correct transcription: {similarity_1:.2%}\n")

    print("Method 2: WhisperMLX")
    print(f"Transcription:\n{text_2}")
    print(f"Time taken: {time_whisper_mlx:.2f} seconds")
    print(f"Similarity to correct transcription: {similarity_2:.2%}\n")

    print("--- Correct Transcription for Reference ---\n")
    print(correct_transcription)
    print(correct_transcription)
