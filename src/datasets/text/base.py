from pathlib import Path

from src.datasets.base import BaseDataset


class TextDataset(BaseDataset):
    """Base class for text datasets"""

    def __init__(
        self,
        cache_dir: Path,
        split: str = "train",
        tokenizer=None,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        super().__init__(cache_dir, split)

    def __getitem__(self, idx):
        text = self.data[idx]
        if self.tokenizer:
            return self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        return text
