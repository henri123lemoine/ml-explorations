import datasets

from ..retrieval import DataPoint
from .base import TextDataset


class EmotionDataset(TextDataset):
    def _create_data(self):
        dataset = datasets.load_dataset(
            "dair-ai/emotion", cache_dir=self.cache_dir, split=self.split
        )
        return [DataPoint(x["text"], x["label"]) for x in dataset.shuffle(seed=1)]
