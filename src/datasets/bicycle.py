from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from datasets import load_dataset
from src.config import DatasetConfig


class BicycleDataset(Dataset):
    """Dataset for bicycle classification with efficient two-level caching."""

    def __init__(
        self,
        processor: Any,
        split: str = "train",
        config: DatasetConfig | None = None,
        transform_fn: Callable | None = None,
    ):
        self.config = config or DatasetConfig()
        self.processor = processor
        self.transform_fn = transform_fn
        self.split = split
        self.images, self.labels = self._load_or_create_dataset()

    def _get_cache_paths(self) -> dict[str, Path]:
        """Get paths for both metadata and processed dataset caches."""
        cache_dir = self.config.cache_dir / "coco2017"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return {
            "metadata": cache_dir / "metadata.pt",
            "processed": cache_dir
            / f"processed_{self.config.max_images}_{self.config.neg_ratio}.pt",
        }

    def _has_bicycle(self, example: dict) -> bool:
        """Check if an example contains a bicycle."""
        try:
            # Labels are in objects['label'] as a list of integers
            return self.config.bicycle_label in example["objects"]["label"]
        except (KeyError, TypeError) as e:
            print(f"\nError in _has_bicycle: {str(e)}")
            print(f"Example type: {type(example)}")
            return False

    def _get_or_create_metadata(self) -> dict[str, np.ndarray]:
        """Load or create cached metadata"""
        cache_paths = self._get_cache_paths()

        if cache_paths["metadata"].exists():
            print("Loading cached metadata...")
            return torch.load(cache_paths["metadata"])  # YOLO NO SAFETY

        print("Creating dataset metadata (this will only happen once)...")
        dataset = load_dataset("rafaelpadilla/coco2017", split="train")

        bicycle_dataset = dataset.filter(
            lambda x: self.config.bicycle_label in x["objects"]["label"],
            num_proc=4,
            load_from_cache_file=True,
        )

        bicycle_indices = bicycle_dataset.select_columns(["image_id"]).to_pandas().index.values
        all_indices = np.arange(len(dataset))
        non_bicycle_indices = np.setdiff1d(all_indices, bicycle_indices)

        metadata = {
            "bicycle_indices": bicycle_indices,
            "non_bicycle_indices": non_bicycle_indices,
            "total_images": len(dataset),
        }

        torch.save(metadata, cache_paths["metadata"])  # YOLO NO SAFETY
        return metadata

    def _load_or_create_dataset(self) -> tuple[list[Image.Image], list[int]]:
        """Load or create dataset"""
        cache_paths = self._get_cache_paths()

        if cache_paths["processed"].exists():
            cache_data = torch.load(cache_paths["processed"])  # YOLO NO SAFETY
            split_data = cache_data[self.split]
            return split_data["images"], split_data["labels"]

        metadata = self._get_or_create_metadata()
        return self._create_new_dataset(metadata)

    def _create_new_dataset(
        self, metadata: dict[str, np.ndarray]
    ) -> tuple[list[Image.Image], list[int]]:
        """Create new dataset"""
        n_pos = min(
            len(metadata["bicycle_indices"]),
            self.config.max_images // (1 + int(self.config.neg_ratio)),
        )
        n_neg = min(int(n_pos * self.config.neg_ratio), len(metadata["non_bicycle_indices"]))

        rng = np.random.RandomState(self.config.random_seed)
        pos_indices = rng.choice(metadata["bicycle_indices"], n_pos, replace=False)
        neg_indices = rng.choice(metadata["non_bicycle_indices"], n_neg, replace=False)

        dataset = load_dataset("rafaelpadilla/coco2017", split="train")
        all_indices = np.concatenate([pos_indices, neg_indices])
        selected_dataset = dataset.select(all_indices)

        images = []
        labels = []

        for i in tqdm(range(n_pos), desc="Loading bicycle images"):
            images.append(selected_dataset[i]["image"])
            labels.append(0)  # Bicycle is 0

        for i in tqdm(range(n_pos, n_pos + n_neg), desc="Loading non-bicycle images"):
            images.append(selected_dataset[i]["image"])
            labels.append(1)  # Non-bicycle is 1

        combined = list(zip(images, labels))
        rng.shuffle(combined)
        images, labels = zip(*combined)
        images, labels = list(images), list(labels)

        split_idx = int(len(images) * self.config.train_ratio)

        # Log statistics before caching
        train_labels = labels[:split_idx]
        val_labels = labels[split_idx:]
        print("\nDataset Statistics:")
        print(f"Training set: {len(train_labels)} images")
        print(f"  Bicycles: {train_labels.count(0)}")
        print(f"  Non-bicycles: {train_labels.count(1)}")
        print(f"Validation set: {len(val_labels)} images")
        print(f"  Bicycles: {val_labels.count(0)}")
        print(f"  Non-bicycles: {val_labels.count(1)}")

        cache_data = {
            "train": {"images": images[:split_idx], "labels": labels[:split_idx]},
            "val": {"images": images[split_idx:], "labels": labels[split_idx:]},
        }
        torch.save(cache_data, self._get_cache_paths()["processed"])

        if self.split == "train":
            return images[:split_idx], labels[:split_idx]
        return images[split_idx:], labels[split_idx:]

    def _split_and_cache_dataset(self, images: list[Image.Image], labels: list[int]) -> None:
        """Split dataset and save to cache."""
        split_idx = int(len(images) * self.config.train_ratio)
        cache_data = {
            "train": {"images": images[:split_idx], "labels": labels[:split_idx]},
            "val": {"images": images[split_idx:], "labels": labels[split_idx:]},
        }

        # Use safer torch.save
        torch.save(cache_data, self._get_cache_paths()["processed"], weights_only=True)

    def _log_stats(self, images: list[Image.Image], labels: list[int]) -> None:
        """Log dataset statistics."""
        n_pos = sum(labels)
        print(
            f"\n{self.split} dataset loaded:"
            f"\nTotal images: {len(images)}"
            f"\nBicycle images: {n_pos}"
            f"\nNon-bicycle images: {len(images) - n_pos}\n"
        )

    def _ensure_rgb(self, image: Any) -> np.ndarray:
        """Convert image to RGB format."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3:
                if image.shape[-1] == 4:
                    return image[..., :3]
                elif image.shape[-1] == 3:
                    return image
                elif image.shape[-1] == 1:
                    return np.repeat(image, 3, axis=-1)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        raise ValueError(f"Unsupported image format: {type(image)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[dict, int]:
        image = self.images[idx]
        label = self.labels[idx]

        # Ensure RGB format
        image = self._ensure_rgb(image)

        # Apply custom transform if provided
        if self.transform_fn is not None:
            image = self.transform_fn(image)

        # Process image for model
        processed = self.processor(images=image, return_tensors="pt")
        processed = {k: v.squeeze(0) for k, v in processed.items()}

        return processed, label
