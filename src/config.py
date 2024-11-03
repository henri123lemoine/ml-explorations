from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import torch
import yaml
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.settings import DATASETS_PATH, MODELS_PATH


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict[str, Any], config_path: str):
    with open(config_path, "w") as f:
        yaml.dump(config, f)


@dataclass
class PretrainedConfig:
    """Configuration for pretrained models"""

    model_name: str
    model_class: Type[PreTrainedModel]
    processor_class: Type[PreTrainedTokenizerBase]
    num_labels: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    freeze_backbone: bool = True
    backbone_attr: str = "base_model"
    classifier_attr: str = "classifier"
    save_path: str | Path = MODELS_PATH
    checkpoint_name: str = "best_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""

    cache_dir: Path = DATASETS_PATH
    max_images: int = 5000
    neg_ratio: float = 1.0
    train_ratio: float = 0.8
    random_seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    bicycle_label: int = 2  # COCO dataset bicycle class label
    use_augmentation: bool = True
