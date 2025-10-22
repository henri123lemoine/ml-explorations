from typing import Any, Callable, Type

from torch.utils.data import DataLoader, Dataset

from src.datasets.image.bicycle import DatasetConfig  # TODO: Fix


def create_dataloaders(
    dataset_class: Type[Dataset],
    processor: Any,
    config: DatasetConfig,
    transform_fn: Callable | None = None,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Generic dataloader creation"""
    train_dataset = dataset_class(
        processor=processor,
        split="train",
        config=config,
        transform_fn=transform_fn,
        **dataset_kwargs,
    )
    val_dataset = dataset_class(processor=processor, split="val", config=config, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader
