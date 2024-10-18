import logging
from dataclasses import dataclass

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, IterableDataset

import datasets
from src.datasets.data_processing import dataset_stats
from src.settings import DATASETS_PATH

logger = logging.getLogger(__name__)


class NumpyDataLoader(IterableDataset):
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield (batch[0].numpy(), batch[1].numpy())

    def __len__(self):
        return len(self.dataloader)


def load_dataset(
    dataset_name,
    batch_size=32,
    normalize=True,
    flatten=True,
    shuffle=True,
    random_state=None,
    data_dir=DATASETS_PATH,
):
    """
    Loads a dataset using PyTorch's torchvision library, applies transformations, and prepares it for training and testing.

    Parameters:
        dataset_name (str): Name of the dataset to be loaded. Should be available in torchvision.datasets.
        batch_size (int): Number of samples per batch.
        normalize (bool): If True, normalize the dataset using its mean and standard deviation.
        flatten (bool): If True, flatten each sample in the dataset.
        shuffle (bool): If True, shuffle the training dataset.
        random_state (int): Seed for reproducibility.
        data_dir (str): Directory to store/load the dataset.

    Returns:
        numpy_trainloader (NumpyDataLoader): Iterable training data loader.
        numpy_testloader (NumpyDataLoader): Iterable test data loader.
    """
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    transform_list = [transforms.ToTensor()]

    # Load the dataset to compute stats
    try:
        dataset_class = getattr(torchvision.datasets, dataset_name)
    except AttributeError:
        raise ValueError(f"{dataset_name} not found in torchvision.datasets")

    preliminary_trainset = dataset_class(
        root=data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    preliminary_trainloader = DataLoader(
        preliminary_trainset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    if normalize:
        mean, std = dataset_stats(preliminary_trainloader)
        transform_list.append(transforms.Normalize(mean, std))

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    # Load the dataset again with the final transformations
    trainset = dataset_class(root=data_dir, train=True, download=True, transform=transform)
    testset = dataset_class(root=data_dir, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    numpy_trainloader = NumpyDataLoader(trainloader)
    numpy_testloader = NumpyDataLoader(testloader)

    return numpy_trainloader, numpy_testloader


@dataclass
class DataPoint:
    text: str
    label: int


def get_dataset(print_info=False):
    # https://huggingface.co/datasets/dair-ai/emotion

    datasets.logging.set_verbosity(datasets.logging.ERROR)

    train_dataset = datasets.load_dataset("dair-ai/emotion", cache_dir=DATASETS_PATH, split="train")
    validation_dataset = datasets.load_dataset(
        "dair-ai/emotion", cache_dir=DATASETS_PATH, split="validation"
    )
    test_dataset = datasets.load_dataset("dair-ai/emotion", cache_dir=DATASETS_PATH, split="test")

    # convert from IterableDataset to shuffled list
    # convert items from {'text': '...', 'label': i} to DataPoint('...', i)
    seed = 1
    train_dataset = list(
        map(lambda x: DataPoint(x["text"], x["label"]), list(train_dataset.shuffle(seed=seed)))
    )
    validation_dataset = list(
        map(lambda x: DataPoint(x["text"], x["label"]), list(validation_dataset.shuffle(seed=seed)))
    )
    test_dataset = list(
        map(lambda x: DataPoint(x["text"], x["label"]), list(test_dataset.shuffle(seed=seed)))
    )

    if print_info:
        logger.info("First entry from each dataset:")
        logger.info(f"train_dataset[0] = {train_dataset[0]}")
        logger.info(f"validation_dataset[0] = {validation_dataset[0]}")
        logger.info(f"test_dataset[0] = {test_dataset[0]}")

    return (train_dataset, validation_dataset, test_dataset)
