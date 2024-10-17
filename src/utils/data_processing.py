import logging
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


def add_bias_term(X):
    # add a feature of all ones to X
    return np.c_[np.ones((X.shape[0], 1)), X]


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot_encode(y, classes):
    return np.eye(len(classes))[np.searchsorted(classes, y)]


def one_hot_decode(y, classes):
    return classes[np.argmax(y, axis=1)]


def k_fold_cross_validation(model, X, y, metric_function, k=5):
    n = len(y)
    indices = np.random.permutation(n)
    fold_sizes = [(n // k) + 1 if p < n % k else n // k for p in range(k)]
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    scores = []

    for fold in folds:
        test_index = fold
        train_index = [idx for idx in indices if idx not in fold]

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = metric_function(y_test, y_pred)
        scores.append(score)

    return scores


def train_test_split(X, y, split_ratio=0.8):
    shuffle_idx = np.random.permutation(len(y))
    train_size = int(split_ratio * len(y))
    X_train = X[shuffle_idx][:train_size]
    y_train = y[shuffle_idx][:train_size]
    X_test = X[shuffle_idx][train_size:]
    y_test = y[shuffle_idx][train_size:]
    return X_train, y_train, X_test, y_test


def normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def standardize(df):
    return (df - df.mean()) / df.std()


class NumpyDataLoader(IterableDataset):
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield (batch[0].numpy(), batch[1].numpy())

    def __len__(self):
        return len(self.dataloader)


def compute_dataset_stats(data_loader):
    # Compute the mean and standard deviation of the dataset for normalization
    mean = None
    M2 = None
    nb_samples = 0

    for batch in data_loader:
        data = batch[0].view(batch[0].size(0), batch[0].size(1), -1)
        batch_mean = torch.mean(data, dim=[0, 2])
        batch_var = torch.var(data, dim=[0, 2], unbiased=False)
        batch_samples = data.size(0)

        if mean is None:
            mean = batch_mean
            M2 = batch_var * (batch_samples - 1)
        else:
            delta = batch_mean - mean
            mean += delta * batch_samples / (nb_samples + batch_samples)
            M2 += batch_var * (batch_samples - 1) + delta**2 * nb_samples * batch_samples / (
                nb_samples + batch_samples
            )

        nb_samples += batch_samples

    std = torch.sqrt(M2 / nb_samples)
    return mean.numpy(), std.numpy()


def load_dataset(
    dataset_name,
    batch_size=32,
    normalize=True,
    flatten=True,
    shuffle=True,
    random_state=None,
    data_dir="./data/datasets",
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

    os.makedirs(data_dir, exist_ok=True)

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
        mean, std = compute_dataset_stats(preliminary_trainloader)
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
