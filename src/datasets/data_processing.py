import logging

import numpy as np
import torch

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


def dataset_stats(data_loader):
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
