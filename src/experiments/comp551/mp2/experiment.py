import logging
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from src.datasets.retrieval import load_dataset
from src.models.legacy.CNN import CNN
from src.models.legacy.MLP import MLP
from src.models.legacy.utils import *
from src.settings import CACHE_PATH
from src.utils.config import load_config
from src.utils.visualization import plot

logging.basicConfig(level=logging.INFO, format="[%(asctime)s - %(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

MP2_PATH = Path(__file__).resolve().parent
DATA_PATH = MP2_PATH / "data"
PLOTS_PATH = DATA_PATH / "plots"

PLOTS_PATH.mkdir(parents=True, exist_ok=True)


def experiment_1(
    n_epochs,
    input_dim,
    hidden_layers,
    num_classes,
    dataset_name,
    batch_size,
    normalize,
    flatten,
    lr,
    log_scale,
    activation_function,
    initializers,
    optimizer,
    regularizer,
    loss_function,
    scheduler,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    log_interval=100,
    **kwargs,
):
    """
    ### 1. Weight Initialization

    First of all, experiment with initializing your model weights in a few
    different ways. Create several MLPs with a single hidden layer having 128
    units, initializing the weights as:
    - (1) all zeros
    - (2) Uniform [-1, 1]
    - (3) Gaussian N(0,1)
    - (4) Xavier
    - (5) Kaiming

    After training these models, compare the effect of weight initialization on
    the training curves and test accuracy on the Fashion MNIST dataset.
    """
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=flatten
    )

    results = {}
    data = np.zeros((2, len(initializers), len(trainloader) * n_epochs, 1))

    for i, initializer in enumerate(initializers):
        logger.info(f"Initializer: {initializer.__name__}")

        try:
            if REUSE_CACHED_RESULTS:
                results[initializer.__name__], data[:, i, :, :] = pickle.load(
                    open(CACHE_PATH / f"experiment_1-{initializer.__name__}.pkl", "rb")
                )
                logger.info("Loaded cached results")
                continue
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            if "initializer" in kwargs:
                kwargs.pop("initializer")
            mlp = MLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                num_classes=num_classes,
                activation_function=activation_function,
                optimizer=optimizer,
                regularizer=regularizer,
                loss_function=loss_function,
                initializer=initializer,
                scheduler=scheduler,
                T_max=len(trainloader) * n_epochs,
                eta_min=lr / 10,
                **kwargs,
            )

            train_accuracies, test_accuracies = mlp.train(
                trainloader,
                testloader,
                epochs=n_epochs,
                print_loss=PRINT_LOSS,
                log_interval=log_interval,
            )

            data[0, i, :, :] = np.array(train_accuracies).reshape(-1, 1)
            data[1, i, :, :] = np.array(test_accuracies).reshape(-1, 1)
            results[initializer.__name__] = (train_accuracies, test_accuracies)

            # cache the results
            if CACHE_PARTIAL_RESULTS:
                pickle.dump(
                    (results[initializer.__name__], data[:, i, :, :]),
                    open(CACHE_PATH / f"experiment_1-{initializer.__name__}.pkl", "wb"),
                )

    plot(
        data=data,
        title="Effect of Weight Initialization on MLP Training",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        algs_info=[
            (initializer.__name__, colour, "line")
            for initializer, colour in zip(initializers, ["r", "g", "b", "c", "m"])
        ],
        log_scale=log_scale,
        include_y0=True,
        filename=(
            PLOTS_PATH / "experiment_1-weight_initialization_accuracy.png" if SAVE_FILES else None
        ),
        show=SHOW_GRAPHS,
    )

    return results


def experiment_2(
    n_epochs,
    input_dim,
    hidden_layerss,
    num_classes,
    dataset_name,
    batch_size,
    normalize,
    flatten,
    lr,
    log_scale,
    activation_function,
    initializer,
    optimizer,
    regularizer,
    loss_function,
    scheduler,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    log_interval=100,
    **kwargs,
):
    """
    ### 2. Effect of Non-linearity and Network Depth

    Create three different models:
    - (1) An MLP with no hidden layers, i.e., it directly maps the inputs to outputs.
    - (2) An MLP with a single hidden layer having 128 units and ReLU activations.
    - (3) An MLP with 2 hidden layers each having 128 units with ReLU activations.

    Note: All of these models should have a softmax layer at the end. After
    training, compare the test accuracy of these three models on the Fashion
    MNIST dataset. Comment on how non-linearity and network depth affects the
    accuracy. Are the results that you obtain expected?
    """
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=flatten
    )

    results = {}
    data = np.zeros((2, len(hidden_layerss), len(trainloader) * n_epochs, 1))

    for i, (name, hidden_layers) in enumerate(hidden_layerss.items()):
        logger.info(f"Model: {name}")
        try:
            if REUSE_CACHED_RESULTS:
                results[name], data[:, i, :, :] = pickle.load(
                    open(CACHE_PATH / f"experiment_2-{name}.pkl", "rb")
                )
                logger.info("Loaded cached results")
                continue
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            if "hidden_layers" in kwargs:
                kwargs.pop("hidden_layers")
            mlp = MLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                num_classes=num_classes,
                activation_function=activation_function,
                optimizer=optimizer,
                regularizer=regularizer,
                loss_function=loss_function,
                initializer=initializer,
                scheduler=scheduler,
                T_max=len(trainloader) * n_epochs,
                eta_min=lr / 10,
                **kwargs,
            )

            train_accuracies, test_accuracies = mlp.train(
                trainloader,
                testloader,
                epochs=n_epochs,
                print_loss=PRINT_LOSS,
                log_interval=log_interval,
            )

            data[0, i, :, :] = np.array(train_accuracies).reshape(-1, 1)
            data[1, i, :, :] = np.array(test_accuracies).reshape(-1, 1)
            results[name] = (train_accuracies, test_accuracies)

            if CACHE_PARTIAL_RESULTS:
                pickle.dump(
                    (results[name], data[:, i, :, :]),
                    open(CACHE_PATH / f"experiment_2-{name}.pkl", "wb"),
                )

    plot(
        data=data,
        title="Effect of Network Depth on MLP Training",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        algs_info=[
            (name, colour, "line") for name, colour in zip(hidden_layerss.keys(), ["r", "g", "b"])
        ],
        log_scale=log_scale,
        include_y0=True,
        filename=PLOTS_PATH / "experiment_2-network_depth_accuracy.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    return results


def experiment_3(
    n_epochs,
    input_dim,
    hidden_layers,
    num_classes,
    dataset_name,
    batch_size,
    normalize,
    flatten,
    lr,
    log_scale,
    activation_functions,
    initializer,
    optimizer,
    scheduler,
    regularizer,
    loss_function,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    log_interval=100,
    **kwargs,
):
    """
    ### 3. Activation Functions

    Take the last model above, the one with 2 hidden layers, and create two
    different copies of it in which you pick two activations of your choice
    (except ReLU) from the course slides. After training these two models on
    Fashion MNIST, compare their test accuracies with the model with ReLU
    activations. Comment on the performances of these models: which one is
    better and why? Are certain activations better than others? If the results
    are not as you expected, what could be the reason?
    """
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=flatten
    )

    results = {}
    data = np.zeros((2, len(activation_functions), len(trainloader) * n_epochs, 1))

    for i, activation_function in enumerate(activation_functions):
        name = activation_function.__name__
        logger.info(f"Activation Function: {name}")
        try:
            if REUSE_CACHED_RESULTS:
                results[name], data[:, i, :, :] = pickle.load(
                    open(CACHE_PATH / f"experiment_3-{name}.pkl", "rb")
                )
                logger.info("Loaded cached results")
                continue
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            if "activation_function" in kwargs:
                kwargs.pop("activation_function")
            mlp = MLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                num_classes=num_classes,
                activation_function=activation_function,
                optimizer=optimizer,
                regularizer=regularizer,
                loss_function=loss_function,
                initializer=initializer,
                scheduler=scheduler,
                T_max=len(trainloader) * n_epochs,
                eta_min=lr / 10,
                **kwargs,
            )

            train_accuracies, test_accuracies = mlp.train(
                trainloader,
                testloader,
                epochs=n_epochs,
                print_loss=PRINT_LOSS,
                log_interval=log_interval,
            )

            data[0, i, :, :] = np.array(train_accuracies).reshape(-1, 1)
            data[1, i, :, :] = np.array(test_accuracies).reshape(-1, 1)
            results[name] = (train_accuracies, test_accuracies)

            if CACHE_PARTIAL_RESULTS:
                pickle.dump(
                    (results[name], data[:, i, :, :]),
                    open(CACHE_PATH / f"experiment_3-{name}.pkl", "wb"),
                )

    plot(
        data=data,
        title="Effect of Activation Functions on MLP Training",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        algs_info=[
            (act_func.__name__, colour, "line")
            for act_func, colour in zip(activation_functions, ["r", "g", "b"])
        ],
        log_scale=log_scale,
        include_y0=True,
        filename=(
            PLOTS_PATH / "experiment_3-activation_functions_accuracy.png" if SAVE_FILES else None
        ),
        show=SHOW_GRAPHS,
    )

    return results


def experiment_4(
    n_epochs,
    input_dim,
    hidden_layers,
    num_classes,
    dataset_name,
    batch_size,
    normalize,
    flatten,
    lr,
    log_scale,
    activation_function,
    initializer,
    optimizer,
    regularizers,
    loss_function,
    scheduler,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    log_interval=100,
    **kwargs,
):
    """
    ### 4. Regularization

    Create an MLP with 2 hidden layers each having 128 units with ReLU
    activations as above. However, this time, independently add L1 and L2
    regularization to the network and train the MLP in this way. How do these
    regularizations affect the accuracy?
    """
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=flatten
    )

    results = {}
    data = np.zeros((2, len(regularizers), len(trainloader) * n_epochs, 1))

    for i, regularizer in enumerate(regularizers):
        name = regularizer.__name__
        if name == "None_":
            name = "No Regularization"
        logger.info(f"Regularizer: {name}")

        try:
            if REUSE_CACHED_RESULTS:
                results[name], data[:, i, :, :] = pickle.load(
                    open(CACHE_PATH / f"experiment_4-{name}.pkl", "rb")
                )
                logger.info(f"Loaded cached results for {name}")
                continue
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            if "regularizer" in kwargs:
                kwargs.pop("regularizer")
            mlp = MLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                num_classes=num_classes,
                activation_function=activation_function,
                optimizer=optimizer,
                regularizer=regularizer,
                loss_function=loss_function,
                initializer=initializer,
                scheduler=scheduler,
                T_max=len(trainloader) * n_epochs,
                eta_min=lr / 10,
                **kwargs,
            )

            train_accuracies, test_accuracies = mlp.train(
                trainloader,
                testloader,
                epochs=n_epochs,
                print_loss=PRINT_LOSS,
                log_interval=log_interval,
            )

            data[0, i, :, :] = np.array(train_accuracies).reshape(-1, 1)
            data[1, i, :, :] = np.array(test_accuracies).reshape(-1, 1)
            results[name] = (train_accuracies, test_accuracies)

            if CACHE_PARTIAL_RESULTS:
                pickle.dump(
                    (results[name], data[:, i, :, :]),
                    open(CACHE_PATH / f"experiment_4-{name}.pkl", "wb"),
                )

    plot(
        data=data,
        title="Effect of Regularization on MLP Training",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        algs_info=[
            (name, colour, "line")
            for name, colour in zip(
                [reg.__name__ for reg in regularizers], ["r", "g", "b", "c", "m"]
            )
        ],
        log_scale=log_scale,
        include_y0=True,
        filename=PLOTS_PATH / "experiment_4-regularization_accuracy.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    return results


def experiment_5(
    n_epochs,
    input_dim,
    hidden_layers,
    num_classes,
    dataset_name,
    batch_size,
    normalizes,
    flatten,
    lr,
    log_scale,
    activation_function,
    initializer,
    optimizer,
    regularizer,
    loss_function,
    scheduler,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    log_interval=100,
    **kwargs,
):
    """
    ### 5. Unnormalized Images

    Create an MLP with 2 hidden layers each having 128 units with ReLU
    activations as above. However, this time, train it with unnormalized
    images. How does this affect the accuracy?
    """
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=True, flatten=flatten
    )

    results = {}
    data = np.zeros((2, len(normalizes), len(trainloader) * n_epochs, 1))

    for i, normalize in enumerate(normalizes):
        trainloader, testloader = load_dataset(
            dataset_name, batch_size=batch_size, normalize=normalize, flatten=flatten
        )
        name = "Normalized" if normalize else "Unnormalized"
        logger.info(f"Images: {name}")

        try:
            if REUSE_CACHED_RESULTS:
                results[name], data[:, i, :, :] = pickle.load(
                    open(CACHE_PATH / f"experiment_5-{name}.pkl", "rb")
                )
                logger.info(f"Loaded cached results for {name}")
                continue
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            if "normalize" in kwargs:
                kwargs.pop("normalize")
            mlp = MLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                num_classes=num_classes,
                activation_function=activation_function,
                optimizer=optimizer,
                regularizer=regularizer,
                loss_function=loss_function,
                initializer=initializer,
                scheduler=scheduler,
                T_max=len(trainloader) * n_epochs,
                eta_min=lr / 10,
                **kwargs,
            )

            train_accuracies, test_accuracies = mlp.train(
                trainloader,
                testloader,
                epochs=n_epochs,
                print_loss=PRINT_LOSS,
                log_interval=log_interval,
            )

            data[0, i, :, :] = np.array(train_accuracies).reshape(-1, 1)
            data[1, i, :, :] = np.array(test_accuracies).reshape(-1, 1)
            results[name] = (train_accuracies, test_accuracies)

            if CACHE_PARTIAL_RESULTS:
                pickle.dump(
                    (results[name], data[:, i, :, :]),
                    open(CACHE_PATH / f"experiment_5-{name}.pkl", "wb"),
                )

    plot(
        data=data,
        title="MLP Training with and without Normalized Images",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        algs_info=[
            (name, colour, "line")
            for name, colour in zip(["Normalized", "Unnormalized"], ["r", "g"])
        ],
        log_scale=log_scale,
        include_y0=True,
        filename=PLOTS_PATH / "experiment_5-normalization_accuracy.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    return results


def experiment_6(
    n_epochs,
    input_dim,
    hidden_layers,
    num_classes,
    dataset_name,
    batch_size,
    normalize,
    flatten,
    lr,
    log_scale,
    activation_function,
    initializer,
    optimizer,
    regularizer,
    loss_function,
    scheduler,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    log_interval=100,
    **kwargs,
):
    """
    ### 6. CNN on Fashion MNIST
    Using PyTorch, create a convolutional neural network (CNN) with 2
    convolutional and 2 fully connected layers. You are free in your choice of
    the hyperparameters of the convolutional layers, but set the number of
    units in the fully connected layers to be 128. Also, set the activations in
    all of the layers to be ReLU. Train this CNN on the Fashion MNIST dataset.
    Does using a CNN increase/decrease the accuracy compared to using MLPs?
    Provide comments on your results.
    """
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=False
    )

    results = {}
    data = np.zeros((2, 1, len(trainloader) * n_epochs, 1))

    try:
        if REUSE_CACHED_RESULTS:
            results, data = pickle.load(open(CACHE_PATH / "experiment_6.pkl", "rb"))
            logger.info("Loaded cached results")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        cnn_model = CNN(
            num_channels=1,
            image_size=28,
            num_classes=num_classes,
            optimizer=optim.Adam,
            loss_function=nn.CrossEntropyLoss,
            lr=lr,
            **kwargs,
        )
        train_accuracies, test_accuracies = cnn_model.train_loop(
            trainloader,
            testloader,
            epochs=n_epochs,
            print_loss=PRINT_LOSS,
            log_interval=log_interval,
        )

        data[0, 0, :, :] = np.array(train_accuracies).reshape(-1, 1)
        data[1, 0, :, :] = np.array(test_accuracies).reshape(-1, 1)
        results["CNN"] = (train_accuracies, test_accuracies)

        if CACHE_PARTIAL_RESULTS:
            pickle.dump((results, data), open(CACHE_PATH / "experiment_6.pkl", "wb"))

    plot(
        data=data,
        title="CNN on Fashion MNIST",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        algs_info=[("CNN", "r", "line")],
        log_scale=log_scale,
        include_y0=True,
        filename=PLOTS_PATH / "experiment_6-cnn_accuracy.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    return results


def experiment_7(
    n_epochs,
    input_dim,
    hidden_layers,
    num_classes,
    dataset_name,
    batch_size,
    normalize,
    flatten,
    lr,
    log_scale,
    activation_function,
    initializer,
    optimizer,
    regularizer,
    loss_function,
    scheduler,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    log_interval=100,
    **kwargs,
):
    """
    ### 7. MLP vs. CNN on CIFAR-10

    Now using the CIFAR-10 dataset, train an MLP using your implementation
    (with whatever layer dimensions you like), and a CNN with the same
    architecture restrictions as in (6). How does using a CNN increase/decrease
    the accuracy compared to using MLPs on this dataset?
    """

    # Train MLP
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=flatten
    )

    results = {}
    data = np.zeros((2, 2, n_epochs * len(trainloader), 1))

    try:
        if REUSE_CACHED_RESULTS:
            results["MLP"], data[:, 0, :, :] = pickle.load(
                open(CACHE_PATH / "experiment_7-MLP.pkl", "rb")
            )
            logger.info("Loaded cached results for MLP")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        mlp = MLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            num_classes=num_classes,
            activation_function=activation_function,
            optimizer=optimizer,
            regularizer=regularizer,
            loss_function=loss_function,
            initializer=initializer,
            scheduler=scheduler,
            T_max=len(trainloader) * n_epochs,
            eta_min=lr / 10,
            **kwargs,
        )

        train_accuracies, test_accuracies = mlp.train(
            trainloader,
            testloader,
            epochs=n_epochs,
            print_loss=PRINT_LOSS,
            log_interval=log_interval,
        )
        data[0, 0, :, :] = np.array(train_accuracies).reshape(-1, 1)
        data[1, 0, :, :] = np.array(test_accuracies).reshape(-1, 1)

        results["MLP"] = (train_accuracies, test_accuracies)
        logger.info(f"MLP Test Accuracy: {np.max(test_accuracies):.2f}%")

        if CACHE_PARTIAL_RESULTS:
            pickle.dump(
                (results["MLP"], data[:, 0, :, :]),
                open(CACHE_PATH / "experiment_7-MLP.pkl", "wb"),
            )

    # Train CNN
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=False
    )
    try:
        if REUSE_CACHED_RESULTS:
            results["CNN"], data[:, 1, :, :] = pickle.load(
                open(CACHE_PATH / "experiment_7-CNN.pkl", "rb")
            )
            logger.info("Loaded cached results for CNN")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        cnn = CNN(
            num_classes=num_classes,
            optimizer=torch.optim.Adam,
            loss_function=nn.CrossEntropyLoss,
            lr=lr,
            **kwargs,
        )

        train_accuracies, test_accuracies = cnn.train_loop(
            trainloader,
            testloader,
            epochs=n_epochs,
            print_loss=PRINT_LOSS,
            log_interval=log_interval,
        )
        data[0, 1, :, :] = np.array(train_accuracies).reshape(-1, 1)
        data[1, 1, :, :] = np.array(test_accuracies).reshape(-1, 1)

        results["CNN"] = (train_accuracies, test_accuracies)
        logger.info(f"CNN Test Accuracy: {np.max(test_accuracies):.2f}%")

        if CACHE_PARTIAL_RESULTS:
            pickle.dump(
                (results["CNN"], data[:, 1, :, :]),
                open(CACHE_PATH / "experiment_7-CNN.pkl", "wb"),
            )

    plot(
        data=data,
        title="MLP vs. CNN on CIFAR-10",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        include_y0=True,
        log_scale=log_scale,
        algs_info=[("MLP", "r", "line"), ("CNN", "b", "line")],
        filename=PLOTS_PATH / "experiment_7-mlp_vs_cnn_accuracy.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    return results


def experiment_8(
    n_epochs,
    input_dim,
    hidden_layers,
    num_classes,
    dataset_name,
    batch_size,
    normalize,
    flatten,
    lr,
    momentum_values,
    log_scale,
    activation_function,
    optimizer,
    scheduler,
    regularizer,
    loss_function,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    log_interval=100,
    **kwargs,
):
    """
    ### 8. Effects of Optimizer on CIFAR-10

    In your CNN implemented with PyTorch, investigate the effects of optimizer
    on performance on the CIFAR-10 dataset. Using an SGD optimizer, set the
    momentum factor to zero, and then try to increase it. How does changing
    this value impact the training and performance of the network in terms of
    convergence speed, final accuracy, and stability? How do these compare if
    you instead use an Adam optimizer?
    """
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=False
    )

    results = {}
    data = np.zeros((2, len(momentum_values) + 1, n_epochs * len(trainloader), 1))

    # Train CNN with SGD and different momentum values
    for i, momentum in enumerate(momentum_values):
        try:
            if REUSE_CACHED_RESULTS:
                results[f"SGD_{momentum}"], data[:, i, :, :] = pickle.load(
                    open(CACHE_PATH / f"experiment_8-SGD_{momentum}.pkl", "rb")
                )
                logger.info(f"Loaded cached results for SGD with momentum {momentum}")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            cnn = CNN(
                num_classes=num_classes,
                optimizer=torch.optim.SGD,
                lr=lr,
                momentum=momentum,
                loss_function=nn.CrossEntropyLoss,
                **kwargs,
            )
            train_accuracies, test_accuracies = cnn.train_loop(
                trainloader,
                testloader,
                epochs=n_epochs,
                print_loss=PRINT_LOSS,
                log_interval=log_interval,
            )
            data[0, i, :, :] = np.array(train_accuracies).reshape(-1, 1)
            data[1, i, :, :] = np.array(test_accuracies).reshape(-1, 1)
            results[f"SGD_{momentum}"] = (train_accuracies, test_accuracies)
            logger.info(
                f"SGD with momentum {momentum} Test Accuracy: {np.max(test_accuracies) * 100:.2f}%"
            )

            if CACHE_PARTIAL_RESULTS:
                pickle.dump(
                    (results[f"SGD_{momentum}"], data[:, i, :, :]),
                    open(CACHE_PATH / f"experiment_8-SGD_{momentum}.pkl", "wb"),
                )

    # Train CNN with Adam
    try:
        if REUSE_CACHED_RESULTS:
            results["Adam"], data[:, -1, :, :] = pickle.load(
                open(CACHE_PATH / "experiment_8-Adam.pkl", "rb")
            )
            logger.info("Loaded cached results for Adam")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        cnn = CNN(
            num_classes=num_classes,
            optimizer=torch.optim.Adam,
            lr=lr,
            loss_function=nn.CrossEntropyLoss,
            **kwargs,
        )
        train_accuracies, test_accuracies = cnn.train_loop(
            trainloader,
            testloader,
            epochs=n_epochs,
            print_loss=PRINT_LOSS,
            log_interval=log_interval,
        )
        data[0, -1, :, :] = np.array(train_accuracies).reshape(-1, 1)
        data[1, -1, :, :] = np.array(test_accuracies).reshape(-1, 1)
        results["Adam"] = (train_accuracies, test_accuracies)
        logger.info(f"Adam Test Accuracy: {np.max(test_accuracies) * 100:.2f}%")

        if CACHE_PARTIAL_RESULTS:
            pickle.dump(
                (results["Adam"], data[:, -1, :, :]),
                open(CACHE_PATH / "experiment_8-Adam.pkl", "wb"),
            )

    algs_info = [
        (f"SGD_{momentum}", colour, "line")
        for momentum, colour in zip(momentum_values, ["r", "g", "c", "m"])
    ]
    algs_info.append(("Adam", "b", "line"))

    plot(
        data=data,
        title="Effects of Optimizer on CIFAR-10",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        include_y0=True,
        algs_info=algs_info,
        log_scale=log_scale,
        filename=PLOTS_PATH / "experiment_8-optimizer_effects.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    return results


def experiment_9(
    n_epochs,
    input_dim,
    hidden_layers,
    num_classes,
    dataset_name,
    batch_size,
    normalize,
    flatten,
    lr,
    log_scale,
    activation_function,
    optimizer,
    scheduler,
    regularizer,
    loss_function,
    SAVE_FILES,
    SHOW_GRAPHS,
    PRINT_LOSS,
    CACHE_PARTIAL_RESULTS,
    REUSE_CACHED_RESULTS,
    **kwargs,
):
    # 1. Load CIFAR-10 Dataset
    trainloader, testloader = load_dataset(
        dataset_name, batch_size=batch_size, normalize=normalize, flatten=flatten
    )

    # 2. Modify Pre-trained Model
    resnet = models.resnet18(weights=True)
    for param in resnet.parameters():
        param.requires_grad = False

    num_features = resnet.fc.in_features
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_classes),
    )

    # 3. Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=lr)
    data = np.zeros((2, 1, n_epochs * len(trainloader), 1))

    for epoch in range(n_epochs):
        resnet.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets).long()

            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Store training accuracy
            _, predicted = outputs.max(1)
            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()
            data[0, 0, epoch * len(trainloader) + batch_idx, 0] = 100.0 * correct / total

            if PRINT_LOSS and batch_idx % 100 == 0:
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.3f}, Accuracy: {100. * correct / total:.3f}"
                )

    # 4. Test the model
    resnet.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets).long()

            outputs = resnet(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            # Store test accuracy
            data[1, 0, epoch * len(testloader) + batch_idx, 0] = 100.0 * correct / targets.size(0)

    test_loss /= len(testloader)
    test_accuracy = 100.0 * correct / len(testloader)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader)} ({test_accuracy:.2f}%)\n"
    )

    if CACHE_PARTIAL_RESULTS:
        pickle.dump((data, test_accuracy), open(CACHE_PATH / "experiment_9.pkl", "wb"))

    algs_info = [("Pre-trained ResNet", "r", "line")]
    plot(
        data=data,
        title="Experiment 9: Pre-trained ResNet on CIFAR-10",
        main_labels=["Batches", "Training Accuracy", "Test Accuracy"],
        ax_titles=["Train Accuracy per Batch", "Test Accuracy per Batch"],
        algs_info=algs_info,
        log_scale=log_scale,
        include_y0=True,
        filename=PLOTS_PATH / "experiment_9-pretrained_resnet.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    return test_accuracy


def replace_strings_with_objects(obj, context):
    if isinstance(obj, dict):
        return {key: replace_strings_with_objects(value, context) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [replace_strings_with_objects(item, context) for item in obj]
    elif isinstance(obj, str) and obj in context:
        return context[obj]
    else:
        if isinstance(obj, str):
            logger.warning(f"String `{obj}` was not found in the context.")
        return obj


def run_experiments(experiments_to_run, experiments_info, global_parameters, default_parameters):
    for exp_num in experiments_to_run:
        exp_args = experiments_info.get(str(exp_num), {})

        # Merge default parameters with experiment-specific parameters
        # Experiment-specific parameters take precedence over default parameters
        combined_parameters = {**default_parameters, **exp_args}

        # Merge global parameters
        combined_parameters = {**combined_parameters, **global_parameters}

        func_name = "experiment_" + str(exp_num)
        try:
            logger.info(f"\n[{exp_num}]\nRunning {func_name}...\n")
            func = globals()[func_name]
            func(**combined_parameters)
        except Exception as e:
            logger.error(f"Error while running {func_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            logger.info("The experiment will be skipped.")
        else:
            logger.info(f"[{exp_num}] Finished running {func_name}.\n")


def main(experiments_to_run):
    config = load_config(MP2_PATH / "config.yaml")

    output_file = config.get("output_file")
    global_parameters = config.get("global_parameters", {})
    default_parameters = config.get("default_parameters", {})
    experiments_info = config.get("experiments_info", {})
    experiments_to_run = experiments_to_run or list(experiments_info.keys())

    # Get the local context, which includes all the objects in the current scope
    local_context = globals()
    local_context.update(locals())

    # Replace strings with objects in the configurations
    experiments_info = cast(
        dict[str, Any], replace_strings_with_objects(experiments_info, local_context)
    )
    global_parameters = cast(
        dict[str, Any], replace_strings_with_objects(global_parameters, local_context)
    )
    default_parameters = cast(
        dict[str, Any], replace_strings_with_objects(default_parameters, local_context)
    )

    # if we have CACHE_PARTIAL_RESULTS and cache directory does not exist, create it
    if global_parameters.get("CACHE_PARTIAL_RESULTS", False):
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)

    if output_file:
        with open(output_file, "w") as file:
            original_stdout = sys.stdout
            sys.stdout = file
            run_experiments(
                experiments_to_run,
                experiments_info,
                global_parameters,
                default_parameters,
            )
            sys.stdout = original_stdout
    else:
        run_experiments(experiments_to_run, experiments_info, global_parameters, default_parameters)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiments_to_run = sys.argv[1:]
        for exp_num in experiments_to_run:
            try:
                int(exp_num)
            except ValueError:
                logger.error(f"Invalid experiment number: {exp_num}")
                sys.exit(1)
        logger.info(f"Running experiment(s) {' '.join(experiments_to_run)}...")
        main(experiments_to_run)

    else:
        logger.info("Running all experiments...")
        main(None)
