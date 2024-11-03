import logging
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Type, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from src.datasets.retrieval import get_dataset
from src.models.legacy.bert import (
    BERT,
    BERTDataProcessor,
    DefaultBERT,
    FrozenBERT,
    InitBERT,
    PrefinetunedBERT,
)
from src.models.legacy.naive_bayes import NaiveBayes, NaiveBayesDataProcessor
from src.visualization import plot

colors = ["r", "b", "g", "c", "m", "y", "k", "w"]
class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

MP3_PATH = Path(__file__).resolve().parent
DATA_PATH = MP3_PATH / "data"
PLOTS_PATH = DATA_PATH / "plots"

PLOTS_PATH.mkdir(parents=True, exist_ok=True)

with open(MP3_PATH / "config.yaml", "r") as file:
    config = yaml.safe_load(file)
logging_config = config.get("logging_config", {})
logging.basicConfig(
    level=getattr(logging, logging_config.get("level", "INFO")),
    format=logging_config.get("format", "[%(asctime)s - %(levelname)s]: %(message)s"),
)
logger = logging.getLogger(__name__)

# we previously had be automatically populating a context dict with all
# globals(), but this was a repeated source of bugs. From now on, if you want to
# be able to reference something in config.yaml, you need to add it to this
# list.
YAML_CONTEXT = {
    obj.__name__: obj
    for obj in [
        NaiveBayes,
        DefaultBERT,
        PrefinetunedBERT,
        InitBERT,
        FrozenBERT,
        Optimizer,
        Adam,
        AdamW,
        LRScheduler,
        ReduceLROnPlateau,
    ]
}


def experiment_naive_bayes_debug(model_name: Type[NaiveBayes], alpha: int):
    """
    Currently for debuggging Naive Bayes. NOT AN EXPERIMENT.
    """
    data_processor = NaiveBayesDataProcessor(
        list(get_dataset()),
        representation_type="count",
        vectorizer_params={"min_df": 15, "max_features": 5000},
    )
    (X_train, y_train), _, (X_test, y_test) = data_processor.preprocess()

    model = model_name(alpha=alpha)
    model.fit(X_train, y_train)
    accuracy = model.evaluate(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy}")

    for param in model.__dict__:
        logger.info(f"{param}: {model.__dict__[param]}")


def experiment_bert_debug(
    model_name: BERT,
    optimizer: Optimizer,
    beta1: float,
    beta2: float,
    epsilon: float,
    learning_rate: float,
    scheduler: LRScheduler,
    mode: str,
    patience: int,
    factor: float,
    epochs: int,
    batch_size: int,
    SAVE_FILES: bool,
):
    """
    Currently for debuggging BERT. ATTENTION.
    """
    data_processor = BERTDataProcessor(get_dataset(), model_name=model_name, batch_size=batch_size)
    train_dataloader, _, test_dataloader = data_processor.preprocess()

    model = model_name(batch_size=batch_size)
    optimizer = optimizer(model.model.parameters(), lr=learning_rate)  # oof
    scheduler = scheduler(optimizer, mode=mode, patience=patience, factor=factor, verbose=True)

    pre_finetuning_accuracy = model.evaluate(test_dataloader)
    logger.info(f"Pre-finetuning test accuracy: {pre_finetuning_accuracy}")

    losses = model.fit(train_dataloader, optimizer=optimizer, scheduler=scheduler, epochs=epochs)

    post_finetuning_accuracy = model.evaluate(test_dataloader)
    logger.info(f"Post-finetuning test accuracy: {post_finetuning_accuracy}")


def experiment_1(model_name: Type[NaiveBayes], representations: list[str]):
    """
    Runs Experiment 1: Feature Representation Comparison

    This experiment assesses the impact of different text representations on the performance of a Naive Bayes classifier.
    We compare the raw term frequency (count), binary occurrence (binary), and term frequency-inverse document frequency
    (tfidf) as feature representations.
    """
    data_processor = NaiveBayesDataProcessor(
        list(get_dataset()), vectorizer_params={"min_df": 15, "max_features": 5000}
    )

    results = {}
    for representation in representations:
        data_processor.representation_type = representation
        (X_train, y_train), _, (X_test, y_test) = data_processor.preprocess()

        model = model_name()
        model.fit(X_train, y_train)
        accuracy = model.evaluate(X_test, y_test)
        logger.info(f"{representation} representation accuracy: {accuracy:.4f}")

        results[representation] = accuracy


def experiment_2(
    model_name: Type[NaiveBayes],
    alpha: dict[str, list[int] | int],
    min_df: dict[str, list[int] | int],
    representation: str,
    SAVE_FILES: bool,
    SHOW_GRAPHS: bool,
):
    """
    Runs Experiment 2: Hyperparameter Tuning

    This experiment performs hyperparameter tuning for the Naive Bayes model to find the best configuration for alpha (Laplace smoothing parameter)
    and min_df (minimum document frequency for feature selection). It evaluates the impact of these hyperparameters on model accuracy.

    Conclusion: Best configuration -- Alpha: 0.60, min_df: 15, test accuracy: 0.8540
    """

    def generate_values(config):
        if "values" in config and isinstance(config["values"], list) and len(config["values"]) > 0:
            return config["values"]
        elif all(key in config and config[key] is not None for key in ["min", "max", "num"]):
            return np.linspace(config["min"], config["max"], config["num"])
        else:
            raise ValueError("Either 'values' or 'min'/'max'/'num' must be specified.")

    alpha_values = generate_values(alpha)
    min_df_values = generate_values(min_df)

    logger.info(
        f'Running experiment 4 with model "{model_name.__name__}", alpha values "{alpha_values}", min df values "{min_df_values}", and representation "{representation}"'
    )

    data_processor = NaiveBayesDataProcessor(
        list(get_dataset()), representation_type=representation, vectorizer_params={}
    )
    best_configuration = {"alpha": None, "min_df": None, "accuracy": 0}

    heatmap_data = np.zeros((len(alpha_values), len(min_df_values)))

    for i, a in enumerate(alpha_values):
        for j, mdf in enumerate(map(int, min_df_values)):
            data_processor.vectorizer_params = {"min_df": mdf, "max_features": 5000}
            (X_train, y_train), (X_validation, y_validation), (X_test, y_test) = (
                data_processor.preprocess()
            )

            model = model_name(alpha=a)
            model.fit(X_train, y_train)
            validation_accuracy = model.evaluate(X_validation, y_validation)

            if validation_accuracy > best_configuration["accuracy"]:
                best_configuration.update(
                    {"alpha": a, "min_df": mdf, "accuracy": validation_accuracy}
                )

            logger.info(
                f"Alpha: {a:.2f}, min_df: {mdf}, validation accuracy: {validation_accuracy:.4f}"
            )
            heatmap_data[i, j] = validation_accuracy

    # After finding the best configuration, evaluate it on the test set.
    best_alpha = best_configuration["alpha"]
    best_min_df = best_configuration["min_df"]

    data_processor.vectorizer_params["min_df"] = best_min_df
    (X_train, y_train), (X_validation, y_validation), (X_test, y_test) = data_processor.preprocess()

    model = model_name(alpha=best_alpha)
    model.fit(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)

    logger.info(
        f"Best configuration -- Alpha: {best_alpha:.2f}, min_df: {best_min_df}, test accuracy: {test_accuracy:.4f}"
    )

    # Plot the heatmap
    plt.figure(figsize=(7, 7))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=[f"{int(mdf)}" for mdf in min_df_values],
        yticklabels=[f"{a:.2f}" for a in alpha_values],
    )

    plt.title("Validation Accuracy as a function of Alpha and min_df")
    plt.xlabel("min_df")
    plt.ylabel("alpha")

    plt.show()

    if SAVE_FILES:
        plt.savefig(PLOTS_PATH / "experiment_2-hyperparameter_tuning.png")


def experiment_3(
    model_names: list[Type[BERT]],
    representation: str,
    min_df: int,
    optimizer: Type[Optimizer],
    beta1: float,
    beta2: float,
    epsilon: float,
    learning_rate: float,
    scheduler: LRScheduler,
    mode: str,
    patience: int,
    factor: float,
    epochs: int,
    batch_size: int,
    SAVE_FILES: bool,
    SHOW_GRAPHS: True,
):
    """
    Runs Experiment 3: BERT Models Comparison
    """
    naive_bayes_models = [
        model_name for model_name in model_names if issubclass(model_name, NaiveBayes)
    ]
    bert_models = [model_name for model_name in model_names if issubclass(model_name, BERT)]

    results = {}

    data = np.zeros((1, len(bert_models), 250 * epochs, 1))

    optimizer_name = optimizer
    scheduler_name = scheduler

    train, validation, test = get_dataset()

    logger.info(
        f"Running {len(model_names)} experiments, with {len(naive_bayes_models)} NaiveBayes models and {len(bert_models)} BERT models."
    )

    # Naive Bayes models
    naive_bayes_data_processor = NaiveBayesDataProcessor(
        [train, validation, test],
        representation_type=representation,
        vectorizer_params={"min_df": min_df, "max_features": 5000},
    )
    for i, naive_bayes_model in enumerate(naive_bayes_models):
        start = time.time()
        logger.info(f'Running experiment 3 with model "{naive_bayes_model.__name__}"')
        naive_bayes_data_processor.representation_type = representation
        (X_train, y_train), _, (X_test, y_test) = naive_bayes_data_processor.preprocess()

        model = naive_bayes_model()
        model.fit(X_train, y_train)

        train_accuracy, test_accuracy = (
            model.evaluate(X_train, y_train),
            model.evaluate(X_test, y_test),
        )

        model.save_complete_model()

        results[naive_bayes_model.__name__] = (train_accuracy, test_accuracy)

        logger.info(
            f"Experiment 3 finished running {naive_bayes_model.__name__} in {time.time() - start:.2f} seconds."
        )
        logger.info(f"Train accuracy: {train_accuracy}")
        logger.info(f"Test accuracy: {test_accuracy}")

    # BERT models
    bert_data_processor = BERTDataProcessor([train, validation, test], batch_size=batch_size)
    for i, bert_model in enumerate(bert_models):
        start = time.time()

        logger.info(f'Running experiment 3 with model "{bert_model.__name__}"')

        bert_data_processor.tokenizer = BERTDataProcessor.get_tokenizer(model_name=bert_model)
        bert_data_processor.model_name = bert_model

        train_dataloader, _, test_dataloader = bert_data_processor.preprocess()

        model = bert_model(batch_size=batch_size)

        optimizer = optimizer_name(model.model.parameters(), lr=learning_rate)
        scheduler = scheduler_name(
            optimizer, mode=mode, patience=patience, factor=factor, verbose=True
        )

        losses = model.fit(
            train_dataloader, optimizer=optimizer, scheduler=scheduler, epochs=epochs
        )
        data[0, i, : len(losses), 0] = losses

        train_accuracy, test_accuracy = (
            model.evaluate(train_dataloader),
            model.evaluate(test_dataloader),
        )

        model.save()
        model.save_complete_model()

        results[bert_model.__name__] = (train_accuracy, test_accuracy)

        logger.info(
            f"Experiment 3 finished running {bert_model.__name__} in {time.time() - start:.2f} seconds."
        )
        logger.info(f"Train accuracy: {train_accuracy}")
        logger.info(f"Test accuracy: {test_accuracy}")

    logger.info("Plotting experiment 3...")
    plot(
        data=data,
        title="BERT Models Comparison",
        main_labels=["Step", "Loss"],
        ax_titles=[""],
        algs_info=[
            (bert_model.__name__, color, "line") for color, bert_model in zip(colors, bert_models)
        ],
        filename=PLOTS_PATH / "experiment_3-bert_models_comparison.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    plt.show()


def experiment_4(model_name: NaiveBayes, SAVE_FILES: bool, SHOW_GRAPHS: bool):
    random.seed(0)

    def visualize_feature_spread(class_index, log_likelihoods, feature_names, n=35):
        """
        Visualize a representation of how much different words contribute to the log likelihood score for a given class.
        """
        features_indices = log_likelihoods[class_index].argsort()
        features = np.array(feature_names)[features_indices]
        features_scores = log_likelihoods[class_index, features_indices]

        # Sort the features by score first, random second (for features with
        # the same score)
        # This is to avoid having the first features in the plot be the same
        # every time, to convey more information about the total spread over
        # multiple.

        features_scores, features = zip(
            *sorted(zip(features_scores, features), key=lambda x: (x[0], random.random()))
        )

        stride = max(1, len(features) // n)

        features_scores = features_scores[::stride]
        features = features[::stride]

        plt.figure(figsize=(6, 12))

        # horizontal barplot with the labels on the bars themselves
        ax = sns.barplot(x=features_scores, y=features)
        ax.set(ylabel=None)
        ax.set(yticklabels=[])
        ax.tick_params(left=False)
        ax.bar_label(ax.containers[0], labels=features, label_type="edge", fontsize=10, padding=5)

        # add padding to left of subplots so that the labels are not cut off
        ax.set_xlim(min(features_scores) - 3)

        plt.title(f"Representative Features Spread for class: {class_names[class_index].upper()}")
        plt.xlabel("Contribution to Log Likelihood Score (more negative = more indicative)")
        if SAVE_FILES:
            plt.savefig(PLOTS_PATH / f"experiment_4-features_class_{class_index}.png")
        if SHOW_GRAPHS:
            plt.show()

    # Load and pre-process data
    train, validation, test = get_dataset()
    data_processor = NaiveBayesDataProcessor(
        [train, validation, test],
        representation_type="count",
        vectorizer_params={"min_df": 15, "max_features": 5000},
    )
    (X_train, y_train), _, (X_test, y_test) = data_processor.preprocess()

    # Initialize and train the NaiveBayes model
    model = NaiveBayes(alpha=1)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Retrieve feature names for visualization
    feature_names = data_processor.get_feature_names()

    # Visualize the top features for each class
    for class_index in np.unique(y_train):
        visualize_feature_spread(class_index, model.log_likelihoods, feature_names)

    # Confusion Matrix
    predictions = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(7, 7))

    # heatmap colour scale by with log-scale to see the off-diagonal differences better
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        norm=LogNorm(),
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plt.show()

    if SAVE_FILES:
        plt.savefig(PLOTS_PATH / "experiment_4-confusion_matrix.png")


def populate_from_context(obj):
    if isinstance(obj, dict):
        return {key: populate_from_context(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [populate_from_context(item) for item in obj]
    elif isinstance(obj, str):
        if obj in YAML_CONTEXT:
            return YAML_CONTEXT[obj]
        else:
            return obj
    else:
        return obj


def main(experiments_to_run):
    default_parameters = config.get("default_parameters", {})
    default_parameters = cast(dict[str, Any], populate_from_context(default_parameters))

    experiments_info = config.get("experiments_info", {})
    experiments_info = cast(dict[str, Any], populate_from_context(experiments_info))

    global_parameters = config.get("global_parameters", {})
    global_parameters = cast(dict[str, Any], populate_from_context(global_parameters))

    experiments_to_run = experiments_to_run or list(experiments_info.keys())

    OUTPUT_FILE = global_parameters["OUTPUT_FILE"]
    sys.stdout = sys.stdout if OUTPUT_FILE is None else open(OUTPUT_FILE, "w")

    for exp_name in experiments_to_run:
        func_name = f"experiment_{exp_name}"
        try:
            func = globals()[func_name]

            # explicit experiment arguments takes precedence over
            # default_parameters, which in turn takes precedence over
            # global_parameters

            parameters = {**experiments_info.get(exp_name, {})}

            func_args = func.__code__.co_varnames[: func.__code__.co_argcount]

            # add all default parameters that func takes, not already in parameters
            for key in default_parameters:
                if key not in parameters and key in func_args:
                    parameters[key] = default_parameters[key]

            # add all global parameters that func takes, not already in parameters
            for key in global_parameters:
                if key not in parameters and key in func_args:
                    parameters[key] = global_parameters[key]

            logger.info(f"Running {func_name}...")
            logger.debug(f"[{func_name}] - Parameters: {parameters}")

            func(**parameters)

        except Exception as e:
            logger.error(f"Error while running {func_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            logger.info("The experiment will be skipped.")
        else:
            logger.info(f"[{exp_name}] Finished running {func_name}.\n")

    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    # if we have arguments, interpret them as a set of experiment names to run.
    # otherwise, run all experiments.
    if len(sys.argv) > 1:
        experiments_to_run = sys.argv[1:]
        for exp_name in experiments_to_run:
            func_name = f"experiment_{exp_name}"
            if not hasattr(sys.modules[__name__], func_name):
                logger.error(f"Experiment function does not exist: {func_name}")
                sys.exit(1)
        logger.info(f"Running experiment(s) {' '.join(experiments_to_run)}...")
        main(experiments_to_run)

    else:
        logger.info("Running all experiments...")
        main(None)
