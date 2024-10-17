import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from src.datasets.data_processing import normalize, standardize, train_test_split
from src.models.legacy.linear_regression import (
    LinearRegressionAnalytic,
    LinearRegressionGD,
    LinearRegressionSGD,
)
from src.models.legacy.logistic_regression import (
    LogisticRegressionGD,
    LogisticRegressionSGD,
)
from src.settings import DATASETS_PATH
from src.utils.metrics import MSE, accuracy, f1_score, precision, recall
from src.utils.visualization import plot1, print_table

# ---------------------------------- controls ----------------------------------

print_output_to_file = True
basic_stats = True
basic_experiment = True
experiment_1 = True
experiment_2 = True
experiment_3 = True
experiment_4 = True
experiment_5 = True
experiment_6 = True
experiment_7 = True
experiment_8 = True

SEED: int = 42
SAVE_FILES: bool = True
SHOW_GRAPHS: bool = False

MP1_PATH = Path(__file__).resolve().parent
DATA_PATH = MP1_PATH / "data"
PLOTS_PATH = DATA_PATH / "plots"

PLOTS_PATH.mkdir(parents=True, exist_ok=True)

if print_output_to_file:
    file = open(DATA_PATH / "experiment_outputs.txt", "w")
    sys.stdout = file


# ----------------------- data retrieval and processing ------------------------

np.random.seed(SEED)

data_boston = pd.read_csv(DATASETS_PATH / "boston.csv")
data_wine = pd.read_csv(DATASETS_PATH / "wine.data")

# remove column "B" from Boston data
data_boston = data_boston.drop(columns=["B"])

# remove malformed or missing examples
data_boston = data_boston.apply(pd.to_numeric, errors="coerce").dropna()
data_wine = data_wine.apply(pd.to_numeric, errors="coerce").dropna()

# basic statistics
if basic_stats:
    # Boston dataset
    title = "Basic Statistics for Boston dataset"
    boston_describe_df = data_boston.describe()

    # Extract headers, row names, and data
    headers = boston_describe_df.columns.tolist()
    row_names = boston_describe_df.index.tolist()
    data = [row_names[i : i + 1] + row.tolist() for i, row in enumerate(boston_describe_df.values)]
    print_table(title, data, [""] + headers)

    # Wine dataset
    title = "Basic Statistics for Wine dataset"
    wine_describe_df = data_wine.describe()

    # Extract headers, row names, and data
    headers = wine_describe_df.columns.tolist()
    row_names = wine_describe_df.index.tolist()
    data = [row_names[i : i + 1] + row.tolist() for i, row in enumerate(wine_describe_df.values)]
    print_table(title, data, [""] + headers)

# Log transform the CRIM column
data_boston["CRIM"] = np.log1p(data_boston["CRIM"])

# Normalize/Standardize datasets
normalize_or_standardize = "standardize"
if normalize_or_standardize == "normalize":
    data_boston = normalize(data_boston)
    data_wine = normalize(data_wine)
elif normalize_or_standardize == "standardize":
    data_boston = standardize(data_boston)
    data_wine = standardize(data_wine)

X_boston = data_boston.drop(columns=["MEDV"])
Y_boston = data_boston["MEDV"]

X_wine = data_wine.drop(columns=["1"])
Y_wine = data_wine["1"]

X_boston = X_boston.to_numpy()
Y_boston = Y_boston.to_numpy()
X_wine = X_wine.to_numpy()
Y_wine = Y_wine.to_numpy()


# ------------------------------ helpers ------------------------------


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


# ------------------------------ basic experiments -----------------------------

if basic_experiment:
    print(" --- Basic Experiments --- ")

    # split data into training and testing sets
    X_train_boston, Y_train_boston, X_test_boston, Y_test_boston = train_test_split(
        X_boston, Y_boston, split_ratio=0.8
    )
    X_train_wine, Y_train_wine, X_test_wine, Y_test_wine = train_test_split(
        X_wine, Y_wine, split_ratio=0.8
    )

    results = []

    # Linear Regression Experiments for Boston Dataset
    models_lr = [
        ("Exact Solution", LinearRegressionAnalytic()),
        ("Gradient Descent", LinearRegressionGD()),
        ("Stochastic Gradient Descent", LinearRegressionSGD()),
    ]

    for name, model in models_lr:
        model.fit(X_train_boston, Y_train_boston)
        score = np.mean(
            k_fold_cross_validation(model, X_test_boston, Y_test_boston, metric_function=MSE, k=10)
        )
        results.append([name, "BOSTON", "MSE", round(score, 5)])

    # Logistic Regression Experiments for Wine Dataset
    models_logistic = [
        ("Gradient Descent", LogisticRegressionGD()),
        ("Stochastic Gradient Descent", LogisticRegressionSGD()),
    ]

    for name, model in models_logistic:
        model.fit(X_train_wine, Y_train_wine)

        acc = accuracy(model.predict(X_test_wine), Y_test_wine)
        prec = precision(model.predict(X_test_wine), Y_test_wine)
        rec = recall(model.predict(X_test_wine), Y_test_wine)
        f1 = f1_score(model.predict(X_test_wine), Y_test_wine)

        results.extend(
            [
                [name, "WINE", "Accuracy", round(acc, 5)],
                [name, "WINE", "Precision", round(prec, 5)],
                [name, "WINE", "Recall", round(rec, 5)],
                [name, "WINE", "F1-score", round(f1, 5)],
            ]
        )

    print_table("Basic Experiments Results", results, ["Model", "Metric", "Dataset", "Value"])
    print()


# -------------------------------- Experiment 1 --------------------------------

# For both datasets, perform an 80/20 train/test split and report the performance
# metrics on both the training set and test set for each model. Please include
# metrics such as Mean Squared Error (MSE) for Linear Regression and accuracy,
# precision, recall, and F1-score for Logistic Regression.
if experiment_1:
    print(" --- Experiment 1 --- ")
    # split data into training and testing sets
    X_train_boston, Y_train_boston, X_test_boston, Y_test_boston = train_test_split(
        X_boston, Y_boston, split_ratio=0.8
    )
    X_train_wine, Y_train_wine, X_test_wine, Y_test_wine = train_test_split(
        X_wine, Y_wine, split_ratio=0.8
    )

    # Linear Regression for Boston
    lr_model = LinearRegressionAnalytic()
    lr_model.fit(X_train_boston, Y_train_boston)

    pred_train = lr_model.predict(X_train_boston)
    mse_train = MSE(Y_train_boston, pred_train)

    pred_test = lr_model.predict(X_test_boston)
    mse_test = MSE(Y_test_boston, pred_test)

    # Logistic Regression for Wine
    log_model = LogisticRegressionGD()
    log_model.fit(X_train_wine, Y_train_wine)

    y_pred_train = log_model.predict(X_train_wine)
    acc_train = accuracy(Y_train_wine, y_pred_train)
    prec_train = precision(Y_train_wine, y_pred_train)
    rec_train = recall(Y_train_wine, y_pred_train)
    f1_train = f1_score(Y_train_wine, y_pred_train)

    y_pred_test = log_model.predict(X_test_wine)
    acc_test = accuracy(Y_test_wine, y_pred_test)
    prec_test = precision(Y_test_wine, y_pred_test)
    rec_test = recall(Y_test_wine, y_pred_test)
    f1_test = f1_score(Y_test_wine, y_pred_test)

    results = [
        ("Linear Regression", "Train", "MSE", "Boston", round(mse_train, 5)),
        ("Linear Regression", "Test", "MSE", "Boston", round(mse_test, 5)),
        ("Logistic Regression", "Train", "Accuracy", "WINE", round(acc_train, 5)),
        ("Logistic Regression", "Train", "Precision", "WINE", round(prec_train, 5)),
        ("Logistic Regression", "Train", "Recall", "WINE", round(rec_train, 5)),
        ("Logistic Regression", "Train", "F1-score", "WINE", round(f1_train, 5)),
        ("Logistic Regression", "Test", "Accuracy", "WINE", round(acc_test, 5)),
        ("Logistic Regression", "Test", "Precision", "WINE", round(prec_test, 5)),
        ("Logistic Regression", "Test", "Recall", "WINE", round(rec_test, 5)),
        ("Logistic Regression", "Test", "F1-score", "WINE", round(f1_test, 5)),
    ]

    print_table(
        "Linear Regression and Logistic Regression (80/20 Split)",
        results,
        ["Regression Type", "Train/Test", "Metric", "Dataset", "Value"],
        sort_headers=["Dataset", "Metric"],
    )
    print()


# -------------------------------- Experiment 2 --------------------------------

# For both datasets, use a 5-fold cross-validation technique and report the
# performance metrics on both the training set and test set for each model.
# Again, include appropriate performance metrics for each model. Check this
# link for more information.
# Note: 5-fold cross-validation is a technique where the dataset is divided
# into five equal parts (folds), and a model is trained and evaluated five
# times, each time using a different fold as the validation set and the remain-
# ing four folds for training.
if experiment_2:
    print(" --- Experiment 2 --- ")
    k = 5

    # --------------------- Linear Regression on Boston Dataset ---------------------
    lr_model = LinearRegressionAnalytic()
    log_model = LogisticRegressionGD()

    lr_mse = k_fold_cross_validation(lr_model, X_boston, Y_boston, MSE, k)
    lr_mse = sum(lr_mse) / k

    acc = k_fold_cross_validation(log_model, X_wine, Y_wine, accuracy, k)
    acc = sum(acc) / k
    prec = k_fold_cross_validation(log_model, X_wine, Y_wine, precision, k)
    prec = sum([mean_precision for mean_precision in prec]) / k
    rec = k_fold_cross_validation(log_model, X_wine, Y_wine, recall, k)
    rec = sum([mean_recall for mean_recall in rec]) / k
    f1 = k_fold_cross_validation(log_model, X_wine, Y_wine, f1_score, k)
    f1 = sum(f1) / k

    logistic_results = [
        ("MSE", "Boston", round(lr_mse, 5)),
        ("Accuracy", "Wine", round(acc, 5)),
        ("Precision", "Wine", round(prec, 5)),
        ("Recall", "Wine", round(rec, 5)),
        ("F1-score", "Wine", round(f1, 5)),
    ]
    print_table(
        "Linear Regression and Logistic Regression (5-fold CV)",
        logistic_results,
        ["Metric", "Dataset", "5-fold CV"],
    )
    print()


# -------------------------------- Experiment 3 --------------------------------

# # For both datasets, Sample growing subsets of the training data (20%, 30%, ...,
# # 80%). Observe and explain how does size of training data affects the
# # performance for both models. Plot two curves as a function of training size,
# # one for performance in train and one for test.
if experiment_3:
    print(" --- Experiment 3 --- ")

    X_train_boston, Y_train_boston, X_test_boston, Y_test_boston = train_test_split(
        X_boston, Y_boston, split_ratio=0.8
    )
    X_train_wine, Y_train_wine, X_test_wine, Y_test_wine = train_test_split(
        X_wine, Y_wine, split_ratio=0.8
    )

    # Lists to store tabling results
    tabling_results = []

    # Lists to store metrics for plotting
    dim_1_boston = ["MSE"]
    dim_2_boston = ["Train", "Test"]

    dim_1_wine = ["Accuracy", "Precision", "Recall", "F1-score"]
    dim_2_wine = ["Train", "Test"]

    train_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    data_boston = np.zeros((len(dim_1_boston), len(dim_2_boston), len(train_sizes), 1))
    data_wine = np.zeros((len(dim_1_wine), len(dim_2_wine), len(train_sizes), 1))

    for i, size in enumerate(train_sizes):
        lr_model = LinearRegressionGD(epochs=100)
        log_model = LogisticRegressionGD(epochs=100)

        lr_model.fit(X_train_boston, Y_train_boston)
        log_model.fit(X_train_wine, Y_train_wine)

        mse_train = MSE(Y_train_boston, lr_model.predict(X_train_boston))
        mse_test = MSE(Y_test_boston, lr_model.predict(X_test_boston))

        acc_train = accuracy(Y_train_wine, log_model.predict(X_train_wine))
        acc_test = accuracy(Y_test_wine, log_model.predict(X_test_wine))
        prec_train = precision(Y_train_wine, log_model.predict(X_train_wine))
        prec_test = precision(Y_test_wine, log_model.predict(X_test_wine))
        rec_train = recall(Y_train_wine, log_model.predict(X_train_wine))
        rec_test = recall(Y_test_wine, log_model.predict(X_test_wine))
        f1_train = f1_score(Y_train_wine, log_model.predict(X_train_wine))
        f1_test = f1_score(Y_test_wine, log_model.predict(X_test_wine))

        # Store metrics for tabling
        tabling_results.extend(
            [
                [
                    "Linear Regression",
                    "Boston",
                    "Train",
                    "MSE",
                    f"{size * 100:.0f}",
                    round(mse_train, 5),
                ],
                [
                    "Linear Regression",
                    "Boston",
                    "Test",
                    "MSE",
                    f"{size * 100:.0f}",
                    round(mse_test, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "Train",
                    "Accuracy",
                    f"{size * 100:.0f}",
                    round(acc_train, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "Test",
                    "Accuracy",
                    f"{size * 100:.0f}",
                    round(acc_test, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "Train",
                    "Precision",
                    f"{size * 100:.0f}",
                    round(prec_train, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "Test",
                    "Precision",
                    f"{size * 100:.0f}",
                    round(prec_test, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "Train",
                    "Recall",
                    f"{size * 100:.0f}",
                    round(rec_train, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "Test",
                    "Recall",
                    f"{size * 100:.0f}",
                    round(rec_test, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "Train",
                    "F1-score",
                    f"{size * 100:.0f}",
                    round(f1_train, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "Test",
                    "F1-score",
                    f"{size * 100:.0f}",
                    round(f1_test, 5),
                ],
            ]
        )

        # Store metrics for plotting
        data_boston[0, :, i, 0] = np.array([mse_train, mse_test])
        data_wine[0, :, i, 0] = np.array([acc_train, acc_test])
        data_wine[1, :, i, 0] = np.array([prec_train, prec_test])
        data_wine[2, :, i, 0] = np.array([rec_train, rec_test])
        data_wine[3, :, i, 0] = np.array([f1_train, f1_test])

    print_table(
        "Performance vs. Training Size",
        tabling_results,
        [
            "Regression Type",
            "Dataset",
            "Train/Test",
            "Metric",
            "Training Size",
            "Value",
        ],
    )

    # Plotting

    # Linear Regression on Boston Dataset
    max_y = data_boston.max() * 1.2

    plot1(
        data=data_boston,
        title="Performance vs. Training Size (Boston Dataset - Linear Regression)",
        main_labels=["Training Size"] + dim_1_boston,
        ax_titles=["Mean Squared Error"],
        algs_info=[("Train", "b", 2), ("Test", "r", 2)],
        range_y=[[0, max_y]],
        x_axis_relative_range=[-1, -0.7],
        custom_x_labels=[f"{size * 100:.0f}%" for size in train_sizes],
        plot_box=Bbox([[0, 0], [10, 10]]),
        filename=PLOTS_PATH / "experiment_3_boston.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    # Logistic Regression on Wine Dataset
    max_y = 1.0
    min_y = data_wine.min() * 0.8

    plot1(
        data=data_wine,
        title="Performance vs. Training Size (Wine Dataset - Logistic Regression)",
        main_labels=["Training Size"] + dim_1_wine,
        ax_titles=dim_1_wine,
        algs_info=[("Train", "b", 2), ("Test", "r", 2)],
        range_y=[[min_y, max_y], [min_y, max_y], [min_y, max_y], [min_y, max_y]],
        x_axis_relative_range=[-1, -0.7],
        custom_x_labels=[f"{size * 100:.0f}%" for size in train_sizes],
        plot_box=Bbox([[0, 0], [10, 10]]),
        filename=PLOTS_PATH / "experiment_3_wine.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )


# -------------------------------- Experiment 4 --------------------------------

# For both datasets, try out growing minibatch sizes, e.g., 8, 16, 32, 64, and
# 128. Compare the convergence speed and final performance of different batch
# sizes to the fully batched baseline. Which configuration works the best among
# the ones you tried?
# Note: This is for SGD only (Task2, third main task).
if experiment_4:
    print(" --- Experiment 4 --- ")

    batch_sizes = [8, 16, 32, 64, 128]

    X_train_boston, Y_train_boston, X_test_boston, Y_test_boston = train_test_split(
        X_boston, Y_boston, split_ratio=0.8
    )
    X_train_wine, Y_train_wine, X_test_wine, Y_test_wine = train_test_split(
        X_wine, Y_wine, split_ratio=0.8
    )

    def plot_experiment(
        model, dataset_name, X_train, Y_train, X_test, Y_test, metric_function, ylabel
    ):
        def metric_function_func(model):
            return metric_function(Y_test, model.predict(X_test))

        def accuracy_func(model):
            return accuracy(Y_test, model.predict(X_test))

        plt.clf()  # clear previous plot
        for size in batch_sizes:
            if dataset_name == "BOSTON":
                model_instance = model(learning_rate=0.01, batch_size=size, epochs=10)
                test_func = metric_function_func
            else:
                model_instance = model(learning_rate=0.01, batch_size=size, epochs=10)
                test_func = accuracy_func

            history = model_instance.fit(X_train, Y_train, test_func=test_func)
            plt.plot(history, label=f"Batch size: {size}")

        plt.title(f"{model.__name__} on {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(
            PLOTS_PATH / f"experiment_4_{model.__name__.lower()}_{dataset_name.lower()}.png"
        )
        plt.show()

    # Linear Regression on Boston Dataset
    plot_experiment(
        LinearRegressionSGD,
        "BOSTON",
        X_train_boston,
        Y_train_boston,
        X_test_boston,
        Y_test_boston,
        MSE,
        "MSE",
    )

    # Logistic Regression on Wine Dataset
    plot_experiment(
        LogisticRegressionSGD,
        "WINE",
        X_train_wine,
        Y_train_wine,
        X_test_wine,
        Y_test_wine,
        accuracy,
        "Accuracy",
    )


# -------------------------------- Experiment 5 --------------------------------

# For both datasets, Present the performance of both linear and logistic
# regression with at least three different learning rates (your own choice).
if experiment_5:
    print(" --- Experiment 5 --- ")

    # Lists to store tabling results
    tabling_results = []

    # Lists to store metrics for plotting
    dim_1_boston = ["MSE"]
    dim_2_boston = ["5-fold cross-validation"]

    dim_1_wine = ["Accuracy", "Precision", "Recall", "F1-score"]
    dim_2_wine = ["5-fold cross-validation"]

    learning_rates = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    k = 5

    data_boston = np.zeros((len(dim_1_boston), len(dim_2_boston), len(learning_rates), k))
    data_wine = np.zeros((len(dim_1_wine), len(dim_2_wine), len(learning_rates), k))

    for i, rate in enumerate(learning_rates):
        # Linear Regression on Boston Dataset
        lr_model = LinearRegressionGD(learning_rate=rate)
        log_model = LogisticRegressionGD(learning_rate=rate)

        lr_mse = k_fold_cross_validation(lr_model, X_boston, Y_boston, MSE, k)
        acc = k_fold_cross_validation(log_model, X_wine, Y_wine, accuracy, k)
        prec = k_fold_cross_validation(log_model, X_wine, Y_wine, precision, k)
        rec = k_fold_cross_validation(log_model, X_wine, Y_wine, recall, k)
        f1 = k_fold_cross_validation(log_model, X_wine, Y_wine, f1_score, k)

        # Store metrics for tabling
        tabling_results.extend(
            [
                [
                    "Linear Regression",
                    "Boston",
                    "5-fold cross-validation",
                    "MSE",
                    f"{rate:.3f}",
                    round(sum(lr_mse) / k, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "5-fold cross-validation",
                    "Accuracy",
                    f"{rate:.3f}",
                    round(sum(acc) / k, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "5-fold cross-validation",
                    "Precision",
                    f"{rate:.3f}",
                    round(sum(prec) / k, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "5-fold cross-validation",
                    "Recall",
                    f"{rate:.3f}",
                    round(sum(rec) / k, 5),
                ],
                [
                    "Logistic Regression",
                    "Wine",
                    "5-fold cross-validation",
                    "F1-score",
                    f"{rate:.3f}",
                    round(sum(f1) / k, 5),
                ],
            ]
        )

        # Store metrics for plotting
        data_boston[0, :, i, :] = np.array([lr_mse])
        print(lr_mse)
        data_wine[0, :, i, :] = np.array([acc])
        data_wine[1, :, i, :] = np.array([prec])
        data_wine[2, :, i, :] = np.array([rec])
        data_wine[3, :, i, :] = np.array([f1])

    print_table(
        "Performance vs. Learning Rate",
        tabling_results,
        [
            "Regression Type",
            "Dataset",
            "Train/Test",
            "Metric",
            "Learning Rate",
            "Value",
        ],
    )

    # Plotting
    min_y = data_boston.min() * 0.8
    max_y = data_boston.max() * 1.2

    plot1(
        data=data_boston,
        title="Performance vs. Learning Rate",
        main_labels=["Learning Rate"] + dim_1_boston,
        ax_titles=dim_1_boston,
        algs_info=[("Learning Rate", "b", 0)],
        range_y=[[0, max_y]],
        x_axis_relative_range=[-0.1, -0.9],
        fill_std=[0],
        custom_x_labels=[str(rate) for rate in learning_rates],
        plot_box=Bbox([[0, 0], [10, 10]]),
        filename=PLOTS_PATH / "experiment_5_boston.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    min_y = data_wine.min() * 0.8
    max_y = 1.0

    plot1(
        data=data_wine,
        title="Performance vs. Learning Rate",
        main_labels=["Learning Rate"] + dim_1_wine,
        ax_titles=dim_1_wine,
        algs_info=[("Learning Rate", "b", 0)],
        range_y=[[min_y, max_y], [min_y, max_y], [min_y, max_y], [min_y, max_y]],
        x_axis_relative_range=[-0.1, -0.9],
        fill_std=[0] * len(dim_1_wine),
        custom_x_labels=[str(rate) for rate in learning_rates],
        plot_box=Bbox([[0, 0], [10, 10]]),
        filename=PLOTS_PATH / "experiment_5_wine.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )


# -------------------------------- Experiment 6 --------------------------------

# For both datasets, Given a variety of parameter configurations,
# select a performance metric and present the optimal parameter choice
# for each dataset. Please provide a rationale for your metric selection,
# along with an explanation of why you opted for that particular metric.
if experiment_6:
    print(" --- Experiment 6 --- ")

    learning_rates = [0.003, 0.01, 0.03, 0.1]
    batch_sizes = [4, 8, 16, 32, 64]

    # Lists to store metrics for plotting
    dim_1_boston = ["MSE"]
    dim_1_wine = ["Accuracy", "Precision", "Recall", "F1-score"]

    k = 5

    data_boston = np.zeros((len(dim_1_boston), len(learning_rates), len(batch_sizes), k))
    data_wine = np.zeros((len(dim_1_wine), len(learning_rates), len(batch_sizes), k))
    heatmap_data_boston = np.zeros((len(learning_rates), len(batch_sizes)))
    heatmap_data_wine = np.zeros((len(learning_rates), len(batch_sizes)))

    best_params_boston = {
        "learning_rate": None,
        "batch_size": None,
        "MSE": float("inf"),
    }
    best_params_wine = {"learning_rate": None, "batch_size": None, "F1": 0}

    for i, rate in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            # Linear Regression on Boston Dataset
            lr_model = LinearRegressionSGD(learning_rate=rate, batch_size=batch_size, epochs=10)
            lr_mse = k_fold_cross_validation(lr_model, X_boston, Y_boston, MSE, k)
            avg_mse = sum(lr_mse) / k

            if avg_mse < best_params_boston["MSE"]:
                best_params_boston = {
                    "learning_rate": rate,
                    "batch_size": batch_size,
                    "MSE": avg_mse,
                }

            # Logistic Regression on Wine Dataset
            log_model = LogisticRegressionSGD(learning_rate=rate, batch_size=batch_size, epochs=10)
            f1 = k_fold_cross_validation(log_model, X_wine, Y_wine, f1_score, k)
            avg_f1 = sum(f1) / k

            if avg_f1 > best_params_wine["F1"]:
                best_params_wine = {
                    "learning_rate": rate,
                    "batch_size": batch_size,
                    "F1": avg_f1,
                }

            # Store metrics for plotting
            data_boston[0, i, j, :] = lr_mse
            data_wine[0, i, j, :] = k_fold_cross_validation(log_model, X_wine, Y_wine, accuracy, k)
            data_wine[1, i, j, :] = k_fold_cross_validation(log_model, X_wine, Y_wine, precision, k)
            data_wine[2, i, j, :] = k_fold_cross_validation(log_model, X_wine, Y_wine, recall, k)
            data_wine[3, i, j, :] = f1

            # Store data for heatmap
            heatmap_data_boston[i, j] = avg_mse
            heatmap_data_wine[i, j] = avg_f1

            print(
                f"Learning Rate: {rate}, Batch Size: {batch_size}, MSE: {avg_mse:.5f}, F1: {avg_f1:.5f}"
            )

    # print(data_boston.shape) # 1, 3, 5, 5 aka 1 plot, 3 learning rates, 5 batch sizes, 5 k-fold cv
    # print(data_wine.shape) # 4, 3, 5, 5 aka 4 plots, 3 learning rates, 5 batch sizes, 5 k-fold cv

    # print(heatmap_data_boston.shape, heatmap_data_wine.shape) # both (3, 5) aka 3 learning rates, 5 batch sizes
    # print(heatmap_data_boston) # [[0.02734825 0.03659205 0.04757617 0.05855891 0.08064244]
    #                            #  [0.0133526  0.01595599 0.0200038  0.02500598 0.03338523]
    #                            #  [0.01193082 0.01196209 0.01218532 0.01287588 0.01486311]]
    # print(heatmap_data_wine)   # [[0.49145976 0.32459179 0.19351852 0.39823762 0.27293533]
    #                            #  [0.67968083 0.51273187 0.50875727 0.42271972 0.49982385]
    #                            #  [0.95815425 0.95873959 0.94320954 0.7968946  0.55544355]]

    # Display Results
    print("\nOptimal Parameters for Linear Regression (Boston):")
    print(
        f"Learning Rate: {best_params_boston['learning_rate']}, Batch Size: {best_params_boston['batch_size']}, MSE: {best_params_boston['MSE']:.5f}"
    )

    print("\nOptimal Parameters for Logistic Regression (Wine):")
    print(
        f"Learning Rate: {best_params_wine['learning_rate']}, Batch Size: {best_params_wine['batch_size']}, F1: {best_params_wine['F1']:.5f}"
    )

    def compute_color(learning_rate, min_lr, max_lr):
        """
        Calculate the color based on the logarithmic position of the learning rate within the provided range.
        Interpolate linearly between yellow (255, 255, 0) and red (255, 0, 0).
        """
        # Calculate logarithmic position
        log_pos = (np.log(learning_rate) - np.log(min_lr)) / (np.log(max_lr) - np.log(min_lr))

        # Interpolate between yellow and red
        r = 255
        g = int(255 * (1 - log_pos))
        b = 0

        # Convert RGB to hex
        color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return color

    alg_info = [
        (
            f"LR: {learning_rate}",
            compute_color(learning_rate, min(learning_rates), max(learning_rates)),
            0,
        )
        for learning_rate in learning_rates
    ]

    # Plotting
    plot1(
        data=data_boston,
        title="Performance vs. Learning Rate and Batch Size (Boston-LR)",
        main_labels=["Batch Size"] + dim_1_boston,
        ax_titles=dim_1_boston,
        algs_info=alg_info,
        fill_std=[0],
        custom_x_labels=[str(batch_size) for batch_size in batch_sizes],
        plot_box=Bbox([[0, 0], [10, 10]]),
        filename=PLOTS_PATH / "experiment_6_boston.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    plot1(
        data=data_wine,
        title="Performance vs. Learning Rate and Batch Size (Wine-LogR)",
        main_labels=["Batch Size"] + dim_1_wine,
        ax_titles=dim_1_wine,
        algs_info=alg_info,
        fill_std=[0] * len(dim_1_wine),
        custom_x_labels=[str(batch_size) for batch_size in batch_sizes],
        plot_box=Bbox([[0, 0], [10, 10]]),
        filename=PLOTS_PATH / "experiment_6_wine.png" if SAVE_FILES else None,
        show=SHOW_GRAPHS,
    )

    # Heatmap
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    cax1 = axs[0].matshow(heatmap_data_boston, cmap="viridis_r")
    cax2 = axs[1].matshow(heatmap_data_wine, cmap="viridis")
    fig.colorbar(cax1, ax=axs[0])
    fig.colorbar(cax2, ax=axs[1])
    axs[0].set_title("Boston Dataset (MSE)")
    axs[1].set_title("Wine Dataset (F1-score)")
    for (i, j), z in np.ndenumerate(heatmap_data_boston):
        axs[0].text(j, i, "{:0.4f}".format(z), ha="center", va="center", color="w", fontsize=10)
    for (i, j), z in np.ndenumerate(heatmap_data_wine):
        axs[1].text(j, i, "{:0.4f}".format(z), ha="center", va="center", color="w", fontsize=10)
    axs[0].set_xticks(np.arange(len(batch_sizes)))
    axs[0].set_yticks(np.arange(len(learning_rates)))
    axs[0].set_xticklabels(batch_sizes)
    axs[0].set_yticklabels(learning_rates)
    axs[1].set_xticks(np.arange(len(batch_sizes)))
    axs[1].set_yticks(np.arange(len(learning_rates)))
    axs[1].set_xticklabels(batch_sizes)
    axs[1].set_yticklabels(learning_rates)
    plt.savefig(PLOTS_PATH / "experiment_6_heatmap.png" if SAVE_FILES else None)


# -------------------------------- Experiment 7 --------------------------------

# Only for dataset1, Gaussian Basis Functions:
# - Utilize Gaussian basis functions to enrich the feature set for Dataset 1.
# - Define each Gaussian basis function as follows:
#
#      phi_j(x) = exp(-||x - mu_j||^2 / 2s^2)
#
# - Employ a total of 5 Gaussian basis functions.
# - Set the spatial scale parameter, s, to a value of 1.
# - Select mu_j values randomly from the training set to determine the centres of these basis functions.
# - Use analytical linear regression to predict the target value.
# Compare the target and predicted values obtained with the new dataset with
# the results obtained with the original feature set, i.e. compare with the
# results obtained without Gaussian basis functions.
if experiment_7:
    print(" --- Experiment 7 --- ")

    # Parameters for Gaussian basis functions
    s = 1.0
    num_basis_functions = 5

    # Randomly select 5 data points from the training set as the centers of the Gaussian basis functions
    random_indices = np.random.choice(X_boston.shape[0], num_basis_functions, replace=False)
    mu_values = X_boston[random_indices]

    def gaussian_basis_function(x, mu, s):
        return np.exp(-(np.linalg.norm(x - mu) ** 2) / (2 * s**2))

    def transform_with_gaussian_basis(X, mu_values, s):
        transformed_X = np.zeros((X.shape[0], num_basis_functions))
        for i in range(X.shape[0]):
            for j, mu in enumerate(mu_values):
                transformed_X[i, j] = gaussian_basis_function(X[i], mu, s)
        return transformed_X

    # Transform the original feature set using Gaussian basis functions
    X_boston_transformed = transform_with_gaussian_basis(X_boston, mu_values, s)

    # Train analytical linear regression models on both original and transformed datasets
    lr_original = LinearRegressionAnalytic()
    lr_transformed = LinearRegressionAnalytic()

    # Split datasets into training and test sets
    (
        X_train_boston_original,
        Y_train_boston_original,
        X_test_boston_original,
        Y_test_boston_original,
    ) = train_test_split(X_boston, Y_boston, split_ratio=0.8)
    (
        X_train_boston_transformed,
        Y_train_boston_transformed,
        X_test_boston_transformed,
        Y_test_boston_transformed,
    ) = train_test_split(X_boston_transformed, Y_boston, split_ratio=0.8)

    lr_original.fit(X_train_boston_original, Y_train_boston_original)
    lr_transformed.fit(X_train_boston_transformed, Y_train_boston_transformed)

    # Make predictions
    Y_pred_original = lr_original.predict(X_test_boston_original)
    Y_pred_transformed = lr_transformed.predict(X_test_boston_transformed)

    # Calculate MSE for both predictions
    mse_original = MSE(Y_test_boston_original, Y_pred_original)
    mse_transformed = MSE(Y_test_boston_transformed, Y_pred_transformed)

    print(f"MSE using original feature set: {mse_original:.5f}")
    print(f"MSE using Gaussian basis functions: {mse_transformed:.5f}")


# -------------------------------- Experiment 8 --------------------------------

# Only for dataset1, Compare analytical linear regression solution with
# mini-batch stochastic gradient descent- based linear regression solution.
# What do you find? Why do you think mini-batch stochastic gradient descent is
# used when an analytical solution is available?
if experiment_8:
    print(" --- Experiment 8 --- ")
    X_train_boston, Y_train_boston, X_test_boston, Y_test_boston = train_test_split(
        X_boston, Y_boston, split_ratio=0.8
    )

    # analytical linear regression
    lr_analytical = LinearRegressionAnalytic()
    lr_analytical.fit(X_train_boston, Y_train_boston)
    mse_analytical = MSE(Y_test_boston, lr_analytical.predict(X_test_boston))

    # mini-batch SGD
    lr_sgd = LinearRegressionSGD(learning_rate=0.01, batch_size=32, epochs=100)
    lr_sgd.fit(X_train_boston, Y_train_boston)
    mse_sgd = MSE(Y_test_boston, lr_sgd.predict(X_test_boston))

    print(f"MSE from Analytical Solution: {mse_analytical:.5f}")
    print(f"MSE from Mini-Batch SGD: {mse_sgd:.5f}")

    if mse_analytical < mse_sgd:
        print("Analytical solution performs better.")
    else:
        print("Mini-Batch SGD performs better.")


# -------------------------------- close file --------------------------------

if print_output_to_file:
    file.close()
