import numpy as np


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
