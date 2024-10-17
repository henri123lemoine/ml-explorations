import numpy as np


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def confusion_matrix_multiclass(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for t, p in zip(y_true, y_pred):
        true_index = np.where(classes == t)[0][0]
        pred_index = np.where(classes == p)[0][0]
        cm[true_index][pred_index] += 1
    return cm


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    cm = confusion_matrix_multiclass(y_true, y_pred)
    tp = np.diag(cm)
    tp_fp = np.sum(cm, axis=0)
    precision_scores = np.divide(tp, tp_fp, out=np.zeros_like(tp, dtype=float), where=tp_fp != 0)
    return np.mean(precision_scores)


def recall(y_true, y_pred):
    cm = confusion_matrix_multiclass(y_true, y_pred)
    tp = np.diag(cm)
    tp_fn = np.sum(cm, axis=1)
    recall_scores = np.divide(tp, tp_fn, out=np.zeros_like(tp, dtype=float), where=tp_fn != 0)
    return np.mean(recall_scores)


def f1_score(y_true, y_pred):
    avg_precision = precision(y_true, y_pred)
    avg_recall = recall(y_true, y_pred)
    f1 = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )
    return f1
