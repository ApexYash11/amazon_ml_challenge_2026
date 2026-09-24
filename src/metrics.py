"""Numerically safe common metrics and the official-metric extension point."""
import numpy as np
from sklearn.metrics import accuracy_score, log_loss as _log_loss, mean_absolute_error, mean_squared_error, r2_score


def competition_metric(y_true, y_pred):
    raise NotImplementedError("Implement the official Amazon ML Challenge 2026 metric here.")


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def mse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, epsilon=1e-8):
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), epsilon)) * 100)


def smape(y_true, y_pred, epsilon=1e-8):
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(actual) + np.abs(predicted), epsilon)
    return float(np.mean(2 * np.abs(predicted - actual) / denominator) * 100)


def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def log_loss(y_true, y_pred, labels=None):
    probabilities = np.asarray(y_pred, dtype=float)
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    if probabilities.ndim == 2:
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    return float(_log_loss(y_true, probabilities, labels=labels))


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def score(name, y_true, y_pred):
    functions = {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape,
                 "smape": smape, "r2": r2, "accuracy": accuracy}
    if name not in functions:
        raise ValueError(f"Unsupported baseline metric: {name}")
    return functions[name](y_true, y_pred)
