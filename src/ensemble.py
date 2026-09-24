"""Small helpers for blending already aligned prediction arrays."""
import numpy as np


def weighted_blend(predictions, weights):
    arrays = [np.asarray(pred, dtype=float) for pred in predictions]
    weights = np.asarray(weights, dtype=float)
    if not arrays or len(arrays) != len(weights):
        raise ValueError("Provide one weight per non-empty prediction array.")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("All prediction arrays must have the same shape and ordering.")
    if not np.isfinite(weights).all() or weights.sum() == 0:
        raise ValueError("Weights must be finite and have a non-zero sum.")
    weights = weights / weights.sum()
    return np.tensordot(weights, np.stack(arrays), axes=(0, 0))


def make_stacking_features(oof_predictions, test_predictions=None):
    """Shape aligned out-of-fold predictions as meta-model inputs."""
    oof = np.column_stack([np.asarray(pred) for pred in oof_predictions])
    if test_predictions is None:
        return oof
    test = np.column_stack([np.asarray(pred) for pred in test_predictions])
    if oof.shape[1] != test.shape[1]:
        raise ValueError("OOF and test prediction lists must contain the same models.")
    return oof, test
