"""Fit a library of training-drug signatures and predict by soft similarity votes."""
import numpy as np
from scipy.linalg import eigh
from scipy.special import softmax

from response_features import center_signatures


def measurements(raw):
    return np.arcsinh(center_signatures(raw)).astype(np.float32)


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def fit_library(x, drugs, labels):
    """Find directions where differences between drugs exceed replicate noise."""
    mean, scale = x.mean(0), x.std(0).clip(0.03)
    features = (x - mean) / scale
    _, first, group, counts = np.unique(drugs, return_index=True, return_inverse=True, return_counts=True)
    profiles = np.asarray(labels[first], dtype=np.uint8)
    if not np.array_equal(profiles[group], labels):
        raise ValueError("Each training drug must have a consistent scored annotation")
    centers = np.zeros((len(counts), x.shape[1]), dtype=np.float32)
    np.add.at(centers, group, features)
    centers /= counts[:, None]
    noise = features - centers[group]
    within = noise.T @ noise / max(len(x) - len(centers), 1)
    ridge = max(0.2 * np.trace(within) / len(within), 1e-6)
    within.flat[::len(within) + 1] += ridge
    between = centers.T @ (centers * counts[:, None]) / counts.sum()
    width = min(128, len(within), len(centers) - 1)
    if width < 1:
        raise ValueError("At least two training drugs are needed")
    _, directions = eigh(between, within, subset_by_index=[len(within) - width, len(within) - 1])
    projection = directions.astype(np.float32)
    return dict(mean=mean, scale=scale, projection=projection,
                centers=unit_rows(centers @ projection), log_count=np.log(counts.astype(np.float32)),
                targets=profiles)


def predict_library(x, state):
    """Return 206 marginal probabilities and the best cosine match for every sample."""
    embedding = unit_rows(((x - state["mean"]) / state["scale"]) @ state["projection"])
    targets = state["targets"].astype(np.float32)
    pred = np.empty((len(x), targets.shape[1]), dtype=np.float32)
    confidence = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), 1024):
        batch = slice(start, start + 1024)
        similarity = embedding[batch] @ state["centers"].T
        confidence[batch] = similarity.max(axis=1).clip(0, 1)
        weights = softmax(24 * similarity + state["log_count"], axis=1)
        pred[batch] = (weights @ targets).clip(0, 1)
    return pred, confidence


def library_for_test(test, mask, models_dir, n_targets, n_folds):
    from infer_engine import GENE_COLS, CELL_COLS

    probabilities = np.zeros((n_folds, len(test), n_targets), dtype=np.float32)
    confidence = np.zeros((n_folds, len(test)), dtype=np.float32)
    if mask.any():
        x = measurements(test.loc[mask, GENE_COLS + CELL_COLS].to_numpy())
        for f in range(n_folds):
            with np.load(models_dir / "drug_library" / f"fold{f}.npz") as state:
                probabilities[f, mask], confidence[f, mask] = predict_library(x, dict(state))
    return probabilities, confidence
