"""Small dependency-free logistic models with explicit split controls."""

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40, 40)
    return 1 / (1 + np.exp(-clipped))


def fit_logistic(features: np.ndarray, labels: np.ndarray, regularization: float) -> dict:
    if features.ndim != 2 or len(features) != len(labels) or regularization <= 0:
        raise ValueError("invalid logistic training data")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("both classes are required")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    positive_weight = len(labels) / (2 * labels.sum())
    negative_weight = len(labels) / (2 * (len(labels) - labels.sum()))
    sample_weights = np.where(labels == 1, positive_weight, negative_weight)
    coefficients = np.zeros(design.shape[1])
    penalty = np.diag([0.0, *([regularization] * features.shape[1])])
    for _ in range(100):
        probability = sigmoid(design @ coefficients)
        curvature = sample_weights * probability * (1 - probability)
        gradient = design.T @ (sample_weights * (probability - labels)) + penalty @ coefficients
        hessian = (design.T * curvature) @ design + penalty
        update = np.linalg.solve(hessian + np.eye(len(coefficients)) * 1e-9, gradient)
        coefficients -= update
        if np.max(np.abs(update)) < 1e-10:
            break
    return {
        "mean": mean.tolist(), "scale": scale.tolist(),
        "intercept": float(coefficients[0]), "weights": coefficients[1:].tolist(),
        "regularization": regularization, "parameters": len(coefficients),
        "examples": len(labels),
    }


def predict(model: dict, features: np.ndarray) -> np.ndarray:
    standardized = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return sigmoid(standardized @ np.asarray(model["weights"]) + model["intercept"])


def fit_accept_threshold(probabilities: np.ndarray, labels: np.ndarray, max_far: float = 0.1) -> dict:
    negatives = labels == 0
    positives = labels == 1
    choices = sorted({float(value) for value in probabilities} | {1.0 + 1e-12})
    feasible = []
    for threshold in choices:
        accepted = probabilities >= threshold
        far = float(accepted[negatives].mean())
        pfa = float((~accepted[positives]).mean())
        if far <= max_far + 1e-12:
            feasible.append((pfa, far, -threshold, threshold))
    pfa, far, _, threshold = min(feasible)
    return {"threshold": threshold, "calibration_far": far, "calibration_pfa": pfa}

