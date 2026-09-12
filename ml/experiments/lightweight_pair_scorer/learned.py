"""Tiny calibration-only logistic scorer over frozen CLIP retrieval features."""

import time

import numpy as np

FEATURE_SCHEMA = [
    "clip_score",
    "reciprocal_rank",
    "gap_from_top",
    "top_two_gap",
    "query_score_mean",
    "query_score_std",
    "query_score_range",
]


def candidate_features(candidates: list[dict]) -> np.ndarray:
    if not candidates:
        return np.empty((0, len(FEATURE_SCHEMA)), dtype=np.float64)
    scores = np.asarray([candidate["score"] for candidate in candidates], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("candidate scores must be finite")
    top = scores[0]
    top_two_gap = top - scores[1] if len(scores) > 1 else 0.0
    common = (top_two_gap, float(scores.mean()), float(scores.std()), float(np.ptp(scores)))
    return np.asarray(
        [
            [score, 1.0 / rank, top - score, *common]
            for rank, score in enumerate(scores, 1)
        ],
        dtype=np.float64,
    )


def is_relevant(timestamp: float, intervals: list[list[float]]) -> bool:
    return any(start <= timestamp < end for start, end in intervals)


def training_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("learned scorer training requires calibration-only rows")
    features, labels = [], []
    for row in rows:
        candidates = row["clip_candidates"]
        features.extend(candidate_features(candidates))
        labels.extend(
            is_relevant(candidate["timestamp"], row["relevant_intervals"])
            for candidate in candidates
        )
    target = np.asarray(labels, dtype=np.float64)
    if not target.any() or target.all():
        raise ValueError("both relevant and irrelevant candidate examples required")
    return np.asarray(features, dtype=np.float64), target


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def fit(rows: list[dict], regularization: float = 1.0) -> dict:
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    started = time.perf_counter()
    features, labels = training_matrix(rows)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    positive_weight = len(labels) / (2 * labels.sum())
    negative_weight = len(labels) / (2 * (len(labels) - labels.sum()))
    sample_weights = np.where(labels == 1, positive_weight, negative_weight)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.diag([0.0, *([regularization] * standardized.shape[1])])
    for _ in range(100):
        probability = _sigmoid(design @ coefficients)
        curvature = sample_weights * probability * (1 - probability)
        gradient = design.T @ (sample_weights * (probability - labels)) + penalty @ coefficients
        hessian = (design.T * curvature) @ design + penalty
        update = np.linalg.solve(hessian + np.eye(len(coefficients)) * 1e-9, gradient)
        coefficients -= update
        if np.max(np.abs(update)) < 1e-10:
            break
    return {
        "feature_schema": FEATURE_SCHEMA,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": coefficients[1:].tolist(),
        "intercept": float(coefficients[0]),
        "regularization": regularization,
        "training_seconds": time.perf_counter() - started,
        "trainable_parameters": len(coefficients),
        "training_examples": len(labels),
        "positive_examples": int(labels.sum()),
    }


def score(candidates: list[dict], model: dict) -> list[float]:
    features = candidate_features(candidates)
    if features.shape[1] != len(model["feature_schema"]):
        raise ValueError("feature schema mismatch")
    standardized = (
        features - np.asarray(model["feature_mean"])
    ) / np.asarray(model["feature_scale"])
    logits = standardized @ np.asarray(model["weights"]) + model["intercept"]
    return _sigmoid(logits).tolist()


def inference_latency(candidates: list[dict], model: dict, repeats: int = 1000) -> dict:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        score(candidates, model)
        timings.append(time.perf_counter() - started)
    return {
        "median_seconds": float(np.median(timings)),
        "p95_seconds": float(np.percentile(timings, 95)),
        "repeats": repeats,
    }
