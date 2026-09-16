"""Small deterministic calibration and fusion helpers.

Nothing in this module is imported by production retrieval.  The functions
operate on already retrieved candidates and preserve the production thumbnail
grouping used by the evaluation trace.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

MODALITY_FEATURE_SETS = {
    "rank_only": ("reciprocal_rank",),
    "raw_only": ("raw_score",),
    "query_relative": (
        "reciprocal_rank",
        "margin_to_top",
        "margin_to_next",
        "robust_z",
    ),
    "combined": (
        "raw_score",
        "reciprocal_rank",
        "margin_to_top",
        "margin_to_next",
        "robust_z",
        "log_query_tokens",
    ),
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40, 40)
    return 1 / (1 + np.exp(-clipped))


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    regularization: float = 1.0,
) -> dict[str, Any]:
    """Fit deterministic, unweighted L2 logistic calibration with Newton steps."""
    if features.ndim != 2 or len(features) != len(labels) or regularization <= 0:
        raise ValueError("invalid logistic training data")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("both classes are required")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    coefficients = np.zeros(design.shape[1])
    penalty = np.diag([0.0, *([regularization] * features.shape[1])])
    for _ in range(100):
        probability = sigmoid(design @ coefficients)
        curvature = probability * (1 - probability)
        gradient = design.T @ (probability - labels) + penalty @ coefficients
        hessian = (design.T * curvature) @ design + penalty
        update = np.linalg.solve(hessian + np.eye(len(coefficients)) * 1e-9, gradient)
        coefficients -= update
        if float(np.max(np.abs(update))) < 1e-10:
            break
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(coefficients[0]),
        "weights": coefficients[1:].tolist(),
        "regularization": regularization,
        "examples": len(labels),
    }


def predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if features.ndim == 1:
        features = features.reshape(1, -1)
    standardized = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return sigmoid(standardized @ np.asarray(model["weights"]) + model["intercept"])


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("AUC requires both classes")
    wins = sum(
        float(value > other) + 0.5 * float(value == other)
        for value in positives
        for other in negatives
    )
    return wins / (len(positives) * len(negatives))


def brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def _overlaps(candidate: dict[str, Any], intervals: list[list[float]]) -> bool:
    start = candidate["timestamp"]
    end = candidate.get("end")
    end = start if end is None else end
    return any(
        start <= interval_end and end >= interval_start
        for interval_start, interval_end in intervals
    )


def required_modalities(category: str) -> tuple[str, ...]:
    category = category.upper()
    if category == "VISUAL":
        return ("visual",)
    if category == "SPEECH":
        return ("speech",)
    if category == "HYBRID" or category == "MULTIMODAL":
        return ("visual", "speech")
    raise ValueError(f"unsupported category: {category}")


def bucket_relevant(bucket: dict[str, Any], row: dict[str, Any]) -> bool:
    if row["negative"]:
        return False
    contributions = bucket["contributions"]
    return all(
        modality in contributions and _overlaps(contributions[modality], row["intervals"])
        for modality in required_modalities(row["category"])
    )


def agreement_features(bucket: dict[str, Any]) -> dict[str, float | bool | None]:
    """Expose agreement shape without assigning it a fusion reward."""
    contributions = bucket.get("contributions", {})
    visual = contributions.get("visual")
    speech = contributions.get("speech")
    return {
        "visual_present": visual is not None,
        "speech_present": speech is not None,
        "cross_modal_present": visual is not None and speech is not None,
        "temporal_distance_seconds": (
            abs(float(visual["timestamp"]) - float(speech["timestamp"]))
            if visual is not None and speech is not None
            else None
        ),
    }


def candidate_feature_rows(candidates: list[dict[str, Any]], query: str) -> list[dict[str, float]]:
    """Describe each ranked item using six declared, query-local features."""
    if not candidates:
        return []
    scores = np.asarray(
        [item["raw_score"] if "raw_score" in item else item["score"] for item in candidates],
        dtype=float,
    )
    median = float(np.median(scores))
    q25, q75 = np.percentile(scores, [25, 75])
    robust_scale = max(float(q75 - q25), 1e-9)
    token_count = len(re.findall(r"\w+", query.casefold()))
    rows = []
    for index, score in enumerate(scores):
        next_score = scores[index + 1] if index + 1 < len(scores) else score
        rows.append(
            {
                "raw_score": float(score),
                "reciprocal_rank": 1 / (index + 1),
                "margin_to_top": float(scores[0] - score),
                "margin_to_next": float(score - next_score),
                "robust_z": float((score - median) / robust_scale),
                "log_query_tokens": math.log1p(token_count),
            }
        )
    return rows


def _raw_candidates(row: dict[str, Any], modality: str) -> list[dict[str, Any]]:
    return row["trace"]["raw_candidates"][modality]


def modality_examples(
    rows: list[dict[str, Any]], modality: str, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features = []
    labels = []
    metadata = []
    for row in rows:
        if not row["negative"] and modality not in required_modalities(row["category"]):
            continue
        candidates = _raw_candidates(row, modality)
        described = candidate_feature_rows(candidates, row["query"])
        for candidate, description in zip(candidates, described):
            label = int(not row["negative"] and _overlaps(candidate, row["intervals"]))
            features.append([description[name] for name in feature_names])
            labels.append(label)
            metadata.append(
                {
                    "source_id": row["source_id"],
                    "query_id": row["query_id"],
                    "query_tokens": len(re.findall(r"\w+", row["query"].casefold())),
                    "rank": candidate["retriever_rank"],
                    "raw_score": candidate["raw_score"],
                    "margin_to_top": description["margin_to_top"],
                    "margin_to_next": description["margin_to_next"],
                    "label": label,
                }
            )
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=float), metadata


def select_calibrator(rows: list[dict[str, Any]], modality: str) -> dict[str, Any]:
    """Choose a feature set by leave-one-source-out calibration evidence."""
    sources = sorted({row["source_id"] for row in rows})
    if len(sources) < 3:
        raise ValueError("at least three calibration sources are required")
    trials = []
    for feature_set, names in MODALITY_FEATURE_SETS.items():
        folds = []
        for heldout_source in sources:
            train = [row for row in rows if row["source_id"] != heldout_source]
            validation = [row for row in rows if row["source_id"] == heldout_source]
            x_train, y_train, _ = modality_examples(train, modality, names)
            x_validation, y_validation, _ = modality_examples(validation, modality, names)
            model = fit_logistic(x_train, y_train)
            probabilities = predict(model, x_validation)
            folds.append(
                {
                    "source_id": heldout_source,
                    "examples": len(y_validation),
                    "positives": int(y_validation.sum()),
                    "auc": auc(y_validation, probabilities),
                    "brier": brier(y_validation, probabilities),
                }
            )
        trials.append(
            {
                "feature_set": feature_set,
                "features": list(names),
                "mean_auc": float(np.mean([fold["auc"] for fold in folds])),
                "mean_brier": float(np.mean([fold["brier"] for fold in folds])),
                "folds": folds,
            }
        )
    winner = min(
        trials,
        key=lambda trial: (
            trial["mean_brier"],
            -trial["mean_auc"],
            len(trial["features"]),
            trial["feature_set"],
        ),
    )
    names = tuple(winner["features"])
    features, labels, _ = modality_examples(rows, modality, names)
    model = fit_logistic(features, labels)
    model.update({"feature_set": winner["feature_set"], "features": list(names)})
    return {"winner": winner, "trials": trials, "model": model}


def model_metrics(
    model: dict[str, Any], rows: list[dict[str, Any]], modality: str
) -> dict[str, float | int]:
    features, labels, _ = modality_examples(rows, modality, tuple(model["features"]))
    probabilities = predict(model, features)
    return {
        "examples": len(labels),
        "positives": int(labels.sum()),
        "auc": auc(labels, probabilities),
        "brier": brier(labels, probabilities),
    }
