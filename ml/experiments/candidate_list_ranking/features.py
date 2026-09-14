"""Deterministic features derived from one CLIP candidate list."""

import math

import numpy as np

CANDIDATE_FEATURES = (
    "clip_score", "score_z", "score_minmax", "reciprocal_rank", "log_rank",
    "gap_from_top", "gap_from_previous", "top_two_gap", "list_mean", "list_std",
    "list_range", "temporal_support_10s", "nearest_time_gap", "top_frame_cosine",
    "list_centroid_cosine", "adjacent_visual_change",
)

QUERY_FEATURES = (
    "best_score", "top_two_gap", "top_five_gap", "mean", "std", "range",
    "entropy", "temporal_support_10s", "top_frame_centroid_cosine",
    "positive_tail_fraction", "candidate_count_log",
)


def _scores(candidates: list[dict]) -> np.ndarray:
    values = np.asarray([row["score"] for row in candidates], dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("candidate list must contain finite scores")
    return values


def candidate_features(candidates: list[dict], embeddings: np.ndarray) -> np.ndarray:
    scores = _scores(candidates)
    indices = [int(row["index"]) for row in candidates]
    if embeddings.ndim != 2 or any(index < 0 or index >= len(embeddings) for index in indices):
        raise ValueError("candidate indices must address an embedding matrix")
    mean, std = float(scores.mean()), float(scores.std())
    spread = float(np.ptp(scores))
    top_gap = float(scores[0] - scores[1]) if len(scores) > 1 else 0.0
    candidate_vectors = embeddings[indices]
    centroid = candidate_vectors.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    rows = []
    for offset, (candidate, score, vector) in enumerate(
        zip(candidates, scores, candidate_vectors, strict=True)
    ):
        timestamp = float(candidate["timestamp"])
        distances = [abs(timestamp - float(other["timestamp"])) for other in candidates]
        neighbors = [distance for i, distance in enumerate(distances) if i != offset]
        support = sum(distance <= 10.05 for distance in neighbors) / max(1, len(neighbors))
        nearest = min(neighbors, default=0.0)
        index = indices[offset]
        changes = []
        if index:
            changes.append(1 - float(vector @ embeddings[index - 1]))
        if index + 1 < len(embeddings):
            changes.append(1 - float(vector @ embeddings[index + 1]))
        rows.append([
            score, (score - mean) / max(std, 1e-12),
            (score - scores[-1]) / max(spread, 1e-12), 1 / (offset + 1),
            1 / math.log2(offset + 2), scores[0] - score,
            (scores[offset - 1] - score) if offset else 0.0, top_gap, mean, std,
            spread, support, nearest, float(vector @ candidate_vectors[0]),
            float(vector @ centroid), max(changes, default=0.0),
        ])
    return np.asarray(rows, dtype=np.float64)


def query_features(candidates: list[dict], embeddings: np.ndarray) -> np.ndarray:
    scores = _scores(candidates)
    matrix = candidate_features(candidates, embeddings)
    shifted = scores - scores.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-15)))
    entropy /= max(math.log(len(scores)), 1.0)
    return np.asarray([
        scores[0], scores[0] - scores[1] if len(scores) > 1 else 0.0,
        scores[0] - scores[min(4, len(scores) - 1)], scores.mean(), scores.std(),
        np.ptp(scores), entropy, matrix[0, CANDIDATE_FEATURES.index("temporal_support_10s")],
        matrix[0, CANDIDATE_FEATURES.index("list_centroid_cosine")],
        float(np.mean(scores >= scores.mean() + scores.std())), math.log1p(len(scores)),
    ], dtype=np.float64)

