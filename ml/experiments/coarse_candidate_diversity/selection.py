"""Deterministic candidate selectors with no learned parameters."""

from dataclasses import dataclass

import numpy as np

TIMESTAMP_TOLERANCE_SECONDS = 0.05


@dataclass(frozen=True)
class Candidate:
    index: int
    timestamp: float
    score: float


def raw_pool(scores: np.ndarray, timestamps: list[float], limit: int) -> list[Candidate]:
    if limit < 1 or len(scores) != len(timestamps):
        raise ValueError("raw pool requires aligned inputs and a positive limit")
    order = np.argsort(-np.asarray(scores), kind="stable")[:limit]
    return [Candidate(int(i), float(timestamps[i]), float(scores[i])) for i in order]


def temporal_nms(
    candidates: list[Candidate], limit: int, radius: float
) -> list[Candidate]:
    if limit < 1 or radius < 0:
        raise ValueError("temporal NMS requires a positive limit and nonnegative radius")
    selected = []
    seen = set()
    for candidate in candidates:
        identity = (candidate.index, candidate.timestamp)
        if identity in seen:
            continue
        seen.add(identity)
        if all(
            abs(candidate.timestamp - row.timestamp) > radius + TIMESTAMP_TOLERANCE_SECONDS
            for row in selected
        ):
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def mmr(
    candidates: list[Candidate], embeddings: np.ndarray, limit: int, relevance_weight: float
) -> list[Candidate]:
    if limit < 1 or not 0 <= relevance_weight <= 1:
        raise ValueError("MMR parameters are outside their valid range")
    if embeddings.ndim != 2:
        raise ValueError("MMR embeddings must be a matrix")
    remaining = list(dict.fromkeys(candidates))
    selected = []
    while remaining and len(selected) < limit:
        if not selected:
            winner = remaining[0]
        else:
            winner = max(
                remaining,
                key=lambda row: (
                    relevance_weight * row.score
                    - (1 - relevance_weight)
                    * max(float(embeddings[row.index] @ embeddings[item.index]) for item in selected),
                    row.score,
                    -row.timestamp,
                ),
            )
        selected.append(winner)
        remaining.remove(winner)
    return selected


def scene_segments(embeddings: np.ndarray, change_threshold: float) -> list[int]:
    if embeddings.ndim != 2 or change_threshold < 0:
        raise ValueError("scene grouping requires a matrix and nonnegative threshold")
    if not len(embeddings):
        return []
    segments = [0]
    segment = 0
    for previous, current in zip(embeddings, embeddings[1:]):
        if 1 - float(previous @ current) >= change_threshold:
            segment += 1
        segments.append(segment)
    return segments


def scene_aware(
    candidates: list[Candidate], segments: list[int], limit: int
) -> list[Candidate]:
    if limit < 1:
        raise ValueError("scene-aware selection requires a positive limit")
    selected = []
    used = set()
    for candidate in candidates:
        segment = segments[candidate.index]
        if segment in used:
            continue
        used.add(segment)
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def represented_intervals(
    candidates: list[Candidate], duration: float, radius: float
) -> list[dict]:
    return [
        {
            "timestamp": row.timestamp,
            "start": max(0.0, row.timestamp - radius),
            "end": min(duration, row.timestamp + radius),
            "score": row.score,
        }
        for row in candidates
    ]
