"""Deterministic local windows, temporal diversification and cache identities."""

import hashlib
from dataclasses import dataclass

POLICY_VERSION = "bounded-secondary-v1"


@dataclass(frozen=True)
class CoarseCandidate:
    timestamp: float
    score: float


def diversify(
    candidates: list[CoarseCandidate], limit: int, minimum_spacing: float
) -> list[CoarseCandidate]:
    if limit < 1 or minimum_spacing < 0:
        raise ValueError("invalid diversification parameters")
    selected = []
    for candidate in candidates:
        if all(abs(candidate.timestamp - row.timestamp) >= minimum_spacing for row in selected):
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def local_windows(
    candidates: list[CoarseCandidate], radius: float, duration: float
) -> list[tuple[float, float]]:
    if radius <= 0 or duration <= 0:
        raise ValueError("radius and duration must be positive")
    windows = sorted(
        (max(0.0, row.timestamp - radius), min(duration, row.timestamp + radius))
        for row in candidates
    )
    merged: list[list[float]] = []
    for start, end in windows:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def secondary_timestamps(
    candidates: list[CoarseCandidate], radius: float, duration: float,
    interval: float = 2.0,
) -> list[float]:
    if interval <= 0:
        raise ValueError("interval must be positive")
    windows = local_windows(candidates, radius, duration)
    count = int(duration // interval) + 1
    grid = [round(index * interval, 3) for index in range(count)]
    return [
        timestamp for timestamp in grid
        if any(start <= timestamp <= end for start, end in windows)
    ]


def cache_key(video_id: str, timestamp: float, policy_version: str = POLICY_VERSION) -> str:
    if not video_id or timestamp < 0:
        raise ValueError("cache identity requires a video ID and nonnegative timestamp")
    identity = f"{policy_version}\0{video_id}\0{timestamp:.3f}".encode()
    return hashlib.sha256(identity).hexdigest()
