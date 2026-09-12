"""Interval-aware retrieval metrics, independent of the model and API."""

import math
from itertools import pairwise


def retrieval_metrics(timestamps, intervals, k):
    if k < 1:
        raise ValueError("k must be positive")
    if any(
        not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start
        for start, end in intervals
    ):
        raise ValueError("Relevance intervals must have finite increasing nonnegative bounds")
    ordered = sorted(intervals)
    if any(previous[1] > current[0] for previous, current in pairwise(ordered)):
        raise ValueError("Relevance intervals must not overlap")
    matched = set()
    reciprocal_rank = 0.0
    for rank, timestamp in enumerate(timestamps[:k], 1):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Retrieved timestamps must be finite and nonnegative")
        for index, (start, end) in enumerate(intervals):
            if start <= timestamp < end:
                matched.add(index)
                if not reciprocal_rank:
                    reciprocal_rank = 1 / rank
                break
    return {
        "recall": len(matched) / len(intervals) if intervals else None,
        "precision": len(matched) / k,
        "reciprocal_rank": reciprocal_rank if intervals else None,
        "returned": min(k, len(timestamps)),
    }
