"""Small deterministic rank-only fusion candidates.

This module is evaluation-only and is never imported by production retrieval.
It consumes the existing production trace, so thumbnail grouping and same-lane
deduplication remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Configuration:
    config_id: str
    strategy: Literal["baseline", "quota", "interleave", "preserve"]
    quota: int | None = None
    start_lane: Literal["visual", "speech"] | None = None
    protected_depth: int | None = None


CONFIGURATIONS = (
    Configuration("rrf60", "baseline"),
    Configuration("quota_1", "quota", quota=1),
    Configuration("quota_2", "quota", quota=2),
    Configuration("interleave_visual", "interleave", start_lane="visual"),
    Configuration("interleave_speech", "interleave", start_lane="speech"),
    Configuration("preserve_depth_1", "preserve", protected_depth=1),
    Configuration("preserve_depth_2", "preserve", protected_depth=2),
    Configuration("preserve_depth_3", "preserve", protected_depth=3),
)


def _baseline(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(trace["candidate_buckets"], key=lambda item: item["post_fusion_rank"])


def _lane_ids(trace: dict[str, Any], lane: str) -> list[str]:
    seen: set[str] = set()
    output = []
    for item in trace["raw_candidates"][lane]:
        candidate_id = item["candidate_id"]
        if candidate_id not in seen:
            seen.add(candidate_id)
            output.append(candidate_id)
    return output


def _bucket_map(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["candidate_id"]: item for item in trace["candidate_buckets"]}


def _append_unique(output: list[str], candidate_id: str, capacity: int) -> None:
    if len(output) < capacity and candidate_id not in output:
        output.append(candidate_id)


def rank_trace(
    trace: dict[str, Any], configuration: Configuration, k: int = 5
) -> list[dict[str, Any]]:
    """Return at most k unique production buckets with stable ordering."""
    if k < 1:
        raise ValueError("k must be positive")
    baseline = _baseline(trace)
    if configuration.strategy == "baseline":
        return baseline[:k]

    buckets = _bucket_map(trace)
    lanes = {lane: _lane_ids(trace, lane) for lane in ("visual", "speech")}
    selected: list[str] = []

    if configuration.strategy == "quota":
        quota = configuration.quota
        if quota is None or quota < 1 or quota * 2 > k:
            raise ValueError("quota must reserve bounded capacity for both lanes")
        mandatory = set(lanes["visual"][:quota] + lanes["speech"][:quota])
        for item in baseline:
            if item["candidate_id"] in mandatory:
                _append_unique(selected, item["candidate_id"], k)
        for item in baseline:
            _append_unique(selected, item["candidate_id"], k)

    elif configuration.strategy == "interleave":
        first = configuration.start_lane
        if first not in {"visual", "speech"}:
            raise ValueError("interleave requires a starting lane")
        order = (first, "speech" if first == "visual" else "visual")
        depth = 0
        while len(selected) < k and any(depth < len(lanes[lane]) for lane in order):
            for lane in order:
                if depth < len(lanes[lane]):
                    _append_unique(selected, lanes[lane][depth], k)
            depth += 1

    elif configuration.strategy == "preserve":
        depth = configuration.protected_depth
        if depth is None or depth < 1:
            raise ValueError("preserve requires a positive rank depth")
        protected = set(lanes["visual"][:depth] + lanes["speech"][:depth])
        protected_items = [item for item in baseline if item["candidate_id"] in protected]
        protected_items.sort(
            key=lambda item: (
                min(
                    item["contributions"][lane]["retriever_rank"] for lane in item["contributions"]
                ),
                item["post_fusion_rank"],
                item["timestamp"],
                item["candidate_id"],
            )
        )
        for item in protected_items:
            _append_unique(selected, item["candidate_id"], k)
        for item in baseline:
            _append_unique(selected, item["candidate_id"], k)
    else:
        raise ValueError(f"unsupported strategy: {configuration.strategy}")

    return [buckets[candidate_id] for candidate_id in selected]


def configuration_dict(configuration: Configuration) -> dict[str, Any]:
    return {
        "config_id": configuration.config_id,
        "strategy": configuration.strategy,
        "quota": configuration.quota,
        "start_lane": configuration.start_lane,
        "protected_depth": configuration.protected_depth,
        "temporal_grouping": "production exact-thumbnail grouping",
        "tie_rule": "declared lane order, then production RRF rank, timestamp, candidate_id",
    }
