"""Offline-only Hybrid fusion variants over frozen development candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RRF_CONSTANT = 60


@dataclass(frozen=True)
class Variant:
    variant_id: str
    family: Literal["baseline", "capped_overlap", "normalized_rank"]
    overlap_cap: float | None = None
    normalization: Literal["normalized_rrf", "percentile"] | None = None
    normalization_constant: int | None = None


VARIANTS = (
    Variant("baseline_rrf60", "baseline"),
    Variant("cap_1_75", "capped_overlap", overlap_cap=1.75),
    Variant("cap_1_50", "capped_overlap", overlap_cap=1.50),
    Variant("cap_1_25", "capped_overlap", overlap_cap=1.25),
    Variant(
        "normalized_rrf60",
        "normalized_rank",
        normalization="normalized_rrf",
        normalization_constant=60,
    ),
    Variant(
        "normalized_rrf10",
        "normalized_rank",
        normalization="normalized_rrf",
        normalization_constant=10,
    ),
    Variant("normalized_percentile", "normalized_rank", normalization="percentile"),
)


def rank_contribution(rank: int, candidate_count: int, variant: Variant) -> float:
    """Return the pre-overlap contribution for one retained retriever candidate."""
    if rank < 1 or candidate_count < rank:
        raise ValueError("rank must be within the retriever candidate list")
    if variant.family in {"baseline", "capped_overlap"}:
        return 1 / (RRF_CONSTANT + rank)
    if variant.normalization == "percentile":
        return (candidate_count - rank + 1) / candidate_count
    if variant.normalization == "normalized_rrf":
        constant = variant.normalization_constant
        if constant is None:
            raise ValueError("normalized RRF requires a constant")
        raw = 1 / (constant + rank)
        lower_anchor = 1 / (constant + candidate_count + 1)
        upper_anchor = 1 / (constant + 1)
        return (raw - lower_anchor) / (upper_anchor - lower_anchor)
    raise ValueError(f"unsupported variant: {variant.variant_id}")


def overlap_score(contributions: list[float], variant: Variant) -> float:
    """Aggregate contributions, applying only Experiment A's declared cap."""
    total = sum(contributions)
    if variant.family != "capped_overlap" or len(contributions) < 2:
        return total
    if variant.overlap_cap is None:
        raise ValueError("capped-overlap variant requires a cap")
    return min(total, variant.overlap_cap * max(contributions))


def rank_candidates(
    visual: list[dict[str, Any]],
    speech: list[dict[str, Any]],
    k: int,
    variant: Variant,
) -> dict[str, Any]:
    """Rank frozen candidates without importing this module into production."""
    merged: dict[str, dict[str, Any]] = {}
    counts = {"visual": len(visual), "speech": len(speech)}
    for modality, results in (("visual", visual), ("speech", speech)):
        seen: set[str] = set()
        for rank, result in enumerate(results, 1):
            key = result["thumbnail"]
            if key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(
                key,
                {
                    **result,
                    "score": 0.0,
                    "evidence": {},
                    "experimental_contributions": {},
                },
            )
            contribution = rank_contribution(rank, counts[modality], variant)
            item["evidence"][modality] = {"rank": rank, "score": result["score"]}
            item["experimental_contributions"][modality] = contribution
            if modality == "speech":
                item.update(text=result["text"], timestamp=result["timestamp"], end=result["end"])
            item["modality"] = "+".join(item["evidence"])

    for item in merged.values():
        contributions = list(item["experimental_contributions"].values())
        item["experimental_uncapped_score"] = sum(contributions)
        item["score"] = overlap_score(contributions, variant)
        item["experimental_cap_reduction"] = item["experimental_uncapped_score"] - item["score"]
    ordered = sorted(merged.values(), key=lambda item: (-item["score"], item["timestamp"]))
    for rank, item in enumerate(ordered, 1):
        item["experimental_rank"] = rank
        item["final_top_k"] = rank <= k
    return {"variant": variant, "ordered": ordered, "results": ordered[:k]}


def production_shape(item: dict[str, Any]) -> dict[str, Any]:
    """Remove evaluation metadata for exact baseline comparison with production."""
    return {
        key: value
        for key, value in item.items()
        if key
        not in {
            "experimental_contributions",
            "experimental_uncapped_score",
            "experimental_cap_reduction",
            "experimental_rank",
            "final_top_k",
        }
    }
