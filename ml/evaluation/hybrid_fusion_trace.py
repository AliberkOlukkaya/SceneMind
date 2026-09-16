"""Evaluation-only observability for the unchanged production Hybrid fusion function."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.hybrid import fuse

RRF_CONSTANT = 60


def _raw_candidate(result: dict[str, Any], modality: str, rank: int) -> dict[str, Any]:
    return {
        "candidate_id": result["thumbnail"],
        "moment_id": None,
        "moment_id_availability": "not_applicable_thumbnail_is_the_bucket_key",
        "modality": modality,
        "retriever": "clip_faiss" if modality == "visual" else "bm25_whisper_segments",
        "retriever_rank": rank,
        "raw_score": result["score"],
        "normalized_score": None,
        "normalized_score_availability": "not_applicable_no_score_normalization",
        "timestamp": result["timestamp"],
        "end": result.get("end"),
        "text": result.get("text"),
    }


def trace_fusion(
    visual: list[dict[str, Any]],
    speech: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    """Describe production fusion without participating in ranking.

    The production result is computed only by :func:`app.hybrid.fuse`. The trace
    independently observes its fixed rules and asserts that its reconstructed
    order agrees, so diagnostics cannot become an alternate ranking path.
    """
    production_results = fuse(visual, speech, k)
    raw = {
        "visual": [_raw_candidate(item, "visual", rank) for rank, item in enumerate(visual, 1)],
        "speech": [_raw_candidate(item, "speech", rank) for rank, item in enumerate(speech, 1)],
    }
    selected: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_ranks: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for modality in ("visual", "speech"):
        seen: set[str] = set()
        for candidate in raw[modality]:
            candidate_id = candidate["candidate_id"]
            if candidate_id in seen:
                duplicate_ranks[candidate_id][modality].append(candidate["retriever_rank"])
                continue
            seen.add(candidate_id)
            selected[candidate_id][modality] = candidate

    buckets = []
    for candidate_id, contributions in selected.items():
        visual_item = contributions.get("visual")
        speech_item = contributions.get("speech")
        chosen = speech_item or visual_item
        contribution_rows = {}
        for modality, item in contributions.items():
            contribution_rows[modality] = {
                "retriever": item["retriever"],
                "retriever_rank": item["retriever_rank"],
                "raw_score": item["raw_score"],
                "timestamp": item["timestamp"],
                "end": item["end"],
                "normalized_score": None,
                "normalized_score_availability": "not_applicable_no_score_normalization",
                "rrf_contribution": 1 / (RRF_CONSTANT + item["retriever_rank"]),
                "other_fusion_contribution": None,
                "other_fusion_contribution_availability": "not_applicable_rrf_only",
            }
        buckets.append({
            "candidate_id": candidate_id,
            "moment_id": None,
            "moment_id_availability": "not_applicable_thumbnail_is_the_bucket_key",
            "timestamp": chosen["timestamp"],
            "end": chosen.get("end"),
            "originating_modalities": list(contributions),
            "exists_in_visual": visual_item is not None,
            "exists_in_speech": speech_item is not None,
            "exists_in_lexical_bm25": speech_item is not None,
            "candidate_overlap": len(contributions) == 2,
            "pre_fusion_rank": None,
            "pre_fusion_rank_availability": "not_applicable_no_cross_retriever_rank",
            "contributions": contribution_rows,
            "total_fusion_score": sum(row["rrf_contribution"] for row in contribution_rows.values()),
            "deduplication": {
                "grouping_key": "thumbnail",
                "selected_visual_rank": visual_item["retriever_rank"] if visual_item else None,
                "selected_speech_rank": speech_item["retriever_rank"] if speech_item else None,
                "discarded_duplicate_visual_ranks": duplicate_ranks[candidate_id].get("visual", []),
                "discarded_duplicate_speech_ranks": duplicate_ranks[candidate_id].get("speech", []),
            },
        })

    buckets.sort(key=lambda item: (-item["total_fusion_score"], item["timestamp"]))
    for post_rank, bucket in enumerate(buckets, 1):
        bucket["post_fusion_rank"] = post_rank
        bucket["final_top_k"] = post_rank <= k
        bucket["rank_displacement"] = {
            modality: post_rank - contribution["retriever_rank"]
            for modality, contribution in bucket["contributions"].items()
        }

    reconstructed = [
        {
            "candidate_id": bucket["candidate_id"],
            "score": bucket["total_fusion_score"],
            "timestamp": bucket["timestamp"],
        }
        for bucket in buckets[:k]
    ]
    observed = [
        {
            "candidate_id": item["thumbnail"],
            "score": item["score"],
            "timestamp": item["timestamp"],
        }
        for item in production_results
    ]
    if reconstructed != observed:
        raise AssertionError("diagnostic reconstruction diverged from production fuse output")

    return {
        "ranking_source": "app.hybrid.fuse",
        "ranking_modified": False,
        "rrf_constant": RRF_CONSTANT,
        "candidate_limits": {"visual": len(visual), "speech": len(speech), "final": k},
        "normalization": {"exists": False, "value": None},
        "temporal_grouping": "nearest production frame thumbnail",
        "raw_candidates": raw,
        "candidate_buckets": buckets,
        "results": production_results,
    }
