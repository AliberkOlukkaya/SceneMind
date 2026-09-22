"""Evaluation helpers for evidence-preserving fusion V1."""

from __future__ import annotations

import re
import statistics
import time
from collections import defaultdict
from typing import Any

import numpy as np

from ml.experiments.calibrated_fusion_v1.calibration import bucket_relevant
from ml.experiments.evidence_preserving_fusion_v1.ranking import Configuration, rank_trace


def normalize_query(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def lexical_similarity(first: str, second: str) -> float:
    left = set(normalize_query(first).split())
    right = set(normalize_query(second).split())
    return len(left & right) / len(left | right) if left or right else 1.0


def leakage_pairs(
    left: list[dict[str, Any]], right: list[dict[str, Any]], threshold: float = 0.8
) -> dict[str, Any]:
    exact = []
    similar = []
    for first in left:
        for second in right:
            normalized_equal = normalize_query(first["query"]) == normalize_query(second["query"])
            similarity = lexical_similarity(first["query"], second["query"])
            record = {
                "left_query_id": first["query_id"],
                "right_query_id": second["query_id"],
                "similarity": similarity,
            }
            if normalized_equal:
                exact.append(record)
            elif similarity >= threshold:
                similar.append(record)
    return {"exact": exact, "lexical_similarity_at_least_0_8": similar}


def relevant_rank(row: dict[str, Any], configuration: Configuration, k: int = 5) -> int | None:
    for rank, bucket in enumerate(rank_trace(row["trace"], configuration, k), 1):
        if bucket_relevant(bucket, row):
            return rank
    return None


def explicit_evidence_available(row: dict[str, Any], depth: int = 5) -> bool:
    if row["negative"]:
        return False
    required = {
        "VISUAL": ("visual",),
        "SPEECH": ("speech",),
        "HYBRID": ("visual", "speech"),
        "MULTIMODAL": ("visual", "speech"),
    }[row["category"].upper()]
    for lane in required:
        candidates = row["trace"]["raw_candidates"][lane][:depth]
        if not any(
            candidate["timestamp"] <= interval_end
            and (candidate.get("end") or candidate["timestamp"]) >= interval_start
            for candidate in candidates
            for interval_start, interval_end in row["intervals"]
        ):
            return False
    return True


def irrelevant_consensus(row: dict[str, Any], configuration: Configuration) -> bool:
    if not explicit_evidence_available(row):
        return False
    results = rank_trace(row["trace"], configuration, 5)
    if not results or any(bucket_relevant(item, row) for item in results):
        return False
    winner = results[0]
    return winner["exists_in_visual"] and winner["exists_in_speech"]


def evaluate(rows: list[dict[str, Any]], configuration: Configuration) -> dict[str, Any]:
    records = []
    for row in rows:
        rank = relevant_rank(row, configuration)
        records.append(
            {
                "query_id": row["query_id"],
                "source_id": row["source_id"],
                "category": row["category"].upper(),
                "negative": row["negative"],
                "relevant_rank": rank,
                "explicit_top5_available": explicit_evidence_available(row),
                "irrelevant_consensus": irrelevant_consensus(row, configuration),
                "top1_candidate_id": rank_trace(row["trace"], configuration, 5)[0]["candidate_id"],
            }
        )

    positives = [record for record in records if not record["negative"]]

    def summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        return {
            "queries": count,
            "top1": sum(item["relevant_rank"] == 1 for item in group),
            "top3": sum(
                item["relevant_rank"] is not None and item["relevant_rank"] <= 3 for item in group
            ),
            "top5": sum(item["relevant_rank"] is not None for item in group),
            "mrr_at_5": (
                sum(
                    1 / item["relevant_rank"] for item in group if item["relevant_rank"] is not None
                )
                / count
                if count
                else None
            ),
        }

    eligible = [record for record in positives if record["explicit_top5_available"]]
    by_category = {}
    for category in sorted({record["category"] for record in positives}):
        by_category[category] = summary(
            [record for record in positives if record["category"] == category]
        )
    negative_records = [record for record in records if record["negative"]]
    return {
        "all": summary(positives),
        "by_category": by_category,
        "explicit_modality_preservation": {
            **summary(eligible),
            "eligible_queries": len(eligible),
            "displacement_count": sum(item["relevant_rank"] is None for item in eligible),
            "displacement_rate": (
                sum(item["relevant_rank"] is None for item in eligible) / len(eligible)
                if eligible
                else None
            ),
        },
        "irrelevant_consensus_count": sum(item["irrelevant_consensus"] for item in positives),
        "negative_queries": {
            "count": len(negative_records),
            "candidate_presentation": "unchanged possible-match UX; no no-match threshold",
            "top1_ids": {item["query_id"]: item["top1_candidate_id"] for item in negative_records},
        },
        "records": records,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    before = {record["query_id"]: record for record in baseline["records"]}
    changes = []
    for after in candidate["records"]:
        prior = before[after["query_id"]]
        if after["negative"]:
            classification = (
                "NEUTRAL REORDERING"
                if after["top1_candidate_id"] != prior["top1_candidate_id"]
                else "UNCHANGED"
            )
        elif prior["relevant_rank"] is None and after["relevant_rank"] is not None:
            classification = "RECOVERY"
        elif prior["relevant_rank"] is not None and after["relevant_rank"] is None:
            classification = "REGRESSION"
        elif prior["relevant_rank"] != after["relevant_rank"]:
            classification = "NEUTRAL REORDERING"
        else:
            classification = "UNCHANGED"
        changes.append(
            {
                "query_id": after["query_id"],
                "category": after["category"],
                "negative": after["negative"],
                "baseline_rank": prior["relevant_rank"],
                "candidate_rank": after["relevant_rank"],
                "classification": classification,
            }
        )
    positive_changes = [item for item in changes if not item["negative"]]
    successes = [item for item in positive_changes if item["baseline_rank"] is not None]
    retained = [item for item in successes if item["candidate_rank"] is not None]
    recoveries = sum(item["classification"] == "RECOVERY" for item in positive_changes)
    regressions = sum(item["classification"] == "REGRESSION" for item in positive_changes)
    return {
        "recoveries": recoveries,
        "regressions": regressions,
        "net_gain": recoveries - regressions,
        "success_retention": len(retained) / len(successes) if successes else None,
        "successes_before": len(successes),
        "successes_retained": len(retained),
        "query_changes": [item for item in changes if item["classification"] != "UNCHANGED"],
    }


def benchmark_overhead(
    rows: list[dict[str, Any]], configuration: Configuration, repeats: int = 200
) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for row in rows:
            rank_trace(row["trace"], configuration, 5)
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "queries_per_repeat": len(rows),
        "median_ms_all_queries": statistics.median(samples),
        "p95_ms_all_queries": float(np.percentile(samples, 95)),
        "median_ms_per_query": statistics.median(samples) / len(rows),
        "p95_ms_per_query": float(np.percentile(samples, 95)) / len(rows),
    }


def category_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, int | float]]:
    output = defaultdict(dict)
    for category in baseline["by_category"]:
        for metric in ("top1", "top3", "top5", "mrr_at_5"):
            output[category][metric] = (
                candidate["by_category"][category][metric]
                - baseline["by_category"][category][metric]
            )
    return dict(output)
