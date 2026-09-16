"""Validate and summarize the frozen Final English Acceptance V2 evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

EVIDENCE_CATEGORIES = {"SPEECH", "VISUAL", "MULTIMODAL", "NEGATIVE"}
OUTCOMES = {"PASS", "PARTIAL", "FAIL", "ACCEPTABLE", "MISLEADING"}


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("annotation_status") != "frozen-before-search":
        raise ValueError("V2 annotations must be frozen before search")
    if manifest.get("primary_mode") != "hybrid":
        raise ValueError("V2 primary mode must be Smart Search / Hybrid")
    if canonical_manifest_sha256(manifest) != manifest.get("manifest_sha256"):
        raise ValueError("manifest checksum does not match its frozen contents")
    video = manifest.get("video", {})
    if not 1800 <= float(video.get("duration_seconds", 0)) <= 3600:
        raise ValueError("acceptance video must be 30-60 minutes")
    required_truths = (
        "real_continuous_content", "previously_unseen", "source_disjoint",
        "full_human_inspection_completed",
    )
    if not all(video.get(field) is True for field in required_truths):
        raise ValueError("acceptance media must be real, new, disjoint, and fully inspected")
    if video.get("used_for_tuning") is not False:
        raise ValueError("acceptance media cannot be used for tuning")
    queries = manifest.get("queries", [])
    if not 28 <= len(queries) <= 36:
        raise ValueError("V2 requires 28-36 queries")
    ids = [query.get("query_id") for query in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("query IDs must be unique")
    counts: Counter[str] = Counter()
    for query in queries:
        category = query.get("evidence_category")
        if category not in EVIDENCE_CATEGORIES:
            raise ValueError(f"invalid evidence category: {query.get('query_id')}")
        counts[category] += 1
        negative = query.get("negative")
        intervals = query.get("intervals")
        if not isinstance(negative, bool) or not isinstance(intervals, list):
            raise ValueError(f"invalid ground truth: {query.get('query_id')}")
        if negative != (category == "NEGATIVE"):
            raise ValueError(f"negative/category mismatch: {query.get('query_id')}")
        if negative == bool(intervals):
            raise ValueError(f"positive queries need intervals and negatives need none: {query.get('query_id')}")
        if not query.get("text", "").strip() or not query.get("evidence_description", "").strip():
            raise ValueError(f"missing human evidence: {query.get('query_id')}")
        for start, end in intervals:
            if not 0 <= start < end <= video["duration_seconds"]:
                raise ValueError(f"invalid interval: {query.get('query_id')}")
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "queries": len(queries),
        "distribution": dict(counts),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def summarize_report(manifest_path: Path, report_path: Path) -> dict[str, Any]:
    frozen = validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != frozen["manifest_sha256"]:
        raise ValueError("report does not match frozen manifest")
    if report.get("primary_mode") != "hybrid" or report.get("auto_used") is not False:
        raise ValueError("every primary V2 query must use Hybrid without AUTO")
    expected = {query["query_id"]: query for query in manifest["queries"]}
    rows = report.get("queries", [])
    if len(rows) != len(expected) or {row.get("query_id") for row in rows} != set(expected):
        raise ValueError("report must contain every frozen query exactly once")
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for row in rows:
        query = expected[row["query_id"]]
        if row.get("requested_mode") != "hybrid" or row.get("selected_route") != "hybrid":
            raise ValueError(f"non-Hybrid primary result: {row['query_id']}")
        outcome = row.get("outcome")
        if outcome not in OUTCOMES:
            raise ValueError(f"invalid outcome: {row['query_id']}")
        if len(row.get("results", [])) != 5:
            raise ValueError(f"Top-5 results are required: {row['query_id']}")
        if query["negative"]:
            if outcome not in {"ACCEPTABLE", "MISLEADING"}:
                raise ValueError(f"negative UX outcome required: {row['query_id']}")
            negatives.append(row)
            continue
        if outcome not in {"PASS", "PARTIAL", "FAIL"}:
            raise ValueError(f"positive usefulness outcome required: {row['query_id']}")
        flags = [row.get(f"useful_top_{depth}") for depth in (1, 3, 5)]
        if not all(isinstance(value, bool) for value in flags) or flags != sorted(flags):
            raise ValueError(f"Top-k usefulness must be monotonic: {row['query_id']}")
        rank = row.get("best_useful_rank")
        if rank is None:
            if any(flags):
                raise ValueError(f"useful rank is missing: {row['query_id']}")
        elif not isinstance(rank, int) or not 1 <= rank <= 5 or flags != [rank == 1, rank <= 3, True]:
            raise ValueError(f"useful rank disagrees with Top-k: {row['query_id']}")
        if outcome != "PASS" and any(flags):
            raise ValueError(f"PARTIAL/FAIL cannot inflate PASS-level Top-k: {row['query_id']}")
        positives.append({**row, "evidence_category": query["evidence_category"]})
    rates = {
        f"useful_top_{depth}": sum(row[f"useful_top_{depth}"] for row in positives) / len(positives)
        for depth in (1, 3, 5)
    }
    by_category = {}
    for category in ("SPEECH", "VISUAL", "MULTIMODAL"):
        group = [row for row in positives if row["evidence_category"] == category]
        by_category[category] = {
            "queries": len(group),
            **{
                f"useful_top_{depth}": sum(row[f"useful_top_{depth}"] for row in group) / len(group)
                for depth in (1, 3, 5)
            },
            "outcomes": dict(Counter(row["outcome"] for row in group)),
        }
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "queries": len(rows),
        "positive_queries": len(positives),
        "negative_queries": len(negatives),
        **rates,
        "mrr_at_5": sum(0 if row["best_useful_rank"] is None else 1 / row["best_useful_rank"] for row in positives) / len(positives),
        "positive_outcomes": dict(Counter(row["outcome"] for row in positives)),
        "category_breakdown": by_category,
        "negative_ux": dict(Counter(row["outcome"] for row in negatives)),
        "search_latency_ms": {
            "median": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "gates": {
            "top_3": rates["useful_top_3"] >= 0.85,
            "top_5": rates["useful_top_5"] >= 0.90,
            "top_1_preferred": rates["useful_top_1"] >= 0.70,
            "pipeline": not report["pipeline"]["failures"],
            "negative_ux": all(row["outcome"] == "ACCEPTABLE" for row in negatives),
        },
    }
