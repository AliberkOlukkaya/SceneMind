"""Validate and summarize the private final English long-video acceptance run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROUTES = {"VISUAL", "SPEECH", "HYBRID"}
CATEGORIES = {
    "spoken_concept", "explained_topic", "mentioned_technical_term",
    "visual_object", "visual_interface_or_scene", "hybrid_spoken_visual",
    "compositional", "plausible_negative",
}
USEFULNESS = {"PASS", "PARTIAL", "FAIL"}
FAILURES = {
    "routing", "ASR", "BM25/speech retrieval", "CLIP visual retrieval",
    "hybrid fusion", "temporal sampling", "timestamp precision", "small object",
    "compositional reasoning", "action/temporal reasoning", "OCR required",
    "unsupported query", "UI/product friction",
}
CATEGORY_GROUPS = {
    "spoken_concept": "Speech", "explained_topic": "Speech",
    "mentioned_technical_term": "Speech", "visual_object": "Visual",
    "visual_interface_or_scene": "Visual", "hybrid_spoken_visual": "Hybrid",
    "compositional": "Compositional", "plausible_negative": "Negative",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _route(value: str) -> str:
    route = value.upper()
    if route not in ROUTES:
        raise ValueError("route must be VISUAL, SPEECH, or HYBRID")
    return route


def validate(path: Path, require_frozen: bool = True) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if require_frozen and manifest.get("annotation_status") != "frozen":
        raise ValueError("freeze human-reviewed annotations before search")
    if require_frozen and not manifest.get("annotation_frozen_at"):
        raise ValueError("record annotation_frozen_at before search")
    videos = manifest.get("videos", [])
    if not videos:
        if not require_frozen and manifest.get("annotation_status") == "draft":
            return {"videos": 0, "queries": 0, "manifest_sha256": sha256(path)}
        raise ValueError("add a real 30-60 minute English-speaking video")
    query_ids: set[str] = set()
    categories: set[str] = set()
    routes: set[str] = set()
    query_count = 0
    for video in videos:
        duration = video.get("duration_seconds")
        if not isinstance(duration, (int, float)) or not 1800 <= duration <= 3600:
            raise ValueError("video duration must be between 30 and 60 minutes")
        if video.get("primary_language") != "English":
            raise ValueError("final acceptance video must contain English speech")
        if video.get("media_kind") != "real_continuous":
            raise ValueError("synthetic or concatenated media cannot establish search quality")
        if video.get("previously_used_for_tuning") is not False:
            raise ValueError("acceptance media must not have been used for tuning")
        if not video.get("usage_rights", "").strip():
            raise ValueError("record legal/local test usage rights")
        if not video.get("human_inspection_completed"):
            raise ValueError("human-inspect the complete video before freezing queries")
        media = Path(video["local_path"])
        if not media.is_file() or sha256(media) != video["sha256"]:
            raise ValueError("acceptance media is missing or changed")
        for query in video.get("queries", []):
            query_count += 1
            query_id = query["query_id"]
            if query_id in query_ids:
                raise ValueError("query IDs must be unique")
            query_ids.add(query_id)
            if not query.get("query", "").strip() or not query.get("human_rationale", "").strip():
                raise ValueError(f"query and human rationale are required: {query_id}")
            category = query.get("category")
            if category not in CATEGORIES:
                raise ValueError(f"invalid category: {query_id}")
            categories.add(category)
            routes.add(_route(query["expected_route"]))
            negative = query.get("negative")
            intervals = query.get("relevant_intervals", [])
            if not isinstance(negative, bool):
                raise ValueError(f"negative must be boolean: {query_id}")
            if category == "plausible_negative" and not negative:
                raise ValueError(f"plausible_negative must be absent: {query_id}")
            if negative == bool(intervals):
                raise ValueError(f"positive needs intervals; negative needs none: {query_id}")
            for interval in intervals:
                if (not isinstance(interval, list) or len(interval) != 2
                        or not all(isinstance(value, (int, float)) for value in interval)
                        or interval[0] < 0 or interval[0] >= interval[1]
                        or interval[1] > duration):
                    raise ValueError(f"invalid interval: {query_id}")
    if not 25 <= query_count <= 40:
        raise ValueError("freeze between 25 and 40 English queries")
    missing = CATEGORIES - categories
    if missing:
        raise ValueError(f"missing query categories: {', '.join(sorted(missing))}")
    if routes != ROUTES:
        raise ValueError("manifest must exercise VISUAL, SPEECH, and HYBRID")
    return {"videos": len(videos), "queries": query_count,
            "manifest_sha256": sha256(path)}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(manifest_path: Path, observations_path: Path) -> dict[str, Any]:
    validated = validate(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if observations.get("manifest_sha256") != validated["manifest_sha256"]:
        raise ValueError("observations do not match the frozen manifest")
    if observations.get("run_status") != "complete":
        raise ValueError("acceptance observations must be complete")
    expected = {query["query_id"]: query for video in manifest["videos"]
                for query in video["queries"]}
    supplied = {row["query_id"] for row in observations.get("queries", [])}
    if supplied != set(expected) or len(supplied) != len(observations.get("queries", [])):
        raise ValueError("record every frozen query exactly once")
    rows = []
    failures: Counter[str] = Counter()
    for row in observations["queries"]:
        query = expected[row["query_id"]]
        route = _route(row["auto_selected_route"])
        top = [row[f"useful_top_{depth}"] for depth in (1, 3, 5)]
        if not all(isinstance(value, bool) for value in top) or top != sorted(top):
            raise ValueError(f"Top-k judgments must be monotonic: {row['query_id']}")
        usefulness = row["user_usefulness"].upper()
        if usefulness not in USEFULNESS:
            raise ValueError(f"invalid usefulness: {row['query_id']}")
        if not row.get("timestamp_quality", "").strip():
            raise ValueError(f"timestamp quality is required: {row['query_id']}")
        if not row.get("failure_reason", "").strip():
            raise ValueError(f"human judgment rationale is required: {row['query_id']}")
        causes = row.get("failure_categories", [])
        if any(cause not in FAILURES for cause in causes):
            raise ValueError(f"invalid failure category: {row['query_id']}")
        failures.update(causes)
        if query["negative"] and not isinstance(row.get("conservative_ux_understandable"), bool):
            raise ValueError(f"negative UX judgment required: {row['query_id']}")
        expected_route = _route(query["expected_route"])
        rows.append({**row, "expected_route": expected_route,
                     "route_correct": route == expected_route,
                     "negative": query["negative"], "category": query["category"],
                     "category_group": CATEGORY_GROUPS[query["category"]]})
    positives = [row for row in rows if not row["negative"]]
    negatives = [row for row in rows if row["negative"]]
    resources = observations.get("pipeline", {})
    required_resources = {
        "upload_seconds", "total_processing_seconds", "peak_worker_ram_bytes",
        "disk_bytes", "frame_count", "asr_segment_count", "failures", "retries",
    }
    if not required_resources <= resources.keys():
        raise ValueError("pipeline observations are incomplete")
    numeric_resources = required_resources - {"failures"}
    if any(not isinstance(resources[key], (int, float)) or resources[key] < 0
           for key in numeric_resources):
        raise ValueError("pipeline measurements must be non-negative numbers")
    if not isinstance(resources["failures"], list):
        raise ValueError("pipeline failures must be a list")
    rates = {f"useful_top_{depth}": sum(row[f"useful_top_{depth}"] for row in positives)
             / len(positives) for depth in (1, 3, 5)}
    reciprocal_ranks = []
    for row in positives:
        rank = row.get("best_useful_result_rank")
        if rank is not None and (not isinstance(rank, int) or not 1 <= rank <= 5):
            raise ValueError(f"best useful rank must be 1-5 or null: {row['query_id']}")
        if (rank is None) != (not row["useful_top_5"]):
            raise ValueError(f"best useful rank must agree with Top-5: {row['query_id']}")
        if rank is not None and (row["useful_top_1"] != (rank == 1)
                                 or row["useful_top_3"] != (rank <= 3)):
            raise ValueError(f"best useful rank must agree with Top-k: {row['query_id']}")
        reciprocal_ranks.append(0 if rank is None else 1 / rank)
    category_breakdown = {}
    for name in ("Speech", "Visual", "Hybrid", "Compositional"):
        group = [row for row in positives if row["category_group"] == name]
        if group:
            category_breakdown[name] = {
                "queries": len(group),
                **{f"useful_top_{depth}": sum(row[f"useful_top_{depth}"] for row in group)
                   / len(group) for depth in (1, 3, 5)},
                "outcomes": dict(Counter(row["user_usefulness"].upper() for row in group)),
            }
    routing = sum(row["route_correct"] for row in rows) / len(rows)
    latencies = [float(row["search_latency_ms"]) for row in rows]
    gate = {
        "routing": routing >= 0.90, "top_3": rates["useful_top_3"] >= 0.85,
        "top_5": rates["useful_top_5"] >= 0.90,
        "ingest": not resources["failures"],
        "conservative_ux": all(row["conservative_ux_understandable"] for row in negatives),
    }
    return {
        "manifest_sha256": validated["manifest_sha256"], "queries": len(rows),
        "positive_queries": len(positives), "negative_queries": len(negatives),
        "auto_routing_accuracy": routing, **rates,
        "mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "positive_outcomes": dict(Counter(row["user_usefulness"].upper()
                                           for row in positives)),
        "category_breakdown": category_breakdown,
        "useful_top_1_preferred_gate": rates["useful_top_1"] >= 0.70,
        "negative_ux_understandable_rate": (
            sum(row["conservative_ux_understandable"] for row in negatives) / len(negatives)
            if negatives else None
        ),
        "negative_misleading_rate": (
            sum(not row["conservative_ux_understandable"] for row in negatives) / len(negatives)
            if negatives else None
        ),
        "search_latency_ms": {"median": _percentile(latencies, 0.5),
                              "p95": _percentile(latencies, 0.95), "max": max(latencies)},
        "pipeline": resources, "failure_categories": dict(failures),
        "gate": {**gate, "passed": all(gate.values())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--allow-draft", action="store_true")
    arguments = parser.parse_args()
    result = (summarize(arguments.manifest, arguments.observations)
              if arguments.observations else validate(arguments.manifest,
                                                       not arguments.allow_draft))
    print(json.dumps(result, indent=2))
