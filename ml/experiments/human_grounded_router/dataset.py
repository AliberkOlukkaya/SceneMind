"""Integrity checks for the frozen human-grounded router annotations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

ROUTES = ("VISUAL", "SPEECH", "HYBRID")
SPLITS = ("train", "validation", "frozen_test")
PROTECTED_TITLE = "evaluating and developing machine learning models: an introduction"


def manifest_hash(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_manifest(manifest: dict, sources: dict) -> dict:
    rows = manifest.get("queries", [])
    required = {"query_id", "source_id", "video_title", "split", "query", "route",
                "evidence_interval", "visual_evidence_rationale",
                "speech_evidence_rationale", "human_grounded", "ambiguous",
                "annotation_notes"}
    if manifest.get("annotation_status") != "frozen-before-model-evaluation":
        raise ValueError("manifest must be frozen before model evaluation")
    if len(rows) < 240 or len({row.get("query_id") for row in rows}) != len(rows):
        raise ValueError("at least 240 uniquely identified annotations are required")
    if len({row.get("query", "").strip().casefold() for row in rows}) != len(rows):
        raise ValueError("queries must be unique")
    source_splits: dict[str, set[str]] = {}
    for row in rows:
        if required - row.keys() or row["route"] not in ROUTES or row["split"] not in SPLITS:
            raise ValueError(f"invalid annotation schema: {row.get('query_id')}")
        if not row["human_grounded"] or not row["query"].strip():
            raise ValueError(f"annotation is not human-grounded: {row['query_id']}")
        interval = row["evidence_interval"]
        if interval["start_seconds"] < 0 or interval["end_seconds"] <= interval["start_seconds"]:
            raise ValueError(f"invalid evidence interval: {row['query_id']}")
        source = sources.get(row["source_id"])
        if source is None or source["split"] != row["split"] or source["title"] != row["video_title"]:
            raise ValueError(f"source metadata mismatch: {row['query_id']}")
        if interval["end_seconds"] > source["duration_seconds"] + 0.5:
            raise ValueError(f"evidence exceeds video: {row['query_id']}")
        if PROTECTED_TITLE in (row["video_title"] + " " + row["query"]).casefold():
            raise ValueError("protected acceptance evidence leaked into manifest")
        source_splits.setdefault(row["source_id"], set()).add(row["split"])
    if any(len(values) != 1 for values in source_splits.values()):
        raise ValueError("video source crosses split boundaries")
    test_sources = {row["source_id"] for row in rows if row["split"] == "frozen_test"}
    if len(test_sources) < 2:
        raise ValueError("frozen test requires at least two unseen videos")
    counts = Counter(row["route"] for row in rows)
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("route classes must be balanced")
    return {"queries": len(rows), "route_counts": dict(counts),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "source_count": len(source_splits), "frozen_test_video_count": len(test_sources),
            "ambiguous_annotations": sum(bool(row["ambiguous"]) for row in rows),
            "source_disjoint_verified": True, "protected_acceptance_leakage_detected": False}
