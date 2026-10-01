"""Deterministic reporting/selection; never changes labels or retrieval."""

import math


def validate_pair(baseline, candidate):
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        raise ValueError("Incomplete run cannot support comparison")
    if baseline["manifest_sha256"] != candidate["manifest_sha256"]:
        raise ValueError("Different frozen manifests")
    if baseline["arm"] != "baseline" or candidate["arm"] != "candidate":
        raise ValueError("Wrong comparison arms")
    fields = ("id", "video_id", "query", "category", "difficulty", "intervals")
    truth = lambda run: {q["id"]: {k: q[k] for k in fields} for q in run["queries"]}  # noqa: E731
    if (len(truth(baseline)) != len(baseline["queries"])
            or len(truth(candidate)) != len(candidate["queries"])
            or truth(baseline) != truth(candidate)):
        raise ValueError("Query identity or ground truth changed")
    if not baseline["queries"]:
        raise ValueError("Empty comparison")


def choose_development(baseline, candidates):
    """Predeclared R5, temporal R5, then lower measured indexing cost tie-break."""
    expected = {(4, 2, 8), (8, 4, 8)}
    configs = [tuple(c["temporal_config"][k] for k in ("length", "stride", "frames"))
               for c in candidates]
    if len(configs) != 2 or set(configs) != expected:
        raise ValueError("Exactly the two predeclared configurations required")
    for candidate in candidates:
        validate_pair(baseline, candidate)
    return max(candidates, key=lambda c: (
        c["metrics"]["overall"]["r5"], c["metrics"]["combined_temporal"]["r5"],
        -sum(v["indexing_seconds"] for v in c["videos"].values()),
    ))["temporal_config"]


def quality_gates(baseline, candidate):
    validate_pair(baseline, candidate)
    b, c = baseline["metrics"], candidate["metrics"]
    catastrophic = [key.removeprefix("video_id:") for key, value in c.items()
                    if key.startswith("video_id:") and value["n"] >= 8 and value["r5"] < 0.6]
    epsilon = 1e-12
    return {
        "overall_r5_at_least_80": c["overall"]["r5"] >= 0.8 - epsilon,
        "overall_gain_at_least_10pp": c["overall"]["r5"] - b["overall"]["r5"] >= 0.1 - epsilon,
        "temporal_r5_at_least_75": c["combined_temporal"]["r5"] >= 0.75 - epsilon,
        "temporal_gain_at_least_10pp": (
            c["combined_temporal"]["r5"] - b["combined_temporal"]["r5"] >= 0.1 - epsilon),
        "static_regression_at_most_5pp": (
            c["combined_static"]["r5"] - b["combined_static"]["r5"] >= -0.05 - epsilon),
        "no_catastrophic_video": not catastrophic,
        "catastrophic_videos": catastrophic,
    }


def performance(run):
    """Measured totals, not deployment guarantees or estimated throughput."""
    videos = run["videos"]
    minutes = sum(v["duration_seconds"] for v in videos.values()) / 60
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("Invalid source duration")
    embedding = sum(v["embedding_seconds"] for v in videos.values())
    frames = sum(v.get("processed_frames", v.get("frames", 0)) for v in videos.values())
    return {
        "source_minutes": minutes,
        "preprocessing_seconds_per_minute": sum(v["preprocess_seconds"] for v in videos.values()) / minutes,
        "indexing_seconds_per_minute": sum(v["indexing_seconds"] for v in videos.values()) / minutes,
        "embedding_frames_per_second": frames / embedding if embedding > 0 else None,
        "processed_frames": frames,
        "frames_per_minute": frames / minutes,
        "clips_per_minute": sum(v.get("temporal_clips", 0) for v in videos.values()) / minutes,
        "index_bytes_per_minute": sum(v["index_bytes"] for v in videos.values()) / minutes,
        "model_load_seconds": run["model_load_seconds"],
        "process_tree_peak_rss_bytes": run["process_tree_peak_rss_bytes"],
        "query_latency_seconds": run["query_latency_seconds"],
    }
