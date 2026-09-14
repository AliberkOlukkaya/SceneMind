"""Validate and summarize a private personal-video acceptance run.

Populated manifests and observations belong under ignored ``data/``. This
module deliberately does not run retrieval: a person must judge whether each
returned moment is useful after using the production UI/API.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROUTES = {"VISUAL", "SPEECH", "HYBRID"}
LANGUAGES = {"EN", "TR"}
SCENARIOS = {"lecture_tutorial", "project_software_demo", "ordinary_real_world"}
CATEGORIES = {
    "spoken_topic", "quoted_mentioned_phrase", "visual_object", "visual_scene",
    "compositional_visual", "mixed_visual_spoken", "difficult_negative_unsupported",
}
USEFULNESS = {"PASS", "PARTIAL", "FAIL"}
FAILURES = {
    "AUTO routing failure", "ASR transcription failure", "speech lexical retrieval failure",
    "Turkish language mismatch", "CLIP semantic retrieval failure",
    "temporal sampling failure", "timestamp granularity issue", "hybrid fusion failure",
    "small-object failure", "temporal/action reasoning required", "OCR required",
    "no-match / false-confidence problem", "UI/product friction", "unsupported query",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _route(value: str) -> str:
    normalized = value.upper()
    if normalized not in ROUTES:
        raise ValueError("route must be VISUAL, SPEECH, or HYBRID")
    return normalized


def _expected_route(query: dict[str, Any]) -> str:
    value = query.get("expected_best_route", query.get("target_route"))
    if value is None:
        raise ValueError("query requires expected_best_route")
    return _route(value)


def validate(path: Path, require_frozen: bool = True) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if require_frozen and manifest.get("annotation_status") != "frozen":
        raise ValueError("freeze manual annotations before retrieval")
    videos = manifest.get("videos", [])
    if not videos:
        raise ValueError("add at least one real personal video")

    query_ids: set[str] = set()
    scenarios: set[str] = set()
    languages: set[str] = set()
    categories: set[str] = set()
    for video in videos:
        video_id = video["video_id"]
        scenario = video.get("scenario")
        if scenario not in SCENARIOS:
            raise ValueError(f"invalid scenario: {video_id}")
        scenarios.add(scenario)
        duration = video.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"invalid duration: {video_id}")
        target_met = (
            1800 <= duration <= 3600 if scenario == "lecture_tutorial"
            else 900 <= duration <= 3600 if scenario == "project_software_demo"
            else True
        )
        if video.get("duration_requirement_met") is not target_met:
            raise ValueError(f"incorrect duration_requirement_met: {video_id}")
        media = Path(video["local_path"])
        if not media.is_file() or sha256(media) != video["sha256"]:
            raise ValueError(f"missing or changed media: {video_id}")
        if not video.get("queries"):
            raise ValueError(f"video requires queries: {video_id}")
        video_languages: set[str] = set()
        video_categories: set[str] = set()
        for query in video["queries"]:
            query_id = query["query_id"]
            if query_id in query_ids:
                raise ValueError("query IDs must be unique")
            query_ids.add(query_id)
            if not query.get("text", "").strip():
                raise ValueError(f"query text is empty: {query_id}")
            language = query.get("language", "").upper()
            if language not in LANGUAGES:
                raise ValueError(f"invalid query language: {query_id}")
            languages.add(language)
            video_languages.add(language)
            category = query.get("category")
            if category not in CATEGORIES:
                raise ValueError(f"invalid query category: {query_id}")
            categories.add(category)
            video_categories.add(category)
            _expected_route(query)
            expected_presence = query.get("expected_presence")
            intervals = query.get("relevant_intervals", [])
            if not isinstance(expected_presence, bool):
                raise ValueError(f"expected_presence must be boolean: {query_id}")
            if expected_presence and not intervals:
                raise ValueError(f"positive query requires reviewed intervals: {query_id}")
            if not expected_presence and intervals:
                raise ValueError(f"negative query cannot have relevant intervals: {query_id}")
            for interval in intervals:
                if (not isinstance(interval, list) or len(interval) != 2
                        or not all(isinstance(value, (int, float)) for value in interval)
                        or interval[0] < 0 or interval[0] >= interval[1]):
                    raise ValueError(f"invalid relevant interval: {query_id}")
        if video_languages != LANGUAGES:
            raise ValueError(f"video requires English and Turkish queries: {video_id}")
        missing_video_categories = CATEGORIES - video_categories
        if missing_video_categories:
            raise ValueError(
                f"video missing query categories ({video_id}): "
                f"{', '.join(sorted(missing_video_categories))}"
            )

    missing_scenarios = SCENARIOS - scenarios
    if missing_scenarios:
        raise ValueError(f"missing scenarios: {', '.join(sorted(missing_scenarios))}")
    missing_languages = LANGUAGES - languages
    if missing_languages:
        raise ValueError(f"missing query languages: {', '.join(sorted(missing_languages))}")
    missing_categories = CATEGORIES - categories
    if missing_categories:
        raise ValueError(f"missing query categories: {', '.join(sorted(missing_categories))}")
    return {"videos": len(videos), "queries": len(query_ids),
            "manifest_sha256": sha256(path)}


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"queries": 0}
    positives = [row for row in rows if row["expected_presence"]]
    negatives = [row for row in rows if not row["expected_presence"]]
    positive_count = len(positives)
    return {
        "queries": count,
        "positive_queries": positive_count,
        "negative_queries": len(negatives),
        "auto_routing_accuracy": round(sum(row["route_correct"] for row in rows) / count, 4),
        "useful_top_1_rate": (
            round(sum(row["useful_top_1"] for row in positives) / positive_count, 4)
            if positive_count else None
        ),
        "useful_top_3_rate": (
            round(sum(row["useful_top_3"] for row in positives) / positive_count, 4)
            if positive_count else None
        ),
        "useful_top_5_rate": (
            round(sum(row["useful_top_5"] for row in positives) / positive_count, 4)
            if positive_count else None
        ),
        "negative_non_misleading_rate": (
            round(sum(not row["negative_misleading"] for row in negatives) / len(negatives), 4)
            if negatives else None
        ),
        "pass_rate": round(sum(row["user_usefulness"] == "PASS" for row in rows) / count, 4),
        "partial_rate": round(sum(row["user_usefulness"] == "PARTIAL" for row in rows) / count, 4),
        "fail_rate": round(sum(row["user_usefulness"] == "FAIL" for row in rows) / count, 4),
    }


def _validate_judgment(
    judgment: dict[str, Any], query_id: str, expected_presence: bool
) -> dict[str, Any]:
    useful = judgment["user_usefulness"].upper()
    if useful not in USEFULNESS:
        raise ValueError(f"invalid usefulness: {query_id}")
    top = [judgment[f"useful_top_{k}"] for k in (1, 3, 5)]
    if not all(isinstance(value, bool) for value in top) or top != sorted(top):
        raise ValueError(f"top-k usefulness must be monotonic: {query_id}")
    failure_categories = judgment.get("failure_categories", [])
    if any(category not in FAILURES for category in failure_categories):
        raise ValueError(f"invalid failure category: {query_id}")
    latency = judgment["search_latency_ms"]
    if not isinstance(latency, (int, float)) or latency < 0:
        raise ValueError(f"invalid search latency: {query_id}")
    timestamp_error = judgment.get("timestamp_error_seconds")
    if timestamp_error is not None and (
            not isinstance(timestamp_error, (int, float)) or timestamp_error < 0):
        raise ValueError(f"invalid timestamp error: {query_id}")
    misleading = judgment.get("negative_misleading")
    if not expected_presence and not isinstance(misleading, bool):
        raise ValueError(f"negative query requires negative_misleading: {query_id}")
    return {**judgment, "user_usefulness": useful,
            "negative_misleading": misleading if not expected_presence else None}


def summarize(manifest_path: Path, observations_path: Path) -> dict[str, Any]:
    manifest_info = validate(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if observations.get("manifest_sha256") != manifest_info["manifest_sha256"]:
        raise ValueError("observations are not bound to this frozen manifest")
    if observations.get("run_status") != "complete":
        raise ValueError("acceptance observations must be complete")

    expected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for video in manifest["videos"]:
        for query in video["queries"]:
            expected[query["query_id"]] = (video, query)
    observation_rows = observations.get("queries", [])
    supplied = {row["query_id"] for row in observation_rows}
    if supplied != set(expected) or len(supplied) != len(observation_rows):
        raise ValueError("observations must contain every frozen query exactly once")

    rows: list[dict[str, Any]] = []
    mode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mode_failures: dict[str, Counter[str]] = defaultdict(Counter)
    for observation in observation_rows:
        video, query = expected[observation["query_id"]]
        selected = _route(observation["auto_selected_route"])
        presence = query["expected_presence"]
        auto = _validate_judgment(observation, observation["query_id"], presence)
        rows.append({
            **auto, "language": query["language"].upper(),
            "scenario": video["scenario"], "category": query["category"],
            "expected_best_route": _expected_route(query),
            "expected_presence": presence,
            "route_correct": selected == _expected_route(query),
        })
        mode_failures["AUTO"].update(auto.get("failure_categories", []))
        explicit = observation.get("explicit_mode_results", {})
        if set(explicit) != ROUTES:
            raise ValueError(f"explicit results require VISUAL, SPEECH, and HYBRID: {observation['query_id']}")
        for mode in sorted(ROUTES):
            result = _validate_judgment(
                explicit[mode], f"{observation['query_id']}:{mode}", presence
            )
            mode_rows[mode].append({**result, "route_correct": mode == _expected_route(query),
                                    "expected_presence": presence})
            mode_failures[mode].update(result.get("failure_categories", []))

    slices: dict[str, dict[str, Any]] = {}
    for dimension in ("language", "scenario", "category"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row[dimension]].append(row)
        slices[dimension] = {key: _rates(value) for key, value in sorted(grouped.items())}
    latencies = [float(row["search_latency_ms"]) for row in rows]
    timestamp_errors = [float(row["timestamp_error_seconds"]) for row in rows
                        if row.get("timestamp_error_seconds") is not None]
    video_observations = observations.get("videos", [])
    expected_video_ids = {video["video_id"] for video in manifest["videos"]}
    observed_video_ids = {video["video_id"] for video in video_observations}
    if (observed_video_ids != expected_video_ids
            or len(observed_video_ids) != len(video_observations)):
        raise ValueError("observations must contain every video exactly once")
    if any(not row.get("transcript_quality_observations", "").strip()
           for row in video_observations):
        raise ValueError("every video requires transcript quality observations")
    processing_times = [float(row["processing_time_seconds"]) for row in video_observations]
    if any(value < 0 for value in processing_times):
        raise ValueError("processing time cannot be negative")
    explicit_summary = {}
    for mode, values in sorted(mode_rows.items()):
        mode_latencies = [float(row["search_latency_ms"]) for row in values]
        mode_rates = _rates(values)
        mode_rates.pop("auto_routing_accuracy")
        explicit_summary[mode] = {
            **mode_rates,
            "latency_ms": {"median": _percentile(mode_latencies, 0.5),
                           "p95": _percentile(mode_latencies, 0.95)},
            "failure_categories": dict(sorted(mode_failures[mode].items())),
        }
    return {
        "suite_id": manifest.get("suite_id"),
        "manifest_sha256": manifest_info["manifest_sha256"],
        "overall": _rates(rows), "slices": slices,
        "latency_ms": {"median": _percentile(latencies, 0.5),
                       "p95": _percentile(latencies, 0.95)},
        "timestamp_error_seconds": {
            "median": _percentile(timestamp_errors, 0.5),
            "p95": _percentile(timestamp_errors, 0.95),
            "rated_queries": len(timestamp_errors),
        },
        "processing": {"videos": len(processing_times),
                       "total_seconds": round(sum(processing_times), 3),
                       "median_seconds": _percentile(processing_times, 0.5)},
        "failure_categories": dict(sorted(mode_failures["AUTO"].items())),
        "explicit_modes": explicit_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    result = (summarize(args.manifest, args.observations) if args.observations
              else validate(args.manifest, not args.allow_draft))
    print(json.dumps(result, indent=2))
