"""Build the committed artifact with visibility, scale and failure analysis."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projected_frames(duration_seconds: int, interval: int) -> int:
    return math.floor(duration_seconds / interval) + 1


def assemble(raw_path: Path, visible_path: Path, max_path: Path) -> dict:
    report = json.loads(raw_path.read_text(encoding="utf-8"))
    visible = json.loads(visible_path.read_text(encoding="utf-8"))
    maximum = json.loads(max_path.read_text(encoding="utf-8"))
    parent_path = ROOT / "ml/evaluation/reports/lightweight-pair-scorer-v1.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    requested_classes: dict[str, set[str]] = {}
    for row in report["rows"]:
        if not row["mapping"]["detector_enabled"] or not row["detector_candidates"]:
            continue
        labels = {
            label for group in row["mapping"]["required_class_groups"] for label in group
        }
        for candidate in row["detector_candidates"]:
            key = f"{row['video_id']}:{candidate['timestamp']:.3f}"
            requested_classes.setdefault(key, set()).update(labels)
    compact_evidence = {}
    for key, evidence in report["detector_evidence"].items():
        labels = requested_classes.get(key, set())
        best = {}
        for detection in evidence:
            label = detection["detected_class"]
            if label in labels and (
                label not in best or detection["confidence"] > best[label]["confidence"]
            ):
                best[label] = detection
        compact_evidence[key] = list(best.values())
    report["detector_evidence"] = compact_evidence
    report["detector_evidence_policy"] = (
        "Highest-confidence box per query-required COCO class and frame; unrelated classes omitted."
    )
    coarse_frames = sum(row["frames"] for row in parent["videos"])
    coarse_bytes = 0
    uuids = {
        row["video_id"]: row["clip_candidates"][0]["thumbnail"].split("/")[2]
        for row in parent["rows"]
    }
    for identifier in uuids.values():
        matches = list((ROOT / "data/benchmarks").glob(f"*/videos/{identifier}/frames/*.jpg"))
        coarse_bytes += sum(path.stat().st_size for path in matches)
    global_two = report["resources"]["global_2s"]
    dense_frames = sum(row["frames"] for row in global_two.values())
    dense_jpeg_bytes = sum(row["jpeg_bytes"] for row in global_two.values())
    dense_embedding_bytes = sum(row["embedding_bytes"] for row in global_two.values())
    dense_work_seconds = sum(
        row["extraction_seconds"] + row["clip_index_seconds"]
        for row in global_two.values()
    )
    jpeg_per_frame = dense_jpeg_bytes / dense_frames
    seconds_per_frame = dense_work_seconds / dense_frames
    scale = {}
    for minutes in (10, 30, 60):
        duration = minutes * 60
        scale[str(minutes)] = {}
        for name, interval in (("global_5s", 5), ("global_2s", 2)):
            frames = projected_frames(duration, interval)
            scale[str(minutes)][name] = {
                "frames": frames,
                "embedding_bytes": frames * 512 * 4,
                "estimated_jpeg_bytes": round(frames * jpeg_per_frame),
                "estimated_extraction_plus_clip_seconds": frames * seconds_per_frame,
                "estimate_basis": "linear extrapolation from the measured ten-video global-2s run",
            }
        scale[str(minutes)]["bounded_per_routed_query"] = {
            "measured_average_frames": report["resources"]["average_secondary_frames_per_query"],
            "maximum_frames_from_five_disjoint_plus_or_minus_2s_windows": 15,
            "observed_mean_added_seconds": sum(
                row["total_added_seconds"] for row in report["resources"]["query_runtime"]
            ) / report["resources"]["routed_queries"],
            "note": "Query work is bounded by policy rather than video duration; cold misses depend on cache history.",
        }
    report["artifact_inputs"] = {
        "raw_measurement_sha256": sha256(raw_path),
        "visible_diagnostic_sha256": sha256(visible_path),
        "maximum_window_diagnostic_sha256": sha256(max_path),
        "parent_report_sha256": sha256(parent_path),
    }
    report["visible_evidence"] = {
        "selected_policy": visible,
        "maximum_tested_policy": maximum,
    }
    report["storage_comparison"] = {
        "global_5s": {
            "frames": coarse_frames, "jpeg_bytes": coarse_bytes,
            "embedding_bytes": coarse_frames * 512 * 4,
        },
        "global_2s": {
            "frames": dense_frames, "jpeg_bytes": dense_jpeg_bytes,
            "embedding_bytes": dense_embedding_bytes,
            "extraction_plus_clip_seconds": dense_work_seconds,
        },
        "bounded_cache": report["resources"]["bounded_cache"],
    }
    report["scale_projection"] = scale
    report["failure_buckets"] = {
        "verified_visible_diagnostic": {
            "coarse_region_or_window_miss": ["dirt-bicycle-pass", "throwball-late-ball"],
            "secondary_clip_ranking_miss_after_evidence_reached": ["throwball-early-ball"],
            "secondary_clip_success": ["pant-bottle-insertion"],
        },
        "benchmark_observations": {
            "detector_weight_selected_as_zero": True,
            "relationship_not_proven_by_object_presence": True,
            "action_temporal_queries_use_coarse_fallback": True,
            "unsupported_detector_queries_use_coarse_fallback": True,
            "threshold_failure": "The <=10% calibration FAR threshold causes 80.95% overall and 100% small-object held-out false abstention.",
            "duplicate_candidate_problem": "Temporal spacing saved frames but reduced calibration routed R@5 from 90.74% to 88.89% at the selected radius/top-K.",
        },
    }
    report["decision"] = {
        "outcome": "C_coarse_candidate_recall_is_dominant",
        "production_promoted": False,
        "second_frozen_run_required": False,
        "recommendation": "Improve coarse candidate generation and semantic temporal diversification before another detector or production integration. Preserve the current production path.",
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--visible", type=Path, required=True)
    parser.add_argument("--maximum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.raw, args.visible, args.maximum)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
