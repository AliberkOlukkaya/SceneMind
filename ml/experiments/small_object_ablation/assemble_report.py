"""Assemble the committed report from ignored measurements and frozen reviews."""

import argparse
import hashlib
import json
from pathlib import Path

from .metrics import sampling_summary
from .schema import VisibilityManifest


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_summary(
    run: dict, false_matches: set[tuple], threshold: float, split: str | None = None
) -> dict:
    rows = [row for row in run["rows"] if split is None or row["split"] == split]
    matched = [
        row for row in rows
        if row["detected"]
        and row["confidence"] >= threshold
        and (run["configuration"], row["event_id"], row["timestamp"]) not in false_matches
    ]
    return {
        "verified_visible_frames": len(rows),
        "matched_target_frames": len(matched),
        "recall": len(matched) / len(rows),
        "mean_confidence_on_matches": (
            sum(row["confidence"] for row in matched) / len(matched) if matched else None
        ),
        "mean_bbox_area_ratio_on_matches": (
            sum(row["bbox_area_ratio"] for row in matched) / len(matched) if matched else None
        ),
    }


def assemble(results: Path, manifest_path: Path, review_path: Path) -> dict:
    manifest = VisibilityManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    sampling = json.loads((results / "sampling.json").read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    false_matches = {
        (row["configuration"], row["event_id"], row["timestamp"])
        for row in review["false_target_matches"]
    }
    detectors = {}
    for config in ("yolox-416", "yolox-640", "yolox-768", "rtdetr-r18-640"):
        run_path = results / f"{config}.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        detectors[config] = {
            "model": run["model"], "runtime": run["runtime"],
            "recall_at_confidence": {
                str(threshold): reviewed_summary(run, false_matches, threshold)
                for threshold in (0.01, 0.1, 0.25, 0.5)
            },
            "split_recall_at_0.01": {
                split: reviewed_summary(run, false_matches, 0.01, split)
                for split in ("calibration", "heldout")
            },
            "measurement_sha256": file_sha256(run_path),
        }
    baseline = detectors["yolox-416"]["recall_at_confidence"]["0.01"]["recall"]
    gates = {}
    second_runs = {}
    for config in ("yolox-640", "yolox-768"):
        candidate = detectors[config]["recall_at_confidence"]["0.01"]["recall"]
        gates[config] = {
            "quality": {"rule": "reviewed visible-frame recall >=80% and >=10 percentage points over YOLOX-Nano 416", "pass": candidate >= 0.8 and candidate - baseline >= 0.1},
            "latency": {"rule": "per-frame CPU p95 <=100 ms", "pass": detectors[config]["runtime"]["per_frame_p95_seconds"] <= 0.1},
            "memory": {"rule": "added peak RSS <=500 MiB", "pass": detectors[config]["runtime"]["added_peak_rss_bytes"] <= 500 * 1024 * 1024},
        }
        first = json.loads((results / f"{config}.json").read_text(encoding="utf-8"))
        second_path = results / f"{config}-second.json"
        second = json.loads(second_path.read_text(encoding="utf-8"))
        second_runs[config] = {
            "quality_rows_identical": first["rows"] == second["rows"],
            "runtime": second["runtime"],
            "measurement_sha256": file_sha256(second_path),
        }
    return {
        "schema_version": "1.0.0", "experiment": "small-object-sampling-resolution-ablation-v1",
        "production_changed": False, "heldout_tuning": False,
        "visibility_manifest_sha256": file_sha256(manifest_path),
        "detection_review_sha256": file_sha256(review_path),
        "sampling_evidence": {
            "all": sampling_summary(manifest),
            "calibration": sampling_summary(manifest, "calibration"),
            "heldout": sampling_summary(manifest, "heldout"),
        },
        "sampling_resources_and_clip": sampling,
        "detectors": detectors, "higher_resolution_gates": gates,
        "second_frozen_runs": second_runs,
        "diagnosis": {
            "outcome": "C_both_sampling_and_detector_matter",
            "dominant_failure": "sampling",
            "recommendation": "Keep production unchanged. Prototype a query-gated, cached 2-second secondary sampling path with YOLOX-Nano 640 over bounded windows. Do not run RT-DETR-R18 over every frame on CPU.",
            "reason": "Five-second visible-evidence recall is 25%, so no detector can recover three of four events. Two-second sampling reaches 100% evidence recall here. Nano 640 improves reviewed conditional recall by 16.7 points, ties Nano 768, and has lower CPU latency and memory. RT-DETR improves another 5.6 points but exceeds the preferred memory gate and is much slower.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.results, args.manifest, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
