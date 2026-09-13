"""Run frozen CLIP top-five candidates through the query-gated detector branch."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2

from ml.experiments.object_detector_branch.detector import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_VERSION,
    YoloXNanoDetector,
)
from ml.experiments.object_detector_branch.mapping import MAPPING_REVISION, map_query
from ml.experiments.object_detector_branch.ranking import (
    enrich_candidates,
    fit_presence_threshold,
    presence_candidates,
    rerank_candidates,
    summarize,
)
from ml.experiments.object_detector_branch.schema import DetectorCalibrationArtifact

ROOT = Path(__file__).resolve().parents[3]
NATURAL_MANIFEST = ROOT / "ml/evaluation/natural_v2.json"
CALIBRATION_V1 = ROOT / "ml/experiments/image_text_verifier/calibration_v1.json"
CALIBRATION_V2 = ROOT / "ml/experiments/lightweight_pair_scorer/calibration_v2.json"
PARENT_REPORT = ROOT / "ml/evaluation/reports/lightweight-pair-scorer-v1.json"
MODEL_PATH = ROOT / "data/models/yolox_nano-0.1.1rc0.onnx"
EXPECTED_MODEL_SHA256 = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
EXPECTED_MODEL_BYTES = 3_659_407
CONFIDENCE_FLOOR = 0.01
NMS_THRESHOLD = 0.45
SMALL_OBJECT_IDS = {
    "street-object-bicycle", "throw-object-ball", "throw-compositional-holding"
}
SMALL_OBJECT_CLASSES = {"bicycle", "sports ball"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_paths() -> dict[str, Path]:
    paths = {}
    for manifest_path in (NATURAL_MANIFEST, CALIBRATION_V1, CALIBRATION_V2):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for video in manifest["videos"]:
            path = ROOT / video["path"]
            if not path.is_file() or digest(path) != video["source"]["source_sha256"]:
                raise ValueError(f"missing or invalid media: {video['video_id']}")
            paths[video["video_id"]] = path
    return paths


def frame_id(video_id: str, timestamp: float) -> str:
    return f"{video_id}:{timestamp:.6f}"


def extract_frame(video: Path, timestamp: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, image = capture.read()
        if not ok or image is None or not cv2.imwrite(str(output), image):
            raise RuntimeError(f"could not extract {video} at {timestamp}")
    finally:
        capture.release()


def runtime_benchmark(model_path: Path, frames: list[Path], threads: int) -> dict:
    command = [
        sys.executable, "-m", "ml.experiments.object_detector_branch.benchmark_runtime",
        "--model", str(model_path), "--threads", str(threads), "--loops", "20",
    ]
    for path in frames:
        command.extend(("--frame", str(path)))
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    return json.loads(completed.stdout)


def _slice_rows(rows: list[dict]) -> dict[str, list[dict]]:
    slices = {name: [] for name in (
        "OBJECT", "SMALL_OBJECT", "RELATIONSHIP", "COMPOSITIONAL", "SCENE",
        "ACTION_TEMPORAL", "NEGATIVE", "DETECTOR_TRIGGERED", "NON_DETECTOR",
        "SMALL_OBJECT_NEGATIVE",
    )}
    for row in rows:
        if row["query_type"] in slices:
            slices[row["query_type"]].append(row)
        if row["query_id"] in SMALL_OBJECT_IDS:
            slices["SMALL_OBJECT"].append(row)
        if row["mapping"].relationship_required:
            slices["RELATIONSHIP"].append(row)
        slices["DETECTOR_TRIGGERED" if row["mapping"].detector_enabled else "NON_DETECTOR"].append(row)
        mapped = {label for group in row["mapping"].required_class_groups for label in group}
        if not row["expected_presence"] and mapped & SMALL_OBJECT_CLASSES:
            slices["SMALL_OBJECT_NEGATIVE"].append(row)
    return slices


def _metrics(rows: list[dict], fields: list[str]) -> dict:
    return {field: summarize(rows, field) for field in fields}


def run(output: Path, artifact_path: Path, model_path: Path = MODEL_PATH, threads: int = 4) -> dict:
    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    if digest(NATURAL_MANIFEST) != parent["natural_manifest_sha256"]:
        raise ValueError("frozen natural benchmark hash changed")
    if digest(CALIBRATION_V1) != parent["previous_calibration_manifest_sha256"]:
        raise ValueError("frozen calibration-v1 hash changed")
    if digest(CALIBRATION_V2) != parent["development_manifest_sha256"]:
        raise ValueError("frozen calibration-v2 hash changed")
    if model_path.stat().st_size != EXPECTED_MODEL_BYTES or digest(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("YOLOX-Nano artifact checksum or size mismatch")
    sources = source_paths()
    run_root = ROOT / "data/object-detector-run"
    rows = []
    frame_paths = {}
    for source_row in parent["rows"]:
        mapping = map_query(source_row["query"], source_row["query_type"])
        candidates = []
        for candidate in source_row["clip_candidates"]:
            identifier = frame_id(source_row["video_id"], candidate["timestamp"])
            path = run_root / "frames" / source_row["video_id"] / f"{candidate['timestamp']:.6f}.jpg"
            extract_frame(sources[source_row["video_id"]], candidate["timestamp"], path)
            frame_paths[identifier] = path
            candidates.append({**candidate, "frame_id": identifier})
        rows.append({
            "query_id": source_row["query_id"], "video_id": source_row["video_id"],
            "split": source_row["split"], "query": source_row["query"],
            "query_type": source_row["query_type"],
            "expected_presence": source_row["expected_presence"],
            "relevant_intervals": source_row["relevant_intervals"],
            "mapping": mapping, "clip_candidates": candidates,
        })

    detector = YoloXNanoDetector(
        model_path, threads=threads, confidence_floor=CONFIDENCE_FLOOR,
        nms_threshold=NMS_THRESHOLD,
    )
    evidences = {}
    inference_seconds = {}
    for identifier, path in frame_paths.items():
        video_id, timestamp = identifier.rsplit(":", 1)
        started = time.perf_counter()
        evidences[identifier] = detector.detect_path(identifier, float(timestamp), path)
        inference_seconds[identifier] = time.perf_counter() - started
    for row in rows:
        row["detector_candidates"] = enrich_candidates(row, evidences)

    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    threshold = fit_presence_threshold(calibration)
    for row in rows:
        row["presence_candidates"] = presence_candidates(row, threshold)

    calibration_triggered = [row for row in calibration if row["mapping"].detector_enabled]
    calibration_presence = summarize(calibration_triggered, "presence_candidates")["5"]
    calibration_clip = summarize(calibration_triggered, "clip_candidates")["5"]
    presence_promising = (
        calibration_presence["positive_false_abstention_rate"] <= 0.2
        and calibration_presence["negative_false_accept_rate"] <= 0.1
        and calibration_clip["recall"] - calibration_presence["recall"] <= 0.02
    )
    selected_weight = None
    if presence_promising:
        scored = []
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            for row in calibration:
                ranked = rerank_candidates(row, weight)
                row["trial_candidates"] = [
                    item for item in ranked
                    if not row["mapping"].detector_enabled
                    or item.get("object_confidence", 0) >= threshold
                ]
            metric = summarize(calibration_triggered, "trial_candidates")
            scored.append((metric["1"]["recall"], metric["5"]["recall"], metric["5"]["mrr"], -weight, weight))
        selected_weight = max(scored)[-1]
        for row in rows:
            ranked = rerank_candidates(row, selected_weight)
            row["reranked_candidates"] = [
                item for item in ranked
                if not row["mapping"].detector_enabled
                or item.get("object_confidence", 0) >= threshold
            ]

    fields = ["clip_candidates", "presence_candidates"]
    if selected_weight is not None:
        fields.append("reranked_candidates")
    heldout_slices = _slice_rows(heldout)
    runtime_frames = [frame_paths[item["frame_id"]] for item in next(
        row for row in rows if row["mapping"].detector_enabled and len(row["clip_candidates"]) == 5
    )["clip_candidates"]]
    runtime = runtime_benchmark(model_path, runtime_frames, threads)
    small_metrics = summarize(heldout_slices["SMALL_OBJECT"], "presence_candidates")["5"]
    small_negative = summarize(
        heldout_slices["SMALL_OBJECT_NEGATIVE"], "presence_candidates"
    )["5"]
    candidate_field = "reranked_candidates" if selected_weight is not None else "detector_candidates"
    raw_small = summarize(heldout_slices["SMALL_OBJECT"], "clip_candidates")["5"]
    branch_small = summarize(heldout_slices["SMALL_OBJECT"], candidate_field)["5"]
    gate_values = {
        "small_object_positive_false_abstention": small_metrics[
            "positive_false_abstention_rate"
        ],
        "small_object_negative_false_accept_rate": small_negative[
            "negative_false_accept_rate"
        ],
        "raw_candidate_recall_at_5_loss": raw_small["recall"] - branch_small["recall"],
        "added_warm_top5_median_seconds": runtime["warm_top5_median_seconds"],
        "added_peak_rss_bytes": runtime["added_peak_rss_bytes"],
    }
    gate_values["passed"] = (
        gate_values["small_object_positive_false_abstention"] <= 0.2
        and gate_values["small_object_negative_false_accept_rate"] is not None
        and gate_values["small_object_negative_false_accept_rate"] <= 0.1
        and gate_values["raw_candidate_recall_at_5_loss"] <= 0.02
        and gate_values["added_warm_top5_median_seconds"] <= 0.4
        and gate_values["added_peak_rss_bytes"] <= 750_000_000
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    report = {
        "experiment_schema_version": "1.0.0", "utc": timestamp,
        "git_commit": git_commit(), "parent_report_sha256": digest(PARENT_REPORT),
        "natural_manifest_sha256": digest(NATURAL_MANIFEST),
        "calibration_v1_sha256": digest(CALIBRATION_V1),
        "calibration_v2_sha256": digest(CALIBRATION_V2),
        "candidate_model": parent["candidate_model"],
        "candidate_revision": parent["candidate_revision"], "candidate_depth": 5,
        "detector": {
            "model_id": MODEL_ID, "model_version": MODEL_VERSION,
            "model_revision": MODEL_REVISION, "model_license": "Apache-2.0",
            "model_sha256": EXPECTED_MODEL_SHA256, "model_bytes": EXPECTED_MODEL_BYTES,
            "runtime": "ONNX Runtime CPUExecutionProvider", "providers": detector.providers,
            "input_size": [416, 416], "confidence_floor": CONFIDENCE_FLOOR,
            "nms_threshold": NMS_THRESHOLD, "threads": threads,
            "unique_candidate_frames": len(frame_paths),
            "evidence_generation_seconds": sum(inference_seconds.values()),
            "runtime_benchmark": runtime,
        },
        "environment": {
            "platform": platform.platform(), "cpu": os.environ.get("PROCESSOR_IDENTIFIER", ""),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "onnxruntime", "opencv-python-headless")},
        },
        "calibration": {
            "rows": len(calibration), "detector_triggered_rows": len(calibration_triggered),
            "threshold": threshold,
            "threshold_rule": "lowest object confidence allowing at most 10% calibration detector-triggered negative-query accepts",
            "presence_promising": presence_promising,
            "reranking_run": selected_weight is not None,
            "selected_detector_weight": selected_weight,
            "metrics": _metrics(calibration, fields),
            "triggered_metrics": _metrics(calibration_triggered, fields),
        },
        "heldout": {
            "rows": len(heldout), "metrics": _metrics(heldout, fields),
            "slices": {
                name: {field: summarize(slice_rows, field) for field in fields}
                for name, slice_rows in heldout_slices.items()
            },
        },
        "gate": gate_values,
        "second_frozen_run": {
            "required": gate_values["passed"],
            "executed": False,
            "reason": "Only required after every first-run production gate passes.",
        },
        "relationship_proof_of_concept": {
            "executed": False,
            "reason": "Only allowed when calibration object-presence evidence is promising."
            if not presence_promising else "Deferred until first-run held-out gate assessment.",
        },
        "evidence": {
            identifier: [item.model_dump(mode="json") for item in items]
            for identifier, items in evidences.items()
        },
        "rows_detail": [
            {
                **{key: row[key] for key in (
                    "query_id", "video_id", "split", "query", "query_type",
                    "expected_presence", "relevant_intervals",
                )},
                "mapping": row["mapping"].model_dump(mode="json"),
                **{field: row[field] for field in fields},
                "detector_candidates": row["detector_candidates"],
            }
            for row in rows
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifact = DetectorCalibrationArtifact(
        schema_version="1.0.0", model_id=MODEL_ID, model_version=MODEL_VERSION,
        model_revision=MODEL_REVISION, model_license="Apache-2.0",
        model_sha256=EXPECTED_MODEL_SHA256, model_bytes=EXPECTED_MODEL_BYTES,
        runtime="ONNX Runtime CPUExecutionProvider", input_size=(416, 416),
        confidence_floor=CONFIDENCE_FLOOR, nms_threshold=NMS_THRESHOLD,
        presence_threshold=threshold,
        threshold_rule=report["calibration"]["threshold_rule"],
        mapping_revision=MAPPING_REVISION, parent_report_sha256=digest(PARENT_REPORT),
        parent_benchmark_sha256=digest(NATURAL_MANIFEST),
        calibration_query_ids=[row["query_id"] for row in calibration],
        heldout_query_ids=[row["query_id"] for row in heldout], created_at=timestamp,
        code_version=report["git_commit"], promotable=gate_values["passed"],
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/object-detector-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "data/object-detector-calibration.json")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.artifact, arguments.model, arguments.threads)
    print(json.dumps({"gate": result["gate"], "calibration": result["calibration"]}, indent=2))
