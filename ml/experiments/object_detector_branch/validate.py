"""Validate the committed detector report, evidence and non-promotable artifact."""

import argparse
import hashlib
import json
from pathlib import Path

from ml.experiments.object_detector_branch.schema import (
    DetectionEvidence,
    DetectorCalibrationArtifact,
)

ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(report_path: Path, artifact_path: Path) -> list[str]:
    errors = []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = DetectorCalibrationArtifact.model_validate_json(
        artifact_path.read_text(encoding="utf-8")
    )
    parents = {
        "natural_manifest_sha256": ROOT / "ml/evaluation/natural_v2.json",
        "calibration_v1_sha256": ROOT / "ml/experiments/image_text_verifier/calibration_v1.json",
        "calibration_v2_sha256": ROOT / "ml/experiments/lightweight_pair_scorer/calibration_v2.json",
        "parent_report_sha256": ROOT / "ml/evaluation/reports/lightweight-pair-scorer-v1.json",
    }
    for field, path in parents.items():
        if report[field] != digest(path):
            errors.append(f"{field} changed")
    if artifact.parent_report_sha256 != report["parent_report_sha256"]:
        errors.append("artifact parent report differs")
    if artifact.parent_benchmark_sha256 != report["natural_manifest_sha256"]:
        errors.append("artifact parent benchmark differs")
    if artifact.presence_threshold != report["calibration"]["threshold"]:
        errors.append("artifact threshold differs")
    if artifact.model_sha256 != report["detector"]["model_sha256"]:
        errors.append("artifact model differs")
    if artifact.promotable != report["gate"]["passed"]:
        errors.append("artifact promotion state differs")
    if report["gate"]["passed"] and not report["second_frozen_run"]["executed"]:
        errors.append("passing gate requires a second frozen run")
    for items in report["evidence"].values():
        for item in items:
            DetectionEvidence.model_validate(item)
    expected_ids = {row["query_id"] for row in report["rows_detail"]}
    artifact_ids = set(artifact.calibration_query_ids) | set(artifact.heldout_query_ids)
    if expected_ids != artifact_ids:
        errors.append("artifact query IDs differ")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report", nargs="?", type=Path,
        default=ROOT / "ml/evaluation/reports/object-detector-v1.json",
    )
    parser.add_argument(
        "artifact", nargs="?", type=Path,
        default=ROOT / "ml/experiments/object_detector_branch/calibration_v1.json",
    )
    arguments = parser.parse_args()
    failures = check(arguments.report, arguments.artifact)
    if failures:
        raise SystemExit("\n".join(failures))
    print("object detector report and artifact validated")
