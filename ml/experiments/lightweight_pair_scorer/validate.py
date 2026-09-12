"""Validate manifests, model files, report and learned scorer metadata."""

import argparse
import hashlib
import json
from pathlib import Path

from ml.evaluation.schema_v2 import load_benchmark
from ml.experiments.image_text_verifier.schema import load_calibration
from ml.experiments.lightweight_pair_scorer.learned import FEATURE_SCHEMA
from ml.experiments.lightweight_pair_scorer.schema import (
    LearnedScorerArtifact,
    assert_all_disjoint,
    load_development,
)

ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(report_path: Path, artifact_path: Path) -> None:
    natural_path = ROOT / "ml/evaluation/natural_v2.json"
    previous_path = ROOT / "ml/experiments/image_text_verifier/calibration_v1.json"
    development_path = Path(__file__).with_name("calibration_v2.json")
    natural = load_benchmark(natural_path)
    previous = load_calibration(previous_path)
    development = load_development(development_path)
    assert_all_disjoint(development, previous, natural.videos)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_hashes = {
        "natural_manifest_sha256": digest(natural_path),
        "previous_calibration_manifest_sha256": digest(previous_path),
        "development_manifest_sha256": digest(development_path),
    }
    if any(report.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("report manifest hash mismatch")
    artifact = LearnedScorerArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
    artifact.assert_compatible(
        FEATURE_SCHEMA, report["candidate_model"], report["candidate_revision"]
    )
    if artifact.threshold != report["calibration"]["learned_threshold"]:
        raise ValueError("learned threshold differs between artifact and report")
    expected_ids = {
        query.query_id
        for video in natural.videos if video.split == "heldout"
        for query in video.queries if "visual" in query.evaluation_modes
    }
    actual_ids = {row["query_id"] for row in report["rows"] if row["split"] == "heldout"}
    if actual_ids != expected_ids:
        raise ValueError("held-out report query set mismatch")
    if sum(report["uform"]["model_files"].values()) != report["uform"]["checkpoint_bytes"]:
        raise ValueError("UForm checkpoint size mismatch")
    print("lightweight pair scorer artifacts valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("artifact", type=Path)
    arguments = parser.parse_args()
    validate(arguments.report, arguments.artifact)
