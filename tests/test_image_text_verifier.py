import json
import math
from pathlib import Path

import pytest

from ml.evaluation.schema_v2 import load_benchmark
from ml.experiments.image_text_verifier.ranking import apply_threshold, fit_threshold, rerank
from ml.experiments.image_text_verifier.regression import check
from ml.experiments.image_text_verifier.schema import (
    CalibrationArtifact,
    Preprocessing,
    assert_disjoint,
    load_calibration,
)
from ml.experiments.image_text_verifier.verifier import batched_text

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_expansion_is_frozen_and_heldout_disjoint():
    calibration = load_calibration(
        ROOT / "ml/experiments/image_text_verifier/calibration_v1.json"
    )
    benchmark = load_benchmark(ROOT / "ml/evaluation/natural_v2.json")

    assert_disjoint(calibration, benchmark.videos)
    assert calibration.annotation_status == "frozen"
    assert sum(len(video.queries) for video in calibration.videos) == 30


def test_batch_text_and_reranking_are_deterministic():
    candidates = [{"timestamp": 1.0}, {"timestamp": 2.0}, {"timestamp": 3.0}]
    ranked = rerank(candidates, [0.2, 0.9, 0.9])

    assert batched_text("find a dog", 3) == ["find a dog"] * 3
    assert [item["timestamp"] for item in ranked] == [2.0, 3.0, 1.0]
    assert [item["clip_rank"] for item in ranked] == [2, 3, 1]
    with pytest.raises(ValueError, match="counts differ"):
        rerank(candidates, [0.1])


def test_threshold_uses_calibration_negatives_only_and_abstains():
    rows = [
        {"split": "calibration", "expected_presence": True,
         "reranked": [{"verifier_score": 0.7}]},
        {"split": "calibration", "expected_presence": False,
         "reranked": [{"verifier_score": 0.4}]},
        {"split": "calibration", "expected_presence": False,
         "reranked": [{"verifier_score": 0.2}]},
    ]
    threshold = fit_threshold(rows)

    assert threshold == math.nextafter(0.4, math.inf)
    assert apply_threshold(rows[0], threshold)
    assert not apply_threshold(rows[1], threshold)
    with pytest.raises(ValueError, match="calibration-only"):
        fit_threshold([{**rows[0], "split": "heldout"}, *rows[1:]])


def test_calibration_artifact_rejects_model_or_preprocessing_mismatch():
    preprocessing = Preprocessing(image_size=384, max_text_tokens=35, processor="pinned")
    artifact = CalibrationArtifact.model_validate(
        {
            "schema_version": "1.0.0",
            "model_id": "model",
            "model_revision": "a" * 40,
            "model_license": "license",
            "preprocessing": preprocessing.model_dump(),
            "match_label_index": 1,
            "threshold": 0.5,
            "threshold_rule": "calibration only",
            "calibration_dataset": "cal-v1",
            "calibration_manifest_sha256": "b" * 64,
            "parent_benchmark": "benchmark",
            "benchmark_schema_version": "2.0.0",
            "created_at": "2026-09-12T00:00:00Z",
            "promotable": False,
        }
    )
    artifact.assert_compatible("model", "a" * 40, preprocessing)
    with pytest.raises(ValueError, match="does not match"):
        artifact.assert_compatible("other", "a" * 40, preprocessing)

    assert json.loads(artifact.model_dump_json())["promotable"] is False


def test_verifier_regression_gate_detects_quality_drop():
    metrics = {
        "recall": 0.8,
        "mrr": 0.7,
        "negative_false_accept_rate": 0.1,
        "positive_false_abstention_rate": 0.2,
    }
    report = {
        **{key: "same" for key in (
            "natural_manifest_sha256", "calibration_manifest_sha256", "candidate_model",
            "candidate_revision", "candidate_depth", "sampling_interval",
        )},
        "verifier": {
            key: "same" for key in (
                "model_id", "revision", "match_label_index", "max_text_tokens", "image_size"
            )
        },
        "heldout": {
            system: {str(k): dict(metrics) for k in (1, 3, 5)}
            for system in (
                "clip", "clip_calibrated", "verifier_reranked", "verifier_calibrated"
            )
        },
    }
    current = json.loads(json.dumps(report))
    current["heldout"]["verifier_calibrated"]["5"]["recall"] = 0.7

    assert check(current, report) == ["verifier_calibrated@5 recall: 0.8 -> 0.7"]
