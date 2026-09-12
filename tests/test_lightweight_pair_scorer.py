import json
from pathlib import Path

import pytest

from ml.evaluation.schema_v2 import load_benchmark
from ml.experiments.image_text_verifier.schema import load_calibration
from ml.experiments.lightweight_pair_scorer.schema import (
    LearnedScorerArtifact,
    assert_all_disjoint,
    load_development,
)

ROOT = Path(__file__).resolve().parents[1]


def test_development_expansion_is_frozen_balanced_and_disjoint():
    development = load_development(
        ROOT / "ml/experiments/lightweight_pair_scorer/calibration_v2.json"
    )
    previous = load_calibration(
        ROOT / "ml/experiments/image_text_verifier/calibration_v1.json"
    )
    benchmark = load_benchmark(ROOT / "ml/evaluation/natural_v2.json")

    assert_all_disjoint(development, previous, benchmark.videos)
    queries = [query for video in development.videos for query in video.queries]
    assert development.annotation_status == "frozen"
    assert len(queries) == 20
    assert sum(query.expected_presence for query in queries) == 10
    assert sum(not query.expected_presence for query in queries) == 10


def test_learned_artifact_serialization_and_compatibility():
    payload = {
        "schema_version": "1.0.0",
        "scorer_type": "logistic-regression",
        "feature_schema": ["score", "rank"],
        "feature_mean": [0.2, 3.0],
        "feature_scale": [0.1, 1.0],
        "weights": [1.0, -0.2],
        "intercept": 0.3,
        "threshold": 0.7,
        "training_dataset": "cal-v2",
        "training_manifest_sha256": "a" * 64,
        "parent_calibration_manifest_sha256": "b" * 64,
        "parent_benchmark_manifest_sha256": "c" * 64,
        "candidate_model_id": "clip",
        "candidate_model_revision": "d" * 40,
        "runtime": "numpy",
        "regularization": 1.0,
        "random_seed": 7,
        "trainable_parameters": 3,
        "training_seconds": 0.01,
        "created_at": "2026-09-12T00:00:00Z",
        "code_version": "e" * 40,
    }
    artifact = LearnedScorerArtifact.model_validate(payload)
    assert json.loads(artifact.model_dump_json())["threshold"] == 0.7
    artifact.assert_compatible(["score", "rank"], "clip", "d" * 40)
    with pytest.raises(ValueError, match="does not match"):
        artifact.assert_compatible(["score"], "clip", "d" * 40)
    with pytest.raises(ValueError, match="dimensions differ"):
        LearnedScorerArtifact.model_validate({**payload, "weights": [1.0]})
