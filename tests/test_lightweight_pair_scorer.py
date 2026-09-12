import json
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation.schema_v2 import load_benchmark
from ml.experiments.image_text_verifier.schema import load_calibration
from ml.experiments.lightweight_pair_scorer.learned import (
    FEATURE_SCHEMA,
    candidate_features,
    fit,
    score,
)
from ml.experiments.lightweight_pair_scorer.ranking import fit_threshold, rerank
from ml.experiments.lightweight_pair_scorer.regression import check
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


def test_feature_extraction_and_tiny_scorer_are_deterministic():
    candidates = [
        {"score": 0.8, "timestamp": 1.0},
        {"score": 0.6, "timestamp": 2.0},
        {"score": 0.5, "timestamp": 3.0},
    ]
    features = candidate_features(candidates)
    assert features.shape == (3, len(FEATURE_SCHEMA))
    assert features[0, 0] == 0.8
    assert features[1, 2] == pytest.approx(0.2)
    assert np.all(features[:, 3] == pytest.approx(0.2))
    rows = [
        {
            "split": "calibration", "expected_presence": True,
            "relevant_intervals": [[0.0, 1.5]], "clip_candidates": candidates,
        },
        {
            "split": "calibration", "expected_presence": False,
            "relevant_intervals": [],
            "clip_candidates": [
                {"score": 0.2, "timestamp": 1.0}, {"score": 0.1, "timestamp": 2.0}
            ],
        },
    ]
    first = fit(rows)
    second = fit(rows)
    assert first["weights"] == pytest.approx(second["weights"])
    probabilities = score(candidates, first)
    assert len(probabilities) == 3
    assert all(0 <= value <= 1 for value in probabilities)
    with pytest.raises(ValueError, match="calibration-only"):
        fit([{**rows[0], "split": "heldout"}, rows[1]])


def test_pair_ranking_and_threshold_behavior():
    candidates = [{"timestamp": 1.0}, {"timestamp": 2.0}]
    ranked = rerank(candidates, [0.2, 0.9], "pair_score")
    assert [item["timestamp"] for item in ranked] == [2.0, 1.0]
    rows = [
        {"split": "calibration", "expected_presence": True, "ranked": ranked},
        {"split": "calibration", "expected_presence": False,
         "ranked": rerank(candidates, [0.1, 0.3], "pair_score")},
    ]
    threshold = fit_threshold(rows, "ranked", "pair_score")
    assert threshold > 0.3
    assert [item for item in ranked if item["pair_score"] >= threshold]


def test_frozen_heldout_comparison_detects_metric_change():
    metrics = {
        "recall": 0.8, "mrr": 0.7, "negative_false_accept_rate": 0.1,
        "positive_false_abstention_rate": 0.2,
    }
    reference = {
        **{key: "same" for key in (
            "natural_manifest_sha256", "previous_calibration_manifest_sha256",
            "development_manifest_sha256", "candidate_model", "candidate_revision",
            "candidate_depth", "sampling_interval",
        )},
        "heldout": {"metrics": {
            system: {str(k): dict(metrics) for k in (1, 3, 5)}
            for system in (
                "clip", "clip_calibrated", "uform_reranked", "uform_calibrated",
                "learned_reranked", "learned_calibrated",
            )
        }},
    }
    current = json.loads(json.dumps(reference))
    current["heldout"]["metrics"]["uform_calibrated"]["5"]["recall"] = 0.7
    assert check(current, reference) == ["uform_calibrated@5 recall: 0.8 -> 0.7"]
