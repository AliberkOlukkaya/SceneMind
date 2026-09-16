import json
from pathlib import Path

import numpy as np
import pytest

from ml.experiments.calibrated_fusion_v1.calibration import (
    agreement_features,
    bucket_relevant,
    candidate_feature_rows,
    fit_logistic,
    predict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_logistic_calibration_is_deterministic_and_normalized():
    features = np.asarray([[0.1], [0.2], [0.8], [0.9]], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=float)
    first = fit_logistic(features, labels)
    second = fit_logistic(features, labels)

    assert first == second
    probabilities = predict(first, features)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert probabilities.tolist() == pytest.approx(sorted(probabilities.tolist()))


def test_query_local_features_preserve_stable_ties():
    candidates = [
        {"raw_score": 0.7},
        {"raw_score": 0.7},
        {"raw_score": 0.4},
    ]
    rows = candidate_feature_rows(candidates, "find the diagram")

    assert [row["reciprocal_rank"] for row in rows] == [1.0, 0.5, 1 / 3]
    assert rows[0]["margin_to_next"] == 0.0
    assert rows[1]["margin_to_top"] == 0.0


def test_missing_modality_and_strong_single_modality_are_explicit():
    visual_only = {
        "contributions": {"visual": {"timestamp": 10.0}},
    }
    visual_row = {
        "negative": False,
        "category": "VISUAL",
        "intervals": [[9.0, 11.0]],
    }
    multimodal_row = {**visual_row, "category": "MULTIMODAL"}

    assert agreement_features(visual_only) == {
        "visual_present": True,
        "speech_present": False,
        "cross_modal_present": False,
        "temporal_distance_seconds": None,
    }
    assert bucket_relevant(visual_only, visual_row) is True
    assert bucket_relevant(visual_only, multimodal_row) is False


def test_weak_two_modality_consensus_does_not_imply_relevance():
    consensus = {
        "contributions": {
            "visual": {"timestamp": 80.0},
            "speech": {"timestamp": 82.0, "end": 84.0},
        }
    }
    row = {
        "negative": False,
        "category": "MULTIMODAL",
        "intervals": [[10.0, 15.0]],
    }

    assert agreement_features(consensus)["cross_modal_present"] is True
    assert agreement_features(consensus)["temporal_distance_seconds"] == 2.0
    assert bucket_relevant(consensus, row) is False


def test_rejected_artifact_loads_and_report_preserves_production():
    artifact = json.loads(
        (ROOT / "ml/experiments/calibrated_fusion_v1/calibration_artifact_v1.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (ROOT / "ml/evaluation/reports/calibrated-fusion-v1.json").read_text(encoding="utf-8")
    )

    assert artifact["status"] == "diagnostic_rejected_before_fusion"
    assert artifact["production_eligible"] is False
    assert report["decision"]["code"] == "C"
    assert report["fusion_candidate_constructed"] is False
    assert report["production_retrieval_modified"] is False
    assert report["data"]["isolation"]["protected_holdout_queries_evaluated"] == 0
