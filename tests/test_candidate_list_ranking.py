import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.experiments.candidate_list_ranking.evaluation import oracle, rank_metrics
from ml.experiments.candidate_list_ranking.features import (
    CANDIDATE_FEATURES,
    QUERY_FEATURES,
    candidate_features,
    query_features,
)
from ml.experiments.candidate_list_ranking.model import (
    fit_accept_threshold,
    fit_logistic,
    predict,
)
from ml.experiments.candidate_list_ranking.run_experiment import route_audit, training_data


def fixture():
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    candidates = [
        {"timestamp": 0.0, "score": 0.8, "index": 0},
        {"timestamp": 5.0, "score": 0.7, "index": 1},
        {"timestamp": 20.0, "score": 0.2, "index": 2},
    ]
    return candidates, vectors


def test_features_are_finite_deterministic_and_schema_aligned():
    candidates, vectors = fixture()
    first = candidate_features(candidates, vectors)
    assert first.shape == (3, len(CANDIDATE_FEATURES))
    assert np.array_equal(first, candidate_features(candidates, vectors))
    query = query_features(candidates, vectors)
    assert query.shape == (len(QUERY_FEATURES),)
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(query))


def test_training_rejects_heldout_leakage():
    candidates, vectors = fixture()
    row = {"split": "heldout", "raw_top50": candidates, "embeddings": vectors,
           "expected_presence": True, "relevant_intervals": [[0, 1]]}
    with pytest.raises(ValueError, match="calibration"):
        training_data([row], 20)


def test_logistic_and_no_match_calibration_are_deterministic():
    x = np.asarray([[0.0], [0.1], [0.9], [1.0]])
    y = np.asarray([0.0, 0.0, 1.0, 1.0])
    first = fit_logistic(x, y, 1.0)
    second = fit_logistic(x, y, 1.0)
    assert first == second
    probabilities = predict(first, x)
    threshold = fit_accept_threshold(probabilities, y, 0.1)
    assert threshold["calibration_far"] <= 0.1


def test_oracle_and_ranking_cover_multiple_intervals_at_20_and_50():
    row = {"expected_presence": True, "relevant_intervals": [[0, 1], [20, 21]],
           "items": [{"timestamp": 0.0}, *({"timestamp": float(i)} for i in range(2, 20)),
                     {"timestamp": 20.0}]}
    assert rank_metrics([row], "items")["5"]["recall"] == 0.5
    assert oracle([row], "items") == {"5": 0.5, "20": 1.0, "50": 1.0}


def test_explicit_routing_uses_frozen_requirement_and_real_cross_path_evidence():
    audit = route_audit()
    assert audit["learned_from_query_text"] is False
    assert audit["counts"] == {"visual": 24, "speech": 6, "hybrid": 1}
    for row in audit["rows"]:
        expected = {"speech": "speech", "hybrid": "hybrid"}.get(
            row["requirement"], "visual"
        )
        assert row["selected_mode"] == expected
        assert set(row["cross_path_features"]) == {"visual", "speech", "hybrid"}


def test_committed_report_is_frozen_source_disjoint_and_not_promoted():
    root = Path(__file__).resolve().parents[1]
    path = root / "ml/evaluation/reports/candidate-list-ranking-v1.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    source = root / "ml/evaluation/reports/lightweight-pair-scorer-v1.json"
    natural = root / "ml/evaluation/natural_v2.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_report_sha256"]
    assert hashlib.sha256(natural.read_bytes()).hexdigest() == report["natural_manifest_sha256"]
    calibration = {row["video_id"] for row in report["rows"] if row["split"] == "calibration"}
    heldout = {row["video_id"] for row in report["rows"] if row["split"] == "heldout"}
    assert calibration.isdisjoint(heldout)
    assert report["heldout_tuning"] is False
    assert set(report["oracle"]["heldout"]) == {"5", "20", "50"}
    assert report["decision"]["outcome"].startswith("E_")
    assert report["production_changed"] is False
    assert report["second_frozen_run"]["required"] is False
