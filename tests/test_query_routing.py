import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app import hybrid
from app.routing import (
    FEATURE_NAMES,
    artifact,
    auto_route,
    resolve_mode,
)
from app.routing import (
    text_features as production_features,
)
from ml.evaluation.personal_acceptance import validate
from ml.experiments.query_routing.router import (
    fit_softmax,
    learned_route,
)
from ml.experiments.query_routing.router import (
    text_features as experiment_features,
)
from ml.experiments.query_routing.run_experiment import training_matrix


@pytest.mark.parametrize("query,route", [
    ("red bicycle", "visual"),
    ("person holding a yellow ball", "visual"),
    ("where does he say gradient descent?", "speech"),
    ("when does he mention Python?", "speech"),
    ("when does she explain captions while YouTube Studio is on screen?", "hybrid"),
])
def test_auto_router_intent_examples(query, route):
    assert auto_route(query)["route"] == route


def test_routing_is_deterministic_and_feature_implementations_match():
    query = 'Where does she say "publish" while the button is visible?'
    assert np.array_equal(production_features(query), experiment_features(query))
    assert len(production_features(query)) == len(FEATURE_NAMES)
    assert auto_route(query) == auto_route(query)


def test_explicit_override_and_disabled_auto_are_safe():
    assert resolve_mode("speech", "red bicycle", True) == {
        "route": "speech", "confidence": None, "reason": "explicit_override"
    }
    assert resolve_mode("auto", "say bicycle", False) == {
        "route": "hybrid", "confidence": None, "reason": "auto_disabled_fallback"
    }


def test_softmax_and_hybrid_fallback_are_deterministic():
    x = np.vstack([experiment_features("red bicycle"),
                   experiment_features("when does he say Python"),
                   experiment_features("explain Python while the screen is visible")])
    y = np.asarray([0, 1, 2])
    first = fit_softmax(x, y, 0.1, iterations=20)
    second = fit_softmax(x, y, 0.1, iterations=20)
    assert first == second
    assert learned_route("unknown", first, fallback_threshold=1.0)[0] == "HYBRID"


def test_training_rejects_heldout_leakage():
    with pytest.raises(ValueError, match="calibration-only"):
        training_matrix([{"split": "heldout", "query": "say x", "target_route": "SPEECH"}])


def test_auto_end_to_end_selects_visual_path(monkeypatch):
    monkeypatch.setattr(hybrid.settings, "auto_routing_enabled", True)
    monkeypatch.setattr(hybrid, "visual_search", lambda video_id, q, k: {
        "query": q, "score_type": "cosine_similarity", "results": [{"timestamp": 1}]
    })
    response = hybrid.search("unused", "red bicycle", 5, "auto")
    assert response["requested_mode"] == "auto"
    assert response["selected_route"] == "visual"
    assert response["results"] == [{"timestamp": 1}]


def test_router_artifact_and_committed_report_are_bound_to_frozen_calibration():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "ml/experiments/query_routing/calibration_v1.json"
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert artifact()["calibration_manifest_sha256"] == expected_hash
    report_path = root / "ml/evaluation/reports/query-routing-v1.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["calibration_manifest_sha256"] == expected_hash
    assert report["dataset"]["source_disjoint"] is True
    assert report["heldout_tuning"] is False
    assert report["gate"]["passed"] is True
    assert report["second_frozen_run"]["passed"] is True
    assert report["production"]["auto_integrated"] is True
    assert report["production"]["explicit_modes_preserved"] is True


def test_personal_acceptance_requires_real_checksum_bound_media(tmp_path):
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"private fixture")
    manifest = {
        "annotation_status": "frozen",
        "videos": [{"video_id": "mine", "local_path": str(media),
                    "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
                    "queries": [{"query_id": "mine-speech", "target_route": "SPEECH",
                                 "relevant_intervals": [[1, 2]]}]}],
    }
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate(path)["queries"] == 1
    manifest["videos"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="changed media"):
        validate(path)
