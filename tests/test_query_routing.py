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
from ml.evaluation.personal_acceptance import summarize, validate
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


def _acceptance_manifest(tmp_path):
    media = []
    for name in ("lecture", "demo", "ordinary"):
        path = tmp_path / f"{name}.mp4"
        path.write_bytes(name.encode())
        media.append(path)
    categories = [
        "spoken_topic", "quoted_mentioned_phrase", "visual_object", "visual_scene",
        "compositional_visual", "mixed_visual_spoken", "difficult_negative_unsupported",
    ]
    def queries(prefix):
        rows = []
        for index, category in enumerate(categories):
            for language in ("EN", "TR"):
                present = category != "difficult_negative_unsupported"
                rows.append({
                    "query_id": f"{prefix}-{index}-{language.lower()}",
                    "text": f"query {index} {language}", "language": language,
                    "category": category,
                    "expected_best_route": ["SPEECH", "VISUAL", "HYBRID"][index % 3],
                    "expected_presence": present,
                    "relevant_intervals": [[1, 2]] if present else [],
                })
        return rows
    return {
        "suite_id": "personal-video-acceptance-v1", "annotation_status": "frozen",
        "videos": [
            {"video_id": "lecture", "scenario": "lecture_tutorial",
             "duration_seconds": 1800, "duration_requirement_met": True,
             "local_path": str(media[0]), "sha256": hashlib.sha256(media[0].read_bytes()).hexdigest(),
             "queries": queries("lecture")},
            {"video_id": "demo", "scenario": "project_software_demo",
             "duration_seconds": 900, "duration_requirement_met": True,
             "local_path": str(media[1]), "sha256": hashlib.sha256(media[1].read_bytes()).hexdigest(),
             "queries": queries("demo")},
            {"video_id": "ordinary", "scenario": "ordinary_real_world",
             "duration_seconds": 60, "duration_requirement_met": True,
             "local_path": str(media[2]), "sha256": hashlib.sha256(media[2].read_bytes()).hexdigest(),
             "queries": queries("ordinary")},
        ],
    }


def test_personal_acceptance_requires_real_checksum_bound_media_and_all_dimensions(tmp_path):
    manifest = _acceptance_manifest(tmp_path)
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate(path)["queries"] == 42
    manifest["videos"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="changed media"):
        validate(path)


def test_personal_acceptance_supports_negative_queries_and_summarizes_human_usefulness(tmp_path):
    manifest = _acceptance_manifest(tmp_path)
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = validate(path)["manifest_sha256"]
    all_queries = [query for video in manifest["videos"] for query in video["queries"]]
    observations = {
        "manifest_sha256": manifest_hash, "run_status": "complete",
        "videos": [{"video_id": video["video_id"], "processing_time_seconds": 10,
                    "transcript_quality_observations": "Reviewed; words are intelligible."}
                   for video in manifest["videos"]],
        "queries": [{
            "query_id": query["query_id"],
            "auto_selected_route": query["expected_best_route"],
            "useful_top_1": query["expected_presence"],
            "useful_top_3": query["expected_presence"],
            "useful_top_5": query["expected_presence"],
            "timestamp_error_seconds": 1 if query["expected_presence"] else None,
            "user_usefulness": "PASS",
            "failure_categories": [], "failure_reason": "", "search_latency_ms": 5,
            **({"negative_misleading": False} if not query["expected_presence"] else {}),
            "explicit_mode_results": {
                mode: {"useful_top_1": query["expected_presence"],
                       "useful_top_3": query["expected_presence"],
                       "useful_top_5": query["expected_presence"],
                       "timestamp_error_seconds": (
                           1 if query["expected_presence"] else None
                       ),
                       "user_usefulness": "PASS", "failure_categories": [],
                       "failure_reason": "", "search_latency_ms": 6,
                       **({"negative_misleading": False}
                          if not query["expected_presence"] else {})}
                for mode in ("VISUAL", "SPEECH", "HYBRID")
            },
        } for query in all_queries],
    }
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(json.dumps(observations), encoding="utf-8")
    report = summarize(path, observations_path)
    assert report["overall"]["auto_routing_accuracy"] == 1.0
    assert report["overall"]["useful_top_5_rate"] == 1.0
    assert report["overall"]["positive_queries"] == 36
    assert report["overall"]["negative_queries"] == 6
    assert report["overall"]["negative_non_misleading_rate"] == 1.0
    assert report["slices"]["language"]["TR"]["queries"] == 21
    assert report["processing"]["total_seconds"] == 30
    assert report["explicit_modes"]["VISUAL"]["useful_top_1_rate"] == 1.0


def test_personal_acceptance_rejects_positive_without_interval(tmp_path):
    manifest = _acceptance_manifest(tmp_path)
    manifest["videos"][0]["queries"][0]["relevant_intervals"] = []
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="positive query requires"):
        validate(path)


def test_personal_acceptance_records_supplied_video_duration_shortfall(tmp_path):
    manifest = _acceptance_manifest(tmp_path)
    manifest["videos"][0]["duration_seconds"] = 1200
    manifest["videos"][0]["duration_requirement_met"] = False
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate(path)["videos"] == 3

    manifest["videos"][0]["duration_requirement_met"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duration_requirement_met"):
        validate(path)
