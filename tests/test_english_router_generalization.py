import copy
import json
from pathlib import Path

import pytest

from app.routing import resolve_mode
from ml.experiments.english_router_generalization.dataset import (
    build_manifest,
    manifest_hash,
    validate_manifest,
)
from ml.experiments.english_router_generalization.model import (
    fit_model,
    load_model,
    predict,
    save_model,
)
from ml.experiments.english_router_generalization.run_experiment import (
    FROZEN_MANIFEST_SHA256,
    _protected_evidence,
)


def _tiny_rows():
    return [
        {"query": "show me the blue window", "route": "VISUAL"},
        {"query": "find the red control panel", "route": "VISUAL"},
        {"query": "where do they explain the warranty", "route": "SPEECH"},
        {"query": "what is said about maintenance", "route": "SPEECH"},
        {"query": "explain the warning while the chart is visible", "route": "HYBRID"},
        {"query": "find the advice with the settings page on screen", "route": "HYBRID"},
    ]


def test_frozen_dataset_is_balanced_source_disjoint_and_bound():
    manifest = build_manifest()
    result = validate_manifest(manifest)
    assert result["sha256"] == FROZEN_MANIFEST_SHA256 == manifest_hash(manifest)
    assert result["sources"] == 15
    assert result["queries"] == 450
    assert result["split_sources"] == {"train": 9, "validation": 3, "frozen_test": 3}
    assert result["route_counts"] == {"VISUAL": 150, "SPEECH": 150, "HYBRID": 150}
    assert result["split_route_counts"]["frozen_test"] == {
        "VISUAL": 30, "SPEECH": 30, "HYBRID": 30
    }
    assert all(row["evidence_required"] for row in manifest["queries"])


def test_split_and_acceptance_leakage_guards_reject_invalid_data():
    manifest = build_manifest()
    leaked = manifest["queries"][0]["query"]
    with pytest.raises(ValueError, match="acceptance query leakage"):
        validate_manifest(manifest, [leaked.upper()])

    invalid = copy.deepcopy(manifest)
    invalid["queries"][0]["split"] = "frozen_test"
    with pytest.raises(ValueError, match="query split"):
        validate_manifest(invalid)


def test_local_protected_acceptance_inventory_is_fully_read_when_available():
    queries, digest = _protected_evidence()
    if digest is not None:
        assert len(queries) == 30
        assert len(digest) == 64


def test_new_source_ids_do_not_overlap_existing_frozen_manifests():
    root = Path(__file__).resolve().parents[1]
    new_manifest = build_manifest()
    new_ids = {source["source_id"] for source in new_manifest["sources"]}
    new_groups = {source["source_group"] for source in new_manifest["sources"]}
    old_ids = set()
    old_groups = set()
    for path in (root / "ml").rglob("*.json"):
        if path.name in {"dataset_v1.json", "english-router-generalization-v1.json"}:
            continue
        try:
            stack = [json.loads(path.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"source_id", "video_id"} and isinstance(child, str):
                        old_ids.add(child)
                    if key == "source_group" and isinstance(child, str):
                        old_groups.add(child)
                    stack.append(child)
            elif isinstance(value, list):
                stack.extend(value)
    assert new_ids.isdisjoint(old_ids)
    assert new_groups.isdisjoint(old_groups)


def test_candidate_training_serialization_and_inference_are_deterministic(tmp_path):
    rows = _tiny_rows()
    first = fit_model(rows, "char_tfidf")
    second = fit_model(rows, "char_tfidf")
    assert first == second
    path = tmp_path / "router.json"
    save_model(first, path)
    loaded = load_model(path)
    assert loaded == json.loads(path.read_text(encoding="utf-8"))
    assert predict(first, ["show me the red panel"])[0] == predict(
        loaded, ["show me the red panel"]
    )[0]


def test_candidate_confidence_fallback_and_malformed_queries():
    model = fit_model(_tiny_rows(), "word_tfidf")
    routes, _, _ = predict(model, ["a completely unfamiliar request"], 1.0)
    assert routes == ["HYBRID"]
    with pytest.raises(ValueError, match="non-empty"):
        predict(model, ["  "])
    with pytest.raises(ValueError, match="non-empty"):
        predict(model, [])


def test_existing_auto_and_explicit_bypass_fallback_remain_unchanged():
    explicit = resolve_mode("visual", "where is the explanation", True)
    assert explicit == {"route": "visual", "confidence": None, "reason": "explicit_override"}
    enabled = resolve_mode("auto", "red bicycle", True)
    assert enabled["route"] in {"visual", "speech", "hybrid"}
    assert resolve_mode("auto", "red bicycle", False) == {
        "route": "hybrid", "confidence": None, "reason": "auto_disabled_fallback"
    }
