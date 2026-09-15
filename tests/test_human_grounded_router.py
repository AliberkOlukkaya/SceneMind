import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.routing import artifact, auto_route, resolve_mode
from ml.experiments.english_router_generalization.model import (
    fit_model,
    load_model,
    predict,
    save_model,
)
from ml.experiments.human_grounded_router.build_annotations import build
from ml.experiments.human_grounded_router.dataset import (
    manifest_hash,
    validate_manifest,
)
from ml.experiments.human_grounded_router.run_experiment import FROZEN_MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "ml/experiments/human_grounded_router"


def documents():
    manifest = json.loads((HERE / "annotations_v1.json").read_text(encoding="utf-8"))
    source_list = json.loads((HERE / "sources_v1.json").read_text(encoding="utf-8"))["sources"]
    return manifest, {source["source_id"]: source for source in source_list}


def test_grounded_manifest_schema_balance_and_frozen_checksum():
    manifest, sources = documents()
    result = validate_manifest(manifest, sources)
    assert result["queries"] == 360
    assert result["route_counts"] == {"VISUAL": 120, "SPEECH": 120, "HYBRID": 120}
    assert result["split_counts"] == {"train": 180, "validation": 120, "frozen_test": 60}
    assert result["frozen_test_video_count"] == 2
    assert manifest_hash(manifest) == FROZEN_MANIFEST_SHA256
    assert build() == manifest


def test_source_disjointness_and_protected_acceptance_leakage_are_rejected():
    manifest, sources = documents()
    crossed = deepcopy(manifest)
    crossed["queries"][0]["split"] = "frozen_test"
    with pytest.raises(ValueError, match="metadata mismatch|crosses split"):
        validate_manifest(crossed, sources)
    leaked = deepcopy(manifest)
    leaked["queries"][0]["video_title"] = "Evaluating and developing machine learning models: an introduction"
    sources = deepcopy(sources)
    sources[leaked["queries"][0]["source_id"]]["title"] = leaked["queries"][0]["video_title"]
    with pytest.raises(ValueError, match="protected acceptance"):
        validate_manifest(leaked, sources)


def test_character_artifact_is_deterministic_loadable_and_checksum_bound(tmp_path):
    manifest, _ = documents()
    train = [row for row in manifest["queries"] if row["split"] == "train"]
    first = fit_model(train, "char_tfidf")
    second = fit_model(train, "char_tfidf")
    assert first == second
    first["fallback_threshold"] = 0.0
    first["dataset_id"] = manifest["dataset_id"]
    first["manifest_sha256"] = FROZEN_MANIFEST_SHA256
    target = tmp_path / "router.json"
    save_model(first, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == "4c1f3febee73275bcb25c110ff160375e77f9a27cde069079f5252ef7585e038"
    loaded = load_model(target)
    assert predict(loaded, ["the red interface panel"])[0][0] in {"VISUAL", "SPEECH", "HYBRID"}
    with pytest.raises(ValueError, match="non-empty"):
        predict(loaded, [""])
    with pytest.raises(ValueError, match="non-empty"):
        predict(loaded, [None])


def test_existing_production_router_and_explicit_modes_remain_available():
    assert artifact()["routes"] == ["VISUAL", "SPEECH", "HYBRID"]
    assert auto_route("where is the red panel")["route"] in {"visual", "speech", "hybrid"}
    for mode in ("visual", "speech", "hybrid"):
        assert resolve_mode(mode, "ambiguous words", True) == {
            "route": mode, "confidence": None, "reason": "explicit_override",
        }
    assert resolve_mode("auto", "anything", False)["route"] == "hybrid"
