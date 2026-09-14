import json
import unicodedata

import numpy as np

from app.routing import resolve_mode
from ml.experiments.turkish_compatibility.build_manifest import SOURCES
from ml.experiments.turkish_compatibility.language import (
    adapt_speech,
    adapt_visual,
    artifact,
    normalize_turkish,
    route_features,
    rule_route,
    words,
)
from ml.experiments.turkish_compatibility.run_experiment import (
    fit_feature_model,
    predict_feature,
    validate_manifest,
)


def test_turkish_casing_and_unicode_normalization():
    decomposed = unicodedata.normalize("NFD", "İSTANBUL IŞIK")
    assert normalize_turkish(decomposed) == "istanbul ışık"


def test_apostrophe_keeps_mixed_technical_term_base():
    assert words("Python'dan ve API'yi anlattığı yer")[:2] == ["python'dan", "ve"]
    assert adapt_visual("Python'dan ve API'yi anlattığı yer") == "python ve api"


def test_turkish_routes_cover_speech_visual_and_hybrid():
    assert rule_route("veritabanını açıkladığı kısmı bul")[0] == "SPEECH"
    assert rule_route("ekranda ayarlar menüsünün açık olduğu bölüm")[0] == "VISUAL"
    assert rule_route("dashboard'u gösterirken API'yi anlattığı yer")[0] == "HYBRID"


def test_query_adaptation_is_path_and_transcript_specific():
    query = "Python'dan bahsettiği yeri bul"
    assert adapt_speech(query, "en") == "python"
    assert adapt_speech(query, "tr") == "python"
    assert "screen" in adapt_visual("ekrandaki mavi kutuyu göster")


def test_explicit_mode_override_bypasses_auto_for_turkish():
    result = resolve_mode("speech", "ekrandaki kadını göster", enabled=True)
    assert result == {
        "route": "speech", "confidence": None, "reason": "explicit_override",
    }


def test_development_sources_are_disjoint_from_personal_acceptance():
    manifest = {
        "annotation_status": "frozen",
        "personal_acceptance_manifest_sha256": __import__("hashlib").sha256(
            __import__("pathlib").Path(
                "ml/evaluation/personal_acceptance_manifest_v1.json"
            ).read_bytes()
        ).hexdigest(),
        "sources": SOURCES,
    }
    validate_manifest(manifest)
    calibration = {
        source["source_group"] for source in SOURCES if source["split"] == "calibration"
    }
    heldout = {
        source["source_group"] for source in SOURCES if source["split"] == "heldout"
    }
    assert calibration.isdisjoint(heldout)


def test_leakage_guard_rejects_personal_media_checksum():
    personal = json.loads(
        __import__("pathlib").Path(
            "ml/evaluation/personal_acceptance_manifest_v1.json"
        ).read_text(encoding="utf-8")
    )
    sources = [dict(source) for source in SOURCES]
    sources[0]["sha256"] = personal["videos"][0]["sha256"]
    manifest = {
        "annotation_status": "frozen",
        "personal_acceptance_manifest_sha256": __import__("hashlib").sha256(
            __import__("pathlib").Path(
                "ml/evaluation/personal_acceptance_manifest_v1.json"
            ).read_bytes()
        ).hexdigest(),
        "sources": sources,
    }
    with __import__("pytest").raises(ValueError, match="leaked"):
        validate_manifest(manifest)


def test_feature_router_is_deterministic_and_serializable():
    rows = [
        {"turkish_query": "konuyu anlattığı yer", "route": "SPEECH"},
        {"turkish_query": "nerede konuşuyor?", "route": "SPEECH"},
        {"turkish_query": "kırmızı arabayı göster", "route": "VISUAL"},
        {"turkish_query": "ekrandaki paneli bul", "route": "VISUAL"},
        {"turkish_query": "anlatırken ekranı gösteriyor", "route": "HYBRID"},
        {"turkish_query": "model kısmı", "route": "HYBRID"},
    ]
    model = fit_feature_model(rows)
    restored = json.loads(json.dumps(model))
    query = "API'yi anlattığı bölüm"
    assert predict_feature(model, query) == predict_feature(restored, query)
    assert np.array_equal(route_features(query), route_features(query))


def test_adapter_artifact_schema_is_stable():
    value = artifact()
    assert value["schema_version"] == "1.0.0"
    assert "translation_lexicon" in value


def test_english_production_router_input_is_unchanged():
    query = "Where does she explain the API?"
    assert normalize_turkish(query) == query.lower()
    assert resolve_mode("auto", query, enabled=False)["route"] == "hybrid"
