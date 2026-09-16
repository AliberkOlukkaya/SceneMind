import app.hybrid as production_hybrid
from app.hybrid import fuse
from app.routing import resolve_mode
from ml.evaluation.development.hybrid_fusion_refinement_v1 import (
    VARIANTS,
    overlap_score,
    production_shape,
    rank_candidates,
    rank_contribution,
)
from ml.evaluation.run_hybrid_fusion_refinement_v1 import DEFAULT_INPUT, run


def _inputs():
    visual = [
        {"thumbnail": "a", "timestamp": 0.0, "score": 0.9, "modality": "visual"},
        {"thumbnail": "b", "timestamp": 5.0, "score": 0.8, "modality": "visual"},
        {"thumbnail": "c", "timestamp": 10.0, "score": 0.7, "modality": "visual"},
    ]
    speech = [
        {
            "thumbnail": "b",
            "timestamp": 6.0,
            "end": 8.0,
            "score": 4.0,
            "text": "shared",
            "modality": "speech",
        },
        {
            "thumbnail": "b",
            "timestamp": 7.0,
            "end": 9.0,
            "score": 3.0,
            "text": "duplicate",
            "modality": "speech",
        },
        {
            "thumbnail": "d",
            "timestamp": 15.0,
            "end": 18.0,
            "score": 2.0,
            "text": "speech only",
            "modality": "speech",
        },
    ]
    return visual, speech


def test_offline_baseline_exactly_matches_production_fuse():
    visual, speech = _inputs()
    baseline = VARIANTS[0]
    experimental = rank_candidates(visual, speech, 4, baseline)["results"]
    assert [production_shape(item) for item in experimental] == fuse(visual, speech, 4)


def test_variants_are_deterministic_and_do_not_mutate_inputs():
    visual, speech = _inputs()
    original = ([dict(item) for item in visual], [dict(item) for item in speech])
    for variant in VARIANTS:
        first = rank_candidates(visual, speech, 4, variant)
        second = rank_candidates(visual, speech, 4, variant)
        assert first == second
    assert (visual, speech) == original


def test_capped_overlap_formula_preserves_singles_and_caps_shared_total():
    mild = next(variant for variant in VARIANTS if variant.variant_id == "cap_1_75")
    assert overlap_score([0.02], mild) == 0.02
    assert overlap_score([0.02, 0.019], mild) == 0.035


def test_normalized_rank_formulas_have_declared_endpoints():
    percentile = next(
        variant for variant in VARIANTS if variant.variant_id == "normalized_percentile"
    )
    normalized = next(variant for variant in VARIANTS if variant.variant_id == "normalized_rrf60")
    assert rank_contribution(1, 50, percentile) == 1.0
    assert rank_contribution(50, 50, percentile) == 0.02
    assert rank_contribution(1, 50, normalized) == 1.0
    assert 0 < rank_contribution(50, 50, normalized) < 0.02


def test_experimental_module_does_not_change_smart_search_or_fuse():
    visual, speech = _inputs()
    before = fuse(visual, speech, 4)
    for variant in VARIANTS:
        rank_candidates(visual, speech, 4, variant)
    assert fuse(visual, speech, 4) == before
    assert resolve_mode("hybrid", "find this", enabled=True)["route"] == "hybrid"


def test_experimental_execution_does_not_alter_normal_api_response(monkeypatch, tmp_path):
    visual, speech = _inputs()
    monkeypatch.setattr(production_hybrid, "folder_for", lambda _video_id: tmp_path / "video")
    monkeypatch.setattr(
        production_hybrid,
        "read_manifest",
        lambda _folder: {"status": "ready", "frames": []},
    )
    monkeypatch.setattr(
        production_hybrid,
        "speech_results",
        lambda _video_id, _query, _frames, _k: (speech, True),
    )
    monkeypatch.setattr(production_hybrid, "index_status", lambda _folder: {"status": "ready"})
    monkeypatch.setattr(
        production_hybrid,
        "visual_search",
        lambda _video_id, _query, _k: {"results": visual},
    )
    before = production_hybrid.search("video", q="find this", k=4, mode="hybrid")
    for variant in VARIANTS:
        rank_candidates(visual, speech, 4, variant)
    after = production_hybrid.search("video", q="find this", k=4, mode="hybrid")
    assert after == before
    assert after["requested_mode"] == "hybrid"
    assert after["selected_route"] == "hybrid"


def test_frozen_development_report_has_exact_baseline_parity(tmp_path):
    report = run(DEFAULT_INPUT, tmp_path / "refinement.json")
    assert report["baseline_parity"] == {
        "passed": True,
        "queries": 32,
        "compared": ["candidate_ids", "timestamps", "ordering", "fusion_scores", "top_k"],
    }
    assert report["production_modified"] is False
