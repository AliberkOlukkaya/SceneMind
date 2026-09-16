import pytest

from app.hybrid import fuse
from app.routing import resolve_mode
from ml.evaluation.hybrid_fusion_trace import trace_fusion


def _inputs():
    visual = [
        {"thumbnail": "frame-a", "timestamp": 0.0, "score": 0.81, "modality": "visual"},
        {"thumbnail": "frame-b", "timestamp": 5.0, "score": 0.72, "modality": "visual"},
        {"thumbnail": "frame-c", "timestamp": 10.0, "score": 0.61, "modality": "visual"},
    ]
    speech = [
        {"thumbnail": "frame-b", "timestamp": 6.0, "end": 8.0, "score": 9.0, "text": "shared", "modality": "speech"},
        {"thumbnail": "frame-b", "timestamp": 7.0, "end": 9.0, "score": 8.0, "text": "duplicate", "modality": "speech"},
        {"thumbnail": "frame-d", "timestamp": 15.0, "end": 18.0, "score": 7.0, "text": "speech only", "modality": "speech"},
    ]
    return visual, speech


def test_diagnostic_trace_does_not_change_production_results():
    visual, speech = _inputs()
    without_diagnostics = fuse(visual, speech, 3)
    trace = trace_fusion(visual, speech, 3)
    assert trace["results"] == without_diagnostics
    assert [(row["thumbnail"], row["timestamp"], row["score"]) for row in trace["results"]] == [
        (row["thumbnail"], row["timestamp"], row["score"]) for row in without_diagnostics
    ]
    assert trace["ranking_source"] == "app.hybrid.fuse"
    assert trace["ranking_modified"] is False


def test_trace_records_overlap_contributions_and_thumbnail_deduplication():
    visual, speech = _inputs()
    trace = trace_fusion(visual, speech, 3)
    shared = next(row for row in trace["candidate_buckets"] if row["candidate_id"] == "frame-b")
    assert shared["candidate_overlap"] is True
    assert shared["contributions"]["visual"]["rrf_contribution"] == pytest.approx(1 / 62)
    assert shared["contributions"]["speech"]["rrf_contribution"] == pytest.approx(1 / 61)
    assert shared["deduplication"]["discarded_duplicate_speech_ranks"] == [2]
    assert shared["timestamp"] == 6.0
    assert trace["normalization"] == {"exists": False, "value": None}


def test_smart_search_mode_still_resolves_directly_to_hybrid():
    routing = resolve_mode("hybrid", "find the relevant moment", enabled=True)
    assert routing == {"route": "hybrid", "confidence": None, "reason": "explicit_override"}
