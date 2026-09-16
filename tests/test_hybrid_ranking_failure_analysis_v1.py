import copy
import json
from pathlib import Path

from app.hybrid import fuse
from ml.evaluation.development.analyze_hybrid_ranking_failures_v1 import summarize_trace
from ml.evaluation.hybrid_fusion_trace import trace_fusion

ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    visual = [
        {"thumbnail": "shared", "timestamp": 0.0, "score": 0.8, "modality": "visual"},
        {"thumbnail": "relevant", "timestamp": 5.0, "score": 0.7, "modality": "visual"},
    ]
    speech = [
        {
            "thumbnail": "shared",
            "timestamp": 1.0,
            "end": 2.0,
            "score": 4.0,
            "text": "shared",
            "modality": "speech",
        },
        {
            "thumbnail": "relevant",
            "timestamp": 6.0,
            "end": 8.0,
            "score": 3.0,
            "text": "relevant",
            "modality": "speech",
        },
    ]
    return visual, speech


def test_analysis_summary_is_observational_and_preserves_production_baseline():
    visual, speech = _inputs()
    expected = fuse(visual, speech, 2)
    trace = trace_fusion(visual, speech, 2)
    original_trace = copy.deepcopy(trace)
    summary = summarize_trace(
        dataset="test",
        source_id="design-free-software-talk",
        query_id="test-query",
        query="find relevant",
        category="MULTIMODAL",
        intervals=[[5.0, 8.0]],
        trace=trace,
    )
    assert trace == original_trace
    assert fuse(visual, speech, 2) == expected == trace["results"]
    assert summary["relevant_candidate"]["thumbnail_key"] == "relevant"
    assert summary["relevant_candidate"]["hybrid_rank"] == 2


def test_analysis_does_not_mutate_candidate_inputs_or_change_tie_order():
    visual, speech = _inputs()
    original = copy.deepcopy((visual, speech))
    before = fuse(visual, speech, 2)
    trace = trace_fusion(visual, speech, 2)
    summarize_trace(
        dataset="test",
        source_id="design-free-software-talk",
        query_id="test-query",
        query="find relevant",
        category="SPEECH",
        intervals=[[5.0, 8.0]],
        trace=trace,
    )
    assert (visual, speech) == original
    assert fuse(visual, speech, 2) == before
    assert [item["thumbnail"] for item in before] == ["shared", "relevant"]


def test_committed_analysis_is_diagnostic_and_keeps_frozen_sets_untouched():
    report = json.loads(
        (ROOT / "ml/evaluation/reports/hybrid-ranking-failure-analysis-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["production_ranker"] == "uncapped RRF60"
    assert report["production_retrieval_modified"] is False
    assert report["parameter_tuning_performed"] is False
    assert report["holdout_labels_or_queries_modified"] is False
    assert report["final_acceptance_v3_started"] is False
    assert report["summary"]["ranking_failures_analyzed"] == 19
    assert report["summary"]["success_controls_analyzed"] == 12
    assert report["v2_trace_reconstruction"]["historical_output_parity_queries"] == 30
