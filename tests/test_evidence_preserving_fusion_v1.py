import hashlib
import json
from pathlib import Path

from ml.evaluation.hybrid_fusion_trace import trace_fusion
from ml.experiments.evidence_preserving_fusion_v1.ranking import (
    CONFIGURATIONS,
    rank_trace,
)


def _item(name, timestamp, score, modality, end=None):
    item = {
        "thumbnail": name,
        "timestamp": timestamp,
        "score": score,
        "modality": modality,
    }
    if modality == "speech":
        item.update(text=name, end=end if end is not None else timestamp + 1)
    return item


def _configuration(config_id):
    return next(item for item in CONFIGURATIONS if item.config_id == config_id)


def _trace(visual=None, speech=None):
    return trace_fusion(visual or [], speech or [], 5)


def test_baseline_is_exactly_the_production_trace_order():
    trace = _trace(
        [_item("a", 1, 0.9, "visual"), _item("b", 2, 0.8, "visual")],
        [_item("b", 2, 8, "speech"), _item("c", 3, 7, "speech")],
    )
    ranked = rank_trace(trace, _configuration("rrf60"), 5)

    assert [item["candidate_id"] for item in ranked] == [
        item["candidate_id"] for item in trace["candidate_buckets"]
    ]


def test_quota_preserves_capacity_for_both_modalities():
    trace = _trace(
        [_item(f"v{i}", i, 1 - i / 10, "visual") for i in range(1, 5)],
        [_item(f"s{i}", 10 + i, 10 - i, "speech") for i in range(1, 5)],
    )
    ranked = rank_trace(trace, _configuration("quota_2"), 5)

    ids = {item["candidate_id"] for item in ranked}
    assert {"v1", "v2", "s1", "s2"} <= ids
    assert len(ranked) == 5


def test_interleaving_is_deterministic_and_honors_start_lane():
    trace = _trace(
        [_item("v1", 1, 0.9, "visual"), _item("v2", 2, 0.8, "visual")],
        [_item("s1", 3, 9, "speech"), _item("s2", 4, 8, "speech")],
    )

    visual_first = rank_trace(trace, _configuration("interleave_visual"), 4)
    speech_first = rank_trace(trace, _configuration("interleave_speech"), 4)
    assert [item["candidate_id"] for item in visual_first] == ["v1", "s1", "v2", "s2"]
    assert [item["candidate_id"] for item in speech_first] == ["s1", "v1", "s2", "v2"]
    assert rank_trace(trace, _configuration("interleave_visual"), 4) == visual_first


def test_missing_visual_or_speech_lane_still_returns_unique_top_k():
    visual_only = _trace([_item(f"v{i}", i, 1 - i / 10, "visual") for i in range(1, 4)])
    speech_only = _trace(speech=[_item(f"s{i}", i, 10 - i, "speech") for i in range(1, 4)])

    for trace in (visual_only, speech_only):
        for config_id in ("quota_1", "interleave_visual", "interleave_speech"):
            ranked = rank_trace(trace, _configuration(config_id), 5)
            ids = [item["candidate_id"] for item in ranked]
            assert len(ids) == 3
            assert len(ids) == len(set(ids))


def test_existing_thumbnail_grouping_removes_temporal_duplicates():
    trace = _trace(
        [_item("shared", 10, 0.9, "visual")],
        [
            _item("shared", 11, 9, "speech", 12),
            _item("shared", 13, 8, "speech", 14),
            _item("other", 20, 7, "speech", 21),
        ],
    )
    ranked = rank_trace(trace, _configuration("interleave_speech"), 5)

    assert [item["candidate_id"] for item in ranked].count("shared") == 1
    shared = next(item for item in ranked if item["candidate_id"] == "shared")
    assert shared["deduplication"]["discarded_duplicate_speech_ranks"] == [2]


def test_preserve_strategy_keeps_strong_single_modality_candidate():
    visual = [_item("strong-visual", 1, 0.99, "visual")]
    speech = []
    for index in range(1, 7):
        name = f"shared-{index}"
        visual.append(_item(name, 10 + index, 0.9 - index / 100, "visual"))
        speech.append(_item(name, 10 + index, 10 - index, "speech"))
    trace = _trace(visual, speech)

    baseline_ids = [item["candidate_id"] for item in rank_trace(trace, _configuration("rrf60"), 5)]
    preserved_ids = [
        item["candidate_id"] for item in rank_trace(trace, _configuration("preserve_depth_1"), 5)
    ]
    assert "strong-visual" not in baseline_ids
    assert "strong-visual" in preserved_ids
    assert len(preserved_ids) == len(set(preserved_ids)) == 5


def test_stable_tie_handling_uses_production_rank_then_timestamp():
    trace = _trace(
        [_item("v", 5, 0.9, "visual")],
        [_item("s", 3, 9, "speech")],
    )
    first = rank_trace(trace, _configuration("quota_1"), 2)
    second = rank_trace(trace, _configuration("quota_1"), 2)

    assert first == second
    assert [item["candidate_id"] for item in first] == ["s", "v"]


def test_frozen_report_stops_before_protected_holdout():
    root = Path(__file__).resolve().parents[1]
    experiment = root / "ml/experiments/evidence_preserving_fusion_v1"
    spec_path = experiment / "frozen_candidate_v1.json"
    recorded = (experiment / "frozen_candidate_v1.sha256").read_text(encoding="utf-8").split()[0]
    observed = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    report = json.loads(
        (root / "ml/evaluation/reports/evidence-preserving-fusion-v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert observed == recorded == report["frozen_spec"]["sha256"]
    assert report["decision"]["code"] == "B"
    assert report["validation"]["gates"]["improvement_not_from_one_category_only"] is False
    assert report["validation"]["all_required_gates_passed"] is False
    assert report["protected_holdout"]["executed"] is False
    assert report["protected_holdout"]["queries_loaded_for_ranking"] == 0
    assert report["production_retrieval_modified"] is False
