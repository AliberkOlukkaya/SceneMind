import hashlib
import json
from pathlib import Path

import pytest

from ml.experiments.bounded_secondary_sampling.cache import SecondaryFrameCache
from ml.experiments.bounded_secondary_sampling.fusion import (
    abstain,
    fit_abstention_threshold,
    fuse,
)
from ml.experiments.bounded_secondary_sampling.policy import (
    CoarseCandidate,
    cache_key,
    diversify,
    local_windows,
    secondary_timestamps,
)
from ml.experiments.object_detector_branch.mapping import map_query


def candidates(*timestamps):
    return [CoarseCandidate(timestamp=value, score=1 - index / 10) for index, value in enumerate(timestamps)]


def test_windows_clip_to_video_boundaries_and_merge():
    assert local_windows(candidates(1, 9), 4, 10) == [(0.0, 10)]
    assert secondary_timestamps(candidates(1), 4, 10) == [0.0, 2.0, 4.0]


def test_temporal_diversification_preserves_score_order():
    rows = diversify(candidates(10, 11, 20, 30), limit=3, minimum_spacing=6)
    assert [row.timestamp for row in rows] == [10, 20, 30]


def test_cache_key_is_versioned_and_stable():
    assert cache_key("video", 2.0) == cache_key("video", 2.0004)
    assert cache_key("video", 2.0) != cache_key("video", 2.0, "v2")
    assert cache_key("video-a", 2.0) != cache_key("video-b", 2.0)


def test_cache_reuses_frames_and_enforces_byte_bound(tmp_path: Path):
    calls = []

    def extract(source, timestamp, output):
        calls.append((source, timestamp))
        output.write_bytes(b"jpeg-bytes")

    cache = SecondaryFrameCache(tmp_path, max_bytes=20, extractor=extract)
    first, hit, _ = cache.get("video", 2.0, Path("source.mp4"))
    assert first.is_file() and not hit
    assert cache.get("video", 2.0, Path("source.mp4"))[1]
    cache.get("video", 4.0, Path("source.mp4"))
    cache.get("video", 6.0, Path("source.mp4"))
    assert len(calls) == 3
    assert cache.stats()["bytes"] <= 20


def test_detector_routing_is_rule_based_and_skips_unsupported_paths():
    assert map_query("find the bicycle", "OBJECT").detector_enabled
    assert map_query("find a person swimming", "ACTION_TEMPORAL").detector_enabled is False
    assert map_query("gradient descent", "SPEECH").detector_enabled is False
    assert map_query("an outdoor scene", "SCENE").detector_enabled is False


def test_fusion_and_abstention_are_calibration_only():
    ranked = fuse([
        {"timestamp": 1, "secondary_clip_score": 0.2, "object_confidence": 0.9},
        {"timestamp": 2, "secondary_clip_score": 0.3, "object_confidence": 0.0},
    ], object_weight=0.75)
    assert ranked[0]["timestamp"] == 1
    calibration = [
        {"split":"calibration","expected_presence":True,"items":[{"combined_score":0.8}]},
        {"split":"calibration","expected_presence":False,"items":[{"combined_score":0.2}]},
    ]
    threshold = fit_abstention_threshold(calibration, "items")
    assert abstain(ranked, threshold)
    with pytest.raises(ValueError, match="calibration-only"):
        fit_abstention_threshold([{**calibration[0], "split":"heldout"}], "items")


def test_committed_report_preserves_frozen_splits_and_rejects_promotion():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((
        root / "ml/evaluation/reports/bounded-secondary-v1.json"
    ).read_text(encoding="utf-8"))
    parent = root / "ml/evaluation/reports/lightweight-pair-scorer-v1.json"
    natural = root / "ml/evaluation/natural_v2.json"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == report["parent_report_sha256"]
    assert hashlib.sha256(natural.read_bytes()).hexdigest() == report["natural_manifest_sha256"]
    calibration = {row["query_id"] for row in report["rows"] if row["split"] == "calibration"}
    heldout = {row["query_id"] for row in report["rows"] if row["split"] == "heldout"}
    assert calibration.isdisjoint(heldout)
    assert report["policy_selection"]["split"] == "calibration"
    assert report["fusion_selection"]["split"] == "calibration"
    assert report["gate"]["passed"] is False
    assert report["production_changed"] is False
    assert report["decision"]["outcome"] == "C_coarse_candidate_recall_is_dominant"
