import hashlib
import json
from pathlib import Path

import pytest

from ml.experiments.small_object_ablation.metrics import (
    conditional_detector_summary,
    sampling_summary,
)
from ml.experiments.small_object_ablation.schema import VisibilityManifest

ROOT = Path(__file__).resolve().parents[1]


def load_manifest():
    payload = json.loads((ROOT / "ml/experiments/small_object_ablation/visibility_v1.json").read_text(encoding="utf-8"))
    return VisibilityManifest.model_validate(payload)


def test_visibility_manifest_is_source_disjoint_and_parent_is_frozen():
    manifest = load_manifest()
    parent = ROOT / "ml/evaluation/natural_v2.json"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == manifest.parent_benchmark_sha256
    calibration = {row.source_group for row in manifest.events if row.split == "calibration"}
    heldout = {row.source_group for row in manifest.events if row.split == "heldout"}
    assert calibration.isdisjoint(heldout)


def test_sampling_recall_uses_actual_frame_visibility():
    result = sampling_summary(load_manifest())
    assert result["5"]["evidence_recall"] == 0.25
    assert result["2"]["evidence_recall"] == 1.0
    assert result["1"]["evidence_recall"] == 1.0
    assert result["5"]["evidence_classes"]["B_only_visible_between_sampled_frames"] == 3


def test_invalid_interval_overlap_cannot_claim_visible_evidence():
    payload = load_manifest().model_dump()
    event = payload["events"][0]
    event["sampling"][0]["evidence_class"] = "A_visible_in_sampled_frame"
    with pytest.raises(ValueError, match="sampled-frame visibility"):
        VisibilityManifest.model_validate(payload)


def test_conditional_detector_recall_excludes_absent_and_uncertain_frames():
    rows = [
        {"visibility":"visible","detected":True,"confidence":0.8,"bbox_area_ratio":0.01},
        {"visibility":"visible","detected":False,"confidence":0.0,"bbox_area_ratio":0.0},
        {"visibility":"not_visible","detected":False,"confidence":0.0,"bbox_area_ratio":0.0},
        {"visibility":"too_small_to_judge","detected":True,"confidence":0.4,"bbox_area_ratio":0.001},
    ]
    summary = conditional_detector_summary(rows)
    assert summary["verified_visible_frames"] == 2
    assert summary["visibility_conditioned_recall"] == pytest.approx(0.5)


def test_committed_report_keeps_production_and_heldout_frozen():
    report = json.loads((
        ROOT / "ml/evaluation/reports/small-object-ablation-v1.json"
    ).read_text(encoding="utf-8"))
    assert report["production_changed"] is False
    assert report["heldout_tuning"] is False
    assert report["sampling_evidence"]["all"]["5"]["evidence_recall"] == 0.25
    assert report["detectors"]["yolox-416"]["recall_at_confidence"]["0.01"]["recall"] == pytest.approx(2 / 3)
    assert report["detectors"]["yolox-768"]["recall_at_confidence"]["0.01"]["recall"] == pytest.approx(5 / 6)
    assert all(
        gate["pass"]
        for config in report["higher_resolution_gates"].values()
        for gate in config.values()
    )
    assert all(
        run["quality_rows_identical"] for run in report["second_frozen_runs"].values()
    )
