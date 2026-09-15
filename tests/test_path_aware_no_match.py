import copy
import json
from pathlib import Path

import pytest

from ml.experiments.path_aware_no_match.build_manifest import MANIFEST, validate_manifest
from ml.experiments.path_aware_no_match.features import (
    apply_rule,
    path_decision,
    select_rule,
    speech_features,
    visual_features,
)


def _rule(feature, threshold):
    return {"winner": {"kind": "threshold", "feature": feature, "threshold": threshold}}


def test_visual_and_speech_evidence_are_path_specific():
    visual = visual_features([
        {"score": 0.8, "timestamp": 0}, {"score": 0.6, "timestamp": 20},
        {"score": 0.5, "timestamp": 40},
    ])
    speech = speech_features(
        "Where does he mention convolution?",
        [{"text": "This section explains circuits."}],
        [0.0],
    )
    assert apply_rule({"v_top1": visual["top1"]}, _rule("v_top1", 0.7)["winner"])
    assert not apply_rule(
        {"s_query_coverage": speech["query_coverage"]},
        _rule("s_query_coverage", 0.5)["winner"],
    )


def test_hybrid_rule_can_require_both_evidence_systems():
    rule = {"kind": "and", "feature": "state_visual", "threshold": 0.5,
            "feature_2": "state_speech", "threshold_2": 0.5}
    assert apply_rule({"state_visual": 1, "state_speech": 1}, rule)
    assert not apply_rule({"state_visual": 1, "state_speech": 0}, rule)


def test_auto_then_path_decision_and_explicit_override():
    rules = {
        "VISUAL": _rule("v", 0.5), "SPEECH": _rule("s", 0.5),
        "HYBRID": _rule("h", 0.5),
    }
    features = {"v": 1.0, "s": 0.0, "h": 0.0}
    auto = path_decision("Find the red bicycle.", "auto", features, rules)
    assert auto == {"selected_route": "VISUAL", "accepted": True, "uncertain": False,
                    "reason": "experimental_path_rule"}
    speech = path_decision("Find the red bicycle.", "speech", features, rules)
    assert speech["selected_route"] == "SPEECH"
    assert speech["uncertain"] is True


def test_disabled_experiment_preserves_ranked_result_behavior():
    rules = {route: _rule("score", 100) for route in ("VISUAL", "SPEECH", "HYBRID")}
    result = path_decision("Find the red bicycle.", "visual", {"score": 0}, rules, False)
    assert result["accepted"] is True
    assert result["reason"] == "experiment_disabled"


def test_rule_selection_is_deterministic_and_preserves_labeled_examples():
    rows = [
        {"expected_presence": True, "features": {"score": 0.9}},
        {"expected_presence": True, "features": {"score": 0.8}},
        {"expected_presence": False, "features": {"score": 0.2}},
        {"expected_presence": False, "features": {"score": 0.1}},
    ]
    assert select_rule(rows) == select_rule(rows)
    winner = select_rule(rows)["winner"]
    assert all(apply_rule(row["features"], winner) == row["expected_presence"] for row in rows)


def test_frozen_manifest_is_balanced_and_rejects_source_leakage(tmp_path):
    validation = validate_manifest()
    assert len(validation["source_groups"]["calibration"]) == 3
    assert len(validation["source_groups"]["heldout"]) == 3
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = copy.deepcopy(data)
    changed["sources"][3]["source_group"] = changed["sources"][0]["source_group"]
    path = tmp_path / "leaked.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap|leakage"):
        validate_manifest(path)


def test_committed_report_records_failed_gate_without_production_promotion():
    report_path = Path(__file__).resolve().parents[1] / (
        "ml/evaluation/reports/path-aware-no-match-v1.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["manifest_sha256"] == validate_manifest()["manifest_sha256"]
    assert report["gate"]["passed"] is False
    assert report["second_frozen_run"]["executed"] is False
    assert report["production"] == {"behavior_modified": False, "feature_flag_added": False}
    assert report["decision"]["outcome"] == "E"
