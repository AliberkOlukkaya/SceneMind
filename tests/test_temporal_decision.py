from copy import deepcopy

import pytest

from ml.evaluation.temporal_decision import choose_development, quality_gates, validate_pair


def fixture():
    query = dict(id="q", video_id="v", query="opens a door", category="HUMAN_ACTION",
                 difficulty="medium", intervals=[[2, 4]])
    baseline = dict(status="complete", manifest_sha256="same", arm="baseline", queries=[query],
                    metrics={"overall": {"r5": 0.7}, "combined_temporal": {"r5": 0.65},
                             "combined_static": {"r5": 0.9}})
    candidate = deepcopy(baseline)
    candidate.update(arm="candidate", temporal_config=dict(length=4, stride=2, frames=8),
                     videos={"v": {"indexing_seconds": 100}})
    candidate["metrics"] = {"overall": {"r5": 0.8}, "combined_temporal": {"r5": 0.75},
                             "combined_static": {"r5": 0.85}, "video_id:v": {"n": 8, "r5": 0.6}}
    return baseline, candidate


def test_selection_tie_break_and_finite_grid():
    baseline, four = fixture()
    eight = deepcopy(four)
    eight["temporal_config"] = dict(length=8, stride=4, frames=8)
    eight["videos"]["v"]["indexing_seconds"] = 50
    assert choose_development(baseline, [four, eight])["length"] == 8
    four["metrics"]["combined_temporal"]["r5"] = 0.8
    assert choose_development(baseline, [four, eight])["length"] == 4
    with pytest.raises(ValueError, match="predeclared"):
        choose_development(baseline, [four, four])


def test_comparison_rejects_partial_or_changed_truth():
    baseline, candidate = fixture()
    candidate["queries"][0]["intervals"] = [[1, 5]]
    with pytest.raises(ValueError, match="ground truth"):
        validate_pair(baseline, candidate)
    baseline, candidate = fixture()
    candidate["status"] = "running"
    with pytest.raises(ValueError, match="Incomplete"):
        validate_pair(baseline, candidate)


def test_quality_gates_exact_boundaries_and_catastrophic_source():
    baseline, candidate = fixture()
    gates = quality_gates(baseline, candidate)
    assert all(value for key, value in gates.items() if key != "catastrophic_videos")
    candidate["metrics"]["video_id:v"]["r5"] = 0.599
    gates = quality_gates(baseline, candidate)
    assert not gates["no_catastrophic_video"] and gates["catastrophic_videos"] == ["v"]
