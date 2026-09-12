import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from ml.evaluation.natural_v2_report import accepted_results, aggregate
from ml.evaluation.regression_natural_v2 import check
from ml.evaluation.schema_v2 import Benchmark, QueryType, load_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_manifest_has_complete_taxonomy_and_is_source_disjoint():
    benchmark = load_benchmark(ROOT / "ml/evaluation/natural_v2.json")
    heldout = [video for video in benchmark.videos if video.split == "heldout"]
    calibration = [video for video in benchmark.videos if video.split == "calibration"]

    assert {query.query_type for video in heldout for query in video.queries} == set(QueryType)
    assert {video.source_group for video in heldout}.isdisjoint(
        video.source_group for video in calibration
    )
    assert all(query.annotation_version == "2.0.0" for video in benchmark.videos for query in video.queries)


def test_schema_rejects_source_group_leakage():
    payload = json.loads((ROOT / "ml/evaluation/natural_v2.json").read_text(encoding="utf-8"))
    heldout = next(video for video in payload["videos"] if video["split"] == "heldout")
    calibration = next(video for video in payload["videos"] if video["split"] == "calibration")
    heldout["source_group"] = calibration["source_group"]

    with pytest.raises(ValidationError, match="source_group leakage"):
        Benchmark.model_validate(payload)


def _row(split, mode, present, scores, query_type="OBJECT", evidence=None):
    return {
        "split": split,
        "mode": mode,
        "expected_presence": present,
        "relevant_intervals": [(0.0, 2.0)] if present else [],
        "results": [
            {
                "timestamp": float(index),
                "score": score,
                "evidence": evidence or {},
            }
            for index, score in enumerate(scores)
        ],
        "query_type": query_type,
        "warm_median_seconds": 0.01,
    }


def test_aggregate_fits_only_calibration_visual_and_reports_paths():
    rows = [
        _row("calibration", "visual", True, [0.8]),
        _row("calibration", "visual", False, [0.4]),
        _row("heldout", "visual", True, [0.6]),
        _row("heldout", "visual", False, [0.3]),
        _row("heldout", "speech", False, []),
    ]
    result = aggregate({"queries": rows})

    assert result["threshold"] == math.nextafter(0.4, math.inf)
    assert result["by_path"]["visual"]["1"]["calibrated"]["recall"] == 1
    assert result["by_path"]["visual"]["1"]["calibrated"]["negative_false_accept_rate"] == 0
    assert result["by_path"]["speech"]["1"]["raw"]["abstention_rate"] == 1


def test_hybrid_calibration_keeps_speech_evidence_and_filters_weak_visual():
    row = _row(
        "heldout",
        "hybrid",
        True,
        [0.03, 0.02],
    )
    row["results"][0]["evidence"] = {"speech": {"score": 1.0}}
    row["results"][1]["evidence"] = {"visual": {"score": 0.2}}

    assert accepted_results(row, 0.5) == [row["results"][0]]


def test_v2_regression_gate_detects_recall_drop():
    baseline = {
        **{key: "same" for key in (
            "schema_version", "benchmark_id", "manifest_sha256", "visual_model",
            "visual_revision", "speech_model", "sampling_interval",
        )},
        "analysis": {"by_path": {"visual": {}}},
    }
    metrics = {
        "recall": 0.8,
        "mrr": 0.8,
        "negative_false_accept_rate": 0.1,
        "positive_false_abstention_rate": 0.1,
    }
    baseline["analysis"]["by_path"]["visual"] = {
        str(k): {state: dict(metrics) for state in ("raw", "calibrated")} for k in (1, 3, 5)
    }
    current = json.loads(json.dumps(baseline))
    current["analysis"]["by_path"]["visual"]["5"]["raw"]["recall"] = 0.7

    assert check(current, baseline) == ["visual raw@5 recall: 0.8 -> 0.7"]
