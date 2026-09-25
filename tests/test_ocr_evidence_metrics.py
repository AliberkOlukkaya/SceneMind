from ml.experiments.ocr_evidence_v1.metrics import (
    best_match,
    character_accuracy,
    edit_distance,
    timestamp_error,
)
from ml.experiments.ocr_evidence_v1.run_experiment import score


def test_edit_distance_and_character_accuracy():
    assert edit_distance("kitten", "sitting") == 3
    assert character_accuracy([("Error", "Error"), ("Value", "Valuf")]) == 0.9


def test_best_match_can_join_split_ocr_lines():
    value, score = best_match("npm install fastapi", ["npm install", "fastapi"])
    assert value == "npm install fastapi"
    assert score == 1


def test_timestamp_error_is_zero_inside_interval():
    intervals = [[5, 10], [20, 25]]
    assert timestamp_error(7, intervals) == 0
    assert timestamp_error(17, intervals) == 3


def test_scoring_separates_sampling_miss_from_visible_ocr_quality():
    manifest = {
        "sources": [
            {
                "source_id": "validation-source",
                "split": "validation",
                "duration_seconds": 10,
                "production_frames": [{"frame_id": "a", "timestamp": 0, "path": "unused"}],
            }
        ],
        "targets": [
            {
                "target_id": "visible",
                "source_id": "validation-source",
                "query": "npm install",
                "reference_text": "npm install",
                "sampled_frame_ids": ["a"],
                "visible_intervals_seconds": [[0, 1]],
            },
            {
                "target_id": "missed-between-frames",
                "source_id": "validation-source",
                "query": "transient error",
                "reference_text": "transient error",
                "sampled_frame_ids": [],
                "visible_intervals_seconds": [[2, 3]],
            },
        ],
    }
    raw = {
        "engine": "mock",
        "split": "validation",
        "sources": [
            {
                "source_id": "validation-source",
                "frames": [
                    {
                        "frame_id": "a",
                        "timestamp": 0,
                        "text": "npm install",
                        "lines": [{"text": "npm install", "confidence": 1.0}],
                    }
                ],
                "latencies_ms": [10],
                "latency_ms": {"median": 10, "p95": 10, "total": 10},
                "peak_rss_delta_bytes": 1,
            }
        ],
    }

    metrics = score(manifest, raw)

    assert metrics["sampling_coverage"] == 0.5
    assert metrics["text_detection_recall_conditional_visible"] == 1.0
    assert metrics["text_accuracy_conditional_visible"] == 1.0
    assert metrics["retrieval_recall_all_targets"]["at_1"] == 0.5
    assert metrics["retrieval_recall_conditional_visible"]["at_1"] == 1.0
