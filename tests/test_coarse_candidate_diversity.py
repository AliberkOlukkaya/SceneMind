import hashlib
import json
from pathlib import Path

import numpy as np

from ml.experiments.coarse_candidate_diversity.selection import (
    Candidate,
    mmr,
    raw_pool,
    represented_intervals,
    scene_aware,
    scene_segments,
    temporal_nms,
)


def test_raw_pool_is_stable_and_bounded():
    rows = raw_pool(np.array([0.2, 0.7, 0.7, 0.1]), [0, 2, 4, 6], 3)
    assert [(row.index, row.timestamp) for row in rows] == [(1, 2), (2, 4), (0, 0)]


def test_temporal_nms_is_deterministic_and_handles_duplicates():
    rows = [
        Candidate(0, 25, 0.9), Candidate(1, 27, 0.8), Candidate(1, 27, 0.8),
        Candidate(2, 70, 0.7), Candidate(3, 110, 0.6),
    ]
    assert [row.timestamp for row in temporal_nms(rows, 3, 2)] == [25, 70, 110]
    assert [row.timestamp for row in temporal_nms(rows, 4, 1)] == [25, 27, 70, 110]
    jittered = [Candidate(0, 10.01, 0.9), Candidate(1, 12.012, 0.8)]
    assert [row.timestamp for row in temporal_nms(jittered, 2, 2)] == [10.01]


def test_mmr_prefers_embedding_diversity_and_preserves_first_candidate():
    rows = [Candidate(0, 0, 0.9), Candidate(1, 2, 0.89), Candidate(2, 10, 0.8)]
    vectors = np.array([[1, 0], [0.999, 0.001], [0, 1]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert [row.index for row in mmr(rows, vectors, 2, 0.5)] == [0, 2]


def test_scene_grouping_and_boundary_intervals():
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    segments = scene_segments(vectors, 0.5)
    rows = [Candidate(0, 0, 0.9), Candidate(1, 2, 0.8), Candidate(2, 9, 0.7)]
    assert segments == [0, 0, 1]
    assert [row.index for row in scene_aware(rows, segments, 5)] == [0, 2]
    assert represented_intervals([rows[0], rows[2]], 10, 2) == [
        {"timestamp": 0, "start": 0.0, "end": 2, "score": 0.9},
        {"timestamp": 9, "start": 7, "end": 10, "score": 0.7},
    ]


def test_committed_report_preserves_frozen_inputs_and_category_gate():
    root = Path(__file__).resolve().parents[1]
    path = root / "ml/evaluation/reports/coarse-candidate-diversity-v1.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    parent = root / "ml/evaluation/reports/bounded-secondary-v1.json"
    natural = root / "ml/evaluation/natural_v2.json"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == report["parent_report_sha256"]
    assert hashlib.sha256(natural.read_bytes()).hexdigest() == report["natural_manifest_sha256"]
    calibration = {row["query_id"] for row in report["rows"] if row["split"] == "calibration"}
    heldout = {row["query_id"] for row in report["rows"] if row["split"] == "heldout"}
    assert calibration.isdisjoint(heldout)
    assert report["selection"]["split"] == "calibration"
    assert report["heldout_tuning"] is False
    assert report["decision"]["outcome"].startswith("F_")
    assert report["gate"]["maximum_category_drop"]["pass"] is False
    assert report["gate"]["maximum_category_drop"]["drops"]["SPEECH"] > 0.05
    if report["gate"]["passed"]:
        assert report["second_frozen_run"]["required"] is True
    assert report["production_changed"] is False
