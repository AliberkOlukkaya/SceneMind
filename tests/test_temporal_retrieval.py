"""Candidate mechanics use mocked inference; no weights/media/network needed."""

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation.temporal_protocol import AUDITS, canonical_source, validate_manifest
from ml.evaluation.temporal_retrieval import (
    aggregate,
    first_rank,
    load_index,
    normalize,
    save_index,
    search,
    windows,
)


def test_windows_boundaries_short_and_long():
    clips = windows(10, 4, 2)
    assert [(w.start, w.end) for w in clips] == [(0, 4), (2, 6), (4, 8), (6, 10)]
    short = windows(0.1, 4, 2)
    assert len(short) == 1 and len(short[0].samples) == 8
    assert 0 < short[0].samples[0] < short[0].samples[-1] < 0.1
    clips = windows(3601, 8, 4)
    assert clips[-1].end == 3601
    assert all(w.start < min(w.samples) < max(w.samples) < w.end for w in clips)
    assert all(a.end >= b.start for a, b in zip(clips, clips[1:]))
    assert clips[-2].end < 3601


@pytest.mark.parametrize("args", [(0, 4, 2), (10, 0, 2), (10, 4, 5),
                                  (float("nan"), 4, 2), (10, 4, 2, 1)])
def test_invalid_windows(args):
    with pytest.raises(ValueError):
        windows(*args)


def test_normalization():
    assert np.allclose(normalize([[3, 4]]), [[0.6, 0.8]])
    for invalid in ([[0, 0]], [[1, float("nan")]], [1, 2]):
        with pytest.raises(ValueError):
            normalize(invalid)


class FakeEncoder:
    def text(self, query):
        assert query == "person sits down"
        return np.ones((1, 512), dtype=np.float32)

    def score(self, vectors, patches, text):
        assert vectors.shape == (2, 512) and patches.shape == (2, 49, 512)
        return np.asarray([0.1, 0.9])


def test_persistence_reload_identity_query_mapping(tmp_path):
    path = tmp_path / "candidate.npz"
    clips = windows(6, 4, 2)
    vectors = np.ones((2, 512), dtype=np.float32)
    patches = np.zeros((2, 49, 512), dtype=np.float32)
    identity = {"media_sha256": "abc", "duration": 4, "stride": 2}
    save_index(path, clips, vectors, patches, identity=identity,
               actual_samples=[list(w.samples) for w in clips])
    loaded, v, p, metadata = load_index(path, identity=identity)
    assert loaded == clips and metadata["identity"] == identity
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
    result = search(path, "person sits down", FakeEncoder(), identity=identity)
    assert [r["timestamp"] for r in result] == [4, 2]
    assert result[0]["start"] == 2 and result[0]["end"] == 6
    assert not path.with_suffix(".tmp").exists()
    with pytest.raises(ValueError, match="identity"):
        load_index(path, identity={"media_sha256": "different"})


def test_multiple_intervals_midpoint_not_overlap():
    results = [{"timestamp": 4, "start": 0, "end": 8}, {"timestamp": 12}]
    assert first_rank(results, [[1, 2], [11, 13]]) == 2
    assert first_rank(results, [[1, 2]]) is None
    assert first_rank(results, [[4, 4]]) == 1
    assert aggregate([1, 3, 7, None]) == dict(n=4, r1=0.25, r3=0.5,
                                            r5=0.5, r10=0.75, mrr5=1 / 3)


def test_production_does_not_import_candidate():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "backend/app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "temporal_retrieval" not in (node.module or "")
            elif isinstance(node, ast.Import):
                assert all("temporal_retrieval" not in name.name for name in node.names)


def valid_manifest():
    sources = [dict(source_id=str(i), source_url=f"https://commons.wikimedia.org/wiki/File:New_{i}",
                    media_sha256=str(i), duration_seconds=1800, license="CC BY",
                    provenance_review="reviewed source provenance", full_timeline_review="review log")
               for i in range(5)]
    categories = ["STATIC_OBJECT", "SCENE", "HUMAN_ACTION", "HUMAN_ACTION", "OBJECT_INTERACTION",
                  "OBJECT_INTERACTION", "TEMPORAL_EVENT", "SMALL_OBJECT_DETAIL",
                  "MULTIPLE_SIMILAR_MOMENTS", "STATIC_OBJECT", "SCENE", "HUMAN_ACTION"]
    queries = [dict(id=f"{i}-{j}", video_id=str(i), query=f"query {j}", difficulty="medium",
                    category=category, intervals=[[1, 3], [20, 24]],
                    audit={**dict.fromkeys(AUDITS, True), "reviewer": "unit-test fixture",
                           "evidence_review": "not actual benchmark truth"})
               for i in range(5) for j, category in enumerate(categories)]
    return dict(split="validation", sources=sources, queries=queries)


def test_source_aliases_and_leakage():
    manifest = valid_manifest()
    assert validate_manifest(manifest)
    alias = deepcopy(manifest["sources"][0])
    alias["source_url"] = "https://commons.wikimedia.org/wiki/File:New%200?tracking=1"
    alias["media_sha256"] = "different_transcode"
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(manifest, [alias])
    assert canonical_source("https://youtu.be/ABC") == canonical_source("https://youtube.com/watch?v=ABC")
    assert canonical_source("https://youtube.com/watch?v=ABC") != canonical_source("https://youtube.com/watch?v=XYZ")


def test_mirrored_source_credit_cannot_bypass_exclusions():
    manifest = valid_manifest()
    manifest["sources"][0]["source_aliases"] = ["https://youtu.be/ORIGINAL"]
    excluded = [{"source_url": "https://youtube.com/watch?v=ORIGINAL"}]
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(manifest, excluded)
    manifest["sources"][1]["source_aliases"] = ["https://youtube.com/watch?v=ORIGINAL"]
    with pytest.raises(ValueError, match="aliases"):
        validate_manifest(manifest)


def test_decoder_exposes_actual_positions_and_fails_on_missing_frame():
    import cv2

    from ml.evaluation.temporal_retrieval import decode_window

    class Capture:
        number = 0
        fail = False

        def get(self, property_id):
            return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: 10,
                    cv2.CAP_PROP_POS_MSEC: self.number * 100}[property_id]

        def set(self, property_id, number):
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            self.number = number

        def read(self):
            return not self.fail, np.zeros((8, 8, 3), dtype=np.uint8)

    capture = Capture()
    frames, actual = decode_window(capture, windows(1, 4, 2)[0])
    assert len(frames) == len(actual) == 8
    assert actual[0] == 0.1 and actual[-1] == 0.9
    capture.fail = True
    with pytest.raises(ValueError, match="decode failed"):
        decode_window(capture, windows(1, 4, 2)[0])


@pytest.mark.parametrize("issue", ["audit", "interval", "long", "count", "duplicate"])
def test_incomplete_or_invalid_truth_refused(issue):
    manifest = valid_manifest()
    if issue == "audit":
        manifest["queries"][0]["audit"]["all_occurrences"] = False
    elif issue == "interval":
        manifest["queries"][0]["intervals"] = [[1800, 1801]]
    elif issue == "long":
        for source in manifest["sources"]:
            source["duration_seconds"] = 1799
    elif issue == "count":
        manifest["queries"].pop()
    else:
        manifest["sources"][1]["media_sha256"] = "0"
    with pytest.raises(ValueError):
        validate_manifest(manifest)
