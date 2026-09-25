import json

import pytest

from app.ocr_evidence import (
    OCREvidence,
    OCRFrame,
    load_evidence,
    normalize_ocr_text,
    save_evidence,
    search_evidence,
    source_fingerprint,
    suppress_duplicates,
)


def test_normalization_retains_command_and_error_punctuation():
    assert normalize_ocr_text("  npm\tinstall  --save\nError: E404  ") == "npm install --save Error: E404"


def test_adjacent_duplicate_frames_become_timestamped_evidence():
    frames = [
        OCRFrame("000001", 0.0, "npm install fastapi", 0.90),
        OCRFrame("000002", 5.0, "npm install fastap1", 0.80),
        OCRFrame("000003", 10.0, "Build complete", 0.95),
    ]
    result = suppress_duplicates(frames, similarity=0.80)
    assert len(result) == 2
    assert (result[0].start_seconds, result[0].end_seconds) == (0.0, 5.0)
    assert result[0].frame_ids == ("000001", "000002")
    assert result[0].confidence == pytest.approx(0.85)


def test_nonadjacent_repeat_is_not_suppressed():
    frames = [OCRFrame("a", 0, "Settings"), OCRFrame("b", 11, "Settings")]
    assert len(suppress_duplicates(frames)) == 2


def test_bm25_search_returns_matching_timestamp():
    evidence = [
        OCREvidence("one", 5, 10, "npm install fastapi", 0.9, ("a",)),
        OCREvidence("two", 15, 15, "Welcome to the dashboard", 0.8, ("b",)),
    ]
    result = search_evidence("What command is shown? npm install fastapi", evidence)
    assert result[0]["evidence_id"] == "one"
    assert result[0]["start_seconds"] == 5


def test_persistence_rejects_stale_source(tmp_path):
    frames = [OCRFrame("a", 0, "Slide title")]
    fingerprint = source_fingerprint(frames, "mock-1", {"width": 480})
    evidence = suppress_duplicates(frames)
    path = tmp_path / "ocr.json"
    save_evidence(path, evidence, engine="mock-1", configuration={"width": 480}, fingerprint=fingerprint)
    assert load_evidence(path, expected_fingerprint=fingerprint) == evidence
    with pytest.raises(ValueError, match="rebuild"):
        load_evidence(path, expected_fingerprint="stale")
    assert json.loads(path.read_text(encoding="utf-8"))["engine"] == "mock-1"
