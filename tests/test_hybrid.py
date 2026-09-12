import pytest
from app.hybrid import bm25, fuse


def test_bm25_matches_terms_not_query_substrings():
    scores = bm25(
        "Where does the speaker explain learning rate?",
        ["The learning rate controls optimization.", "A bicycle rolls down a street."],
    )
    assert scores[0] > 0
    assert scores[1] == 0
    assert bm25("missing", ["learning rate"]) == [0]


def test_fusion_rewards_shared_evidence_and_deduplicates():
    visual = [
        {"thumbnail": "a", "timestamp": 0, "score": 0.8},
        {"thumbnail": "b", "timestamp": 5, "score": 0.7},
    ]
    speech = [
        {"thumbnail": "b", "timestamp": 5.2, "end": 6, "score": 4, "text": "car"},
        {"thumbnail": "b", "timestamp": 5.4, "end": 7, "score": 3, "text": "car"},
    ]
    result = fuse(visual, speech, 10)
    assert len(result) == 2
    assert result[0]["thumbnail"] == "b"
    assert result[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert result[0]["timestamp"] == 5.2
    assert result[0]["modality"] == "visual+speech"
