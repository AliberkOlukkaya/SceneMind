import pytest

from ml.evaluation.metrics import retrieval_metrics


def test_interval_metrics_penalize_duplicate_hits():
    result = retrieval_metrics([1, 2, 8], [(0, 3), (7, 9)], 3)
    assert result["recall"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["reciprocal_rank"] == 1


def test_mrr_and_misses():
    result = retrieval_metrics([4, 8], [(7, 9)], 3)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["reciprocal_rank"] == 0.5
    assert retrieval_metrics([], [(7, 9)], 3)["recall"] == 0
    assert retrieval_metrics([4], [], 1)["recall"] is None


def test_half_open_intervals_and_invalid_labels():
    assert retrieval_metrics([5], [(0, 5)], 1)["recall"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics([], [(0, 4), (3, 7)], 1)
