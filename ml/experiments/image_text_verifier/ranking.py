"""Pure reranking, calibration and metric functions for verifier experiments."""

import math
import statistics
from collections import defaultdict

from ml.evaluation.metrics import retrieval_metrics


def rerank(candidates: list[dict], scores: list[float]) -> list[dict]:
    if len(candidates) != len(scores):
        raise ValueError("candidate and verifier score counts differ")
    enriched = [
        {**candidate, "clip_rank": rank, "verifier_score": score}
        for rank, (candidate, score) in enumerate(zip(candidates, scores, strict=True), 1)
    ]
    return sorted(
        enriched,
        key=lambda candidate: (-candidate["verifier_score"], candidate["clip_rank"]),
    )


def fit_threshold(rows: list[dict], max_negative_far: float = 0.1) -> float:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("threshold fit requires calibration-only rows")
    negatives = [row for row in rows if not row["expected_presence"]]
    positives = [row for row in rows if row["expected_presence"]]
    if not negatives or not positives:
        raise ValueError("positive and negative calibration rows required")
    allowed = math.floor(max_negative_far * len(negatives))
    maxima = sorted(
        (max((item["verifier_score"] for item in row["reranked"]), default=-math.inf)
         for row in negatives),
        reverse=True,
    )
    if allowed >= len(maxima):
        return -math.inf
    return math.nextafter(maxima[allowed], math.inf)


def apply_threshold(row: dict, threshold: float) -> list[dict]:
    return [item for item in row["reranked"] if item["verifier_score"] >= threshold]


def summarize(rows: list[dict], candidate_field: str, threshold=None, score_key="verifier_score") -> dict:
    output = {}
    for k in (1, 3, 5):
        measured = []
        for row in rows:
            candidates = row[candidate_field]
            if threshold is not None:
                candidates = [item for item in candidates if item[score_key] >= threshold]
            metric = retrieval_metrics(
                [item["timestamp"] for item in candidates[:k]], row["relevant_intervals"], k
            )
            measured.append((row, metric))
        positives = [metric for row, metric in measured if row["expected_presence"]]
        negatives = [metric for row, metric in measured if not row["expected_presence"]]
        output[str(k)] = {
            "positive_queries": len(positives),
            "negative_queries": len(negatives),
            "recall": statistics.mean(metric["recall"] for metric in positives)
            if positives else None,
            "mrr": statistics.mean(metric["reciprocal_rank"] for metric in positives)
            if positives else None,
            "precision": statistics.mean(metric["precision"] for metric in positives)
            if positives else None,
            "negative_false_accept_rate": statistics.mean(
                metric["returned"] > 0 for metric in negatives
            ) if negatives else None,
            "positive_false_abstention_rate": statistics.mean(
                metric["returned"] == 0 for metric in positives
            ) if positives else None,
        }
    return output


def analyze(rows: list[dict], threshold: float, clip_threshold: float) -> dict:
    heldout = [row for row in rows if row["split"] == "heldout"]
    by_category = defaultdict(list)
    for row in heldout:
        by_category[row["query_type"]].append(row)
    return {
        "clip": summarize(heldout, "clip_candidates"),
        "clip_calibrated": summarize(
            heldout, "clip_candidates", clip_threshold, score_key="score"
        ),
        "verifier_reranked": summarize(heldout, "reranked"),
        "verifier_calibrated": summarize(heldout, "reranked", threshold),
        "by_category": {
            category: {
                "clip": summarize(category_rows, "clip_candidates"),
                "clip_calibrated": summarize(
                    category_rows, "clip_candidates", clip_threshold, score_key="score"
                ),
                "verifier_reranked": summarize(category_rows, "reranked"),
                "verifier_calibrated": summarize(category_rows, "reranked", threshold),
            }
            for category, category_rows in by_category.items()
        },
    }
