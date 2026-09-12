"""Pure ranking, threshold, metric and slice helpers."""

import math
import statistics
from collections import defaultdict

from ml.evaluation.metrics import retrieval_metrics


def rerank(candidates: list[dict], scores: list[float], score_key: str) -> list[dict]:
    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    enriched = [
        {**candidate, "clip_rank": rank, score_key: float(score)}
        for rank, (candidate, score) in enumerate(zip(candidates, scores, strict=True), 1)
    ]
    return sorted(enriched, key=lambda item: (-item[score_key], item["clip_rank"]))


def fit_threshold(
    rows: list[dict], candidate_field: str, score_key: str, max_negative_far: float = 0.1
) -> float:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("threshold fit requires calibration-only rows")
    negatives = [row for row in rows if not row["expected_presence"]]
    positives = [row for row in rows if row["expected_presence"]]
    if not negatives or not positives:
        raise ValueError("positive and negative calibration rows required")
    allowed = math.floor(max_negative_far * len(negatives))
    maxima = sorted(
        (
            max((item[score_key] for item in row[candidate_field]), default=-math.inf)
            for row in negatives
        ),
        reverse=True,
    )
    return -math.inf if allowed >= len(maxima) else math.nextafter(maxima[allowed], math.inf)


def summarize(
    rows: list[dict], candidate_field: str, threshold: float | None = None,
    score_key: str | None = None
) -> dict:
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
            "recall": statistics.mean(metric["recall"] for metric in positives) if positives else None,
            "mrr": statistics.mean(metric["reciprocal_rank"] for metric in positives) if positives else None,
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


def slices(rows: list[dict], systems: dict[str, tuple[str, float | None, str | None]]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_type"].lower()].append(row)
        if row["query_id"] in {
            "street-object-bicycle", "throw-object-ball", "throw-compositional-holding"
        }:
            groups["small_object"].append(row)
        if row["query_type"] == "COMPOSITIONAL":
            groups["relationship_compositional"].append(row)
        if row["query_type"] == "ACTION_TEMPORAL":
            groups["temporal_action"].append(row)
    return {
        group: {
            name: summarize(group_rows, field, threshold, score_key)["5"]
            for name, (field, threshold, score_key) in systems.items()
        }
        for group, group_rows in groups.items()
    }
