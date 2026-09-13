"""Presence features, calibration-only thresholds and transparent reranking."""

import math
import statistics

from ml.evaluation.metrics import retrieval_metrics
from ml.experiments.object_detector_branch.schema import DetectionEvidence, QueryObjectMapping


def object_features(
    mapping: QueryObjectMapping, detections: list[DetectionEvidence]
) -> dict[str, float]:
    if not mapping.detector_enabled:
        return {"object_confidence": 0.0, "object_area_ratio": 0.0, "matched_groups": 0.0}
    group_scores = []
    group_areas = []
    for group in mapping.required_class_groups:
        matches = [item for item in detections if item.detected_class in group]
        if matches:
            best = max(matches, key=lambda item: item.confidence)
            group_scores.append(best.confidence)
            group_areas.append(best.bbox_area_ratio)
        else:
            group_scores.append(0.0)
            group_areas.append(0.0)
    return {
        "object_confidence": min(group_scores),
        "object_area_ratio": min(group_areas),
        "matched_groups": float(sum(score > 0 for score in group_scores)),
    }


def enrich_candidates(row: dict, evidences: dict[str, list[DetectionEvidence]]) -> list[dict]:
    output = []
    for rank, candidate in enumerate(row["clip_candidates"], 1):
        features = object_features(row["mapping"], evidences.get(candidate["frame_id"], []))
        output.append({**candidate, **features, "clip_rank": rank})
    return output


def fit_presence_threshold(rows: list[dict], max_negative_far: float = 0.1) -> float:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("presence threshold fit requires calibration-only rows")
    triggered = [row for row in rows if row["mapping"].detector_enabled]
    negatives = [row for row in triggered if not row["expected_presence"]]
    positives = [row for row in triggered if row["expected_presence"]]
    if not negatives or not positives:
        raise ValueError("triggered positive and negative calibration rows required")
    maxima = sorted(
        (max((item["object_confidence"] for item in row["detector_candidates"]), default=0.0)
         for row in negatives),
        reverse=True,
    )
    allowed = math.floor(max_negative_far * len(negatives))
    return 1.0 if allowed >= len(maxima) else math.nextafter(maxima[allowed], math.inf)


def presence_candidates(row: dict, threshold: float) -> list[dict]:
    if not row["mapping"].detector_enabled:
        return list(row["clip_candidates"])
    return [
        item for item in row["detector_candidates"] if item["object_confidence"] >= threshold
    ]


def rerank_candidates(row: dict, detector_weight: float) -> list[dict]:
    if not 0 <= detector_weight <= 1:
        raise ValueError("detector weight must be between zero and one")
    candidates = row["detector_candidates"]
    if not row["mapping"].detector_enabled or not candidates:
        return list(row["clip_candidates"])
    clip_scores = [item["score"] for item in candidates]
    low, high = min(clip_scores), max(clip_scores)
    span = high - low
    enriched = []
    for item in candidates:
        normalized_clip = (item["score"] - low) / span if span else 1.0
        score = (1 - detector_weight) * normalized_clip + detector_weight * item[
            "object_confidence"
        ]
        enriched.append({**item, "combined_score": score})
    return sorted(enriched, key=lambda item: (-item["combined_score"], item["clip_rank"]))


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarize(rows: list[dict], candidate_field: str) -> dict[str, dict]:
    summary = {}
    for k in (1, 3, 5):
        measured = [
            (row, retrieval_metrics(
                [item["timestamp"] for item in row[candidate_field]][:k],
                row["relevant_intervals"],
                k,
            ))
            for row in rows
        ]
        positives = [metric for row, metric in measured if row["expected_presence"]]
        negatives = [metric for row, metric in measured if not row["expected_presence"]]
        summary[str(k)] = {
            "positive_queries": len(positives),
            "negative_queries": len(negatives),
            "recall": statistics.mean(metric["recall"] for metric in positives)
            if positives else None,
            "mrr": statistics.mean(metric["reciprocal_rank"] for metric in positives)
            if positives else None,
            "negative_false_accept_rate": statistics.mean(
                metric["returned"] > 0 for metric in negatives
            ) if negatives else None,
            "positive_false_abstention_rate": statistics.mean(
                metric["returned"] == 0 for metric in positives
            ) if positives else None,
        }
    return summary
