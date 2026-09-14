"""Explainable CLIP/object fusion and calibration-only abstention."""

import math


def fuse(candidates: list[dict], object_weight: float) -> list[dict]:
    if not 0 <= object_weight <= 1:
        raise ValueError("object weight must be between zero and one")
    if not candidates:
        return []
    output = []
    for rank, row in enumerate(candidates, 1):
        combined = row["secondary_clip_score"] + object_weight * row.get(
            "object_confidence", 0.0
        )
        output.append({**row, "secondary_clip_rank": rank, "combined_score": combined})
    return sorted(output, key=lambda row: (-row["combined_score"], row["secondary_clip_rank"]))


def fit_abstention_threshold(
    rows: list[dict], candidate_field: str, max_negative_far: float = 0.1
) -> float:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("threshold fitting requires calibration-only rows")
    positives = [row for row in rows if row["expected_presence"]]
    negatives = [row for row in rows if not row["expected_presence"]]
    if not positives or not negatives:
        raise ValueError("positive and negative calibration rows are required")
    maxima = sorted(
        (
            max((item["combined_score"] for item in row[candidate_field]), default=0.0)
            for row in negatives
        ),
        reverse=True,
    )
    allowed = math.floor(max_negative_far * len(negatives))
    return 1.0 if allowed >= len(maxima) else math.nextafter(maxima[allowed], math.inf)


def abstain(candidates: list[dict], threshold: float) -> list[dict]:
    return [row for row in candidates if row["combined_score"] >= threshold]
