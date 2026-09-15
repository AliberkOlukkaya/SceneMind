"""Cheap query-level evidence and deterministic calibration rules."""

from __future__ import annotations

import math

import numpy as np

from app.hybrid import tokens
from app.routing import resolve_mode

INTENT_WORDS = {
    "where", "when", "find", "show", "speaker", "narrator", "lecturer", "presenter",
    "say", "says", "mention", "mentions", "discuss", "discusses", "explain", "explains",
    "visible", "shown", "appear", "appears", "standing", "does", "while", "part", "section",
}


def content_tokens(query: str) -> list[str]:
    return [word for word in tokens(query) if word not in INTENT_WORDS and len(word) > 1]


def visual_features(results: list[dict]) -> dict[str, float]:
    scores = np.asarray([row["score"] for row in results], dtype=np.float64)
    if not len(scores):
        return {name: 0.0 for name in (
            "top1", "margin_1_2", "margin_1_5", "mean_5", "std_5",
            "entropy_strength", "temporal_support", "separated_support",
        )}
    top = scores[:5]
    probabilities = np.exp((top - top.max()) / 0.02)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    times = [row["timestamp"] for row in results[:10]]
    temporal = sum(abs(value - times[0]) <= 10 for value in times) / max(1, len(times))
    separated = 1
    anchors = [times[0]]
    for value in times[1:]:
        if all(abs(value - anchor) >= 15 for anchor in anchors):
            anchors.append(value)
            separated += 1
    return {
        "top1": float(top[0]),
        "margin_1_2": float(top[0] - top[min(1, len(top) - 1)]),
        "margin_1_5": float(top[0] - top[-1]),
        "mean_5": float(top.mean()),
        "std_5": float(top.std()),
        "entropy_strength": 1 - entropy / math.log(max(2, len(top))),
        "temporal_support": temporal,
        "separated_support": separated / max(1, len(times)),
    }


def speech_features(query: str, segments: list[dict], scores: list[float]) -> dict[str, float]:
    terms = content_tokens(query)
    documents = [set(tokens(segment["text"])) for segment in segments]
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    positive = [index for index in ranked if scores[index] > 0]
    top_index = positive[0] if positive else None
    all_words = set().union(*documents) if documents else set()
    covered = sum(term in all_words for term in set(terms))
    top_covered = (sum(term in documents[top_index] for term in set(terms))
                   if top_index is not None else 0)
    denominator = max(1, len(set(terms)))
    ordered_terms = list(dict.fromkeys(terms))
    bigrams = [f"{left} {right}" for left, right in zip(ordered_terms, ordered_terms[1:])]
    normalized_transcript = " ".join(tokens(" ".join(segment["text"] for segment in segments)))
    technical = {term for term in terms if len(term) >= 6}
    return {
        "top1": float(scores[top_index]) if top_index is not None else 0.0,
        "margin_1_2": (float(scores[positive[0]] - scores[positive[1]])
                       if len(positive) > 1 else float(scores[top_index]) if positive else 0.0),
        "query_coverage": covered / denominator,
        "top_segment_coverage": top_covered / denominator,
        "matching_segments": math.log1p(len(positive)),
        "evidence_concentration": (float(scores[top_index]) / max(1e-12, sum(scores[index]
                                    for index in positive)) if positive else 0.0),
        "phrase_overlap": float(any(bigram in normalized_transcript for bigram in bigrams)),
        "technical_coverage": (sum(term in all_words for term in technical) / len(technical)
                               if technical else covered / denominator),
    }


def candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return [math.inf]
    return [unique[0] - 1e-9, *[(a + b) / 2 for a, b in zip(unique, unique[1:])],
            unique[-1] + 1e-9]


def decision_metrics(rows: list[dict], decisions: list[bool]) -> dict[str, float | int]:
    positives = [index for index, row in enumerate(rows) if row["expected_presence"]]
    negatives = [index for index, row in enumerate(rows) if not row["expected_presence"]]
    accepted_positive = sum(decisions[index] for index in positives)
    false_accepts = sum(decisions[index] for index in negatives)
    accepted = sum(decisions)
    return {
        "positive_queries": len(positives), "negative_queries": len(negatives),
        "accepted_positive_rate": accepted_positive / max(1, len(positives)),
        "positive_false_abstention_rate": 1 - accepted_positive / max(1, len(positives)),
        "negative_false_accept_rate": false_accepts / max(1, len(negatives)),
        "negative_rejection_rate": 1 - false_accepts / max(1, len(negatives)),
        "accept_precision": accepted_positive / max(1, accepted),
        "coverage": accepted / max(1, len(rows)),
    }


def _quality(metrics: dict) -> tuple:
    far = metrics["negative_false_accept_rate"]
    abstain = metrics["positive_false_abstention_rate"]
    passes = far <= 0.15 and abstain <= 0.15
    return (not passes, max(far / 0.15, abstain / 0.15), abstain, far)


def select_rule(rows: list[dict], feature_prefix: str = "") -> dict:
    names = sorted(rows[0]["features"])
    if feature_prefix:
        names = [name for name in names if name.startswith(feature_prefix)]
    candidates = []
    for name in names:
        for threshold in candidate_thresholds([row["features"][name] for row in rows]):
            decisions = [row["features"][name] >= threshold for row in rows]
            candidates.append({"kind": "threshold", "feature": name, "threshold": threshold,
                               "metrics": decision_metrics(rows, decisions)})
    for first, second in ((names[i], names[j]) for i in range(len(names))
                          for j in range(i + 1, len(names))):
        first_values = candidate_thresholds([row["features"][first] for row in rows])
        second_values = candidate_thresholds([row["features"][second] for row in rows])
        for one in first_values:
            for two in second_values:
                decisions = [row["features"][first] >= one and row["features"][second] >= two
                             for row in rows]
                candidates.append({"kind": "and", "feature": first, "threshold": one,
                                   "feature_2": second, "threshold_2": two,
                                   "metrics": decision_metrics(rows, decisions)})
    winner = min(candidates, key=lambda candidate: (*_quality(candidate["metrics"]),
                                                     candidate["kind"] == "and",
                                                     candidate["feature"]))
    return {"winner": winner, "candidate_count": len(candidates)}


def apply_rule(features: dict[str, float], rule: dict) -> bool:
    accepted = features[rule["feature"]] >= rule["threshold"]
    if rule["kind"] == "and":
        accepted = accepted and features[rule["feature_2"]] >= rule["threshold_2"]
    return bool(accepted)


def path_decision(
    query: str,
    requested_mode: str,
    features: dict[str, float],
    rules: dict,
    enabled: bool = True,
) -> dict:
    """Exercise AUTO/override flow without changing the production API."""
    route = resolve_mode(requested_mode.casefold(), query, True)["route"].upper()
    accepted = apply_rule(features, rules[route]["winner"]) if enabled else True
    return {
        "selected_route": route,
        "accepted": accepted,
        "uncertain": not accepted,
        "reason": "experimental_path_rule" if enabled else "experiment_disabled",
    }


def distribution(rows: list[dict]) -> dict:
    output = {}
    for name in sorted(rows[0]["features"]):
        output[name] = {}
        for label, presence in (("positive", True), ("negative", False)):
            values = [row["features"][name] for row in rows
                      if row["expected_presence"] is presence]
            output[name][label] = {
                "min": min(values), "median": float(np.median(values)),
                "mean": float(np.mean(values)), "max": max(values),
            }
    return output
