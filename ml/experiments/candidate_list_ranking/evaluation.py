"""Ranking, abstention, oracle, and routing metrics."""

import statistics


def relevant(timestamp: float, intervals: list[list[float]]) -> bool:
    return any(start <= timestamp < end for start, end in intervals)


def rank_metrics(rows: list[dict], field: str) -> dict:
    positives = [row for row in rows if row["expected_presence"]]
    result = {}
    for depth in (1, 3, 5):
        reciprocal = []
        recalls = []
        for row in positives:
            ranks = [rank for rank, item in enumerate(row[field][:depth], 1)
                     if relevant(item["timestamp"], row["relevant_intervals"])]
            matched = {
                index for item in row[field][:depth]
                for index, (start, end) in enumerate(row["relevant_intervals"])
                if start <= item["timestamp"] < end
            }
            recalls.append(len(matched) / len(row["relevant_intervals"]))
            reciprocal.append(1 / ranks[0] if ranks else 0.0)
        result[str(depth)] = {
            "positive_queries": len(positives), "recall": statistics.mean(recalls),
            "mrr": statistics.mean(reciprocal),
        }
    return result


def oracle(rows: list[dict], field: str) -> dict:
    positives = [row for row in rows if row["expected_presence"]]
    return {
        str(depth): statistics.mean(
            len({index for item in row[field][:depth]
                 for index, (start, end) in enumerate(row["relevant_intervals"])
                 if start <= item["timestamp"] < end}) / len(row["relevant_intervals"])
            for row in positives
        )
        for depth in (5, 20, 50)
    }


def no_match_metrics(rows: list[dict], threshold: float) -> dict:
    accepted = [row["match_probability"] >= threshold for row in rows]
    negatives = [i for i, row in enumerate(rows) if not row["expected_presence"]]
    positives = [i for i, row in enumerate(rows) if row["expected_presence"]]
    true_accepts = sum(accepted[i] for i in positives)
    all_accepts = sum(accepted)
    return {
        "queries": len(rows), "positive_queries": len(positives),
        "negative_queries": len(negatives),
        "false_accept_rate": (sum(accepted[i] for i in negatives) / len(negatives)
                              if negatives else None),
        "positive_false_abstention_rate": (sum(not accepted[i] for i in positives) / len(positives)
                                           if positives else None),
        "accepted": all_accepts, "rejected": len(rows) - all_accepts,
        "accepted_precision": true_accepts / all_accepts if all_accepts else None,
    }


def classify_failures(rows: list[dict], ranked_field: str, threshold: float) -> dict:
    buckets = {name: [] for name in (
        "retrieval_failure", "ranking_failure", "routing_failure",
        "false_abstention", "false_accept",
    )}
    for row in rows:
        if row["expected_presence"]:
            if not any(relevant(item["timestamp"], row["relevant_intervals"])
                       for item in row["raw_top50"]):
                buckets["retrieval_failure"].append(row["query_id"])
            elif not any(relevant(item["timestamp"], row["relevant_intervals"])
                         for item in row[ranked_field][:5]):
                buckets["ranking_failure"].append(row["query_id"])
            if row.get("modality_requirement") == "speech" or row["query_type"] == "SPEECH":
                buckets["routing_failure"].append(row["query_id"])
            if row["match_probability"] < threshold:
                buckets["false_abstention"].append(row["query_id"])
        elif row["match_probability"] >= threshold:
            buckets["false_accept"].append(row["query_id"])
    return buckets
