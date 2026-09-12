"""Aggregate Natural Video Benchmark V2 without fitting on held-out data."""

import math
import statistics
from collections import defaultdict

from ml.evaluation.calibrate import fit
from ml.evaluation.metrics import retrieval_metrics

KS = (1, 3, 5)


def visual_threshold(rows: list[dict]) -> float:
    calibration = [
        row for row in rows if row["split"] == "calibration" and row["mode"] == "visual"
    ]
    return fit(
        [
            {
                "split": row["split"],
                "mode": row["mode"],
                "intervals": row["relevant_intervals"],
                "scores": [result["score"] for result in row["results"]],
            }
            for row in calibration
        ]
    )


def accepted_results(row: dict, threshold: float | None) -> list[dict]:
    if threshold is None or row["mode"] == "speech":
        return row["results"]
    if row["mode"] == "visual":
        return [result for result in row["results"] if result["score"] >= threshold]
    return [
        result
        for result in row["results"]
        if "speech" in result.get("evidence", {})
        or result.get("evidence", {}).get("visual", {}).get("score", -math.inf) >= threshold
    ]


def summarize(rows: list[dict], threshold: float | None, k: int) -> dict:
    measured = []
    for row in rows:
        results = accepted_results(row, threshold)[:k]
        metric = retrieval_metrics(
            [result["timestamp"] for result in results], row["relevant_intervals"], k
        )
        measured.append((row, metric))
    positives = [metric for row, metric in measured if row["expected_presence"]]
    negatives = [metric for row, metric in measured if not row["expected_presence"]]
    latencies = [row["warm_median_seconds"] for row, _ in measured]
    return {
        "queries": len(measured),
        "positive_queries": len(positives),
        "negative_queries": len(negatives),
        "recall": statistics.mean(metric["recall"] for metric in positives)
        if positives
        else None,
        "mrr": statistics.mean(metric["reciprocal_rank"] for metric in positives)
        if positives
        else None,
        "precision": statistics.mean(metric["precision"] for metric in positives)
        if positives
        else None,
        "negative_false_accept_rate": statistics.mean(metric["returned"] > 0 for metric in negatives)
        if negatives
        else None,
        "abstention_rate": statistics.mean(metric["returned"] == 0 for _, metric in measured)
        if measured
        else None,
        "positive_false_abstention_rate": statistics.mean(
            metric["returned"] == 0 for metric in positives
        )
        if positives
        else None,
        "latency_median_seconds": statistics.median(latencies) if latencies else None,
        "latency_p95_seconds": sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)]
        if latencies
        else None,
    }


def aggregate(report: dict) -> dict:
    rows = report["queries"]
    threshold = visual_threshold(rows)
    result = {
        "threshold": threshold,
        "threshold_strategy": "above-largest-calibration-visual-negative",
        "threshold_fit": "calibration split visual rows only",
        "overall": {},
        "by_path": {},
        "by_category": {},
    }
    heldout = [row for row in rows if row["split"] == "heldout"]
    groups = {
        "overall": {"heldout": heldout},
        "by_path": defaultdict(list),
        "by_category": defaultdict(list),
    }
    for row in heldout:
        groups["by_path"][row["mode"]].append(row)
        groups["by_category"][row["query_type"]].append(row)
    for section, section_groups in groups.items():
        result[section] = {
            name: {
                str(k): {
                    "raw": summarize(group_rows, None, k),
                    "calibrated": summarize(group_rows, threshold, k),
                }
                for k in KS
            }
            for name, group_rows in section_groups.items()
        }
    return result


def failures(report: dict, analysis: dict) -> list[dict]:
    threshold = analysis["threshold"]
    output = []
    for row in report["queries"]:
        if row["split"] != "heldout":
            continue
        results = accepted_results(row, threshold)[:5]
        metrics = retrieval_metrics(
            [result["timestamp"] for result in results], row["relevant_intervals"], 5
        )
        failure = None
        if row["expected_presence"] and not results:
            failure = "false abstention"
        elif row["expected_presence"] and metrics["recall"] < 1:
            failure = "missed relevant interval"
        elif not row["expected_presence"] and results:
            failure = "false accept on absent query"
        if failure:
            output.append(
                {
                    "query_id": row["query_id"],
                    "video_id": row["video_id"],
                    "mode": row["mode"],
                    "query_type": row["query_type"],
                    "failure": failure,
                    "likely_component": _likely_component(row, failure),
                    "raw_top_results": [_result_summary(item) for item in row["results"][:3]],
                    "top_results": [
                        _result_summary(item)
                        for item in results[:3]
                    ],
                }
            )
    return output


def _result_summary(item: dict) -> dict:
    return {
        "timestamp": item["timestamp"],
        "score": item["score"],
        "modality": item.get("modality"),
        "text": item.get("text"),
        "evidence": item.get("evidence"),
    }


def _likely_component(row: dict, failure: str) -> str:
    if failure == "false accept on absent query":
        return "open-set rejection / score calibration"
    if row["query_type"] == "ACTION_TEMPORAL":
        return "single-frame sampling and temporal representation"
    if row["query_type"] == "COMPOSITIONAL":
        return "visual composition binding"
    if row["query_type"] == "SPEECH" and row["mode"] == "visual":
        return "modality mismatch: visual path cannot retrieve spoken content"
    if row["query_type"] == "SPEECH":
        return "ASR transcript quality or lexical BM25 matching"
    return "visual embedding or five-second frame sampling"
