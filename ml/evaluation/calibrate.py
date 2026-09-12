"""Fit on calibration only; evaluate a frozen threshold on held-out queries."""

import argparse
import json
import math
import statistics
from pathlib import Path

from ml.evaluation.metrics import retrieval_metrics


def fit(rows):
    if not rows or any(r["split"] != "calibration" or r["mode"] != "visual" for r in rows):
        raise ValueError("Fit requires calibration-only visual queries")
    if not any(r["intervals"] for r in rows) or not any(not r["intervals"] for r in rows):
        raise ValueError("Calibration needs positives and negatives")
    scores = [s for r in rows for s in r["scores"]]
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError("Expected finite measured scores")
    negatives = [max(r["scores"], default=-1) for r in rows if not r["intervals"]]
    return math.nextafter(max(negatives), math.inf)


def summarize(rows, threshold, k):
    measured = []
    for row in rows:
        timestamps = [
            t for t, s in zip(row["timestamps"], row["scores"], strict=True) if s >= threshold
        ]
        measured.append((row, retrieval_metrics(timestamps, row["intervals"], k)))
    positive = [m for r, m in measured if r["intervals"]]
    negative = [m for r, m in measured if not r["intervals"]]
    return {
        "positive_queries": len(positive),
        "negative_queries": len(negative),
        "recall": statistics.mean(m["recall"] for m in positive) if positive else None,
        "mrr": statistics.mean(m["reciprocal_rank"] for m in positive) if positive else None,
        "positive_abstention_rate": statistics.mean(m["returned"] == 0 for m in positive)
        if positive
        else None,
        "negative_false_accept_rate": statistics.mean(m["returned"] > 0 for m in negative)
        if negative
        else None,
    }


def calibrate(report):
    calibration = [r for r in report["queries"] if r["split"] == "calibration"]
    heldout = [r for r in report["queries"] if r["split"] == "heldout"]
    if report["k"] < 5:
        raise ValueError("Retrieve at least five candidates for Recall@1/3/5")
    if not heldout or any(r["mode"] != "visual" for r in heldout):
        raise ValueError("Held-out visual queries required")
    for field in ("video", "source_group"):
        if {r[field] for r in calibration} & {r[field] for r in heldout}:
            raise ValueError(f"Split leakage: {field}")
    hashes = {v["sha256"] for v in report["videos"] if v["split"] == "calibration"}
    if hashes & {v["sha256"] for v in report["videos"] if v["split"] == "heldout"}:
        raise ValueError("Identical video content crosses splits")
    threshold = fit(calibration)
    result = {
        key: report[key]
        for key in ("visual_model", "revision", "sampling_interval", "manifest_sha256")
    }
    result.update(
        mode="visual",
        threshold=threshold,
        strategy="above-largest-calibration-negative",
        warning="Pilot threshold, not a probability or an unseen-domain guarantee",
        metrics={},
    )
    for split, rows in [("calibration", calibration), ("heldout", heldout)]:
        result["metrics"][split] = {
            str(k): {
                "raw": summarize(rows, -math.inf, k),
                "calibrated": summarize(rows, threshold, k),
            }
            for k in (1, 3, 5)
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/calibration.json"))
    args = parser.parse_args()
    result = calibrate(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
