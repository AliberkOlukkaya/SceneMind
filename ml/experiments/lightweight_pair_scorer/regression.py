"""Deterministic regression checks for frozen lightweight-scorer reports."""

import argparse
import json
from pathlib import Path


def check(current: dict, reference: dict) -> list[str]:
    failures = []
    for key in (
        "natural_manifest_sha256",
        "previous_calibration_manifest_sha256",
        "development_manifest_sha256",
        "candidate_model",
        "candidate_revision",
        "candidate_depth",
        "sampling_interval",
    ):
        if current.get(key) != reference.get(key):
            failures.append(f"{key}: {reference.get(key)!r} -> {current.get(key)!r}")
    for system in (
        "clip", "clip_calibrated", "uform_reranked", "uform_calibrated",
        "learned_reranked", "learned_calibrated",
    ):
        for k in ("1", "3", "5"):
            for metric in (
                "recall", "mrr", "negative_false_accept_rate",
                "positive_false_abstention_rate",
            ):
                expected = reference["heldout"]["metrics"][system][k][metric]
                actual = current["heldout"]["metrics"][system][k][metric]
                if actual != expected:
                    failures.append(f"{system}@{k} {metric}: {expected!r} -> {actual!r}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("ml/evaluation/reports/lightweight-pair-scorer-v1.json"),
    )
    arguments = parser.parse_args()
    failures = check(
        json.loads(arguments.current.read_text(encoding="utf-8")),
        json.loads(arguments.reference.read_text(encoding="utf-8")),
    )
    if failures:
        raise SystemExit("\n".join(failures))
    print("lightweight scorer regression gate passed")
