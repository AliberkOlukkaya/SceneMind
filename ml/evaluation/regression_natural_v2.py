"""Regression gate for compatible frozen Natural Video Benchmark V2 reports."""

import argparse
import json
from pathlib import Path


def check(current: dict, baseline: dict, tolerance: float = 0.01) -> list[str]:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    for key in (
        "schema_version",
        "benchmark_id",
        "manifest_sha256",
        "visual_model",
        "visual_revision",
        "speech_model",
        "sampling_interval",
    ):
        if current[key] != baseline[key]:
            raise ValueError(f"incompatible baseline: {key}")
    failures = []
    for path, baseline_values in baseline["analysis"]["by_path"].items():
        if path not in current["analysis"]["by_path"]:
            failures.append(f"missing path: {path}")
            continue
        for k in ("1", "3", "5"):
            for state in ("raw", "calibrated"):
                actual = current["analysis"]["by_path"][path][k][state]
                expected = baseline_values[k][state]
                for metric in ("recall", "mrr"):
                    if _regressed(actual[metric], expected[metric], tolerance, higher_is_better=True):
                        failures.append(
                            f"{path} {state}@{k} {metric}: {expected[metric]} -> {actual[metric]}"
                        )
                for metric in ("negative_false_accept_rate", "positive_false_abstention_rate"):
                    if _regressed(actual[metric], expected[metric], tolerance, higher_is_better=False):
                        failures.append(
                            f"{path} {state}@{k} {metric}: {expected[metric]} -> {actual[metric]}"
                        )
    return failures


def _regressed(actual, expected, tolerance, higher_is_better):
    if actual is None and expected is None:
        return False
    if actual is None or expected is None:
        raise ValueError("missing metric in compatible report")
    change = actual - expected
    return change < -tolerance if higher_is_better else change > tolerance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01)
    arguments = parser.parse_args()
    problems = check(
        json.loads(arguments.current.read_text(encoding="utf-8")),
        json.loads(arguments.baseline.read_text(encoding="utf-8")),
        arguments.tolerance,
    )
    if problems:
        raise SystemExit("Regression detected:\n" + "\n".join(problems))
    print("Natural V2 held-out regression gate passed")
