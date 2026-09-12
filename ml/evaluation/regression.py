"""Compare frozen held-out reports; never update a baseline automatically."""

import argparse
import json
from pathlib import Path


def check(current, baseline, tolerance=0.01):
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    for key in ("manifest_sha256", "visual_model", "revision", "sampling_interval", "strategy"):
        if current[key] != baseline[key]:
            raise ValueError(f"Incompatible baseline: {key}")
    failures = []
    for k in ("1", "3", "5"):
        for mode in ("raw", "calibrated"):
            actual = current["metrics"]["heldout"][k][mode]
            expected = baseline["metrics"]["heldout"][k][mode]
            for metric in (
                "recall",
                "mrr",
                "negative_false_accept_rate",
                "positive_abstention_rate",
            ):
                sign = -1 if metric in ("recall", "mrr") else 1
                if actual[metric] is None or expected[metric] is None:
                    raise ValueError("Missing held-out metrics")
                if sign * (actual[metric] - expected[metric]) > tolerance:
                    failures.append(f"{mode}@{k} {metric}: {expected[metric]} -> {actual[metric]}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    failures = check(
        json.loads(args.current.read_text()), json.loads(args.baseline.read_text()), args.tolerance
    )
    if failures:
        raise SystemExit("Regression detected:\n" + "\n".join(failures))
    print("Held-out metric regression gate passed")
