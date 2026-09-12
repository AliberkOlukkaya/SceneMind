"""Stable comparison gate for compatible verifier experiment reports."""

import argparse
import json
from pathlib import Path


def check(current: dict, baseline: dict, tolerance: float = 0.01) -> list[str]:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    for key in ("natural_manifest_sha256", "calibration_manifest_sha256", "candidate_model",
                "candidate_revision", "candidate_depth", "sampling_interval"):
        if current[key] != baseline[key]:
            raise ValueError(f"incompatible report: {key}")
    for key in ("model_id", "revision", "match_label_index", "max_text_tokens", "image_size"):
        if current["verifier"][key] != baseline["verifier"][key]:
            raise ValueError(f"incompatible verifier: {key}")
    failures = []
    for system in ("clip", "clip_calibrated", "verifier_reranked", "verifier_calibrated"):
        for k in ("1", "3", "5"):
            actual = current["heldout"][system][k]
            expected = baseline["heldout"][system][k]
            for metric in ("recall", "mrr"):
                if _worse(actual[metric], expected[metric], tolerance, True):
                    failures.append(f"{system}@{k} {metric}: {expected[metric]} -> {actual[metric]}")
            for metric in ("negative_false_accept_rate", "positive_false_abstention_rate"):
                if _worse(actual[metric], expected[metric], tolerance, False):
                    failures.append(f"{system}@{k} {metric}: {expected[metric]} -> {actual[metric]}")
    return failures


def _worse(actual, expected, tolerance, higher_is_better):
    if actual is None and expected is None:
        return False
    if actual is None or expected is None:
        raise ValueError("missing metric")
    delta = actual - expected
    return delta < -tolerance if higher_is_better else delta > tolerance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01)
    arguments = parser.parse_args()
    problems = check(
        json.loads(arguments.current.read_text(encoding="utf-8-sig")),
        json.loads(arguments.baseline.read_text(encoding="utf-8-sig")),
        arguments.tolerance,
    )
    if problems:
        raise SystemExit("Regression detected:\n" + "\n".join(problems))
    print("Verifier experiment regression gate passed")
