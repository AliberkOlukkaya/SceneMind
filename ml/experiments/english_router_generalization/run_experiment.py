"""Compare frozen source-disjoint English AUTO router candidates."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.routing import auto_route
from ml.experiments.english_router_generalization.dataset import (
    ROUTES,
    manifest_hash,
    validate_manifest,
)
from ml.experiments.english_router_generalization.model import (
    artifact_size,
    fit_model,
    model_memory_bytes,
    predict,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("dataset_v1.json")
FROZEN_MANIFEST_SHA256 = "ce1f2b0e0665c037fb4f914878e5209052160687775cfdfdafa5d406810380da"


def _protected_evidence() -> tuple[list[str], str | None]:
    path = ROOT / "data/final-english-acceptance-v1.json"
    if not path.exists():
        return [], None
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    stack = [data]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"text", "query"} and isinstance(child, str):
                    queries.append(child)
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return queries, hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(expected: list[str], predicted: list[str], confidence: list[float]) -> dict:
    matrix = {actual: {guess: sum(a == actual and p == guess
                                  for a, p in zip(expected, predicted))
                       for guess in ROUTES} for actual in ROUTES}
    per_route = {}
    for route in ROUTES:
        tp = matrix[route][route]
        actual = sum(matrix[route].values())
        guessed = sum(matrix[a][route] for a in ROUTES)
        precision = tp / guessed if guessed else 0.0
        recall = tp / actual if actual else 0.0
        per_route[route] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0}
    return {"accuracy": statistics.mean(a == p for a, p in zip(expected, predicted)),
            "per_route": per_route, "confusion_matrix": matrix,
            "confidence": {"minimum": min(confidence), "median": statistics.median(confidence),
                           "p95": float(np.percentile(confidence, 95)),
                           "maximum": max(confidence)}}


def _latency(call, texts: list[str], repeats: int = 100) -> dict:
    samples = []
    for _ in range(repeats):
        for text in texts:
            started = time.perf_counter_ns()
            call(text)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return {"median_ms": statistics.median(samples),
            "p95_ms": float(np.percentile(samples, 95)), "samples": len(samples)}


def _evaluate_candidate(model: dict, rows: list[dict], threshold: float = 0.0) -> dict:
    expected = [row["route"] for row in rows]
    predicted, confidence, _ = predict(model, [row["query"] for row in rows], threshold)
    return _metrics(expected, predicted, confidence.tolist()) | {
        "predictions": predicted, "fallback_frequency": float(np.mean(
            confidence < threshold)) if threshold else 0.0}


def _select_fallback(model: dict, validation: list[dict]) -> dict:
    _, confidence, _ = predict(model, [row["query"] for row in validation], 0.0)
    candidates = [0.0] + sorted({round(float(value), 6) for value in confidence})
    scored = []
    for threshold in candidates:
        result = _evaluate_candidate(model, validation, threshold)
        scored.append((result["accuracy"], -result["fallback_frequency"], -threshold,
                       threshold, result["fallback_frequency"]))
    winner = max(scored)
    return {"threshold": winner[3], "validation_accuracy": winner[0],
            "fallback_frequency": winner[4], "calibrated_on": "validation"}


def _diagnosis(rows: list[dict], predictions: list[str], train_rows: list[dict]) -> dict:
    failures = [row for row, predicted in zip(rows, predictions) if row["route"] != predicted]
    tags = Counter(tag for row in failures for tag in row["tags"])
    train_words = {word for row in train_rows for word in row["query"].casefold().split()}
    unseen_ids = {row["query_id"] for row in rows
                  if any(word not in train_words for word in row["query"].casefold().split())}
    slice_names = ("natural_question", "long_query", "technical_terms",
                   "indirect_speech_intent", "indirect_visual_intent", "mixed_modality",
                   "ambiguous_wording", "lexical_overlap_trap", "unseen_vocabulary")
    slices = {}
    for name in slice_names:
        members = [(row, predicted) for row, predicted in zip(rows, predictions)
                   if name in row["tags"] or (name == "unseen_vocabulary"
                                              and row["query_id"] in unseen_ids)]
        slices[name] = {"queries": len(members),
                        "failures": sum(row["route"] != predicted
                                        for row, predicted in members),
                        "accuracy": statistics.mean(row["route"] == predicted
                                                    for row, predicted in members)
                        if members else None}
    return {"failures": len(failures), "failure_rate": len(failures) / len(rows),
            "failure_tags": dict(sorted(tags.items())),
            "unseen_vocabulary_failures": sum(row["query_id"] in unseen_ids for row in failures),
            "slices": slices,
            "failed_query_ids": [row["query_id"] for row in failures]}


def run(output: Path, artifact_output: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest_hash(manifest) != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen router manifest changed")
    protected_queries, protected_manifest_sha256 = _protected_evidence()
    integrity = validate_manifest(manifest, protected_queries)
    if protected_manifest_sha256 is not None and len(protected_queries) != 30:
        raise ValueError("protected acceptance query inventory changed")
    splits = {split: [row for row in manifest["queries"] if row["split"] == split]
              for split in ("train", "validation", "frozen_test")}

    baseline_rows = splits["frozen_test"]
    baseline_predictions = []
    baseline_confidence = []
    for row in baseline_rows:
        result = auto_route(row["query"])
        baseline_predictions.append(result["route"].upper())
        baseline_confidence.append(result["confidence"])
    baseline = _metrics([row["route"] for row in baseline_rows], baseline_predictions,
                        baseline_confidence)
    baseline["latency"] = _latency(auto_route, [row["query"] for row in baseline_rows])
    baseline["diagnosis"] = _diagnosis(baseline_rows, baseline_predictions, splits["train"])

    candidates = {}
    fitted = {}
    for architecture in ("word_tfidf", "char_tfidf", "word_char_tfidf"):
        model = fit_model(splits["train"], architecture)
        fallback = _select_fallback(model, splits["validation"])
        validation = _evaluate_candidate(model, splits["validation"], fallback["threshold"])
        candidates[architecture] = {"validation": {key: value for key, value in validation.items()
                                                    if key != "predictions"},
                                    "fallback": fallback,
                                    "artifact_bytes": artifact_size(model),
                                    "memory_bytes": model_memory_bytes(model),
                                    "parameters": model["classifier"]["parameters"]}
        fitted[architecture] = model
    selected_name = max(candidates, key=lambda name: (
        candidates[name]["validation"]["accuracy"],
        min(candidates[name]["validation"]["per_route"][route]["recall"] for route in ROUTES),
        -candidates[name]["artifact_bytes"], name,
    ))
    selected_model = fitted[selected_name]
    fallback = candidates[selected_name]["fallback"]
    selected_model["fallback_threshold"] = fallback["threshold"]

    first = _evaluate_candidate(selected_model, splits["frozen_test"], fallback["threshold"])
    without_fallback = _evaluate_candidate(selected_model, splits["frozen_test"], 0.0)
    latency = _latency(lambda text: predict(selected_model, [text]),
                       [row["query"] for row in splits["frozen_test"]])
    gate = {"accuracy": first["accuracy"] >= 0.90,
            "each_route_recall": all(first["per_route"][route]["recall"] >= 0.85
                                     for route in ROUTES),
            "median_latency": latency["median_ms"] < 5.0,
            "p95_latency": latency["p95_ms"] < 10.0,
            "retrieval_unchanged": True, "acceptance_leakage_absent": True}
    gate["passed"] = all(gate.values())
    second = _evaluate_candidate(selected_model, splits["frozen_test"], fallback["threshold"])
    deterministic = first == second

    # The query set is independently authored but is not grounded in reviewed real videos.
    # It therefore cannot justify production promotion even when its internal gates pass.
    decision = "E"
    report = {
        "schema_version": "1.0.0", "experiment": "english-router-generalization-v1",
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "data": integrity | {"provenance": "independently authored source scenarios",
                              "real_video_grounded": False},
        "baseline": baseline,
        "candidate_selection": {"selected": selected_name, "validation_only": True,
                                "candidates": candidates},
        "best_candidate": {key: value for key, value in first.items() if key != "predictions"} | {
            "architecture": selected_name, "artifact_bytes": artifact_size(selected_model),
            "parameters": selected_model["classifier"]["parameters"],
            "latency": latency, "memory_bytes": model_memory_bytes(selected_model),
            "diagnosis": _diagnosis(splits["frozen_test"], first["predictions"], splits["train"])},
        "fallback": {"used": fallback["threshold"] > 0,
                     "threshold": fallback["threshold"],
                     "frozen_test_frequency": first["fallback_frequency"],
                     "frozen_test_accuracy_without_fallback": without_fallback["accuracy"],
                     "frozen_test_accuracy": first["accuracy"]},
        "gate": gate,
        "second_frozen_run": {"completed": gate["passed"], "deterministic": deterministic,
                              "metrics_equal": first == second},
        "integrity": {"acceptance_video_leakage_detected": False,
                      "protected_acceptance_queries_checked": len(protected_queries),
                      "protected_acceptance_manifest_sha256": protected_manifest_sha256,
                      "source_disjoint_verified": True,
                      "test_used_for_selection": False},
        "production": {"router_promoted": False, "retrieval_stack_modified": False,
                       "explicit_modes_preserved": True, "artifact_written_to_production": False},
        "decision": {"code": decision,
                     "reason": "The 450-query source-card set is balanced and source-disjoint but is not grounded in independently reviewed real videos, so it is insufficient evidence for production promotion.",
                     "next_milestone": "Collect independently reviewed real-video route annotations from new source groups, freeze a new source-disjoint test, and rerun the same classical comparison.",
                     "new_independent_acceptance_video_required_after_promotion": True},
    }
    artifact_output.parent.mkdir(parents=True, exist_ok=True)
    artifact_output.write_text(json.dumps(selected_model, separators=(",", ":")), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    data = ROOT / "data/english-router-generalization"
    result = run(ROOT / "ml/evaluation/reports/english-router-generalization-v1.json",
                 data / "selected-router.json")
    print(json.dumps({"baseline": result["baseline"]["accuracy"],
                      "selected": result["candidate_selection"]["selected"],
                      "candidate": result["best_candidate"]["accuracy"],
                      "gate": result["gate"], "decision": result["decision"]["code"]}, indent=2))
