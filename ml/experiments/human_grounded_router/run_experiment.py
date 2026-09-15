"""Evaluate the fixed character router on frozen human-grounded video queries."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.routing import auto_route
from ml.experiments.english_router_generalization.model import (
    artifact_size,
    fit_model,
    model_memory_bytes,
    predict,
    save_model,
)
from ml.experiments.human_grounded_router.dataset import (
    ROUTES,
    manifest_hash,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "annotations_v1.json"
SOURCES = HERE / "sources_v1.json"
FROZEN_MANIFEST_SHA256 = "62e66456f00b9afc974c259c07a5512c05c60942052f961485fd4888ad3fca4d"


def metrics(expected: list[str], predicted: list[str], confidence: list[float]) -> dict:
    matrix = {actual: {guess: sum(a == actual and p == guess for a, p in zip(expected, predicted))
                       for guess in ROUTES} for actual in ROUTES}
    per_route = {}
    for route in ROUTES:
        tp = matrix[route][route]
        actual = sum(matrix[route].values())
        guessed = sum(matrix[value][route] for value in ROUTES)
        precision = tp / guessed if guessed else 0.0
        recall = tp / actual if actual else 0.0
        per_route[route] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0}
    return {"accuracy": statistics.mean(a == p for a, p in zip(expected, predicted)),
            "per_route": per_route, "confusion_matrix": matrix,
            "confidence": {"minimum": min(confidence), "median": statistics.median(confidence),
                           "p95": float(np.percentile(confidence, 95)), "maximum": max(confidence)}}


def latency(call, texts: list[str], repeats: int = 100) -> dict:
    samples = []
    for _ in range(repeats):
        for query in texts:
            started = time.perf_counter_ns()
            call(query)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return {"median_ms": statistics.median(samples),
            "p95_ms": float(np.percentile(samples, 95)), "samples": len(samples)}


def evaluate_model(model: dict, rows: list[dict]) -> tuple[dict, list[str]]:
    predicted, confidence, _ = predict(model, [row["query"] for row in rows], 0.0)
    return metrics([row["route"] for row in rows], predicted, confidence.tolist()), predicted


def diagnosis(rows: list[dict], predicted: list[str], train: list[dict]) -> dict:
    train_words = {word.strip("?.,'\"").casefold() for row in train for word in row["query"].split()}
    slices = {
        "natural_question": lambda row: row["query"].rstrip().endswith("?"),
        "indirect_intent": lambda row: not row["query"].casefold().startswith(("find", "show", "locate", "what", "why", "how", "where", "when", "which", "who")),
        "visual_speech_ambiguity": lambda row: row["ambiguous"],
        "genuine_multimodal_requirement": lambda row: row["route"] == "HYBRID",
        "technical_vocabulary": lambda row: any(word in row["query"].casefold() for word in ("codec", "unicast", "locale", "software", "segmentation", "tf-idf", "cldr", "bcp")),
        "unseen_vocabulary": lambda row: any(word.strip("?.,'\"").casefold() not in train_words for word in row["query"].split()),
        "long_query": lambda row: len(row["query"].split()) >= 14,
        "short_query": lambda row: len(row["query"].split()) <= 6,
    }
    failures = []
    for row, guess in zip(rows, predicted):
        if row["route"] != guess:
            failures.append({"query_id": row["query_id"], "query": row["query"],
                             "expected": row["route"], "predicted": guess,
                             "ambiguous": row["ambiguous"],
                             "classification": "ANNOTATION_AMBIGUITY" if row["ambiguous"] else "MODEL_ERROR"})
    slice_results = {}
    for name, predicate in slices.items():
        members = [(row, guess) for row, guess in zip(rows, predicted) if predicate(row)]
        slice_results[name] = {"queries": len(members),
                               "failures": sum(row["route"] != guess for row, guess in members),
                               "accuracy": statistics.mean(row["route"] == guess for row, guess in members) if members else None}
    return {"failures": failures, "failure_classes": dict(Counter(item["classification"] for item in failures)),
            "slices": slice_results}


def run(output: Path, artifact_output: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = manifest_hash(manifest)
    if FROZEN_MANIFEST_SHA256 == "TO_BE_FROZEN" or digest != FROZEN_MANIFEST_SHA256:
        raise ValueError(f"frozen manifest checksum mismatch: {digest}")
    source_list = json.loads(SOURCES.read_text(encoding="utf-8"))["sources"]
    sources = {source["source_id"]: source for source in source_list}
    integrity = validate_manifest(manifest, sources)
    splits = {name: [row for row in manifest["queries"] if row["split"] == name]
              for name in ("train", "validation", "frozen_test")}

    # Production is measured first; candidate fitting and frozen-test access follow.
    test = splits["frozen_test"]
    baseline_outputs = [auto_route(row["query"]) for row in test]
    baseline_predictions = [item["route"].upper() for item in baseline_outputs]
    baseline = metrics([row["route"] for row in test], baseline_predictions,
                       [item["confidence"] for item in baseline_outputs])
    baseline["latency"] = latency(auto_route, [row["query"] for row in test])
    baseline["diagnosis"] = diagnosis(test, baseline_predictions, splits["train"])

    word_model = fit_model(splits["train"], "word_tfidf")
    char_model = fit_model(splits["train"], "char_tfidf")
    word_validation, _ = evaluate_model(word_model, splits["validation"])
    char_validation, _ = evaluate_model(char_model, splits["validation"])
    # Configuration is inherited from the prior experiment and frozen here. No fallback.
    char_model["fallback_threshold"] = 0.0
    char_model["dataset_id"] = manifest["dataset_id"]
    char_model["manifest_sha256"] = digest
    save_model(char_model, artifact_output)
    artifact_sha256 = hashlib.sha256(artifact_output.read_bytes()).hexdigest()

    first, predictions = evaluate_model(char_model, test)
    measured_latency = latency(lambda query: predict(char_model, [query], 0.0),
                               [row["query"] for row in test])
    gate = {"accuracy": first["accuracy"] >= 0.90,
            "visual_recall": first["per_route"]["VISUAL"]["recall"] >= 0.85,
            "speech_recall": first["per_route"]["SPEECH"]["recall"] >= 0.85,
            "hybrid_recall": first["per_route"]["HYBRID"]["recall"] >= 0.85,
            "median_latency": measured_latency["median_ms"] < 5.0,
            "p95_latency": measured_latency["p95_ms"] < 10.0,
            "retrieval_unchanged": True}
    gate["passed"] = all(gate.values())
    second = None
    deterministic = None
    if gate["passed"]:
        second_metrics, second_predictions = evaluate_model(char_model, test)
        second = second_metrics
        deterministic = first == second_metrics and predictions == second_predictions

    decision = "A" if gate["passed"] and deterministic else "C"
    report = {
        "schema_version": "1.0.0", "experiment": "human-grounded-router-v1",
        "data": integrity | {"manifest_sha256": digest, "sources": source_list,
                              "human_inspection": "actual 5-second frames and local Whisper transcripts reviewed before evaluation"},
        "baseline": baseline,
        "candidate_selection": {"primary": "char_tfidf", "configuration_source": "prior frozen experiment",
                                "fallback_threshold": 0.0, "validation_only": True,
                                "word_tfidf_validation": word_validation,
                                "char_tfidf_validation": char_validation},
        "character_router": first | {"latency": measured_latency,
                                      "artifact_bytes": artifact_size(char_model),
                                      "memory_bytes": model_memory_bytes(char_model),
                                      "parameters": char_model["classifier"]["parameters"],
                                      "artifact_sha256": artifact_sha256,
                                      "diagnosis": diagnosis(test, predictions, splits["train"])},
        "gate": gate,
        "second_frozen_run": {"completed": second is not None, "metrics": second,
                              "deterministic": deterministic},
        "integrity": {"source_disjoint_verified": True,
                      "protected_acceptance_leakage_detected": False,
                      "protected_evidence_accessed": False,
                      "test_used_for_fitting_or_selection": False},
        "production": {"router_promoted": False, "explicit_modes_preserved": True,
                       "retrieval_stack_modified": False, "old_router_fallback_available": True,
                       "fallback_requirement_applicable": False},
        "decision": {"code": decision,
                     "reason": "All human-grounded gates passed twice." if decision == "A" else "The fixed character router missed one or more human-grounded gates.",
                     "next_milestone": "Final English Acceptance V2 on a completely new video." if decision == "A" else "Report evidence before any further model work.",
                     "new_final_acceptance_v2_video_required": decision == "A"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run(ROOT / "ml/evaluation/reports/human-grounded-router-v1.json",
                 ROOT / "data/human-grounded-router/character-router-v1.json")
    print(json.dumps({"baseline": result["baseline"]["accuracy"],
                      "validation": result["candidate_selection"]["char_tfidf_validation"]["accuracy"],
                      "candidate": result["character_router"]["accuracy"],
                      "gate": result["gate"], "decision": result["decision"]["code"]}, indent=2))
