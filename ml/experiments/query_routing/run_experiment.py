"""Run calibration-only routing selection and one frozen held-out evaluation."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml.evaluation.metrics import retrieval_metrics
from ml.experiments.query_routing.prepare import prepare
from ml.experiments.query_routing.router import (
    ROUTES,
    fit_softmax,
    heuristic_route,
    learned_route,
    predict_softmax,
    text_features,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "ml/experiments/query_routing/calibration_v1.json"
NATURAL_MANIFEST = ROOT / "ml/evaluation/natural_v2.json"
NATURAL_REPORT = ROOT / "ml/evaluation/reports/natural-v2.json"
DATA_ROOT = ROOT / "data/query-routing"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def target_route(requirement: str) -> str:
    return {"visual": "VISUAL", "temporal": "VISUAL", "speech": "SPEECH",
            "hybrid": "HYBRID"}[requirement]


def path_metrics(rows: list[dict], route_field: str) -> dict:
    positives = [row for row in rows if row["expected_presence"]]
    output = {}
    for depth in (1, 3, 5):
        metrics = [retrieval_metrics(
            [item["timestamp"] for item in row["paths"][row[route_field]][:depth]],
            row["relevant_intervals"], depth,
        ) for row in positives]
        output[str(depth)] = {
            "positive_queries": len(positives),
            "recall": statistics.mean(item["recall"] for item in metrics),
            "mrr": statistics.mean(item["reciprocal_rank"] for item in metrics),
        }
    return output


def fixed_path_metrics(rows: list[dict], route: str) -> dict:
    copies = [{**row, "fixed": route} for row in rows]
    return path_metrics(copies, "fixed")


def routing_accuracy(rows: list[dict], field: str) -> float:
    return statistics.mean(row[field] == row["target_route"] for row in rows)


def confusion(rows: list[dict], field: str) -> dict:
    return {actual: {predicted: sum(row["target_route"] == actual and row[field] == predicted
                                   for row in rows) for predicted in ROUTES}
            for actual in ROUTES}


def evaluate_method(rows: list[dict], field: str) -> dict:
    return {"routing_accuracy": routing_accuracy(rows, field),
            "confusion_matrix": confusion(rows, field),
            "search": path_metrics(rows, field)}


def calibration_assets(manifest: dict) -> dict:
    from app.encoder import encoder
    from app.hybrid import bm25, fuse

    clip = encoder()
    assets = {}
    for source in manifest["sources"]:
        stem = Path(source["file"]).stem
        frames_root = DATA_ROOT / "sources" / f"{stem}-frames"
        paths = sorted(frames_root.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"prepare five-second frames for {source['source_id']}")
        timestamps_path = DATA_ROOT / "sources" / f"{stem}.timestamps.json"
        timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))
        if len(timestamps) != len(paths):
            raise ValueError(f"timestamp/frame mismatch for {source['source_id']}")
        vector_path = DATA_ROOT / "vectors" / f"{source['source_id']}-production-480.npy"
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        if vector_path.exists():
            vectors = np.load(vector_path, allow_pickle=False)
        else:
            vectors = clip.images(paths)
            with vector_path.open("wb") as output:
                np.save(output, vectors, allow_pickle=False)
        transcript_path = DATA_ROOT / "sources" / f"{stem}.transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        assets[source["source_id"]] = {
            "vectors": vectors, "timestamps": timestamps,
            "thumbnails": [f"/{source['source_id']}/{path.name}" for path in paths],
            "segments": transcript["segments"], "asr_language": transcript["language"],
            "asr_seconds": transcript["seconds"], "bm25": bm25, "fuse": fuse,
        }
    return assets


def build_calibration_rows(manifest: dict) -> list[dict]:
    from app.encoder import encoder, rank_vectors

    assets = calibration_assets(manifest)
    clip = encoder()
    rows = []
    for source in manifest["sources"]:
        asset = assets[source["source_id"]]
        for annotation in source["queries"]:
            scores, indices = rank_vectors(asset["vectors"], clip.text(annotation["query"]), 50)
            visual = [{"timestamp": asset["timestamps"][index],
                       "thumbnail": asset["thumbnails"][index], "score": float(score),
                       "modality": "visual"} for score, index in zip(scores, indices)]
            speech_scores = asset["bm25"](
                annotation["query"], [segment["text"] for segment in asset["segments"]]
            )
            order = sorted(range(len(speech_scores)), key=lambda i: (-speech_scores[i],
                                                                      asset["segments"][i]["start"]))
            speech = []
            for index in order:
                if speech_scores[index] <= 0:
                    continue
                segment = asset["segments"][index]
                nearest = min(range(len(asset["timestamps"])),
                              key=lambda i: abs(asset["timestamps"][i] - segment["start"]))
                speech.append({"timestamp": segment["start"], "end": segment["end"],
                               "thumbnail": asset["thumbnails"][nearest],
                               "score": float(speech_scores[index]), "text": segment["text"],
                               "modality": "speech"})
            hybrid = asset["fuse"](visual, speech, 50)
            rows.append({
                "query_id": annotation["query_id"], "query": annotation["query"],
                "source_id": source["source_id"], "source_group": source["source_group"],
                "split": "calibration", "target_route": annotation["route"],
                "expected_presence": True,
                "relevant_intervals": annotation["relevant_intervals"],
                "hard_case": annotation["hard_case"],
                "paths": {"VISUAL": visual, "SPEECH": speech, "HYBRID": hybrid},
            })
    return rows


def heldout_rows() -> list[dict]:
    manifest = json.loads(NATURAL_MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(NATURAL_REPORT.read_text(encoding="utf-8"))
    annotations = {query["query_id"]: {**query, "source_group": video["source_group"]}
                   for video in manifest["videos"] if video["split"] == "heldout"
                   for query in video["queries"]}
    variants = {}
    for row in report["queries"]:
        if row["split"] == "heldout":
            variants.setdefault(row["query_id"], {})[row["mode"].upper()] = row["results"]
    rows = []
    for query_id, annotation in annotations.items():
        paths = variants[query_id]
        visual = paths["VISUAL"]
        # The production hybrid endpoint falls back to its available visual path.
        hybrid = paths.get("HYBRID", visual)
        rows.append({
            "query_id": query_id, "query": annotation["query_text"],
            "source_group": annotation["source_group"], "split": "heldout",
            "query_type": annotation["query_type"],
            "target_route": target_route(annotation["modality_requirement"]),
            "expected_presence": annotation["expected_presence"],
            "relevant_intervals": annotation["relevant_intervals"],
            "paths": {"VISUAL": visual, "SPEECH": paths.get("SPEECH", []),
                      "HYBRID": hybrid},
            "transcript_available": "SPEECH" in paths,
        })
    return rows


def training_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("router fitting requires calibration-only rows")
    return (np.stack([text_features(row["query"]) for row in rows]),
            np.asarray([ROUTES.index(row["target_route"]) for row in rows]))


def fit_learned(calibration: list[dict]) -> tuple[dict, dict]:
    groups = sorted({row["source_group"] for row in calibration})
    trials = []
    for regularization in (0.01, 0.1, 1.0):
        predictions = []
        expected = []
        for group in groups:
            train = [row for row in calibration if row["source_group"] != group]
            validation = [row for row in calibration if row["source_group"] == group]
            x, y = training_matrix(train)
            model = fit_softmax(x, y, regularization)
            vx, vy = training_matrix(validation)
            predictions.extend(np.argmax(predict_softmax(model, vx), axis=1).tolist())
            expected.extend(vy.tolist())
        accuracy = statistics.mean(a == b for a, b in zip(predictions, expected))
        trials.append({"regularization": regularization,
                       "leave_one_source_out_accuracy": accuracy})
    winner = max(trials, key=lambda row: (row["leave_one_source_out_accuracy"],
                                          -row["regularization"]))
    x, y = training_matrix(calibration)
    return fit_softmax(x, y, winner["regularization"]), {"winner": winner, "trials": trials}


def apply_routes(rows: list[dict], model: dict, fallback_threshold: float | None = None):
    output = []
    for original in rows:
        row = dict(original)
        heuristic, heuristic_confidence, reasons = heuristic_route(row["query"])
        learned, learned_confidence = learned_route(row["query"], model)
        fallback, _ = learned_route(row["query"], model, fallback_threshold)
        row.update(heuristic=heuristic, heuristic_confidence=heuristic_confidence,
                   heuristic_reasons=reasons, learned=learned,
                   learned_confidence=learned_confidence, learned_fallback=fallback,
                   oracle=row["target_route"], current_default="HYBRID",
                   always_visual="VISUAL", always_speech="SPEECH", always_hybrid="HYBRID")
        output.append(row)
    return output


def select_fallback(calibration: list[dict], model: dict) -> dict:
    trials = []
    for threshold in (0.0, 0.45, 0.55, 0.65, 0.75):
        rows = apply_routes(calibration, model, threshold)
        result = evaluate_method(rows, "learned_fallback")
        trials.append({"threshold": threshold, "routing_accuracy": result["routing_accuracy"],
                       "search": result["search"]})
    winner = max(trials, key=lambda row: (row["search"]["5"]["recall"],
                                          row["search"]["1"]["recall"],
                                          row["search"]["5"]["mrr"],
                                          row["routing_accuracy"], -row["threshold"]))
    return {"winner": winner, "trials": trials}


def category_recall(rows: list[dict], field: str) -> dict:
    categories = sorted({row["query_type"] for row in rows if row["expected_presence"]})
    return {category: path_metrics([row for row in rows
                                    if row["query_type"] == category], field)["5"]["recall"]
            for category in categories}


def failure_buckets(rows: list[dict], field: str) -> dict:
    buckets = {name: [] for name in (
        "speech_intent_misclassification", "visual_intent_misclassification",
        "hybrid_over_specialized", "ambiguous_query", "transcript_unavailable", "weak_transcript_retrieval",
        "weak_clip_retrieval", "hybrid_fusion_failure", "temporal_reasoning",
        "ocr_required", "unsupported_capability",
    )}
    for row in rows:
        selected = row[field]
        if row["target_route"] == "SPEECH" and selected != "SPEECH":
            buckets["speech_intent_misclassification"].append(row["query_id"])
        if row["target_route"] == "VISUAL" and selected == "SPEECH":
            buckets["visual_intent_misclassification"].append(row["query_id"])
        if row["target_route"] == "HYBRID" and selected != "HYBRID":
            buckets["hybrid_over_specialized"].append(row["query_id"])
        if row.get("heuristic_reasons") == ["ambiguous"]:
            buckets["ambiguous_query"].append(row["query_id"])
        if selected in {"SPEECH", "HYBRID"} and not row["transcript_available"]:
            buckets["transcript_unavailable"].append(row["query_id"])
        if not row["expected_presence"]:
            continue
        result = retrieval_metrics([item["timestamp"] for item in row["paths"][selected][:5]],
                                   row["relevant_intervals"], 5)
        if result["recall"]:
            continue
        if selected == "SPEECH":
            buckets["weak_transcript_retrieval"].append(row["query_id"])
        elif selected == "VISUAL":
            buckets["weak_clip_retrieval"].append(row["query_id"])
        else:
            buckets["hybrid_fusion_failure"].append(row["query_id"])
        if row["query_type"] == "ACTION_TEMPORAL":
            buckets["temporal_reasoning"].append(row["query_id"])
    return buckets


def compact_rows(rows: list[dict], field: str) -> list[dict]:
    return [{key: row[key] for key in (
        "query_id", "query", "source_group", "split", "target_route",
        "expected_presence", "relevant_intervals", "heuristic", "heuristic_confidence",
        "heuristic_reasons", "learned", "learned_confidence", "learned_fallback",
    )} | {"selected_route": row[field], "path_top5": {
        route: [{k: item[k] for k in ("timestamp", "score", "modality")}
                for item in row["paths"][route][:5]] for route in ROUTES}}
        for row in rows]


def run(output: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    natural = json.loads(NATURAL_MANIFEST.read_text(encoding="utf-8"))
    calibration_groups = {source["source_group"] for source in manifest["sources"]}
    heldout_groups = {video["source_group"] for video in natural["videos"]
                      if video["split"] == "heldout"}
    if calibration_groups & heldout_groups:
        raise ValueError("calibration and held-out source groups overlap")
    prepare(DATA_ROOT / "sources", verify_only=True)
    calibration = build_calibration_rows(manifest)
    model, learned_selection = fit_learned(calibration)
    fallback_selection = select_fallback(calibration, model)
    threshold = fallback_selection["winner"]["threshold"]
    calibration = apply_routes(calibration, model, threshold)
    calibration_methods = {field: evaluate_method(calibration, field)
                           for field in ("heuristic", "learned", "learned_fallback")}
    selected_field = max(calibration_methods, key=lambda field: (
        calibration_methods[field]["search"]["5"]["recall"],
        calibration_methods[field]["search"]["1"]["recall"],
        calibration_methods[field]["search"]["5"]["mrr"],
        calibration_methods[field]["routing_accuracy"], field == "heuristic",
    ))

    heldout = apply_routes(heldout_rows(), model, threshold)
    started = time.perf_counter()
    for _ in range(1000):
        for row in heldout:
            if selected_field == "heuristic":
                heuristic_route(row["query"])
            else:
                learned_route(row["query"], model,
                              threshold if selected_field == "learned_fallback" else None)
    per_query = (time.perf_counter() - started) / (1000 * len(heldout))
    latency_samples = []
    for row in heldout:
        for _ in range(200):
            tick = time.perf_counter_ns()
            if selected_field == "heuristic":
                heuristic_route(row["query"])
            else:
                learned_route(row["query"], model,
                              threshold if selected_field == "learned_fallback" else None)
            latency_samples.append((time.perf_counter_ns() - tick) / 1e9)
    methods = {field: evaluate_method(heldout, field) for field in (
        "current_default", "always_visual", "always_speech", "always_hybrid",
        "oracle", "heuristic", "learned", "learned_fallback")}
    selected = methods[selected_field]
    oracle_categories = category_recall(heldout, "oracle")
    selected_categories = category_recall(heldout, selected_field)
    drops = {name: oracle_categories[name] - selected_categories[name]
             for name in oracle_categories}
    gate = {
        "end_to_end_recall_at_5": {"value": selected["search"]["5"]["recall"],
                                    "minimum": 0.93,
                                    "pass": selected["search"]["5"]["recall"] >= 0.93},
        "maximum_category_drop": {"drops": drops, "limit": 0.05,
                                  "pass": max(drops.values(), default=0) <= 0.05},
        "median_latency_seconds": {"value": float(np.median(latency_samples)),
                                   "preferred_maximum": 0.005,
                                   "pass": float(np.median(latency_samples)) < 0.005},
        "artifact_bytes": {"value": len(json.dumps(model).encode()), "negligible": True,
                           "pass": len(json.dumps(model).encode()) < 100_000},
    }
    gate["passed"] = all(item["pass"] for item in gate.values() if isinstance(item, dict))
    second_rows = apply_routes(heldout_rows(), model, threshold) if gate["passed"] else []
    second_result = evaluate_method(second_rows, selected_field) if second_rows else None
    second_passed = bool(second_result == selected) if gate["passed"] else None
    report = {
        "schema_version": "1.0.0", "experiment": "query-routing-v1",
        "utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "platform": platform.platform(), "python": platform.python_version(),
        "calibration_manifest_sha256": digest(MANIFEST),
        "natural_manifest_sha256": digest(NATURAL_MANIFEST),
        "natural_report_sha256": digest(NATURAL_REPORT), "heldout_tuning": False,
        "dataset": {"calibration_sources": len(manifest["sources"]),
                    "calibration_queries": len(calibration),
                    "calibration_asr_segments": sum(len(json.loads(
                        (DATA_ROOT / "sources" / f"{Path(source['file']).stem}.transcript.json")
                        .read_text(encoding="utf-8"))["segments"])
                        for source in manifest["sources"]),
                    "route_counts": dict(Counter(row["target_route"] for row in calibration)),
                    "duration_seconds": sum(source["duration_seconds"]
                                            for source in manifest["sources"]),
                    "heldout_sources": len(heldout_groups), "heldout_queries": len(heldout),
                    "source_disjoint": True,
                    "sources": [{key: source[key] for key in (
                        "source_id", "domain", "title", "source_group", "page", "download_url", "license",
                        "license_url", "attribution", "sha256", "duration_seconds",
                        "speech_characteristics")}
                        for source in manifest["sources"]]},
        "selection": {"split": "calibration", "selected_method": selected_field,
                      "learned": learned_selection, "hybrid_fallback": fallback_selection,
                      "calibration_methods": calibration_methods},
        "learned_model": model, "heldout_methods": methods,
        "heldout_category_recall_at_5": {"oracle": oracle_categories,
                                         "selected": selected_categories},
        "resources": {"median_seconds": float(np.median(latency_samples)),
                      "p95_seconds": float(np.percentile(latency_samples, 95)),
                      "mean_seconds": per_query,
                      "artifact_bytes": len(json.dumps(model).encode()),
                      "parameters": model["parameters"]},
        "gate": gate,
        "second_frozen_run": {"required": gate["passed"], "executed": gate["passed"],
                              "passed": second_passed, "result": second_result},
        "failures": failure_buckets(heldout, selected_field),
        "production": {"auto_integrated": False, "explicit_modes_preserved": True},
        "decision": {"outcome": "pending", "recommendation": "pending"},
        "calibration_rows": compact_rows(calibration, selected_field),
        "heldout_rows": compact_rows(heldout, selected_field),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "report.json")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"selected": result["selection"]["selected_method"],
                      "methods": {name: value["search"]["5"]["recall"]
                                  for name, value in result["heldout_methods"].items()},
                      "gate": result["gate"],
                      "second": result["second_frozen_run"]}, indent=2))
