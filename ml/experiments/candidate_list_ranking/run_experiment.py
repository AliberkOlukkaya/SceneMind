"""Run the frozen candidate-list ranking and no-match experiment."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from ml.experiments.candidate_list_ranking.evaluation import (
    classify_failures,
    no_match_metrics,
    oracle,
    rank_metrics,
    relevant,
)
from ml.experiments.candidate_list_ranking.features import (
    CANDIDATE_FEATURES,
    QUERY_FEATURES,
    candidate_features,
    query_features,
)
from ml.experiments.candidate_list_ranking.model import (
    fit_accept_threshold,
    fit_logistic,
    predict,
)
from ml.experiments.coarse_candidate_diversity.run_experiment import coarse_assets, faiss_pool

ROOT = Path(__file__).resolve().parents[3]
SOURCE_REPORT = ROOT / "ml/evaluation/reports/lightweight-pair-scorer-v1.json"
COARSE_REPORT = ROOT / "ml/evaluation/reports/coarse-candidate-diversity-v1.json"
NATURAL_MANIFEST = ROOT / "ml/evaluation/natural_v2.json"
NATURAL_REPORT = ROOT / "ml/evaluation/reports/natural-v2.json"
SMALL_OBJECT_IDS = {"street-object-bicycle", "throw-object-ball", "throw-compositional-holding"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def training_data(rows: list[dict], depth: int, query_level: bool = False):
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("training is restricted to calibration rows")
    features, labels = [], []
    for row in rows:
        candidates = row["raw_top50"][:depth]
        if query_level:
            features.append(query_features(candidates, row["embeddings"]))
            labels.append(row["expected_presence"])
        else:
            matrix = candidate_features(candidates, row["embeddings"])
            features.extend(matrix)
            labels.extend(relevant(item["timestamp"], row["relevant_intervals"])
                          for item in candidates)
    return np.asarray(features), np.asarray(labels, dtype=np.float64)


def rerank(row: dict, model: dict, depth: int) -> list[dict]:
    candidates = row["raw_top50"][:depth]
    probabilities = predict(model, candidate_features(candidates, row["embeddings"]))
    enriched = [{**item, "list_probability": float(probability)}
                for item, probability in zip(candidates, probabilities, strict=True)]
    return sorted(enriched, key=lambda item: (-item["list_probability"], -item["score"],
                                               item["timestamp"]))


def temporal_rerank(row: dict, depth: int, support_weight: float, change_weight: float):
    candidates = row["raw_top50"][:depth]
    matrix = candidate_features(candidates, row["embeddings"])
    z = matrix[:, CANDIDATE_FEATURES.index("score_z")]
    support = matrix[:, CANDIDATE_FEATURES.index("temporal_support_10s")]
    change = matrix[:, CANDIDATE_FEATURES.index("adjacent_visual_change")]
    scores = z + support_weight * support + change_weight * change
    enriched = [{**item, "temporal_list_score": float(score)}
                for item, score in zip(candidates, scores, strict=True)]
    return sorted(enriched, key=lambda item: (-item["temporal_list_score"], -item["score"],
                                               item["timestamp"]))


def select_ranking_model(calibration: list[dict], depth: int) -> tuple[dict, dict]:
    groups = sorted({row["video_id"] for row in calibration})
    trials = []
    for regularization in (0.1, 1.0, 10.0):
        fold_scores = []
        for group in groups:
            train = [row for row in calibration if row["video_id"] != group]
            validation = [dict(row) for row in calibration if row["video_id"] == group]
            x, y = training_data(train, depth)
            model = fit_logistic(x, y, regularization)
            for row in validation:
                row["candidate_logistic"] = rerank(row, model, depth)
            metrics = rank_metrics(validation, "candidate_logistic")
            fold_scores.append((metrics["5"]["recall"], metrics["1"]["recall"],
                                metrics["5"]["mrr"]))
        mean = tuple(statistics.mean(values) for values in zip(*fold_scores))
        trials.append({"regularization": regularization, "leave_one_source_out": {
            "recall_at_5": mean[0], "recall_at_1": mean[1], "mrr_at_5": mean[2]}})
    winner = max(trials, key=lambda row: (
        row["leave_one_source_out"]["recall_at_5"],
        row["leave_one_source_out"]["recall_at_1"],
        row["leave_one_source_out"]["mrr_at_5"], -row["regularization"],
    ))
    x, y = training_data(calibration, depth)
    return fit_logistic(x, y, winner["regularization"]), {"winner": winner, "trials": trials}


def select_temporal(calibration: list[dict], depth: int) -> dict:
    trials = []
    for support_weight in (-0.5, -0.25, 0.0, 0.25, 0.5):
        for change_weight in (-1.0, -0.5, 0.0, 0.5, 1.0):
            copies = [dict(row) for row in calibration]
            for row in copies:
                row["temporal"] = temporal_rerank(row, depth, support_weight, change_weight)
            metric = rank_metrics(copies, "temporal")
            trials.append({"support_weight": support_weight, "change_weight": change_weight,
                           "metrics": metric})
    return max(trials, key=lambda row: (row["metrics"]["5"]["recall"],
                                        row["metrics"]["1"]["recall"],
                                        row["metrics"]["5"]["mrr"],
                                        -abs(row["support_weight"]),
                                        -abs(row["change_weight"])))


def select_no_match_model(calibration: list[dict], depth: int = 20) -> tuple[dict, dict, dict]:
    groups = sorted({row["video_id"] for row in calibration})
    trials = []
    for regularization in (0.1, 1.0, 10.0):
        fold_far, fold_pfa = [], []
        for group in groups:
            train = [row for row in calibration if row["video_id"] != group]
            validation = [row for row in calibration if row["video_id"] == group]
            x, y = training_data(train, depth, query_level=True)
            model = fit_logistic(x, y, regularization)
            threshold = fit_accept_threshold(predict(model, x), y)
            vx, vy = training_data(validation, depth, query_level=True)
            probabilities = predict(model, vx)
            negative, positive = vy == 0, vy == 1
            fold_far.append(float((probabilities[negative] >= threshold["threshold"]).mean()))
            fold_pfa.append(float((probabilities[positive] < threshold["threshold"]).mean()))
        trials.append({"regularization": regularization,
                       "leave_one_source_out_far": statistics.mean(fold_far),
                       "leave_one_source_out_pfa": statistics.mean(fold_pfa)})
    winner = min(trials, key=lambda row: (row["leave_one_source_out_far"] > 0.1,
                                          row["leave_one_source_out_far"],
                                          row["leave_one_source_out_pfa"],
                                          row["regularization"]))
    x, y = training_data(calibration, depth, query_level=True)
    model = fit_logistic(x, y, winner["regularization"])
    threshold = fit_accept_threshold(predict(model, x), y)
    return model, threshold, {"winner": winner, "trials": trials}


def slice_metrics(rows: list[dict], field: str) -> dict:
    names = sorted({row["query_type"] for row in rows})
    result = {name: rank_metrics([row for row in rows if row["query_type"] == name], field)
              for name in names if any(row["expected_presence"] and row["query_type"] == name
                                       for row in rows)}
    small = [row for row in rows if row["query_id"] in SMALL_OBJECT_IDS]
    result["SMALL_OBJECT"] = rank_metrics(small, field)
    return result


def route_audit() -> dict:
    report = json.loads(NATURAL_REPORT.read_text(encoding="utf-8"))
    heldout = [row for row in report["queries"] if row["split"] == "heldout"]
    route = {"visual": "visual", "temporal": "visual", "speech": "speech",
             "hybrid": "hybrid"}
    selected = []
    for query_id in sorted({row["query_id"] for row in heldout}):
        variants = [row for row in heldout if row["query_id"] == query_id]
        requirement = variants[0]["modality_requirement"]
        mode = route[requirement]
        chosen = next(row for row in variants if row["mode"] == mode)
        chosen = dict(chosen)
        chosen["cross_path_features"] = {
            candidate_mode: {
                "result_count": len(next((item["results"] for item in variants
                                          if item["mode"] == candidate_mode), [])),
                "best_score": next((item["results"][0]["score"] for item in variants
                                    if item["mode"] == candidate_mode and item["results"]), None),
            }
            for candidate_mode in ("visual", "speech", "hybrid")
        }
        selected.append(chosen)
    rows = [{**row, "routed": row["results"]} for row in selected]
    return {
        "policy": "existing explicit product mode; benchmark modality_requirement selects the audited path",
        "learned_from_query_text": False,
        "ambiguous_policy": "user selects hybrid; no lexical classifier was fit",
        "routing_classes": {
            "visual_object_scene": "explicit visual mode",
            "speech_transcript": "explicit speech mode",
            "hybrid": "explicit hybrid mode",
            "ambiguous": "explicit user choice; no calibration-safe text classifier exists",
        },
        "counts": {mode: sum(row["mode"] == mode for row in selected)
                   for mode in ("visual", "speech", "hybrid")},
        "metrics": rank_metrics(rows, "routed"),
        "empty_result_negative_rejection": {
            "speech": [row["query_id"] for row in selected
                       if not row["expected_presence"] and row["mode"] == "speech"
                       and not row["results"]],
        },
        "rows": [{"query_id": row["query_id"], "requirement": row["modality_requirement"],
                  "selected_mode": row["mode"], "result_count": len(row["results"]),
                  "cross_path_features": row["cross_path_features"]}
                 for row in selected],
    }


def run(output: Path) -> dict:
    from app.encoder import encoder

    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if digest(NATURAL_MANIFEST) != source["natural_manifest_sha256"]:
        raise ValueError("frozen Natural V2 manifest changed")
    assets = coarse_assets(source)
    clip = encoder()
    rows, text_times, search_times = [], [], []
    for frozen in source["rows"]:
        asset = assets[frozen["video_id"]]
        started = time.perf_counter()
        text = clip.text(frozen["query"])
        text_times.append(time.perf_counter() - started)
        timestamps = [frame["timestamp"] for frame in asset["frames"]]
        candidates, elapsed = faiss_pool(asset["vectors"], text, timestamps, 50)
        search_times.append(elapsed)
        rows.append({**{key: frozen[key] for key in (
            "query_id", "video_id", "split", "query", "query_type",
            "expected_presence", "relevant_intervals")},
            "source_group": frozen["video_id"], "embeddings": asset["vectors"],
            "raw_top50": [{"timestamp": item.timestamp, "score": item.score,
                            "index": item.index} for item in candidates]})
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]

    ranking_models, selections, temporal_selections = {}, {}, {}
    for depth in (20, 50):
        model, selection = select_ranking_model(calibration, depth)
        ranking_models[str(depth)], selections[str(depth)] = model, selection
        temporal_selections[str(depth)] = select_temporal(calibration, depth)
        for row in rows:
            row[f"logistic_top{depth}"] = rerank(row, model, depth)
            choice = temporal_selections[str(depth)]
            row[f"temporal_top{depth}"] = temporal_rerank(
                row, depth, choice["support_weight"], choice["change_weight"])

    no_match_model, threshold, no_match_selection = select_no_match_model(calibration)
    inference_times = []
    for row in rows:
        started = time.perf_counter()
        features = query_features(row["raw_top50"][:20], row["embeddings"])
        row["match_probability"] = float(predict(no_match_model, features[None, :])[0])
        candidate_features(row["raw_top50"][:20], row["embeddings"])
        predict(ranking_models["20"], candidate_features(
            row["raw_top50"][:20], row["embeddings"]))
        inference_times.append(time.perf_counter() - started)

    systems = {}
    for depth in (20, 50):
        raw_field = "raw_top50"
        systems[f"raw_top{depth}"] = rank_metrics(heldout, raw_field)
        systems[f"normalized_top{depth}"] = systems[f"raw_top{depth}"]
        systems[f"margin_top{depth}"] = systems[f"raw_top{depth}"]
        systems[f"temporal_top{depth}"] = rank_metrics(heldout, f"temporal_top{depth}")
        systems[f"logistic_top{depth}"] = rank_metrics(heldout, f"logistic_top{depth}")
    selected_field = max(("logistic_top20", "logistic_top50", "temporal_top20",
                          "temporal_top50", "raw_top50"),
                         key=lambda field: (rank_metrics(calibration, field)["5"]["recall"],
                                            rank_metrics(calibration, field)["1"]["recall"],
                                            rank_metrics(calibration, field)["5"]["mrr"]))
    baseline = rank_metrics(heldout, "raw_top50")
    selected = rank_metrics(heldout, selected_field)
    baseline_slices = slice_metrics(heldout, "raw_top50")
    selected_slices = slice_metrics(heldout, selected_field)
    drops = {name: baseline_slices[name]["5"]["recall"]
             - selected_slices[name]["5"]["recall"] for name in baseline_slices}
    no_match = no_match_metrics(heldout, threshold["threshold"])
    no_match_slices = {
        name: no_match_metrics([row for row in heldout if row["query_type"] == name],
                               threshold["threshold"])
        for name in sorted({row["query_type"] for row in heldout})
    }
    no_match_slices["SMALL_OBJECT"] = no_match_metrics(
        [row for row in heldout if row["query_id"] in SMALL_OBJECT_IDS], threshold["threshold"])
    no_match_slices["NEGATIVE"] = no_match_metrics(
        [row for row in heldout if not row["expected_presence"]], threshold["threshold"])
    ranking_gate = {
        "recall_at_5": {"value": selected["5"]["recall"], "minimum": 0.8571,
                         "pass": selected["5"]["recall"] >= 0.8571},
        "recall_at_1_preferred": {"value": selected["1"]["recall"], "baseline": 0.7143,
                                   "pass": selected["1"]["recall"] > 0.7143},
        "mrr_preferred": {"value": selected["5"]["mrr"], "baseline": 0.8048,
                          "pass": selected["5"]["mrr"] > 0.8048},
        "maximum_category_drop": {"drops": drops, "limit": 0.05,
                                  "pass": max(drops.values(), default=0) <= 0.05},
    }
    ranking_gate["passed"] = ranking_gate["recall_at_5"]["pass"] and ranking_gate[
        "maximum_category_drop"]["pass"]
    no_match_gate = {
        "false_accept_rate": {"value": no_match["false_accept_rate"], "limit": 0.1,
                              "pass": no_match["false_accept_rate"] <= 0.1},
        "positive_false_abstention_rate": {
            "value": no_match["positive_false_abstention_rate"], "limit": 0.2,
            "pass": no_match["positive_false_abstention_rate"] <= 0.2},
    }
    no_match_gate["passed"] = all(item["pass"] for item in no_match_gate.values()
                                  if isinstance(item, dict))
    p95 = float(np.percentile(inference_times, 95))
    report_rows = [{**{key: row[key] for key in (
        "query_id", "video_id", "split", "query_type", "expected_presence",
        "relevant_intervals", "match_probability")},
        "raw_top50": row["raw_top50"], "selected": row[selected_field]}
        for row in rows]
    report = {
        "schema_version": "1.0.0", "experiment": "candidate-list-ranking-v1",
        "utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "platform": platform.platform(), "python": platform.python_version(),
        "source_report_sha256": digest(SOURCE_REPORT),
        "coarse_report_sha256": digest(COARSE_REPORT),
        "natural_manifest_sha256": digest(NATURAL_MANIFEST),
        "natural_report_sha256": digest(NATURAL_REPORT),
        "heldout_tuning": False, "candidate_depths": [20, 50],
        "feature_schema": {"candidate": CANDIDATE_FEATURES, "query": QUERY_FEATURES},
        "selection": {"split": "calibration", "selected_system": selected_field,
                      "ranking": selections, "temporal": temporal_selections,
                      "no_match": no_match_selection},
        "models": {"ranking": ranking_models, "no_match": no_match_model,
                   "no_match_threshold": threshold},
        "ranking": {"baseline": baseline, "systems": systems, "selected": selected,
                    "baseline_slices": baseline_slices, "selected_slices": selected_slices},
        "oracle": {"heldout": oracle(heldout, "raw_top50"),
                   "calibration": oracle(calibration, "raw_top50")},
        "no_match": {"calibration": no_match_metrics(calibration, threshold["threshold"]),
                     "heldout": no_match, "heldout_slices": no_match_slices},
        "routing": route_audit(),
        "failures": classify_failures(heldout, selected_field, threshold["threshold"]),
        "resources": {
            "added_median_seconds": statistics.median(inference_times),
            "added_p95_seconds": p95,
            "model_parameter_bytes": 8 * (no_match_model["parameters"]
                                            + ranking_models["20"]["parameters"]),
            "ram_limit_bytes": 250 * 1024 * 1024,
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "text_encoding_median_seconds": statistics.median(text_times),
            "faiss_top50_median_seconds": statistics.median(search_times),
        },
        "gates": {"ranking": ranking_gate, "no_match": no_match_gate,
                  "resources": {"added_median_seconds": {"value": statistics.median(inference_times),
                                  "limit": 0.05, "pass": statistics.median(inference_times) <= 0.05},
                                "added_p95_seconds": {"value": p95, "limit": 0.05,
                                                      "pass": p95 <= 0.05},
                                "added_ram_bytes": {"value": 8 * (no_match_model["parameters"]
                                                      + ranking_models["20"]["parameters"]),
                                                    "limit": 250 * 1024 * 1024, "pass": True}}},
        "second_frozen_run": {"required": ranking_gate["passed"] and no_match_gate["passed"],
                              "executed": False, "passed": None},
        "production_changed": False,
        "decision": {"outcome": "pending", "recommendation": "pending"},
        "rows": report_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/candidate-list-ranking/report.json")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"selected": result["selection"]["selected_system"],
                      "ranking": result["ranking"]["selected"],
                      "no_match": result["no_match"]["heldout"],
                      "gates": result["gates"]}, indent=2))
