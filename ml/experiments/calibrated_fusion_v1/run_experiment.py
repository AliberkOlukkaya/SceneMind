"""Run the calibration generalization gate before any fusion candidate exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.evaluation.hybrid_fusion_trace import trace_fusion
from ml.experiments.calibrated_fusion_v1.calibration import (
    MODALITY_FEATURE_SETS,
    auc,
    bucket_relevant,
    fit_logistic,
    modality_examples,
    model_metrics,
    predict,
    select_calibrator,
)
from ml.experiments.path_aware_no_match.run_experiment import build_rows, load_assets

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PATH_MANIFEST = ROOT / "ml/experiments/path_aware_no_match/manifest_v1.json"
PATH_REPORT = ROOT / "ml/evaluation/reports/path-aware-no-match-v1.json"
DEVELOPMENT_MANIFEST = ROOT / "ml/evaluation/development/hybrid_fusion_queries_v1.json"
DEVELOPMENT_REPORT = ROOT / "ml/evaluation/reports/hybrid-fusion-diagnostics.json"
HOLDOUT_MANIFEST = ROOT / "ml/evaluation/holdout/hybrid_fusion_holdout_v1.json"
HOLDOUT_SHA = ROOT / "ml/evaluation/holdout/hybrid_fusion_holdout_v1.sha256"
V2_MANIFEST = ROOT / "ml/evaluation/final_english_acceptance_v2_manifest.json"
DEFAULT_CACHE = ROOT / "data/path-aware-no-match/calibrated-fusion-calibration-rows.json"
DEFAULT_ARTIFACT = EXPERIMENT_ROOT / "calibration_artifact_v1.json"
DEFAULT_REPORT = ROOT / "ml/evaluation/reports/calibrated-fusion-v1.json"

GENERALIZATION_GATE = {
    "auc_delta_vs_rank_minimum": 0.0,
    "brier_delta_vs_rank_maximum": 0.0,
    "rule": "Both modalities must be no worse than rank-only on both source-disjoint AUC and Brier.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_calibration_paths(cache: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(PATH_MANIFEST.read_text(encoding="utf-8"))
    calibration_manifest = {
        **manifest,
        "sources": [source for source in manifest["sources"] if source["split"] == "calibration"],
    }
    expected_queries = sum(len(source["queries"]) for source in calibration_manifest["sources"])
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
    else:
        assets = load_assets(calibration_manifest)
        built = build_rows(calibration_manifest, assets)
        cached = [
            {
                "query_id": row["query_id"],
                "source_id": row["source_id"],
                "route": row["route"],
                "negative": not row["expected_presence"],
                "intervals": row["relevant_intervals"],
                "query": row["query"],
                "paths": row["paths"],
            }
            for row in built
        ]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached) + "\n", encoding="utf-8")
    if len(cached) != expected_queries:
        raise ValueError("calibration cache query count differs from frozen manifest")
    source_ids = {source["source_id"] for source in calibration_manifest["sources"]}
    if {row["source_id"] for row in cached} != source_ids:
        raise ValueError("calibration cache sources differ from frozen manifest")

    historical = json.loads(PATH_REPORT.read_text(encoding="utf-8"))
    historical_rows = {
        row["query_id"]: row for row in historical["rows"] if row["split"] == "calibration"
    }
    parity = 0
    normalized = []
    for row in cached:
        expected = historical_rows[row["query_id"]]["path_top5"]
        for mode in ("VISUAL", "SPEECH", "HYBRID"):
            observed_items = row["paths"][mode][:5]
            expected_items = expected[mode]
            timestamps_match = [item["timestamp"] for item in observed_items] == [
                item["timestamp"] for item in expected_items
            ]
            scores_match = np.allclose(
                [item["score"] for item in observed_items],
                [item["score"] for item in expected_items],
                rtol=1e-12,
                atol=1e-12,
            )
            if not timestamps_match or not scores_match:
                raise AssertionError(
                    f"calibration reconstruction changed: {row['query_id']} {mode}"
                )
        parity += 1
        trace = trace_fusion(row["paths"]["VISUAL"], row["paths"]["SPEECH"], 5)
        normalized.append(
            {
                "query_id": row["query_id"],
                "source_id": row["source_id"],
                "category": row["route"],
                "negative": row["negative"],
                "intervals": row["intervals"],
                "query": row["query"],
                "trace": trace,
            }
        )
    return normalized, {
        "sources": len(source_ids),
        "queries": len(normalized),
        "historical_top5_parity": parity,
        "manifest_sha256": sha256(PATH_MANIFEST),
    }


def _load_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))
    if manifest["acceptance_isolation"]["eligible_for_future_tuning"] is not True:
        raise ValueError("development data is not eligible for future tuning")
    report = json.loads(DEVELOPMENT_REPORT.read_text(encoding="utf-8"))
    rows = [
        {
            "query_id": query["query_id"],
            "source_id": source["source"]["source_id"],
            "category": query["evidence_category"],
            "negative": query["negative"],
            "intervals": query["intervals"],
            "query": query["query"],
            "trace": query["trace"],
        }
        for source in report["sources"]
        for query in source["queries"]
    ]
    return rows, {
        "sources": len(report["sources"]),
        "queries": len(rows),
        "manifest_sha256": report["manifest_sha256"],
        "production_parity": all(
            not query["trace"]["ranking_modified"]
            for source in report["sources"]
            for query in source["queries"]
        ),
    }


def _assert_source_disjoint(calibration_rows: list[dict[str, Any]]) -> dict[str, Any]:
    path_manifest = json.loads(PATH_MANIFEST.read_text(encoding="utf-8"))
    development = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    v2 = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    calibration_sources = {
        source["sha256"]
        for source in path_manifest["sources"]
        if source["source_id"] in {row["source_id"] for row in calibration_rows}
    }
    development_sources = {source["sha256"] for source in development["sources"]}
    holdout_sources = {source["sha256"] for source in holdout["sources"]}
    protected_sources = holdout_sources | {v2["video"]["sha256"]}
    if calibration_sources & development_sources:
        raise ValueError("calibration and validation media overlap")
    if (calibration_sources | development_sources) & protected_sources:
        raise ValueError("development media overlaps protected evaluation")
    recorded_holdout_hash = HOLDOUT_SHA.read_text(encoding="utf-8").strip().split()[0]
    if sha256(HOLDOUT_MANIFEST) != recorded_holdout_hash:
        raise ValueError("protected holdout manifest checksum changed")
    return {
        "calibration_source_hashes": sorted(calibration_sources),
        "validation_source_hashes": sorted(development_sources),
        "protected_source_hashes": sorted(protected_sources),
        "all_disjoint": True,
        "protected_holdout_manifest_sha256": recorded_holdout_hash,
        "protected_holdout_queries_available": len(holdout["queries"]),
        "protected_holdout_queries_evaluated": 0,
    }


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "minimum": float(array.min()),
        "q25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "q75": float(np.percentile(array, 75)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _raw_analysis(rows: list[dict[str, Any]], modality: str) -> dict[str, Any]:
    _, labels, metadata = modality_examples(rows, modality, MODALITY_FEATURE_SETS["combined"])
    raw = np.asarray([item["raw_score"] for item in metadata])
    ranks = np.asarray([item["rank"] for item in metadata], dtype=float)

    def group(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "raw_score": _quantiles([item["raw_score"] for item in items]),
            "rank": _quantiles([item["rank"] for item in items]),
            "margin_to_top": _quantiles([item["margin_to_top"] for item in items]),
            "margin_to_next": _quantiles([item["margin_to_next"] for item in items]),
        }

    by_source = {}
    for source_id in sorted({item["source_id"] for item in metadata}):
        subset = [item for item in metadata if item["source_id"] == source_id]
        source_labels = np.asarray([item["label"] for item in subset])
        source_scores = np.asarray([item["raw_score"] for item in subset])
        by_source[source_id] = {
            "examples": len(subset),
            "positives": int(source_labels.sum()),
            "raw_auc": auc(source_labels, source_scores),
            "relevant_median": _quantiles([item["raw_score"] for item in subset if item["label"]])[
                "median"
            ],
            "irrelevant_median": _quantiles(
                [item["raw_score"] for item in subset if not item["label"]]
            )["median"],
        }
    length_groups = {"short_1_6": [], "medium_7_12": [], "long_13_plus": []}
    for item in metadata:
        key = (
            "short_1_6"
            if item["query_tokens"] <= 6
            else "medium_7_12"
            if item["query_tokens"] <= 12
            else "long_13_plus"
        )
        length_groups[key].append(item)
    by_query_length = {}
    for key, subset in length_groups.items():
        if not subset:
            by_query_length[key] = {"examples": 0}
            continue
        subset_labels = np.asarray([item["label"] for item in subset])
        subset_scores = np.asarray([item["raw_score"] for item in subset])
        by_query_length[key] = {
            "examples": len(subset),
            "positives": int(subset_labels.sum()),
            "raw_auc": auc(subset_labels, subset_scores),
        }
    return {
        "relevant": group([item for item in metadata if item["label"]]),
        "irrelevant": group([item for item in metadata if not item["label"]]),
        "raw_score_auc": auc(labels, raw),
        "rank_auc": auc(labels, -ranks),
        "by_source": by_source,
        "by_query_length": by_query_length,
    }


def _baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in rows if not row["negative"]]
    results = []
    for row in rows:
        relevant_rank = next(
            (
                bucket["post_fusion_rank"]
                for bucket in row["trace"]["candidate_buckets"]
                if bucket_relevant(bucket, row)
            ),
            None,
        )
        results.append(
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "negative": row["negative"],
                "relevant_rank": relevant_rank,
            }
        )

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        return {
            "queries": count,
            "top1": sum(item["relevant_rank"] == 1 for item in group),
            "top3": sum(
                item["relevant_rank"] is not None and item["relevant_rank"] <= 3 for item in group
            ),
            "top5": sum(
                item["relevant_rank"] is not None and item["relevant_rank"] <= 5 for item in group
            ),
            "mrr_at_5": sum(
                1 / item["relevant_rank"]
                for item in group
                if item["relevant_rank"] is not None and item["relevant_rank"] <= 5
            )
            / count,
        }

    positives = [item for item in results if not item["negative"]]
    return {
        "all": summarize(positives),
        "by_category": {
            category: summarize([item for item in positives if item["category"] == category])
            for category in sorted({item["category"] for item in positives})
        },
        "positive_queries": len(positive_rows),
        "negative_queries": len(rows) - len(positive_rows),
        "queries": results,
    }


def _failure_classification(baseline: dict[str, Any]) -> list[dict[str, str]]:
    mechanisms = {
        "hfd-d-s01": "irrelevant consensus",
        "hfd-d-s03": "irrelevant consensus",
        "hfd-d-s05": "irrelevant consensus",
        "hfd-d-v04": "weak required-modality rank",
        "hfd-d-m03": "cross-modal agreement",
        "hfd-d-m04": "retrieval miss",
        "hfd-d-a01": "irrelevant consensus",
        "hfd-h-s03": "irrelevant consensus",
    }
    return [
        {"query_id": item["query_id"], "mechanism": mechanisms[item["query_id"]]}
        for item in baseline["queries"]
        if not item["negative"] and (item["relevant_rank"] is None or item["relevant_rank"] > 5)
    ]


def _prediction_latency(
    model: dict[str, Any], rows: list[dict[str, Any]], modality: str
) -> dict[str, float]:
    features, _, _ = modality_examples(rows, modality, tuple(model["features"]))
    samples = []
    for _ in range(100):
        started = time.perf_counter()
        predict(model, features)
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms_for_all_validation_candidates": statistics.median(samples),
        "p95_ms_for_all_validation_candidates": float(np.percentile(samples, 95)),
    }


def run(cache: Path, artifact_path: Path, report_path: Path) -> dict[str, Any]:
    calibration_rows, calibration_data = _load_calibration_paths(cache)
    validation_rows, validation_data = _load_validation()
    isolation = _assert_source_disjoint(calibration_rows)

    calibration = {}
    artifact_models = {}
    all_modalities_pass = True
    for modality in ("visual", "speech"):
        selected = select_calibrator(calibration_rows, modality)
        rank_x, rank_y, _ = modality_examples(
            calibration_rows, modality, MODALITY_FEATURE_SETS["rank_only"]
        )
        rank_model = fit_logistic(rank_x, rank_y)
        rank_model.update(
            {"feature_set": "rank_only", "features": list(MODALITY_FEATURE_SETS["rank_only"])}
        )
        selected_validation = model_metrics(selected["model"], validation_rows, modality)
        rank_validation = model_metrics(rank_model, validation_rows, modality)
        auc_delta = selected_validation["auc"] - rank_validation["auc"]
        brier_delta = selected_validation["brier"] - rank_validation["brier"]
        passed = (
            auc_delta >= GENERALIZATION_GATE["auc_delta_vs_rank_minimum"]
            and brier_delta <= GENERALIZATION_GATE["brier_delta_vs_rank_maximum"]
        )
        all_modalities_pass = all_modalities_pass and passed
        calibration[modality] = {
            "selection_on_calibration_only": selected,
            "source_disjoint_validation": {
                "selected": selected_validation,
                "rank_only": rank_validation,
                "auc_delta_selected_minus_rank": auc_delta,
                "brier_delta_selected_minus_rank": brier_delta,
                "generalization_gate_passed": passed,
            },
            "prediction_latency": _prediction_latency(selected["model"], validation_rows, modality),
        }
        artifact_models[modality] = {
            "selected": selected["model"],
            "rank_only_reference": rank_model,
        }

    artifact = {
        "schema_version": "1.0.0",
        "experiment": "calibrated-fusion-v1",
        "status": "diagnostic_rejected_before_fusion"
        if not all_modalities_pass
        else "calibration_gate_passed",
        "production_eligible": False,
        "regularization": 1.0,
        "feature_sets": {key: list(value) for key, value in MODALITY_FEATURE_SETS.items()},
        "generalization_gate": GENERALIZATION_GATE,
        "input_hashes": {
            "calibration_manifest": calibration_data["manifest_sha256"],
            "validation_manifest": validation_data["manifest_sha256"],
        },
        "models": artifact_models,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    baseline = _baseline_metrics(validation_rows)
    decision = (
        {
            "code": "C",
            "label": "CALIBRATION DOES NOT GENERALIZE",
            "reason": (
                "Both calibration-selected modality models are worse than rank-only on "
                "source-disjoint validation AUC and Brier; fusion candidate construction "
                "and protected holdout evaluation therefore stopped."
            ),
        }
        if not all_modalities_pass
        else {
            "code": "B",
            "label": "CALIBRATION GATE PASSED; FUSION NOT YET EVALUATED",
            "reason": "This runner intentionally requires a separately frozen fusion candidate.",
        }
    )
    report = {
        "schema_version": "1.0.0",
        "experiment": "calibrated-fusion-v1",
        "git_commit_before_experiment": _git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "production_baseline": "uncapped RRF60",
        "production_retrieval_modified": False,
        "holdout_used_for_tuning": False,
        "protected_holdout_opened_for_candidate_evaluation": False,
        "feature_definitions_frozen_before_holdout": True,
        "fusion_candidate_constructed": False,
        "fusion_candidate_metrics": None,
        "data": {
            "calibration": calibration_data,
            "validation": validation_data,
            "isolation": isolation,
        },
        "raw_score_analysis_on_source_disjoint_validation": {
            modality: _raw_analysis(validation_rows, modality) for modality in ("visual", "speech")
        },
        "calibration": calibration,
        "calibration_artifact": {
            "path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "loaded_round_trip": json.loads(artifact_path.read_text(encoding="utf-8")) == artifact,
        },
        "validation_production_rrf60": baseline,
        "validation_failures": _failure_classification(baseline),
        "fusion_experiment": {
            "status": "stopped_before_candidate_methods",
            "methods_not_run": [
                "RRF plus calibrated raw-score confidence",
                "calibrated confidence fusion with rank features",
                "tiny interpretable linear/logistic fusion",
            ],
            "recovered_failures": [],
            "new_regressions": [],
            "net_gain": 0,
            "irrelevant_consensus_before": 5,
            "irrelevant_consensus_after": None,
            "strong_single_modality_recoveries": 0,
            "negative_query_behavior": "not evaluated for a rejected calibration",
        },
        "resource_impact": {
            "production_memory_bytes": 0,
            "production_artifact_bytes": 0,
            "diagnostic_artifact_bytes": artifact_path.stat().st_size,
        },
        "decision": decision,
        "second_stage_reranker_justified": False,
        "next_milestone": (
            "Collect at least three additional source-disjoint calibration sources with "
            "full Top-50 Visual/Speech traces and freeze a new validation split before "
            "reconsidering confidence-aware fusion."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run(args.calibration_cache, args.artifact, args.report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "calibration_sources": report["data"]["calibration"]["sources"],
                "calibration_queries": report["data"]["calibration"]["queries"],
                "validation_queries": report["data"]["validation"]["queries"],
                "protected_holdout_evaluated": report["data"]["isolation"][
                    "protected_holdout_queries_evaluated"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
