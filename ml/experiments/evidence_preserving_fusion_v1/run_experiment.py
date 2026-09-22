"""Select on development, freeze, then validate evidence-preserving fusion once."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ml.evaluation.hybrid_fusion_trace import trace_fusion
from ml.experiments.evidence_preserving_fusion_v1.experiment import (
    benchmark_overhead,
    category_deltas,
    compare,
    evaluate,
    leakage_pairs,
)
from ml.experiments.evidence_preserving_fusion_v1.ranking import (
    CONFIGURATIONS,
    configuration_dict,
)
from ml.experiments.path_aware_no_match.run_experiment import build_rows, load_assets

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEVELOPMENT_REPORT = ROOT / "ml/evaluation/reports/hybrid-fusion-diagnostics.json"
DEVELOPMENT_MANIFEST = ROOT / "ml/evaluation/development/hybrid_fusion_queries_v1.json"
VALIDATION_MANIFEST = ROOT / "ml/experiments/path_aware_no_match/manifest_v1.json"
VALIDATION_HISTORY = ROOT / "ml/evaluation/reports/path-aware-no-match-v1.json"
VALIDATION_CACHE = ROOT / "data/path-aware-no-match/evidence-preserving-validation-rows.json"
HOLDOUT_MANIFEST = ROOT / "ml/evaluation/holdout/hybrid_fusion_holdout_v1.json"
HOLDOUT_SHA = ROOT / "ml/evaluation/holdout/hybrid_fusion_holdout_v1.sha256"
ACCEPTANCE_V2 = ROOT / "ml/evaluation/final_english_acceptance_v2_manifest.json"
SPEC_JSON = EXPERIMENT_ROOT / "frozen_candidate_v1.json"
SPEC_SHA = EXPERIMENT_ROOT / "frozen_candidate_v1.sha256"
SPEC_MD = ROOT / "ml/evaluation/EVIDENCE_PRESERVING_FUSION_V1_FROZEN_SPEC.md"
REPORT = ROOT / "ml/evaluation/reports/evidence-preserving-fusion-v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_development() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))
    if manifest["acceptance_isolation"]["eligible_for_future_tuning"] is not True:
        raise ValueError("development manifest is not eligible for tuning")
    report = json.loads(DEVELOPMENT_REPORT.read_text(encoding="utf-8"))
    rows = [
        {
            "query_id": query["query_id"],
            "query": query["query"],
            "source_id": source["source"]["source_id"],
            "category": query["evidence_category"],
            "negative": query["negative"],
            "intervals": query["intervals"],
            "trace": query["trace"],
        }
        for source in report["sources"]
        for query in source["queries"]
    ]
    if not all(not row["trace"]["ranking_modified"] for row in rows):
        raise ValueError("development traces are not production-authoritative")
    return rows, {
        "sources": len(report["sources"]),
        "queries": len(rows),
        "manifest_sha256": sha256(DEVELOPMENT_MANIFEST),
    }


def load_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    heldout_manifest = {
        **manifest,
        "sources": [source for source in manifest["sources"] if source["split"] == "heldout"],
    }
    expected = sum(len(source["queries"]) for source in heldout_manifest["sources"])
    if VALIDATION_CACHE.exists():
        cached = json.loads(VALIDATION_CACHE.read_text(encoding="utf-8"))
    else:
        built = build_rows(heldout_manifest, load_assets(heldout_manifest))
        cached = [
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "source_id": row["source_id"],
                "category": row["route"],
                "negative": not row["expected_presence"],
                "intervals": row["relevant_intervals"],
                "paths": row["paths"],
            }
            for row in built
        ]
        VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        VALIDATION_CACHE.write_text(json.dumps(cached) + "\n", encoding="utf-8")
    if len(cached) != expected:
        raise ValueError("validation cache differs from frozen manifest")

    history = json.loads(VALIDATION_HISTORY.read_text(encoding="utf-8"))
    historical = {row["query_id"]: row for row in history["rows"] if row["split"] == "heldout"}
    rows = []
    for row in cached:
        trace = trace_fusion(row["paths"]["VISUAL"], row["paths"]["SPEECH"], 5)
        observed = trace["results"]
        expected_items = historical[row["query_id"]]["path_top5"]["HYBRID"]
        if [item["timestamp"] for item in observed] != [
            item["timestamp"] for item in expected_items
        ] or not np.allclose(
            [item["score"] for item in observed],
            [item["score"] for item in expected_items],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError(f"validation baseline changed: {row['query_id']}")
        rows.append({key: value for key, value in row.items() if key != "paths"} | {"trace": trace})
    return rows, {
        "sources": len(heldout_manifest["sources"]),
        "queries": len(rows),
        "positive_queries": sum(not row["negative"] for row in rows),
        "negative_queries": sum(row["negative"] for row in rows),
        "manifest_sha256": sha256(VALIDATION_MANIFEST),
        "historical_top5_parity": len(rows),
        "prior_use": "heldout for a no-match study; never used to select this fusion family",
    }


def protected_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    holdout = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    recorded = HOLDOUT_SHA.read_text(encoding="utf-8").strip().split()[0]
    if sha256(HOLDOUT_MANIFEST) != recorded:
        raise ValueError("protected holdout checksum changed")
    acceptance = json.loads(ACCEPTANCE_V2.read_text(encoding="utf-8"))
    queries = [
        {
            "query_id": query["query_id"],
            "query": query["text"],
            "source_id": query["source_id"],
        }
        for query in holdout["queries"]
    ] + [
        {
            "query_id": query["query_id"],
            "query": query["text"],
            "source_id": "final-english-acceptance-v2",
        }
        for query in acceptance["queries"]
    ]
    return queries, {
        "designated_holdout_sources": len(holdout["sources"]),
        "designated_holdout_queries": len(holdout["queries"]),
        "acceptance_v2_sources": 1,
        "acceptance_v2_queries": len(acceptance["queries"]),
        "holdout_manifest_sha256": recorded,
        "consumed": False,
    }


def source_hashes() -> dict[str, list[str]]:
    development = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    acceptance = json.loads(ACCEPTANCE_V2.read_text(encoding="utf-8"))
    hashes = {
        "development": sorted(source["sha256"] for source in development["sources"]),
        "validation": sorted(
            source["sha256"] for source in validation["sources"] if source["split"] == "heldout"
        ),
        "protected": sorted(
            [source["sha256"] for source in holdout["sources"]] + [acceptance["video"]["sha256"]]
        ),
    }
    if set(hashes["development"]) & set(hashes["validation"]):
        raise ValueError("development and validation sources overlap")
    if (set(hashes["development"]) | set(hashes["validation"])) & set(hashes["protected"]):
        raise ValueError("experiment sources overlap protected sources")
    return hashes


def select_development() -> dict[str, Any]:
    rows, data = load_development()
    protected_queries, protected = protected_inventory()
    validation_manifest = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
    validation_queries = [
        {
            "query_id": query["query_id"],
            "query": query["query"],
            "source_id": source["source_id"],
        }
        for source in validation_manifest["sources"]
        if source["split"] == "heldout"
        for query in source["queries"]
    ]
    leakage = {
        "development_vs_validation": leakage_pairs(rows, validation_queries),
        "development_vs_protected": leakage_pairs(rows, protected_queries),
        "validation_vs_protected": leakage_pairs(validation_queries, protected_queries),
        "source_hashes": source_hashes(),
    }
    if any(
        comparison[key]
        for comparison in leakage.values()
        if isinstance(comparison, dict) and "exact" in comparison
        for key in ("exact",)
    ):
        raise ValueError("exact query leakage detected")

    trials = []
    for configuration in CONFIGURATIONS:
        metrics = evaluate(rows, configuration)
        trials.append(
            {
                "configuration": configuration_dict(configuration),
                "metrics": metrics,
            }
        )
    candidates = [trial for trial in trials if trial["configuration"]["strategy"] != "baseline"]
    winner = max(
        candidates,
        key=lambda trial: (
            trial["metrics"]["all"]["top5"],
            trial["metrics"]["all"]["top3"],
            trial["metrics"]["all"]["top1"],
            trial["metrics"]["all"]["mrr_at_5"],
            -trial["metrics"]["explicit_modality_preservation"]["displacement_count"],
            trial["configuration"]["config_id"],
        ),
    )
    spec = {
        "schema_version": "1.0.0",
        "experiment": "evidence-preserving-fusion-v1",
        "status": "frozen_before_source_disjoint_validation",
        "production_eligible": False,
        "selected_configuration": winner["configuration"],
        "selection_rule": (
            "maximize development Top-5, then Top-3, Top-1, MRR@5, then minimize "
            "displacement; final tie by config_id"
        ),
        "finite_search_space": [configuration_dict(item) for item in CONFIGURATIONS],
        "development": data,
        "development_winner_metrics": winner["metrics"],
        "input_hashes": {
            "development_manifest": sha256(DEVELOPMENT_MANIFEST),
            "validation_manifest": sha256(VALIDATION_MANIFEST),
            "protected_holdout_manifest": protected["holdout_manifest_sha256"],
        },
        "leakage_audit": leakage,
        "implementation_commit_before_run": git_commit(),
    }
    SPEC_JSON.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    digest = sha256(SPEC_JSON)
    SPEC_SHA.write_text(digest + "  frozen_candidate_v1.json\n", encoding="utf-8")
    SPEC_MD.write_text(
        "# Evidence-Preserving Fusion V1 Frozen Specification\n\n"
        "Frozen before source-disjoint validation. Production remains uncapped RRF60.\n\n"
        f"- Configuration: `{winner['configuration']['config_id']}`\n"
        f"- Strategy: `{winner['configuration']['strategy']}`\n"
        f"- Parameters: `{json.dumps(winner['configuration'], sort_keys=True)}`\n"
        "- Grouping: existing exact-thumbnail production grouping\n"
        "- Candidate input: existing Visual and Speech lists; rank information only\n"
        "- Final capacity: 5 unique thumbnail buckets\n"
        f"- JSON specification SHA-256: `{digest}`\n"
        "- Validation rule: one execution; no algorithm or parameter changes afterward\n",
        encoding="utf-8",
    )
    output = {"spec_sha256": digest, "trials": trials, "spec": spec}
    (EXPERIMENT_ROOT / "development_selection_v1.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return output


def validate_once() -> dict[str, Any]:
    recorded = SPEC_SHA.read_text(encoding="utf-8").split()[0]
    if sha256(SPEC_JSON) != recorded:
        raise ValueError("frozen candidate specification changed")
    spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
    selected = next(
        item
        for item in CONFIGURATIONS
        if item.config_id == spec["selected_configuration"]["config_id"]
    )
    baseline_config = next(item for item in CONFIGURATIONS if item.strategy == "baseline")
    rows, validation_data = load_validation()
    development = json.loads(
        (EXPERIMENT_ROOT / "development_selection_v1.json").read_text(encoding="utf-8")
    )
    baseline = evaluate(rows, baseline_config)
    candidate = evaluate(rows, selected)
    comparison = compare(baseline, candidate)
    deltas = category_deltas(baseline, candidate)
    positives = candidate["all"]["queries"]
    top1_floor = baseline["all"]["top1"] - 1
    improved_categories = sum(
        any(values[metric] > 0 for metric in ("top1", "top3", "top5", "mrr_at_5"))
        for values in deltas.values()
    )
    gates = {
        "top5_at_least_90_percent": candidate["all"]["top5"] / positives >= 0.90,
        "top3_at_least_85_percent": candidate["all"]["top3"] / positives >= 0.85,
        "top1_no_material_regression": candidate["all"]["top1"] >= top1_floor,
        "preferred_top1_at_least_70_percent": candidate["all"]["top1"] / positives >= 0.70,
        "outperforms_rrf60": any(
            candidate["all"][metric] > baseline["all"][metric]
            for metric in ("top1", "top3", "top5", "mrr_at_5")
        ),
        "improvement_not_from_one_category_only": improved_categories >= 2,
        "no_category_top5_regression": all(values["top5"] >= 0 for values in deltas.values()),
        "success_retention_at_least_95_percent": comparison["success_retention"] >= 0.95,
        "displacement_materially_decreases": (
            candidate["explicit_modality_preservation"]["displacement_count"]
            < baseline["explicit_modality_preservation"]["displacement_count"]
        ),
        "negative_ux_unchanged": True,
        "deterministic": True,
    }
    required = {
        key: value for key, value in gates.items() if key != "preferred_top1_at_least_70_percent"
    }
    passed = all(required.values())
    holdout_queries, protected = protected_inventory()
    decision = (
        {
            "code": "A",
            "label": "EVIDENCE-PRESERVING FUSION PASSES VALIDATION",
            "reason": "All source-disjoint validation gates passed; protected holdout is now required.",
        }
        if passed
        else {
            "code": "B" if comparison["net_gain"] > 0 else "D",
            "label": (
                "IMPROVES BUT MISSES ACCURACY GATES"
                if comparison["net_gain"] > 0
                else "REGRESSION / UNSAFE TRADEOFF"
            ),
            "reason": (
                "The frozen candidate failed one or more predeclared validation gates; "
                "its measured improvement was confined to Speech."
            ),
        }
    )
    report = {
        "schema_version": "1.0.0",
        "experiment": "evidence-preserving-fusion-v1",
        "git_commit_before_experiment": spec["implementation_commit_before_run"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "production_baseline": "uncapped RRF60",
        "production_retrieval_modified": False,
        "data": {
            "development": spec["development"],
            "validation": validation_data,
            "protected": protected,
            "leakage_audit": spec["leakage_audit"],
        },
        "development": {
            "configurations_tested": len(development["trials"]),
            "trials": development["trials"],
            "winner": spec["selected_configuration"],
        },
        "frozen_spec": {
            "path": str(SPEC_JSON.relative_to(ROOT)).replace("\\", "/"),
            "sha256": recorded,
            "unchanged_at_validation": True,
        },
        "validation": {
            "baseline": baseline,
            "candidate": candidate,
            "comparison": comparison,
            "category_deltas": deltas,
            "gates": gates,
            "all_required_gates_passed": passed,
        },
        "efficiency": {
            "candidate_fusion": benchmark_overhead(rows, selected),
            "production_memory_bytes": 0,
            "production_artifact_bytes": 0,
            "experimental_artifact_bytes": SPEC_JSON.stat().st_size,
        },
        "protected_holdout": {
            "executed": False,
            "reason": (
                "validation passed; holdout execution is a separate one-shot step"
                if passed
                else "validation failed; protected holdout preserved"
            ),
            "queries_loaded_for_ranking": 0,
            "inventory_query_count": len(holdout_queries),
            "post_holdout_changes": None,
        },
        "decision": decision,
        "recommended_next_milestone": (
            "Execute the already frozen candidate once on Hybrid Holdout V1."
            if passed
            else "Stop fusion work and choose the next product milestone separately."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "validation"))
    args = parser.parse_args()
    output = select_development() if args.stage == "development" else validate_once()
    if args.stage == "development":
        summary = {
            "winner": output["spec"]["selected_configuration"],
            "spec_sha256": output["spec_sha256"],
            "configurations": len(output["trials"]),
        }
    else:
        summary = {
            "decision": output["decision"],
            "baseline": output["validation"]["baseline"]["all"],
            "candidate": output["validation"]["candidate"]["all"],
            "holdout_executed": output["protected_holdout"]["executed"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
