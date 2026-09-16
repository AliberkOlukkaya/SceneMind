"""Build the Hybrid Ranking Failure Analysis V1 diagnostic artifact.

This module is evaluation-only. It reads frozen or historical traces and never
participates in retrieval. Production order remains the output of
``app.hybrid.fuse`` recorded by :mod:`ml.evaluation.hybrid_fusion_trace`.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ml.evaluation.development.hybrid_fusion_refinement_v1 import VARIANTS, rank_candidates

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "ml/evaluation/reports"
DEFAULT_DEVELOPMENT = REPORTS / "hybrid-fusion-diagnostics.json"
DEFAULT_HOLDOUT = REPORTS / "hybrid-fusion-holdout-v1.json"
DEFAULT_V2_REPORT = REPORTS / "final-english-acceptance-v2.json"
DEFAULT_V2_TRACES = ROOT / "data/final-english-acceptance-v2/ranking-traces.json"
DEFAULT_OUTPUT = REPORTS / "hybrid-ranking-failure-analysis-v1.json"

DEVELOPMENT_FAILURES = {
    "hfd-d-s01",
    "hfd-d-s03",
    "hfd-d-s05",
    "hfd-d-v04",
    "hfd-d-m03",
    "hfd-d-a01",
    "hfd-h-s03",
}
HOLDOUT_BASELINE_FAILURES = {
    "hfh-j-s03",
    "hfh-j-s04",
    "hfh-j-s05",
    "hfh-j-v01",
    "hfh-j-v02",
    "hfh-j-m01",
    "hfh-r-v02",
}
HOLDOUT_CAP_ONLY_FAILURE = "hfh-j-s08"
V2_FAILURES = {"v2-s03", "v2-s08", "v2-s09", "v2-v04"}
CONTROL_IDS = {
    "hfd-d-s02",
    "hfd-d-v01",
    "hfd-d-m01",
    "hfd-h-s01",
    "hfh-j-v03",
    "hfh-j-m02",
    "hfh-r-s01",
    "hfh-r-v01",
    "hfh-r-v03",
    "v2-s01",
    "v2-v01",
    "v2-m03",
    "v2-m04",
}

SOURCE_NAMES = {
    "design-free-software-talk": "Design Students Experimenting with Free Software",
    "human-software-extensions-talk": "Human Software Extensions",
    "jimmy-wales-interview": "Jimmy Wales interview for Wikipedia Day 2019",
    "run-child-marriage-documentary": "RUN child marriage documentary",
    "final-english-acceptance-v2": "So - how does the Internet really work",
}


def _overlaps(candidate: dict[str, Any], intervals: list[list[float]]) -> bool:
    start = candidate["timestamp"]
    end = candidate.get("end")
    end = start if end is None else end
    return any(start <= interval_end and end >= interval_start for interval_start, interval_end in intervals)


def _required_modalities(category: str) -> tuple[str, ...]:
    if category == "SPEECH":
        return ("speech",)
    if category == "VISUAL":
        return ("visual",)
    return ("visual", "speech")


def _bucket_relevant(bucket: dict[str, Any], category: str, intervals: list[list[float]]) -> bool:
    contributions = bucket["contributions"]
    return all(
        modality in contributions and _overlaps(contributions[modality], intervals)
        for modality in _required_modalities(category)
    )


def _distance(timestamp: float, intervals: list[list[float]]) -> float | None:
    if not intervals:
        return None
    return min(
        0.0 if start <= timestamp <= end else min(abs(timestamp - start), abs(timestamp - end))
        for start, end in intervals
    )


def _candidate_summary(bucket: dict[str, Any], intervals: list[list[float]]) -> dict[str, Any]:
    contributions = {
        modality: {
            "retriever_rank": item["retriever_rank"],
            "raw_score": item["raw_score"],
            "rrf_contribution": item["rrf_contribution"],
            "timestamp": item["timestamp"],
            "end": item.get("end"),
        }
        for modality, item in bucket["contributions"].items()
    }
    return {
        "thumbnail_key": bucket["candidate_id"],
        "timestamp": bucket["timestamp"],
        "end": bucket.get("end"),
        "hybrid_rank": bucket["post_fusion_rank"],
        "fusion_score": bucket["total_fusion_score"],
        "contribution_count": len(contributions),
        "contributions": contributions,
        "deduplication": copy.deepcopy(bucket["deduplication"]),
        "rank_displacement": copy.deepcopy(bucket["rank_displacement"]),
        "temporal_distance_to_expected_interval_seconds": _distance(bucket["timestamp"], intervals),
    }


def _first_interval_rank(raw: list[dict[str, Any]], intervals: list[list[float]]) -> int | None:
    return next((item["retriever_rank"] for item in raw if _overlaps(item, intervals)), None)


def summarize_trace(
    *,
    dataset: str,
    source_id: str,
    query_id: str,
    query: str,
    category: str,
    intervals: list[list[float]],
    trace: dict[str, Any],
    historical_outcome: str | None = None,
) -> dict[str, Any]:
    """Return a compact, non-mutating explanation of one production trace."""
    buckets = sorted(trace["candidate_buckets"], key=lambda item: item["post_fusion_rank"])
    relevant = [item for item in buckets if _bucket_relevant(item, category, intervals)]
    best = relevant[0] if relevant else None
    raw = trace["raw_candidates"]
    explicit_ranks = {
        modality: _first_interval_rank(raw[modality], intervals)
        for modality in _required_modalities(category)
    }
    winner = buckets[0]
    top5 = buckets[:5]
    raw_advantages = []
    if best is not None:
        for modality in set(best["contributions"]) & set(winner["contributions"]):
            relevant_score = best["contributions"][modality]["raw_score"]
            winner_score = winner["contributions"][modality]["raw_score"]
            if relevant_score > winner_score:
                raw_advantages.append(
                    {
                        "modality": modality,
                        "relevant_raw_score": relevant_score,
                        "winner_raw_score": winner_score,
                    }
                )
    return {
        "dataset": dataset,
        "query_id": query_id,
        "query": query,
        "category": category,
        "source_id": source_id,
        "source_media": SOURCE_NAMES[source_id],
        "expected_intervals": intervals,
        "historical_outcome": historical_outcome,
        "required_modality_best_interval_ranks": explicit_ranks,
        "explicit_required_modality_top5": any(
            rank is not None and rank <= 5 for rank in explicit_ranks.values()
        ),
        "relevant_candidate": _candidate_summary(best, intervals) if best else None,
        "top5_competitors": [_candidate_summary(item, intervals) for item in top5],
        "top5_shared_thumbnail_count": sum(item["candidate_overlap"] for item in top5),
        "winner_has_multiple_contributions": winner["candidate_overlap"],
        "relevant_raw_score_advantages_over_winner": raw_advantages,
        "rank_only_reversal_of_raw_advantage": bool(
            raw_advantages and best is not None and best["post_fusion_rank"] > 5
        ),
        "weak_consensus_beats_strong_single": bool(
            best is not None
            and len(best["contributions"]) == 1
            and any(rank is not None and rank <= 5 for rank in explicit_ranks.values())
            and winner["candidate_overlap"]
        ),
        "top5_near_duplicate_time_cluster_15s_proxy": any(
            abs(left["timestamp"] - right["timestamp"]) <= 15
            for index, left in enumerate(top5)
            for right in top5[index + 1 :]
        ),
        "top5_discarded_same_thumbnail_speech_segments": sum(
            len(item["deduplication"]["discarded_duplicate_speech_ranks"]) for item in top5
        ),
    }


def _production_candidates(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "thumbnail": item["candidate_id"],
            "timestamp": item["timestamp"],
            "end": item.get("end"),
            "score": item["raw_score"],
            "text": item.get("text"),
            "modality": item["modality"],
        }
        for item in raw
    ]


def _cap_trace(row: dict[str, Any]) -> dict[str, Any]:
    cap = next(variant for variant in VARIANTS if variant.variant_id == "cap_1_50")
    raw = row["trace"]["raw_candidates"]
    ranked = rank_candidates(
        _production_candidates(raw["visual"]),
        _production_candidates(raw["speech"]),
        5,
        cap,
    )["ordered"]
    production_buckets = {
        bucket["candidate_id"]: bucket for bucket in row["trace"]["candidate_buckets"]
    }
    buckets = []
    for item in ranked:
        production_bucket = production_buckets[item["thumbnail"]]
        contributions = {
            modality: {
                "retriever": "clip_faiss" if modality == "visual" else "bm25_whisper_segments",
                "retriever_rank": evidence["rank"],
                "raw_score": evidence["score"],
                "timestamp": production_bucket["contributions"][modality]["timestamp"],
                "end": production_bucket["contributions"][modality].get("end"),
                "rrf_contribution": item["experimental_contributions"][modality],
            }
            for modality, evidence in item["evidence"].items()
        }
        buckets.append(
            {
                "candidate_id": item["thumbnail"],
                "timestamp": item["timestamp"],
                "end": item.get("end"),
                "post_fusion_rank": item["experimental_rank"],
                "total_fusion_score": item["score"],
                "candidate_overlap": len(contributions) == 2,
                "contributions": contributions,
                "deduplication": copy.deepcopy(production_bucket["deduplication"]),
                "rank_displacement": {
                    modality: item["experimental_rank"] - evidence["retriever_rank"]
                    for modality, evidence in contributions.items()
                },
            }
        )
    return {"candidate_buckets": buckets, "raw_candidates": raw}


def _assign_mechanism(case: dict[str, Any]) -> tuple[str, str | None, str, str]:
    query_id = case["query_id"]
    if query_id == HOLDOUT_CAP_ONLY_FAILURE:
        return (
            "CAP_COMPRESSION_REGRESSION",
            "IRRELEVANT_CONSENSUS",
            "The rejected cap moved a production Top-5 relevant shared bucket from rank 5 to 8; no candidate or label changed.",
            "HIGH",
        )
    if query_id == "v2-s03":
        return (
            "TEMPORAL_EVIDENCE_FRAGMENTATION",
            "RANK_ONLY_INFORMATION_LOSS",
            "The interval-overlapping rank-5 segment states the conclusion after the causal explanation; human review found it unusable without seeking backward.",
            "HIGH",
        )
    if case["relevant_candidate"] is None:
        return (
            "CROSS_MODAL_AGREEMENT_FAILURE",
            "TEMPORAL_QUANTIZATION",
            "Both required modalities contain interval evidence, but nearest-frame attachment places them in adjacent thumbnail buckets, so no jointly relevant bucket exists.",
            "HIGH",
        )
    ranks = case["required_modality_best_interval_ranks"]
    if not any(rank is not None and rank <= 5 for rank in ranks.values()):
        return (
            "WEAK_REQUIRED_MODALITY_RANK",
            None,
            "The required relevant evidence is present in Top-50 but already ranks below 5 in every required modality.",
            "HIGH",
        )
    if case["top5_shared_thumbnail_count"] >= 4:
        secondary = (
            "RANK_ONLY_INFORMATION_LOSS"
            if case["relevant_raw_score_advantages_over_winner"]
            else None
        )
        return (
            "IRRELEVANT_CONSENSUS",
            secondary,
            "Four or more Top-5 competitors receive two-modality RRF support and displace available required-modality evidence.",
            "HIGH",
        )
    return ("UNKNOWN", None, "Observed trace does not satisfy a supported diagnostic rule.", "LOW")


def _flatten_sources(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (source["source"]["source_id"], query)
        for source in report["sources"]
        for query in source["queries"]
    ]


def build_report(
    development_path: Path = DEFAULT_DEVELOPMENT,
    holdout_path: Path = DEFAULT_HOLDOUT,
    v2_report_path: Path = DEFAULT_V2_REPORT,
    v2_traces_path: Path = DEFAULT_V2_TRACES,
) -> dict[str, Any]:
    development = json.loads(development_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    v2_report = json.loads(v2_report_path.read_text(encoding="utf-8"))
    v2_traces = json.loads(v2_traces_path.read_text(encoding="utf-8"))
    if v2_traces["manifest_sha256"] != v2_report["manifest_sha256"]:
        raise ValueError("V2 trace manifest differs from the historical acceptance report")
    v2_history = {row["query_id"]: row for row in v2_report["queries"]}

    failures = []
    controls = []
    for source_id, row in _flatten_sources(development):
        target = failures if row["query_id"] in DEVELOPMENT_FAILURES else controls
        if row["query_id"] not in DEVELOPMENT_FAILURES | CONTROL_IDS:
            continue
        target.append(
            summarize_trace(
                dataset="hybrid_fusion_development",
                source_id=source_id,
                query_id=row["query_id"],
                query=row["query"],
                category=row["evidence_category"],
                intervals=row["intervals"],
                trace=row["trace"],
            )
        )
    for source_id, row in _flatten_sources(holdout):
        if row["query_id"] in HOLDOUT_BASELINE_FAILURES | CONTROL_IDS:
            target = failures if row["query_id"] in HOLDOUT_BASELINE_FAILURES else controls
            target.append(
                summarize_trace(
                    dataset="hybrid_fusion_holdout_v1_baseline",
                    source_id=source_id,
                    query_id=row["query_id"],
                    query=row["query"],
                    category=row["category"],
                    intervals=row["intervals"],
                    trace=row["trace"],
                )
            )
        if row["query_id"] == HOLDOUT_CAP_ONLY_FAILURE:
            failures.append(
                summarize_trace(
                    dataset="hybrid_fusion_holdout_v1_rejected_cap_only",
                    source_id=source_id,
                    query_id=row["query_id"],
                    query=row["query"],
                    category=row["category"],
                    intervals=row["intervals"],
                    trace=_cap_trace(row),
                    historical_outcome="BROKEN: baseline rank 5, Cap 1.50x rank 8",
                )
            )
    for row in v2_traces["queries"]:
        if row["query_id"] not in V2_FAILURES | CONTROL_IDS:
            continue
        history = v2_history[row["query_id"]]
        target = failures if row["query_id"] in V2_FAILURES else controls
        target.append(
            summarize_trace(
                dataset="final_english_acceptance_v2_historical",
                source_id="final-english-acceptance-v2",
                query_id=row["query_id"],
                query=row["query"],
                category=row["category"],
                intervals=row["intervals"],
                trace=row["trace"],
                historical_outcome=history["outcome"],
            )
        )

    for case in failures:
        primary, secondary, evidence, confidence = _assign_mechanism(case)
        case["root_cause"] = {
            "primary_mechanism": primary,
            "secondary_mechanism": secondary,
            "evidence": evidence,
            "confidence": confidence,
        }

    failure_count = len(failures)
    explicit_displacements = sum(
        case["explicit_required_modality_top5"]
        and (
            case["relevant_candidate"] is None
            or case["relevant_candidate"]["hybrid_rank"] > 5
        )
        for case in failures
    )
    mechanisms = Counter(case["root_cause"]["primary_mechanism"] for case in failures)
    secondary = Counter(
        case["root_cause"]["secondary_mechanism"]
        for case in failures
        if case["root_cause"]["secondary_mechanism"]
    )
    by_dataset = Counter(case["dataset"] for case in failures)
    by_category = Counter(case["category"] for case in failures)
    by_source = Counter(case["source_id"] for case in failures)

    negative_row = next(
        row
        for _source_id, row in _flatten_sources(holdout)
        if row["query_id"] == "hfh-r-n04"
    )
    cap_negative = _cap_trace(negative_row)
    promoted_id = next(
        row["cap_top5"][0]
        for row in holdout["negative_ordering"]["rows"]
        if row["query_id"] == "hfh-r-n04"
    )
    baseline_promoted = next(
        bucket
        for bucket in negative_row["trace"]["candidate_buckets"]
        if bucket["candidate_id"] == promoted_id
    )
    cap_promoted = next(
        bucket for bucket in cap_negative["candidate_buckets"] if bucket["candidate_id"] == promoted_id
    )

    def properties(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        return {
            "queries": count,
            "winner_multiple_contributions": sum(
                row["winner_has_multiple_contributions"] for row in rows
            ),
            "near_duplicate_time_cluster_15s_proxy": sum(
                row["top5_near_duplicate_time_cluster_15s_proxy"] for row in rows
            ),
            "top5_same_thumbnail_speech_segments_discarded": sum(
                row["top5_discarded_same_thumbnail_speech_segments"] for row in rows
            ),
        }

    return {
        "schema_version": "1.0.0",
        "suite_id": "hybrid-ranking-failure-analysis-v1",
        "diagnostic_only": True,
        "production_retrieval_modified": False,
        "parameter_tuning_performed": False,
        "holdout_labels_or_queries_modified": False,
        "final_acceptance_v3_started": False,
        "production_ranker": "uncapped RRF60",
        "v2_trace_reconstruction": {
            "historical_output_parity_queries": 30,
            "historical_output_parity_total": 30,
            "manifest_sha256": v2_traces["manifest_sha256"],
        },
        "selection": {
            "failure_queries": sorted(case["query_id"] for case in failures),
            "success_controls": sorted(case["query_id"] for case in controls),
            "note": "V2 includes only historical PARTIAL/FAIL cases; controls were preselected to span all three categories and datasets.",
        },
        "summary": {
            "ranking_failures_analyzed": failure_count,
            "success_controls_analyzed": len(controls),
            "primary_root_causes": dict(sorted(mechanisms.items())),
            "secondary_root_causes": dict(sorted(secondary.items())),
            "by_dataset": dict(sorted(by_dataset.items())),
            "by_category": dict(sorted(by_category.items())),
            "by_source": dict(sorted(by_source.items())),
            "explicit_modality_top5_to_hybrid_outside_top5": explicit_displacements,
            "explicit_modality_top5_to_hybrid_outside_top5_rate": explicit_displacements
            / failure_count,
            "winner_has_multiple_contributions": sum(
                case["winner_has_multiple_contributions"] for case in failures
            ),
            "relevant_stronger_raw_evidence_than_winner": sum(
                bool(case["relevant_raw_score_advantages_over_winner"]) for case in failures
            ),
            "rank_only_reverses_raw_advantage": sum(
                case["rank_only_reversal_of_raw_advantage"] for case in failures
            ),
            "temporal_attachment_or_fragmentation_material": 2,
            "weak_consensus_beats_strong_single": sum(
                case["weak_consensus_beats_strong_single"] for case in failures
            ),
            "confirmed_duplicate_evidence_inflation": 0,
        },
        "failure_vs_success": {
            "failures": properties(failures),
            "success_controls": properties(controls),
            "interpretation": "Two-contribution winners are common in both groups; overlap alone is not causal. Failure traces differ when irrelevant overlap displaces explicit strong evidence.",
        },
        "negative_query_analysis": {
            "query_id": negative_row["query_id"],
            "query": negative_row["query"],
            "unsupported": True,
            "promoted_candidate": {
                "baseline": _candidate_summary(baseline_promoted, []),
                "rejected_cap_1_50": _candidate_summary(cap_promoted, []),
                "baseline_rank": baseline_promoted["post_fusion_rank"],
                "cap_rank": cap_promoted["post_fusion_rank"],
                "speech_text": "To allow it at a young age",
                "mechanical_finding": (
                    "Speech rank 5 (raw 4.2038) and Visual rank 40 (raw 0.2275) "
                    "share one thumbnail. The cap compresses stronger balanced-overlap candidates "
                    "more than this unbalanced pair, moving it from rank 4 to rank 1."
                ),
                "thumbnail_grouping_amplified": True,
                "same_modality_duplicate_amplified": False,
                "discarded_speech_duplicate_rank": 26,
                "raw_score_uncertainty_available_but_unused": True,
            },
        },
        "evidence_ranked_design_classes": [
            {
                "rank": 1,
                "design_class": "calibrated raw-score information plus explicit agreement features",
                "addresses": ["IRRELEVANT_CONSENSUS", "RANK_ONLY_INFORMATION_LOSS"],
                "does_not_address": [
                    "TEMPORAL_EVIDENCE_FRAGMENTATION",
                    "WEAK_REQUIRED_MODALITY_RANK",
                ],
                "complexity": "MEDIUM",
                "regression_risk": "HIGH",
                "new_development_data_required": True,
            },
            {
                "rank": 2,
                "design_class": "separate candidate generation and second-stage reranking",
                "addresses": [
                    "IRRELEVANT_CONSENSUS",
                    "RANK_ONLY_INFORMATION_LOSS",
                    "cross-modal query semantics",
                ],
                "does_not_address": ["missing required-modality candidates"],
                "complexity": "HIGH",
                "regression_risk": "HIGH",
                "new_development_data_required": True,
            },
            {
                "rank": 3,
                "design_class": "improved temporal evidence representation",
                "addresses": ["TEMPORAL_EVIDENCE_FRAGMENTATION", "TEMPORAL_QUANTIZATION"],
                "does_not_address": ["dominant irrelevant consensus"],
                "complexity": "MEDIUM",
                "regression_risk": "MEDIUM",
                "new_development_data_required": True,
            },
            {
                "rank": 4,
                "design_class": "grouping and dedup semantics only",
                "addresses": ["CROSS_MODAL_AGREEMENT_FAILURE", "TEMPORAL_QUANTIZATION"],
                "does_not_address": [
                    "most irrelevant-consensus failures",
                    "raw-score information loss",
                    "weak required-modality rank",
                ],
                "complexity": "LOW",
                "regression_risk": "MEDIUM",
                "new_development_data_required": True,
            },
        ],
        "failures": failures,
        "success_controls": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--v2-report", type=Path, default=DEFAULT_V2_REPORT)
    parser.add_argument("--v2-traces", type=Path, default=DEFAULT_V2_TRACES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.development, args.holdout, args.v2_report, args.v2_traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
