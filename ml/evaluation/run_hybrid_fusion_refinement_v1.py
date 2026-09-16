"""Evaluate predeclared offline Hybrid fusion variants on frozen diagnostic traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ml.evaluation.development.hybrid_fusion_refinement_v1 import (
    VARIANTS,
    Variant,
    production_shape,
    rank_candidates,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(__file__).parent / "reports/hybrid-fusion-diagnostics.json"
DEFAULT_OUTPUT = Path(__file__).parent / "reports/hybrid-fusion-refinement-v1.json"
PROTECTED_V2 = Path(__file__).parent / "final_english_acceptance_v2_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _production_candidates(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for item in raw:
        candidate = {
            "thumbnail": item["candidate_id"],
            "timestamp": item["timestamp"],
            "score": item["raw_score"],
            "modality": item["modality"],
        }
        if item["modality"] == "speech":
            candidate.update(end=item["end"], text=item["text"])
        candidates.append(candidate)
    return candidates


def _query_input(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = row["trace"]["raw_candidates"]
    return _production_candidates(raw["visual"]), _production_candidates(raw["speech"])


def _required_modalities(category: str) -> tuple[str, ...]:
    if category == "SPEECH":
        return ("speech",)
    if category == "VISUAL":
        return ("visual",)
    if category == "MULTIMODAL":
        return ("speech", "visual")
    return ()


def _overlaps(candidate: dict[str, Any], intervals: list[list[float]]) -> bool:
    start = candidate["timestamp"]
    end = candidate.get("end")
    end = start if end is None else end
    return any(start <= interval_end and end >= interval_start for interval_start, interval_end in intervals)


def _strong_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    strong = []
    raw = row["trace"]["raw_candidates"]
    for modality in _required_modalities(row["evidence_category"]):
        match = next(
            (
                item
                for item in raw[modality]
                if item["retriever_rank"] <= 5 and _overlaps(item, row["intervals"])
            ),
            None,
        )
        if match:
            strong.append({"modality": modality, "candidate_id": match["candidate_id"]})
    return strong


def _score_variant(row: dict[str, Any], variant: Variant) -> dict[str, Any]:
    visual, speech = _query_input(row)
    ranked = rank_candidates(visual, speech, 5, variant)
    relevant_ids = {
        bucket["candidate_id"]
        for bucket in row["trace"]["candidate_buckets"]
        if bucket["matches_development_evidence"]
    }
    relevant_rank = next(
        (
            item["experimental_rank"]
            for item in ranked["ordered"]
            if item["thumbnail"] in relevant_ids
        ),
        None,
    )
    top5_ids = [item["thumbnail"] for item in ranked["results"]]
    strong = _strong_candidates(row)
    return {
        "query_id": row["query_id"],
        "source_id": row["source_id"],
        "category": row["evidence_category"],
        "negative": row["negative"],
        "intervals": row["intervals"],
        "relevant_rank": relevant_rank,
        "top1": relevant_rank == 1,
        "top3": relevant_rank is not None and relevant_rank <= 3,
        "top5": relevant_rank is not None and relevant_rank <= 5,
        "reciprocal_rank_at_5": 1 / relevant_rank if relevant_rank and relevant_rank <= 5 else 0.0,
        "top5_ids": top5_ids,
        "top5_shared": sum(len(item["evidence"]) == 2 for item in ranked["results"]),
        "strong_candidates": [
            {
                **candidate,
                "retained": candidate["candidate_id"] in top5_ids,
                "post_fusion_rank": next(
                    item["experimental_rank"]
                    for item in ranked["ordered"]
                    if item["thumbnail"] == candidate["candidate_id"]
                ),
            }
            for candidate in strong
        ],
        "ranked": ranked,
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if not row["negative"]]
    strong = [candidate for row in positives for candidate in row["strong_candidates"]]
    return {
        "queries": len(positives),
        "top1": sum(row["top1"] for row in positives),
        "top3": sum(row["top3"] for row in positives),
        "top5": sum(row["top5"] for row in positives),
        "mrr_at_5": statistics.fmean(row["reciprocal_rank_at_5"] for row in positives),
        "shared_thumbnail_top5": sum(row["top5_shared"] for row in positives),
        "shared_thumbnail_top5_rate": sum(row["top5_shared"] for row in positives)
        / (5 * len(positives)),
        "strong_candidates": len(strong),
        "strong_candidates_retained": sum(candidate["retained"] for candidate in strong),
        "strong_candidate_retention_rate": (
            sum(candidate["retained"] for candidate in strong) / len(strong) if strong else 0.0
        ),
        "strong_candidate_mean_rank": (
            statistics.fmean(candidate["post_fusion_rank"] for candidate in strong)
            if strong
            else None
        ),
    }


def _variant_summary(rows: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_by_id = {row["query_id"]: row for row in baseline}
    classifications = Counter()
    for row in rows:
        if row["negative"]:
            continue
        before = baseline_by_id[row["query_id"]]["top5"]
        after = row["top5"]
        label = (
            "RESCUED"
            if not before and after
            else "BROKEN"
            if before and not after
            else "UNCHANGED_SUCCESS"
            if before
            else "UNCHANGED_FAILURE"
        )
        classifications[label] += 1
    categories = {}
    for category in ("ALL", "SPEECH", "VISUAL", "MULTIMODAL"):
        group = rows if category == "ALL" else [row for row in rows if row["category"] == category]
        categories[category] = _metrics(group)
    negatives = [row for row in rows if row["negative"]]
    baseline_negatives = {row["query_id"]: row for row in baseline if row["negative"]}
    negative_changes = []
    for row in negatives:
        before = baseline_negatives[row["query_id"]]["top5_ids"]
        after = row["top5_ids"]
        negative_changes.append({
            "query_id": row["query_id"],
            "identical_order": before == after,
            "same_candidate_set": set(before) == set(after),
            "baseline_top5": before,
            "experimental_top5": after,
        })
    return {
        "metrics": categories,
        "outcomes": dict(classifications),
        "negative_ordering": {
            "queries": len(negative_changes),
            "identical_order": sum(change["identical_order"] for change in negative_changes),
            "same_candidate_set": sum(change["same_candidate_set"] for change in negative_changes),
            "changes": negative_changes,
            "interpretation": "ordering_only; no no-match claim",
        },
    }


def _best_relevant(row: dict[str, Any]) -> dict[str, Any] | None:
    relevant_ids = {
        bucket["candidate_id"]
        for bucket in row["diagnostic_trace"]["candidate_buckets"]
        if bucket["matches_development_evidence"]
    }
    return next(
        (item for item in row["ranked"]["ordered"] if item["thumbnail"] in relevant_ids),
        None,
    )


def _query_comparisons(
    original: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    experimental: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source = {row["query_id"]: row for row in original}
    before_by_id = {row["query_id"]: row for row in baseline}
    comparisons = []
    for after in experimental:
        if after["negative"]:
            continue
        before = before_by_id[after["query_id"]]
        original_row = source[after["query_id"]]
        before_with_trace = {**before, "diagnostic_trace": original_row["trace"]}
        after_with_trace = {**after, "diagnostic_trace": original_row["trace"]}
        before_best = _best_relevant(before_with_trace)
        after_best = _best_relevant(after_with_trace)
        classification = (
            "RESCUED"
            if not before["top5"] and after["top5"]
            else "BROKEN"
            if before["top5"] and not after["top5"]
            else "UNCHANGED_SUCCESS"
            if before["top5"]
            else "UNCHANGED_FAILURE"
        )
        comparisons.append({
            "query_id": after["query_id"],
            "category": after["category"],
            "expected_intervals": after["intervals"],
            "classification": classification,
            "baseline_top5": before["top5"],
            "experimental_top5": after["top5"],
            "baseline_relevant_rank": before["relevant_rank"],
            "experimental_relevant_rank": after["relevant_rank"],
            "baseline_candidate_id": before_best["thumbnail"] if before_best else None,
            "experimental_candidate_id": after_best["thumbnail"] if after_best else None,
            "baseline_retriever_sources": list(before_best["evidence"]) if before_best else [],
            "experimental_retriever_sources": list(after_best["evidence"]) if after_best else [],
            "baseline_overlap_count": before["top5_shared"],
            "experimental_overlap_count": after["top5_shared"],
            "baseline_fusion_score": before_best["score"] if before_best else None,
            "experimental_fusion_score": after_best["score"] if after_best else None,
            "baseline_uncapped_score": (
                before_best["experimental_uncapped_score"] if before_best else None
            ),
            "experimental_uncapped_score": (
                after_best["experimental_uncapped_score"] if after_best else None
            ),
            "baseline_cap_reduction": (
                before_best["experimental_cap_reduction"] if before_best else None
            ),
            "experimental_cap_reduction": (
                after_best["experimental_cap_reduction"] if after_best else None
            ),
            "baseline_contributions": (
                before_best["experimental_contributions"] if before_best else {}
            ),
            "experimental_contributions": (
                after_best["experimental_contributions"] if after_best else {}
            ),
            "rank_change_reason": (
                "The declared variant changed only generic rank contributions or the shared-bucket cap; "
                "candidate generation, grouping and relevance labels stayed fixed."
            ),
        })
    return comparisons


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    diagnostic = json.loads(input_path.read_text(encoding="utf-8"))
    protected = json.loads(PROTECTED_V2.read_text(encoding="utf-8"))
    if diagnostic["acceptance_v2_used"] or any(
        source["source"]["sha256"] == protected["video"]["sha256"]
        for source in diagnostic["sources"]
    ):
        raise ValueError("protected Final English Acceptance V2 evidence cannot be used")

    original_rows = []
    for source in diagnostic["sources"]:
        for row in source["queries"]:
            original_rows.append({**row, "source_id": source["source"]["source_id"]})

    variant_rows: dict[str, list[dict[str, Any]]] = {}
    parity_queries = 0
    for variant in VARIANTS:
        rows = [_score_variant(row, variant) for row in original_rows]
        variant_rows[variant.variant_id] = rows
        if variant.family == "baseline":
            for original, scored in zip(original_rows, rows, strict=True):
                expected = original["trace"]["results"]
                actual = [production_shape(item) for item in scored["ranked"]["results"]]
                if actual != expected:
                    raise AssertionError(f"offline baseline parity failed: {original['query_id']}")
                parity_queries += 1

    baseline = variant_rows["baseline_rrf60"]
    variants = []
    for variant in VARIANTS:
        rows = variant_rows[variant.variant_id]
        summary = _variant_summary(rows, baseline)
        source_metrics = {
            source_id: _metrics([row for row in rows if row["source_id"] == source_id])
            for source_id in sorted({row["source_id"] for row in rows})
        }
        variants.append({
            "configuration": {
                "variant_id": variant.variant_id,
                "family": variant.family,
                "overlap_cap": variant.overlap_cap,
                "normalization": variant.normalization,
                "normalization_constant": variant.normalization_constant,
            },
            **summary,
            "per_source": source_metrics,
            "query_comparisons": _query_comparisons(original_rows, baseline, rows),
        })

    report = {
        "schema_version": "1.0.0",
        "suite_id": "hybrid-fusion-refinement-v1",
        "diagnostic_input_sha256": sha256(input_path),
        "acceptance_v2_used": False,
        "production_modified": False,
        "baseline_parity": {
            "passed": parity_queries == len(original_rows),
            "queries": parity_queries,
            "compared": ["candidate_ids", "timestamps", "ordering", "fusion_scores", "top_k"],
        },
        "predeclared_variants": [variant.variant_id for variant in VARIANTS],
        "variants": variants,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = run(arguments.input, arguments.output)
    print(json.dumps({
        "baseline_parity": result["baseline_parity"],
        "variants": [
            {
                "variant": variant["configuration"]["variant_id"],
                "all": variant["metrics"]["ALL"],
                "speech_top5": variant["metrics"]["SPEECH"]["top5"],
                "visual_top5": variant["metrics"]["VISUAL"]["top5"],
                "multimodal_top5": variant["metrics"]["MULTIMODAL"]["top5"],
                "outcomes": variant["outcomes"],
            }
            for variant in result["variants"]
        ],
    }, indent=2))
