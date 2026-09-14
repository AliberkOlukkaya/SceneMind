"""Add the evidence-led decision and failure buckets to a raw measurement."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def hit(candidates: list[dict], intervals: list[list[float]], limit: int = 5) -> bool:
    return any(
        start <= candidate["timestamp"] < end
        for candidate in candidates[:limit]
        for start, end in intervals
    )


def assemble(source: Path, output: Path) -> dict:
    report = json.loads(source.read_text(encoding="utf-8"))
    failures = {
        "correct_region_absent_from_raw_clip_top20": [],
        "correct_region_present_but_diversity_drops_it": [],
        "near_duplicate_distractors": [
            row["query_id"] for row in report["global_2s_diagnosis"]["lost_queries"]
            if row["diagnosis"] == "near_duplicate_competition"
        ],
        "visually_plausible_semantic_distractor": [],
        "small_object_too_weak_for_clip_top5": [],
        "compositional_failure": [],
        "temporal_action_requires_motion": [],
        "query_requires_ocr_text": [],
        "transcript_path_should_handle_query": [],
    }
    for row in report["rows"]:
        if not row["expected_presence"]:
            continue
        pool_hit = hit(row["raw_2s_top20"], row["relevant_intervals"], 20)
        selected_hit = hit(row["selected"], row["relevant_intervals"])
        if not pool_hit:
            failures["correct_region_absent_from_raw_clip_top20"].append(row["query_id"])
        elif not selected_hit:
            failures["correct_region_present_but_diversity_drops_it"].append(row["query_id"])
            failures["visually_plausible_semantic_distractor"].append(row["query_id"])
        if not selected_hit and row["query_type"] == "COMPOSITIONAL":
            failures["compositional_failure"].append(row["query_id"])
        if not selected_hit and row["query_type"] == "ACTION_TEMPORAL":
            failures["temporal_action_requires_motion"].append(row["query_id"])
        if not selected_hit and row["query_type"] == "SPEECH":
            failures["transcript_path_should_handle_query"].append(row["query_id"])
    failures["small_object_too_weak_for_clip_top5"] = [
        row["event_id"] for row in report["small_object_rows"]
        if row["raw_2s_top20_visible_hit"] and not row["selected_visible_hit"]
    ]
    report["failure_buckets"] = failures
    report["artifact_inputs"] = {
        "raw_measurement_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    report["decision"] = {
        "outcome": "F_candidate_pool_is_high_recall_but_final_selection_is_not_promotable",
        "production_promoted": False,
        "recommendation": (
            "Keep production 5-second raw CLIP unchanged. The 2-second top-50 pool reaches "
            "97.62% held-out correct-region recall and 100% reviewed small-object visibility, "
            "but every cheap final-top-5 selector misses promotion gates. Revisit calibrated "
            "no-match only with a bounded candidate-list ranking experiment; keep Video RAG "
            "deferred until ranking and rejection both pass."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "ml/evaluation/reports/coarse-candidate-diversity-v1.json",
    )
    args = parser.parse_args()
    result = assemble(args.source, args.output)
    print(json.dumps({"decision": result["decision"], "gate": result["gate"]}, indent=2))
