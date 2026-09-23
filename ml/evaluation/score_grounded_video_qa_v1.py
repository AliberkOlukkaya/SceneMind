"""Combine the one frozen provider run with manual claim/evidence review."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    for row in raw["results"]:
        row["human_review"] = review["rows"][row["id"]]
    answerable = [row for row in raw["results"] if row["answerable"]]
    unanswerable = [row for row in raw["results"] if not row["answerable"]]
    substantive = [row for row in raw["results"] if row["generated"]["answerable"]]
    claims = sum(row["human_review"]["material_claims"] for row in substantive)
    citations = sum(row["human_review"]["citations"] for row in substantive)
    metrics = {
        **raw["automatic_summary"],
        "answer_correctness": sum(
            row["human_review"]["answer_correct"] is True for row in answerable
        )
        / len(answerable),
        "grounded_answer_rate": sum(
            row["human_review"]["fully_supported"] is True for row in substantive
        )
        / len(substantive),
        "citation_precision": sum(
            row["human_review"]["supporting_citations"] for row in substantive
        )
        / citations,
        "citation_recall": sum(row["human_review"]["supported_claims"] for row in substantive)
        / claims,
        "unsupported_claim_rate": sum(
            row["human_review"]["material_claims"] - row["human_review"]["supported_claims"]
            for row in substantive
        )
        / claims,
        "correct_abstention_rate": sum(
            row["human_review"]["appropriate_abstention"] is True for row in unanswerable
        )
        / len(unanswerable),
        "false_answer_rate": sum(row["human_review"]["false_answer"] for row in unanswerable)
        / len(unanswerable),
        "estimated_cost_usd": raw["automatic_summary"]["input_tokens"] * 0.75 / 1_000_000
        + raw["automatic_summary"]["output_tokens"] * 4.50 / 1_000_000,
    }
    report = {
        "schema_version": 1,
        "suite_id": "grounded-video-qa-v1",
        "run_date": "2026-09-23",
        "manifest_sha256": raw["manifest_sha256"],
        "provider": "openai",
        "model": next(row["usage"]["model"] for row in raw["results"] if row["usage"].get("model")),
        "segment_count": raw["segment_count"],
        "chunk_count": raw["chunk_count"],
        "validation_run_count": 1,
        "metrics": metrics,
        "pricing": {
            "input_per_million_usd": 0.75,
            "output_per_million_usd": 4.50,
            "source": "Official OpenAI GPT-5.4 Mini model page checked 2026-09-23",
        },
        "gate_results": {
            "evidence_recall_at_5": metrics["evidence_recall_at_5"] >= 0.90,
            "answer_correctness": metrics["answer_correctness"] >= 0.90,
            "grounded_answer_rate": metrics["grounded_answer_rate"] >= 0.95,
            "citation_precision": metrics["citation_precision"] >= 0.95,
            "unsupported_claim_rate": metrics["unsupported_claim_rate"] <= 0.05,
            "correct_abstention_rate": metrics["correct_abstention_rate"] >= 0.90,
            "false_answer_rate": metrics["false_answer_rate"] <= 0.10,
        },
        "failure_counts": {
            "retrieval_misses_at_5": sum(not row["evidence_recall_hit"]["5"] for row in answerable),
            "generation_errors": 0,
            "answers_with_unsupported_claims": sum(
                row["human_review"]["unsupported_claim"] for row in substantive
            ),
            "citation_failure_answers": sum(
                row["human_review"]["citation_complete"] is False for row in substantive
            ),
            "abstention_failures": sum(
                row["human_review"]["appropriate_abstention"] is False for row in unanswerable
            ),
        },
        "decision": {
            "code": "D",
            "label": "ABSTENTION SAFETY FAILED",
            "reason": "Evidence Recall@5 passed, but one of three unanswerable questions received a substantive response; correctness, grounding, citation precision, unsupported-claim and abstention gates failed.",
            "ask_video_promoted": False,
            "next_milestone": "Q&A ABSTENTION SAFETY V1 ON NEW DEVELOPMENT EVIDENCE; KEEP THIS VALIDATION SET FROZEN AND EVALUATION-ONLY",
        },
        "human_review_method": review["review_method"],
        "results": raw["results"],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
