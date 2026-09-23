"""Combine the frozen Q&A safety run with manual claim and citation review."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-replay", type=Path)
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
        "answer_correctness": ratio(
            sum(row["human_review"]["answer_correct"] is True for row in answerable),
            len(answerable),
        ),
        "grounded_answer_rate": ratio(
            sum(row["human_review"]["fully_supported"] is True for row in substantive),
            len(substantive),
        ),
        "citation_precision": ratio(
            sum(row["human_review"]["supporting_citations"] for row in substantive),
            citations,
        ),
        "citation_recall": ratio(
            sum(row["human_review"]["supported_claims"] for row in substantive), claims
        ),
        "unsupported_claim_rate": ratio(
            sum(
                row["human_review"]["material_claims"] - row["human_review"]["supported_claims"]
                for row in substantive
            ),
            claims,
        ),
        "correct_abstention_rate": ratio(
            sum(row["human_review"]["appropriate_abstention"] is True for row in unanswerable),
            len(unanswerable),
        ),
        "false_answer_rate": ratio(
            sum(row["human_review"]["false_answer"] for row in unanswerable), len(unanswerable)
        ),
        "false_abstention_rate": ratio(
            sum(not row["generated"]["answerable"] for row in answerable), len(answerable)
        ),
    }
    metrics["estimated_cost_usd"] = (
        metrics["input_tokens"] * 0.75 / 1_000_000 + metrics["output_tokens"] * 4.50 / 1_000_000
    )

    grouped = defaultdict(list)
    for row in raw["results"]:
        grouped[row["category"]].append(row)
    categories = {}
    for category, rows in sorted(grouped.items()):
        positives = [row for row in rows if row["answerable"]]
        negatives = [row for row in rows if not row["answerable"]]
        categories[category] = {
            "questions": len(rows),
            "answerable": len(positives),
            "answer_correctness": ratio(
                sum(row["human_review"]["answer_correct"] is True for row in positives),
                len(positives),
            ),
            "correct_abstention_rate": ratio(
                sum(row["human_review"]["appropriate_abstention"] is True for row in negatives),
                len(negatives),
            ),
            "false_answer_rate": ratio(
                sum(row["human_review"]["false_answer"] for row in negatives), len(negatives)
            ),
        }

    gates = {
        "evidence_recall_at_5": metrics["evidence_recall_at_5"] >= 0.90,
        "answer_correctness": metrics["answer_correctness"] >= 0.90,
        "grounded_answer_rate": metrics["grounded_answer_rate"] >= 0.95,
        "citation_precision": metrics["citation_precision"] >= 0.95,
        "citation_recall": metrics["citation_recall"] >= 0.90,
        "unsupported_claim_rate": metrics["unsupported_claim_rate"] <= 0.05,
        "correct_abstention_rate": metrics["correct_abstention_rate"] >= 0.90,
        "false_answer_rate": metrics["false_answer_rate"] <= 0.10,
        "no_catastrophic_category_failure": False,
    }
    historical_replay = None
    if args.historical_replay:
        replay = json.loads(args.historical_replay.read_text(encoding="utf-8"))
        historical_replay = {
            "excluded_from_validation_metrics": True,
            "rows": [
                {
                    "historical_id": row["historical_id"],
                    "question": row["question"],
                    "answerable": row["generated"]["answerable"],
                    "safe_abstention": not row["generated"]["answerable"],
                    "input_tokens": row["usage"].get("input_tokens"),
                    "output_tokens": row["usage"].get("output_tokens"),
                    "generation_ms": row["generation_ms"],
                }
                for row in replay
            ],
        }

    report = {
        "schema_version": 1,
        "suite_id": raw["suite_id"],
        "run_date": "2026-09-23",
        "manifest_sha256": raw["manifest_sha256"],
        "provider": "openai",
        "model": next(row["usage"]["model"] for row in raw["results"] if row["usage"].get("model")),
        "validation_run_count": 1,
        "data": {
            "source_count": raw["source_count"],
            "questions": len(raw["results"]),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "hard_negatives": sum(row["category"] == "HARD_NEGATIVE" for row in raw["results"]),
            "source_disjoint_verified": True,
            "previous_qa_validation_reused": False,
        },
        "development_selection": {
            "questions": 30,
            "answerable": 18,
            "unanswerable": 12,
            "evidence_recall_at_5": 1.0,
            "variants": [
                {
                    "id": "v1_strict_contract",
                    "answerable_questions_answered": 12,
                    "correct_abstentions": 12,
                    "false_answers": 0,
                    "selected": False,
                },
                {
                    "id": "v2_asr_tolerant_claim_contract",
                    "answerable_questions_answered": 16,
                    "correct_abstentions": 12,
                    "false_answers": 0,
                    "selected": True,
                    "input_tokens": 20735,
                    "output_tokens": 2581,
                },
                {
                    "id": "v3_more_permissive_asr_instruction",
                    "answerable_questions_answered": 15,
                    "correct_abstentions": 12,
                    "false_answers": 0,
                    "selected": False,
                },
            ],
        },
        "metrics": metrics,
        "categories": categories,
        "pricing": {
            "input_per_million_usd": 0.75,
            "output_per_million_usd": 4.50,
            "source": "Official OpenAI GPT-5.4 Mini model page checked 2026-09-23",
        },
        "gate_results": gates,
        "historical_replay": historical_replay,
        "failure_counts": {
            "retrieval_misses_at_5": sum(not row["evidence_recall_hit_at_5"] for row in answerable),
            "false_abstentions": sum(not row["generated"]["answerable"] for row in answerable),
            "incorrect_substantive_answers": sum(
                row["generated"]["answerable"] and row["human_review"]["answer_correct"] is False
                for row in answerable
            ),
            "answers_with_unsupported_claims": sum(
                row["human_review"]["unsupported_claim"] for row in substantive
            ),
            "false_answers": sum(row["human_review"]["false_answer"] for row in unanswerable),
        },
        "decision": {
            "code": "D",
            "label": "TEMPORAL / STRUCTURED QUESTION FAILURE",
            "reason": "Safety gates pass, but answer correctness is 14/18 and structured categories fail: frozen LIST_COUNT correctness is 0/2 and TEMPORAL correctness is 1/2. Two answerable questions abstain and two substantive list/decision answers omit required facts.",
            "ask_video_eligible_for_enablement": False,
            "next_milestone": "Q&A STRUCTURED-QUESTION EVIDENCE V1 ON NEW DEVELOPMENT SOURCES; KEEP ASK VIDEO DISABLED",
        },
        "human_review_method": review["review_method"],
        "results": raw["results"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
