"""Aggregate completed human review for structured Q&A evidence V1."""

import argparse
import json
from pathlib import Path


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    rows = report["results"]
    if any(row["human_review"] is None for row in rows):
        raise SystemExit("Every result requires human_review before scoring")

    answered = [row for row in rows if row["generated"]["answerable"]]
    answerable = [row for row in rows if row["answerable"]]
    negatives = [row for row in rows if not row["answerable"]]
    lists = [row for row in rows if row["question_type"] == "LIST_COUNT"]
    temporal = [row for row in rows if row["question_type"] == "TEMPORAL"]
    ordinary = [row for row in rows if row["question_type"] == "ORDINARY"]

    reviews = [row["human_review"] for row in rows]
    answer_reviews = [row["human_review"] for row in answered]
    metrics = {
        **report["automatic_summary"],
        "answer_correctness": ratio(sum(review["answer_correct"] for review in reviews), len(rows)),
        "grounded_answer_rate": ratio(
            sum(review["grounded"] for review in answer_reviews), len(answer_reviews)
        ),
        "citation_precision": ratio(
            sum(review["precise_citation_count"] for review in reviews),
            sum(review["citation_count"] for review in reviews),
        ),
        "citation_recall": ratio(
            sum(review["cited_required_fact_count"] for review in reviews),
            sum(review["required_fact_count"] for review in reviews),
        ),
        "unsupported_claim_rate": ratio(
            sum(review["unsupported_claim_count"] for review in answer_reviews),
            sum(review["claim_count"] for review in answer_reviews),
        ),
        "correct_abstention_rate": ratio(
            sum(not row["generated"]["answerable"] for row in negatives), len(negatives)
        ),
        "false_answer_rate": ratio(
            sum(row["generated"]["answerable"] for row in negatives), len(negatives)
        ),
        "false_abstention_rate": ratio(
            sum(not row["generated"]["answerable"] for row in answerable), len(answerable)
        ),
        "estimated_cost_usd": (
            report["automatic_summary"]["input_tokens"] * 0.75
            + report["automatic_summary"]["output_tokens"] * 4.5
        )
        / 1_000_000,
    }
    categories = {
        "ordinary_answer_correctness": ratio(
            sum(row["human_review"]["answer_correct"] for row in ordinary), len(ordinary)
        ),
        "list_answer_correctness": ratio(
            sum(row["human_review"]["answer_correct"] for row in lists), len(lists)
        ),
        "list_supported_item_precision": ratio(
            sum(row["human_review"]["list_supported_items"] for row in lists),
            sum(row["human_review"]["list_predicted_items"] for row in lists),
        ),
        "list_supported_item_recall": ratio(
            sum(row["human_review"]["list_supported_items"] for row in lists),
            sum(row["human_review"]["list_expected_items"] for row in lists),
        ),
        "requested_count_compliance": ratio(
            sum(row["human_review"]["count_compliant"] for row in lists), len(lists)
        ),
        "temporal_answer_correctness": ratio(
            sum(row["human_review"]["answer_correct"] for row in temporal), len(temporal)
        ),
        "temporal_direction_accuracy": ratio(
            sum(row["human_review"]["temporal_direction_correct"] for row in temporal),
            len(temporal),
        ),
        "temporal_citation_correctness": ratio(
            sum(row["human_review"]["temporal_citations_correct"] for row in temporal),
            len(temporal),
        ),
        "hard_negative_false_answer_rate": ratio(
            sum(row["generated"]["answerable"] for row in negatives), len(negatives)
        ),
    }
    gates = {
        "overall_answer_correctness": metrics["answer_correctness"] >= 0.90,
        "grounded_answer_rate": metrics["grounded_answer_rate"] >= 0.95,
        "citation_precision": metrics["citation_precision"] >= 0.95,
        "citation_recall": metrics["citation_recall"] >= 0.90,
        "unsupported_claim_rate": metrics["unsupported_claim_rate"] <= 0.05,
        "correct_abstention_rate": metrics["correct_abstention_rate"] >= 0.90,
        "false_answer_rate": metrics["false_answer_rate"] <= 0.10,
        "list_answer_correctness": categories["list_answer_correctness"] >= 0.85,
        "temporal_answer_correctness": categories["temporal_answer_correctness"] >= 0.85,
        "temporal_direction_accuracy": categories["temporal_direction_accuracy"] >= 0.95,
        "hard_negative_false_answer_rate": categories["hard_negative_false_answer_rate"] <= 0.10,
        "ordinary_no_material_regression": categories["ordinary_answer_correctness"] >= 0.90,
    }
    report.update(
        {
            "metrics": metrics,
            "categories": categories,
            "pricing": {
                "input_per_million_usd": 0.75,
                "output_per_million_usd": 4.5,
                "source": "Official OpenAI GPT-5.4 Mini model page checked 2026-09-23",
            },
            "gate_results": gates,
            "all_promotion_gates_pass": all(gates.values()),
        }
    )
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
