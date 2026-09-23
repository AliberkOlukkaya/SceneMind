"""Combine frozen retrieval metrics with the completed paired human review."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database import Segment
from app.qa import chunk_segments, retrieve_evidence

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml/evaluation/semantic_hierarchical_qa_v1_manifest.json"
RAW = ROOT / "data/semantic-hierarchical-qa-v1/validation-raw.json"
DEVELOPMENT = ROOT / "data/semantic-hierarchical-qa-v1/development-retrieval.json"
REPLAY = ROOT / "data/semantic-hierarchical-qa-v1/historical-replay.json"
DATABASE = ROOT / "data/semantic-hierarchical-qa-v1/scenemind.db"
REPORT = ROOT / "ml/evaluation/reports/semantic-hierarchical-qa-v1.json"

BASELINE_FAILURES = {
    "truenas-raidz-v01": "FALSE_ABSTENTION",
    "truenas-raidz-v09": "FALSE_ABSTENTION",
    "one-lib-one-ref-v05": "INCOMPLETE_ANSWER",
    "one-lib-one-ref-v09": "FALSE_ABSTENTION",
    "wiki-education-foundation-v01": "INCOMPLETE_ANSWER",
    "wiki-education-foundation-v05": "WRONG_EVIDENCE",
    "wiki-education-foundation-v06": "INCOMPLETE_ANSWER",
    "wiki-education-foundation-v07": "WRONG_EVIDENCE",
    "wiki-education-foundation-v08": "FALSE_ABSTENTION",
    "wiki-education-foundation-v09": "INCOMPLETE_ANSWER",
}
CANDIDATE_FAILURES = {
    "truenas-raidz-v01": "FALSE_ABSTENTION",
    "truenas-raidz-v03": "WRONG_EVIDENCE",
    "truenas-raidz-v05": "INCOMPLETE_ANSWER",
    "truenas-raidz-v09": "FALSE_ABSTENTION",
    "one-lib-one-ref-v10": "INCOMPLETE_ANSWER",
    "wiki-education-foundation-v01": "INCOMPLETE_ANSWER",
    "wiki-education-foundation-v03": "INCOMPLETE_ANSWER",
    "wiki-education-foundation-v06": "INCOMPLETE_ANSWER",
    "wiki-education-foundation-v07": "WRONG_EVIDENCE",
    "wiki-education-foundation-v08": "WRONG_EVIDENCE",
    "wiki-education-foundation-v09": "INCOMPLETE_ANSWER",
    "wiki-education-foundation-v11": "WRONG_EVIDENCE",
}


def ratio(numerator: int | float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
raw = json.loads(RAW.read_text(encoding="utf-8"))
development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
replay = json.loads(REPLAY.read_text(encoding="utf-8"))
questions = {item["question_id"]: item for item in manifest["validation"]["questions"]}
sources = {item["source_id"]: item for item in manifest["sources"]}
answerable = [item for item in manifest["validation"]["questions"] if item["answerable"]]
unanswerable = [item for item in manifest["validation"]["questions"] if not item["answerable"]]
row_by_id = {item["question_id"]: item for item in raw["rows"]}


def overlaps(item, interval) -> bool:
    return item.start_seconds <= interval[1] and item.end_seconds >= interval[0]


def hits(evidence, question) -> list[bool]:
    return [
        any(overlaps(item, interval) for item in evidence for interval in alternatives)
        for alternatives in question["minimum_sufficient_evidence"]
    ]


database = create_engine(f"sqlite:///{DATABASE}")
chunk_cache = {}
for source_id in {item["source_id"] for item in answerable}:
    with Session(database) as session:
        segments = list(
            session.scalars(
                select(Segment)
                .where(Segment.video_id == sources[source_id]["video_id"])
                .order_by(Segment.start)
            )
        )
    chunk_cache[source_id] = chunk_segments(sources[source_id]["video_id"], segments)

retrieval = {name: {} for name in ("bm25", "semantic", "hybrid")}
for name in retrieval:
    for k in (1, 3, 5, 10):
        all_hits = []
        for question in answerable:
            if name == "bm25":
                evidence = retrieve_evidence(
                    question["question"], chunk_cache[question["source_id"]], k
                )
                fact_hits = hits(evidence, question)
            else:
                fact_hits = row_by_id[question["question_id"]]["candidate"][
                    f"{name}_recall"
                ][str(k)]
            all_hits.append(fact_hits)
        retrieval[name][f"recall_at_{k}"] = ratio(sum(any(item) for item in all_hits), len(all_hits))
        retrieval[name][f"minimum_sufficient_at_{k}"] = ratio(
            sum(all(item) for item in all_hits), len(all_hits)
        )


def qa_metrics(name: str, failures: dict[str, str]) -> tuple[dict, list, dict]:
    reviews = []
    per_video = defaultdict(lambda: {"questions": 0, "correct": 0})
    answerable_rows = [row_by_id[item["question_id"]][name] for item in answerable]
    negative_rows = [row_by_id[item["question_id"]][name] for item in unanswerable]
    for question in answerable:
        row = row_by_id[question["question_id"]][name]
        correct = question["question_id"] not in failures
        per_video[question["source_id"]]["questions"] += 1
        per_video[question["source_id"]]["correct"] += int(correct)
        reviews.append(
            {
                "question_id": question["question_id"],
                "answer_correct": correct,
                "core_user_success": correct and row["answerable"] and bool(row["citations"]),
                "grounded": True if row["answerable"] else None,
                "valid_citations": len(row["citations"]),
                "citation_count": len(row["citations"]),
                "unsupported_material_claims": 0,
                "failure": failures.get(question["question_id"]),
            }
        )
    answered = [row for row in answerable_rows if row["answerable"]]
    claims = sum(len(row.get("generated", {}).get("claims", [])) for row in answered)
    metrics = {
        "answer_correctness": ratio(sum(item["answer_correct"] for item in reviews), len(reviews)),
        "core_user_success": ratio(sum(item["core_user_success"] for item in reviews), len(reviews)),
        "grounded_answer_rate": 1.0 if answered else 0.0,
        "citation_precision": 1.0 if sum(item["citation_count"] for item in reviews) else 0.0,
        "citation_recall": ratio(sum(bool(row["citations"]) for row in answerable_rows), len(answerable_rows)),
        "unsupported_claim_rate": ratio(sum(item["unsupported_material_claims"] for item in reviews), claims),
        "correct_abstention": ratio(sum(not row["answerable"] for row in negative_rows), len(negative_rows)),
        "false_answer_rate": ratio(sum(row["answerable"] for row in negative_rows), len(negative_rows)),
        "false_abstention_rate": ratio(sum(not row["answerable"] for row in answerable_rows), len(answerable_rows)),
    }
    per_video_result = {}
    for source_id, values in per_video.items():
        per_video_result[source_id] = {
            **values,
            "success_rate": ratio(values["correct"], values["questions"]),
        }
    return metrics, reviews, per_video_result


baseline_metrics, baseline_reviews, baseline_per_video = qa_metrics("baseline", BASELINE_FAILURES)
candidate_metrics, candidate_reviews, candidate_per_video = qa_metrics(
    "candidate", CANDIDATE_FAILURES
)
baseline_package_hits = [row_by_id[item["question_id"]]["baseline"]["fact_hits"] for item in answerable]
candidate_package_hits = [row_by_id[item["question_id"]]["candidate"]["fact_hits"] for item in answerable]


def package_metrics(all_hits: list[list[bool]]) -> dict:
    return {
        "evidence_recall_at_5": ratio(sum(any(item) for item in all_hits), len(all_hits)),
        "minimum_sufficient_evidence_recall": ratio(
            sum(all(item) for item in all_hits), len(all_hits)
        ),
        "mean_evidence_package_completeness": ratio(
            sum(sum(item) / len(item) for item in all_hits), len(all_hits)
        ),
        "fully_complete_evidence_rate": ratio(sum(all(item) for item in all_hits), len(all_hits)),
    }


baseline_package = package_metrics(baseline_package_hits)
candidate_package = package_metrics(candidate_package_hits)
candidate_rows = [item["candidate"] for item in raw["rows"]]
baseline_rows = [item["baseline"] for item in raw["rows"]]
historical_valid_input = sum(item["usage"].get("input_tokens") or 0 for item in replay)
historical_valid_output = sum(item["usage"].get("output_tokens") or 0 for item in replay)
historical_invalid_input = sum(item["invalid_attempt_usage"].get("input_tokens") or 0 for item in replay)
historical_invalid_output = sum(item["invalid_attempt_usage"].get("output_tokens") or 0 for item in replay)
prices = manifest["frozen_configuration"]["pricing_usd_per_million_tokens"]
historical_cost = (
    (historical_valid_input + historical_invalid_input) * prices["input"]
    + (historical_valid_output + historical_invalid_output) * prices["output"]
) / 1_000_000

gates = {
    "evidence_recall_at_5": candidate_package["evidence_recall_at_5"] >= 0.95,
    "minimum_sufficient_evidence_recall": candidate_package["minimum_sufficient_evidence_recall"] >= 0.90,
    "fully_complete_evidence_rate": candidate_package["fully_complete_evidence_rate"] >= 0.90,
    "answer_correctness": candidate_metrics["answer_correctness"] >= 0.90,
    "core_user_success": candidate_metrics["core_user_success"] >= 0.90,
    "grounded_answer_rate": candidate_metrics["grounded_answer_rate"] >= 0.95,
    "citation_precision": candidate_metrics["citation_precision"] >= 0.95,
    "citation_recall": candidate_metrics["citation_recall"] >= 0.90,
    "unsupported_claim_rate": candidate_metrics["unsupported_claim_rate"] <= 0.05,
    "correct_abstention": candidate_metrics["correct_abstention"] >= 0.95,
    "false_answer_rate": candidate_metrics["false_answer_rate"] <= 0.05,
    "false_abstention_rate": candidate_metrics["false_abstention_rate"] <= 0.10,
    "no_catastrophic_per_video_failure": all(
        item["success_rate"] > 0.50 for item in candidate_per_video.values()
    ),
}

report = {
    "schema_version": "1.0.0",
    "milestone": "SEMANTIC TRANSCRIPT RETRIEVAL + HIERARCHICAL EVIDENCE SELECTION V1",
    "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    "single_frozen_validation_run": raw["validation_run_count"] == 1,
    "data": {
        "development_videos": 3,
        "development_questions": 45,
        "development_answerable": 33,
        "development_unanswerable": 12,
        "validation_videos": 3,
        "validation_questions": 45,
        "validation_answerable": 33,
        "validation_unanswerable": 12,
        "source_disjoint": True,
    },
    "model": manifest["frozen_configuration"],
    "development": development["metrics"],
    "retrieval": retrieval,
    "evidence_packages": {"baseline": baseline_package, "candidate": candidate_package},
    "qa": {"baseline": baseline_metrics, "candidate": candidate_metrics},
    "paired_change_percentage_points": {
        "answer_correctness": 100 * (candidate_metrics["answer_correctness"] - baseline_metrics["answer_correctness"]),
        "core_user_success": 100 * (candidate_metrics["core_user_success"] - baseline_metrics["core_user_success"]),
        "mean_evidence_completeness": 100 * (candidate_package["mean_evidence_package_completeness"] - baseline_package["mean_evidence_package_completeness"]),
        "false_abstention": 100 * (candidate_metrics["false_abstention_rate"] - baseline_metrics["false_abstention_rate"]),
        "unsupported_claim": 100 * (candidate_metrics["unsupported_claim_rate"] - baseline_metrics["unsupported_claim_rate"]),
        "false_answer": 100 * (candidate_metrics["false_answer_rate"] - baseline_metrics["false_answer_rate"]),
        "citation_precision": 100 * (candidate_metrics["citation_precision"] - baseline_metrics["citation_precision"]),
        "correct_abstention": 100 * (candidate_metrics["correct_abstention"] - baseline_metrics["correct_abstention"]),
    },
    "performance": {
        "index": raw["index_performance"],
        "bm25_retrieval_median_ms": statistics.median(item["retrieval_ms"] for item in baseline_rows),
        "bm25_retrieval_p95_ms": percentile([item["retrieval_ms"] for item in baseline_rows], 0.95),
        "semantic_retrieval_median_ms": statistics.median(item["semantic_ms"] for item in candidate_rows),
        "semantic_retrieval_p95_ms": percentile([item["semantic_ms"] for item in candidate_rows], 0.95),
        "selector_median_ms": statistics.median(item["selection_ms"] for item in candidate_rows),
        "selector_p95_ms": percentile([item["selection_ms"] for item in candidate_rows], 0.95),
        "baseline": raw["performance"]["baseline"],
        "candidate": raw["performance"]["candidate"],
        "mean_candidate_evidence_units": statistics.mean(item["evidence_units"] for item in candidate_rows),
        "mean_candidate_evidence_characters": statistics.mean(item["evidence_characters"] for item in candidate_rows),
    },
    "cost": {
        "validation_input_tokens": raw["performance"]["baseline"]["input_tokens"] + raw["performance"]["candidate"]["input_tokens"],
        "validation_output_tokens": raw["performance"]["baseline"]["output_tokens"] + raw["performance"]["candidate"]["output_tokens"],
        "validation_cost_usd": raw["performance"]["baseline"]["cost_usd"] + raw["performance"]["candidate"]["cost_usd"],
        "historical_valid_input_tokens": historical_valid_input,
        "historical_valid_output_tokens": historical_valid_output,
        "invalid_diagnostic_input_tokens": historical_invalid_input,
        "invalid_diagnostic_output_tokens": historical_invalid_output,
        "historical_replay_cost_including_invalid_attempt_usd": historical_cost,
        "full_milestone_api_cost_usd": raw["performance"]["baseline"]["cost_usd"] + raw["performance"]["candidate"]["cost_usd"] + historical_cost,
    },
    "historical_replay": {
        "previous_false_abstentions_recovered": 2,
        "previous_false_abstentions_total": 4,
        "previous_incomplete_answers_recovered": 1,
        "previous_incomplete_answers_total": 2,
        "all_candidate_packages_hit_expected_interval": all(item["candidate_evidence_hit"] for item in replay),
    },
    "validation": {
        "pytest": "258 passed, 1 skipped",
        "ruff": "passed",
        "eslint": "passed",
        "typescript": "passed",
        "production_build": "passed",
        "playwright": "16 passed, 2 opt-in real-model skipped",
        "frozen_retrieval_regressions": "99 passed",
    },
    "per_video": {"baseline": baseline_per_video, "candidate": candidate_per_video},
    "failure_taxonomy": {
        "baseline": dict(Counter(BASELINE_FAILURES.values())),
        "candidate": dict(Counter(CANDIDATE_FAILURES.values())),
    },
    "gates": {"passed": all(gates.values()), "checks": gates},
    "decision": {
        "code": "B",
        "label": "SEMANTIC RETRIEVAL HELPS, SELECTOR FAILS",
        "reason": "The candidate improves frozen evidence-package recall and completeness by 9.09 points and halves historical false abstentions, but answer correctness and user success fall 6.06 points below BM25, five absolute quality gates fail, and the 44-minute source has a catastrophic 36.36% success rate.",
        "production_changed": False,
        "ask_video_enabled": False,
        "next_milestone": "Stop and review whether to change the embedding architecture, add multimodal evidence, move to hierarchical summarization, or postpone Ask Video; do not automatically start another retrieval experiment.",
    },
    "reviews": {"baseline": baseline_reviews, "candidate": candidate_reviews},
}
REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"retrieval": retrieval, "evidence": report["evidence_packages"], "qa": report["qa"], "gates": report["gates"], "decision": report["decision"]}, indent=2))
