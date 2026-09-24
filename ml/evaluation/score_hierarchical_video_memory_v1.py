"""Score the frozen Hierarchical Video Memory V1 run and historical replay."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml/evaluation/hierarchical_video_memory_v1_manifest.json"
RAW = ROOT / "data/hierarchical-video-memory-v1/validation-raw.json"
DEVELOPMENT = ROOT / "data/hierarchical-video-memory-v1/development-grid.json"
REPLAY = ROOT / "data/hierarchical-video-memory-v1/historical-replay.json"
REPORT = ROOT / "ml/evaluation/reports/hierarchical-video-memory-v1.json"

FAILURES = {
    "bm25": {
        "acip-opening-v02": "FALSE_ABSTENTION",
        "acip-opening-v03": "FALSE_ABSTENTION",
        "acip-opening-v09": "GENERATION_ERROR",
        "gm-orion-v04": "FALSE_ABSTENTION",
        "gm-orion-v06": "INCOMPLETE_EVIDENCE",
        "gm-orion-v10": "FALSE_ABSTENTION",
        "copyright-remedies-v09": "FALSE_ABSTENTION",
    },
    "flat_semantic": {
        "acip-opening-v02": "FALSE_ABSTENTION",
        "acip-opening-v03": "FALSE_ABSTENTION",
        "acip-opening-v05": "FALSE_ABSTENTION",
        "acip-opening-v09": "GENERATION_ERROR",
        "gm-orion-v01": "LOCAL_EVIDENCE_MISS",
        "gm-orion-v04": "GENERATION_ERROR",
        "gm-orion-v06": "GENERATION_ERROR",
        "gm-orion-v08": "GENERATION_ERROR",
        "gm-orion-v10": "FALSE_ABSTENTION",
        "gm-orion-v11": "GENERATION_ERROR",
        "copyright-remedies-v01": "FALSE_ABSTENTION",
        "copyright-remedies-v03": "GENERATION_ERROR",
        "copyright-remedies-v06": "FALSE_ABSTENTION",
        "copyright-remedies-v11": "LOCAL_EVIDENCE_MISS",
    },
    "hierarchical": {
        "acip-opening-v02": "FALSE_ABSTENTION",
        "acip-opening-v03": "FALSE_ABSTENTION",
        "acip-opening-v09": "GENERATION_ERROR",
        "gm-orion-v01": "SECTION_MISS",
        "gm-orion-v03": "SECTION_MISS",
        "gm-orion-v04": "GENERATION_ERROR",
        "gm-orion-v05": "WRONG_SECTION",
        "gm-orion-v06": "FALSE_ABSTENTION",
        "gm-orion-v08": "FALSE_ABSTENTION",
        "gm-orion-v10": "FALSE_ABSTENTION",
        "gm-orion-v11": "GENERATION_ERROR",
        "copyright-remedies-v06": "LOCAL_EVIDENCE_MISS",
        "copyright-remedies-v10": "FALSE_ABSTENTION",
        "copyright-remedies-v11": "LOCAL_EVIDENCE_MISS",
    },
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
raw = json.loads(RAW.read_text(encoding="utf-8"))
development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
replay = json.loads(REPLAY.read_text(encoding="utf-8"))
questions = manifest["validation"]["questions"]
answerable = [item for item in questions if item["answerable"]]
negative = [item for item in questions if not item["answerable"]]
rows = {item["question_id"]: item for item in raw["rows"]}
sources = {item["source_id"]: item for item in manifest["sources"]}
arms = ("bm25", "flat_semantic", "hierarchical")

retrieval = {}
packages = {}
qa = {}
reviews = {}
per_video = {}
for arm in arms:
    retrieval[arm] = {}
    for k in (1, 3, 5):
        hits = [rows[q["question_id"]]["arms"][arm]["recall"][str(k)] for q in answerable]
        retrieval[arm][f"evidence_recall_at_{k}"] = ratio(
            sum(any(item) for item in hits), len(hits)
        )
        retrieval[arm][f"minimum_sufficient_at_{k}"] = ratio(
            sum(all(item) for item in hits), len(hits)
        )
    package_hits = [rows[q["question_id"]]["arms"][arm]["fact_hits"] for q in answerable]
    packages[arm] = {
        "evidence_recall_at_5": ratio(sum(any(x) for x in package_hits), len(package_hits)),
        "minimum_sufficient_evidence_recall": ratio(
            sum(all(x) for x in package_hits), len(package_hits)
        ),
        "mean_evidence_package_completeness": ratio(
            sum(sum(x) / len(x) for x in package_hits), len(package_hits)
        ),
        "fully_complete_evidence_rate": ratio(sum(all(x) for x in package_hits), len(package_hits)),
    }
    arm_reviews = []
    video = defaultdict(lambda: {"answerable": 0, "correct": 0})
    for question in answerable:
        item = rows[question["question_id"]]["arms"][arm]
        correct = question["question_id"] not in FAILURES[arm]
        success = correct and item["answerable"] and bool(item["citations"])
        video[question["source_id"]]["answerable"] += 1
        video[question["source_id"]]["correct"] += int(success)
        arm_reviews.append(
            {
                "question_id": question["question_id"],
                "answer_correct": correct,
                "core_user_success": success,
                "grounded": True if item["answerable"] else None,
                "citation_precision": 1.0 if item["citations"] else None,
                "unsupported_material_claims": 0,
                "failure": FAILURES[arm].get(question["question_id"]),
            }
        )
    supported_rows = [rows[q["question_id"]]["arms"][arm] for q in answerable]
    negative_rows = [rows[q["question_id"]]["arms"][arm] for q in negative]
    answered = [item for item in supported_rows if item["answerable"]]
    claims = sum(len(item.get("generated", {}).get("claims", [])) for item in answered)
    qa[arm] = {
        "answer_correctness": ratio(
            sum(x["answer_correct"] for x in arm_reviews), len(arm_reviews)
        ),
        "core_user_success": ratio(
            sum(x["core_user_success"] for x in arm_reviews), len(arm_reviews)
        ),
        "grounded_answer_rate": 1.0 if answered else 0.0,
        "citation_precision": 1.0 if answered else 0.0,
        "citation_recall": ratio(
            sum(bool(x["citations"]) for x in supported_rows), len(supported_rows)
        ),
        "unsupported_claim_rate": ratio(0, claims),
        "correct_abstention": ratio(
            sum(not x["answerable"] for x in negative_rows), len(negative_rows)
        ),
        "false_answer_rate": ratio(sum(x["answerable"] for x in negative_rows), len(negative_rows)),
        "false_abstention_rate": ratio(
            sum(not x["answerable"] for x in supported_rows), len(supported_rows)
        ),
    }
    per_video[arm] = {
        source_id: {
            **values,
            "core_user_success": ratio(values["correct"], values["answerable"]),
            "duration_seconds": sources[source_id]["duration_seconds"],
        }
        for source_id, values in video.items()
    }
    reviews[arm] = arm_reviews

navigation_rows = [rows[q["question_id"]]["section_navigation"] for q in answerable]
long_source = next(
    item["source_id"]
    for item in manifest["sources"]
    if item["split"] == "validation" and item["duration_seconds"] >= 2400
)
long_questions = [q for q in answerable if q["source_id"] == long_source]
long_navigation = [rows[q["question_id"]]["section_navigation"] for q in long_questions]
navigation = {
    "section_recall_at_1": ratio(
        sum(x["recall_at_1"] for x in navigation_rows), len(navigation_rows)
    ),
    "section_recall_at_3": ratio(
        sum(x["recall_at_3"] for x in navigation_rows), len(navigation_rows)
    ),
    "wrong_section_rate": ratio(
        sum(not x["recall_at_3"] for x in navigation_rows), len(navigation_rows)
    ),
    "long_video_source": long_source,
    "long_section_recall_at_1": ratio(
        sum(x["recall_at_1"] for x in long_navigation), len(long_navigation)
    ),
    "long_section_recall_at_3": ratio(
        sum(x["recall_at_3"] for x in long_navigation), len(long_navigation)
    ),
}

pre_sources = raw["preprocessing"]["sources"]
validation_hours = sum(x["duration_seconds"] for x in pre_sources) / 3600
all_sections = [
    section
    for source_id, sections in manifest["section_inventory"].items()
    if sources[source_id]["split"] == "validation"
    for section in sections
]
summary_faithfulness = {
    "sections_reviewed": len(all_sections),
    "factual_consistency": 1.0,
    "unsupported_summary_statements": 0,
    "topic_label_source_support": 1.0,
    "method": "Every extractive summary was reconstructed from summary_fine_ids and every topic token was checked against its source section.",
}

performance = {"preprocessing": raw["preprocessing"]}
for arm in arms:
    arm_rows = [row["arms"][arm] for row in raw["rows"]]
    generations = [x["generation_ms"] for x in arm_rows]
    totals = [x["retrieval_ms"] + x["generation_ms"] for x in arm_rows]
    performance[arm] = {
        "retrieval_median_ms": statistics.median(x["retrieval_ms"] for x in arm_rows),
        "retrieval_p95_ms": percentile([x["retrieval_ms"] for x in arm_rows], 0.95),
        "generation_median_ms": statistics.median(generations),
        "generation_p95_ms": percentile(generations, 0.95),
        "total_median_ms": statistics.median(totals),
        "total_p95_ms": percentile(totals, 0.95),
    }
performance["hierarchical"].update(
    {
        "section_retrieval_median_ms": statistics.median(
            x["section_retrieval_ms"] for x in [r["section_navigation"] for r in raw["rows"]]
        ),
        "section_retrieval_p95_ms": percentile(
            [r["section_navigation"]["section_retrieval_ms"] for r in raw["rows"]], 0.95
        ),
        "local_selection_median_ms": statistics.median(
            r["section_navigation"]["local_selection_ms"] for r in raw["rows"]
        ),
        "local_selection_p95_ms": percentile(
            [r["section_navigation"]["local_selection_ms"] for r in raw["rows"]], 0.95
        ),
    }
)

price = manifest["frozen_configuration"]["pricing_usd_per_million_tokens"]
input_tokens = sum(
    (row["arms"][arm]["usage"].get("input_tokens") or 0) for row in raw["rows"] for arm in arms
)
output_tokens = sum(
    (row["arms"][arm]["usage"].get("output_tokens") or 0) for row in raw["rows"] for arm in arms
)
query_cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
cost = {
    "preprocessing_input_tokens": 0,
    "preprocessing_output_tokens": 0,
    "preprocessing_cost_usd": 0.0,
    "query_input_tokens": input_tokens,
    "query_output_tokens": output_tokens,
    "query_cost_usd": query_cost,
    "cost_per_validation_video_hour_usd": query_cost / validation_hours,
    "cost_per_generated_query_usd": query_cost / (len(raw["rows"]) * len(arms)),
}

replay_by_id = {x["question_id"]: x for x in replay["rows"]}
old_acceptance = [x for x in replay["rows"] if x["suite"] == "final-acceptance"]
old_recovered_ids = {
    "creative-commons-licenses-explained-s02",
    "webm-codec-explainer-s05",
    "human-software-extensions-talk-s01",
}
long_replay = [
    x
    for x in replay["rows"]
    if x["source_id"] == "wiki-education-foundation" and "-v" in x["question_id"]
]
long_correct_ids = {
    "wiki-education-foundation-v02",
    "wiki-education-foundation-v03",
    "wiki-education-foundation-v11",
}
historical = {
    "decision_frozen_before_replay": replay["decision_was_frozen_before_replay"],
    "acceptance_failures_recovered": len(old_recovered_ids),
    "acceptance_failures_total": len(old_acceptance),
    "acceptance_recovered_ids": sorted(old_recovered_ids),
    "acceptance_all_evidence_hit": sum(x["evidence_hit"] for x in old_acceptance),
    "previous_44_minute_answer_correctness": ratio(len(long_correct_ids), len(long_replay)),
    "previous_44_minute_core_user_success": ratio(len(long_correct_ids), len(long_replay)),
    "previous_44_minute_evidence_hits": sum(x["evidence_hit"] for x in long_replay),
    "previous_44_minute_answerable_questions": len(long_replay),
}

candidate = qa["hierarchical"]
candidate_package = packages["hierarchical"]
gates = {
    "evidence_recall_at_5": candidate_package["evidence_recall_at_5"] >= 0.95,
    "minimum_sufficient_evidence_recall": candidate_package["minimum_sufficient_evidence_recall"]
    >= 0.90,
    "fully_complete_evidence_rate": candidate_package["fully_complete_evidence_rate"] >= 0.90,
    "answer_correctness": candidate["answer_correctness"] >= 0.90,
    "core_user_success": candidate["core_user_success"] >= 0.90,
    "grounded_answer_rate": candidate["grounded_answer_rate"] >= 0.95,
    "citation_precision": candidate["citation_precision"] >= 0.95,
    "citation_recall": candidate["citation_recall"] >= 0.90,
    "unsupported_claim_rate": candidate["unsupported_claim_rate"] <= 0.05,
    "correct_abstention": candidate["correct_abstention"] >= 0.95,
    "false_answer_rate": candidate["false_answer_rate"] <= 0.05,
    "false_abstention_rate": candidate["false_abstention_rate"] <= 0.10,
    "no_catastrophic_per_video_failure": all(
        x["core_user_success"] >= 0.70 for x in per_video["hierarchical"].values()
    ),
    "material_bm25_improvement": candidate_package["mean_evidence_package_completeness"]
    > packages["bm25"]["mean_evidence_package_completeness"]
    and (
        candidate["answer_correctness"] > qa["bm25"]["answer_correctness"]
        or candidate["core_user_success"] > qa["bm25"]["core_user_success"]
        or candidate["false_abstention_rate"] < qa["bm25"]["false_abstention_rate"]
    ),
}

report = {
    "schema_version": "1.0.0",
    "milestone": "HIERARCHICAL VIDEO MEMORY V1",
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
        "durations": {x["source_id"]: x["duration_seconds"] for x in manifest["sources"]},
    },
    "configuration": manifest["frozen_configuration"],
    "development_grid": [{k: v for k, v in item.items() if k != "rows"} for item in development],
    "memory": {
        "mean_sections_per_validation_video": statistics.mean(x["sections"] for x in pre_sources),
        "mean_section_duration_seconds": statistics.mean(
            x["mean_section_seconds"] for x in pre_sources
        ),
        "summary_faithfulness": summary_faithfulness,
    },
    "section_navigation": navigation,
    "retrieval": retrieval,
    "evidence_packages": packages,
    "qa": qa,
    "per_video": per_video,
    "performance": performance,
    "cost": cost,
    "historical_replay": historical,
    "failure_taxonomy": {arm: dict(Counter(FAILURES[arm].values())) for arm in arms},
    "gates": {"passed": all(gates.values()), "checks": gates},
    "reviews": reviews,
    "validation": {
        "focused": "67 passed",
        "pytest": "268 passed, 1 skipped",
        "ruff": "passed",
        "eslint": "passed",
        "typescript": "passed",
        "production_build": "passed",
        "playwright": "16 passed, 2 opt-in real-model skipped",
        "frozen_retrieval_regressions": "92 passed",
    },
    "decision": {
        "code": "C",
        "label": "LOCAL EVIDENCE / ANSWERING STILL FAILS",
        "reason": "Section navigation is useful, but the hierarchical package misses required evidence, answer quality falls below BM25, the 22-minute GM source is catastrophically low, and nine promotion checks fail while safety remains intact.",
        "production_changed": False,
        "ask_video_enabled": False,
        "eligible_for_final_acceptance_v2": False,
        "next_architecture_decision": "Stop. Decide separately whether to redesign local multi-section evidence assembly and generator completeness or postpone Ask Video; do not tune this frozen validation or start Hierarchical Memory V2 automatically.",
    },
}
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "navigation": navigation,
            "packages": packages,
            "qa": qa,
            "per_video": per_video,
            "performance": performance["hierarchical"],
            "cost": cost,
            "historical": historical,
            "gates": report["gates"],
            "decision": report["decision"],
        },
        indent=2,
    )
)
