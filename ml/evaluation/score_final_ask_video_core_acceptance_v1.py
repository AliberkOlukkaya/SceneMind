"""Apply the frozen human review to the single Final Ask Video acceptance run."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml/evaluation/final_ask_video_core_acceptance_v1_manifest.json"
RAW = ROOT / "data/final-ask-video-core-acceptance-v1/acceptance-raw.json"
PRODUCT = ROOT / "data/final-ask-video-core-acceptance-v1/product-ask-verification.json"
REPORT = ROOT / "ml/evaluation/reports/final-ask-video-core-acceptance-v1.json"

FALSE_ABSTENTIONS = {
    "creative-commons-licenses-explained-s02": "The retrieved evidence directly says Kerry granted permission with a Creative Commons license.",
    "creative-commons-licenses-explained-s05": "The retrieved evidence directly explains that no derivatives withholds permission to change the photo.",
    "webm-codec-explainer-s05": "The retrieved evidence directly says the server must send and manage one thousand streams.",
    "demo-video-production-tutorial-s03": "The retrieved evidence directly says time is needed to resolve issues before posting.",
}
INCOMPLETE_ANSWERS = {
    "human-software-extensions-talk-s01": "The answer omits bodies, senses, cognition, and the plug-in/discard aspect of the frozen definition.",
    "demo-video-production-tutorial-s04": "The answer gives audience characteristics but omits who the audience is and the intended outcome.",
}


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
raw = json.loads(RAW.read_text(encoding="utf-8"))
product = json.loads(PRODUCT.read_text(encoding="utf-8"))
annotations = {row["question_id"]: row for row in manifest["questions"]}
reviews = []
failure_counts = Counter()
per_video = defaultdict(lambda: {"supported_answerable": 0, "core_success": 0, "failures": 0})

for row in raw["rows"]:
    annotation = annotations[row["question_id"]]
    review = {
        "question_id": row["question_id"],
        "answer_correct": None,
        "grounded": None,
        "valid_citations": len(row["citations"]),
        "citation_count": len(row["citations"]),
        "unsupported_material_claims": 0,
        "core_user_success": None,
        "failure": None,
        "note": None,
    }
    if annotation["supported_scope"] and annotation["answerable_from_transcript"]:
        per_video[row["source_id"]]["supported_answerable"] += 1
        review["answer_correct"] = row["answerable"]
        review["grounded"] = True if row["answerable"] else None
        if row["question_id"] in FALSE_ABSTENTIONS:
            review.update(
                answer_correct=False,
                core_user_success=False,
                failure="FALSE_ABSTENTION",
                note=FALSE_ABSTENTIONS[row["question_id"]],
            )
        elif row["question_id"] in INCOMPLETE_ANSWERS:
            review.update(
                answer_correct=False,
                core_user_success=False,
                failure="GENERATION_ERROR",
                note=INCOMPLETE_ANSWERS[row["question_id"]],
            )
        else:
            review["core_user_success"] = bool(
                row["answerable"] and row["citations"] and review["grounded"]
            )
        if review["core_user_success"]:
            per_video[row["source_id"]]["core_success"] += 1
        else:
            per_video[row["source_id"]]["failures"] += 1
    elif annotation["supported_scope"]:
        review["answer_correct"] = not row["answerable"]
        if row["answerable"]:
            review.update(failure="MISSED_ABSTENTION", note="Frozen hard negative received an answer.")
    else:
        rejected = not row["scope"]["supported"] and not row["generator_called"]
        review["answer_correct"] = rejected
        if not rejected:
            review.update(failure="SCOPE_FALSE_ACCEPT", note="Out-of-scope question reached generation.")
    if review["failure"]:
        failure_counts[review["failure"]] += 1
    reviews.append(review)

core_rows = [
    row
    for row in raw["rows"]
    if annotations[row["question_id"]]["supported_scope"]
    and annotations[row["question_id"]]["answerable_from_transcript"]
]
core_reviews = [reviews[index] for index, row in enumerate(raw["rows"]) if row in core_rows]
unanswerable = [
    row
    for row in raw["rows"]
    if annotations[row["question_id"]]["supported_scope"]
    and not annotations[row["question_id"]]["answerable_from_transcript"]
]
out_scope = [row for row in raw["rows"] if not annotations[row["question_id"]]["supported_scope"]]
supported = [row for row in raw["rows"] if annotations[row["question_id"]]["supported_scope"]]
allowed = [row for row in raw["rows"] if row["scope"]["supported"]]
answered_core = [row for row in core_rows if row["answerable"]]
claims = sum(len(row.get("generated", {}).get("claims", [])) for row in answered_core)

metrics = {
    "scope_gate": {
        "supported_question_recall": ratio(sum(row["scope"]["supported"] for row in supported), len(supported)),
        "supported_question_precision": ratio(sum(annotations[row["question_id"]]["supported_scope"] for row in allowed), len(allowed)),
        "false_scope_rejection_rate": ratio(sum(not row["scope"]["supported"] for row in supported), len(supported)),
        "out_of_scope_rejection_rate": ratio(sum(not row["scope"]["supported"] for row in out_scope), len(out_scope)),
        "unsafe_generation_rate": ratio(sum(row["generator_called"] for row in out_scope), len(out_scope)),
    },
    "core_qa": {
        "evidence_recall_at_5": ratio(sum(row["evidence_recall_at_5"] is True for row in core_rows), len(core_rows)),
        "answer_correctness": ratio(sum(review["answer_correct"] is True for review in core_reviews), len(core_reviews)),
        "grounded_answer_rate": ratio(sum(review["grounded"] is True for review in core_reviews), len(answered_core)),
        "citation_precision": ratio(sum(review["valid_citations"] for review in core_reviews), sum(review["citation_count"] for review in core_reviews)),
        "citation_recall": ratio(sum(bool(row["citations"]) for row in core_rows), len(core_rows)),
        "unsupported_claim_rate": ratio(sum(review["unsupported_material_claims"] for review in core_reviews), claims),
        "core_user_success_rate": ratio(sum(review["core_user_success"] is True for review in core_reviews), len(core_reviews)),
        "false_abstention_rate": ratio(sum(not row["answerable"] for row in core_rows), len(core_rows)),
    },
    "unanswerable": {
        "correct_abstention_rate": ratio(sum(not row["answerable"] for row in unanswerable), len(unanswerable)),
        "false_answer_rate": ratio(sum(row["answerable"] for row in unanswerable), len(unanswerable)),
    },
}

product_tokens = {
    "input": sum(row["usage"]["input_tokens"] for row in product["rows"]),
    "output": sum(row["usage"]["output_tokens"] for row in product["rows"]),
}
product_cost = (product_tokens["input"] * 0.75 + product_tokens["output"] * 4.5) / 1_000_000
for source_id, values in per_video.items():
    values["core_user_success_rate"] = ratio(values["core_success"], values["supported_answerable"])

report = {
    "schema_version": "1.0.0",
    "milestone": "FINAL ASK VIDEO CORE ACCEPTANCE V1",
    "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    "single_frozen_run": True,
    "data": {
        "videos": len(manifest["sources"]),
        "questions": len(raw["rows"]),
        "supported_answerable": len(core_rows),
        "unanswerable": len(unanswerable),
        "out_of_scope": len(out_scope),
        "source_independent_from_prior_qa": True,
    },
    "metrics": metrics,
    "performance": {
        **raw["performance"],
        "product_flow_input_tokens": product_tokens["input"],
        "product_flow_output_tokens": product_tokens["output"],
        "product_flow_api_cost_usd": product_cost,
        "full_milestone_api_cost_usd": raw["performance"]["total_api_cost_usd"] + product_cost,
    },
    "product_flow": {
        "youtube": product["rows"][0],
        "upload": product["rows"][1],
        "production_default_remained_disabled": True,
        "isolated_runtime_enabled_for_endpoint_verification": True,
        "ask_ui": "passed on desktop and mobile",
        "citation_click_to_seek": "passed on desktop and mobile",
        "mobile_behavior": "passed",
    },
    "validation": {
        "pytest": "248 passed, 1 skipped",
        "ruff": "passed",
        "eslint": "passed",
        "typescript": "passed",
        "production_build": "passed",
        "playwright": "16 passed, 2 opt-in real-model skipped",
        "frozen_retrieval_regressions": "75 passed",
    },
    "failures": {"total": sum(failure_counts.values()), "taxonomy": dict(failure_counts)},
    "per_video": dict(per_video),
    "catastrophic_source_failure": False,
    "gates": {
        "passed": False,
        "failed": ["answer_correctness", "citation_recall", "core_user_success_rate"],
    },
    "decision": {
        "code": "B",
        "label": "CORE ANSWER QUALITY FAILED",
        "ask_video_enabled": False,
        "reason": "Retrieval and safety passed, but four false abstentions and two incomplete answers left correctness, citation recall, and core user success below fixed gates.",
        "next_milestone": "Choose a fundamentally different architecture; begin with a measured semantic transcript retrieval and hierarchical evidence design, not another prompt, BM25, neighbor-expansion, or rule tweak.",
    },
    "reviews": reviews,
}
REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"metrics": metrics, "failures": report["failures"], "decision": report["decision"]}, indent=2))
