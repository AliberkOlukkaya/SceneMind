"""Run the checksum-frozen Final Ask Video Core Acceptance exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Segment, engine, engine_for
from app.qa import (
    QA_INSTRUCTIONS,
    OpenAIAnswerGenerator,
    chunk_segments,
    classify_question_scope,
    resolve_answer,
    retrieve_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml/evaluation/final_ask_video_core_acceptance_v1_manifest.json"
CHECKSUM = MANIFEST.with_suffix(".sha256")
RUN_ROOT = ROOT / "data/final-ask-video-core-acceptance-v1"
RAW = RUN_ROOT / "acceptance-raw.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def interval_hit(evidence: list, expected: list[list[float]]) -> bool:
    return any(
        item.start_seconds <= end and item.end_seconds >= start
        for item in evidence
        for start, end in expected
    )


def verify(manifest: dict) -> None:
    expected = CHECKSUM.read_text(encoding="ascii").split()[0]
    if sha256(MANIFEST) != expected:
        raise RuntimeError("frozen manifest checksum mismatch")
    frozen = manifest["frozen_configuration"]
    if sha256(ROOT / "backend/app/qa.py") != frozen["scope_gate_implementation_sha256"]:
        raise RuntimeError("frozen Q&A implementation mismatch")
    if hashlib.sha256(QA_INSTRUCTIONS.encode()).hexdigest() != frozen["prompt_sha256"]:
        raise RuntimeError("frozen Q&A prompt mismatch")
    if settings.qa_model != frozen["model"]:
        raise RuntimeError("frozen Q&A model mismatch")
    if settings.qa_top_k != 5:
        raise RuntimeError("BM25 Top-5 is not frozen")


def main(resume: bool) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify(manifest)
    settings.data_dir = RUN_ROOT / "videos"
    settings.database_url = f"sqlite:///{RUN_ROOT / 'scenemind.db'}"
    settings.model_cache = ROOT / "data/models"
    settings.qa_enabled = False
    engine_for.cache_clear()
    if RAW.exists():
        if not resume:
            raise RuntimeError("raw acceptance output already exists; refusing a second run")
        report = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        report = {
            "schema_version": "1.0.0",
            "manifest_sha256": sha256(MANIFEST),
            "run_started_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
            "qa_product_default_enabled": settings.qa_enabled,
            "rows": [],
        }
    completed = {row["question_id"] for row in report["rows"]}
    sources = {row["source_id"]: row for row in manifest["sources"]}
    generator = OpenAIAnswerGenerator()
    if not generator.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured locally")

    segment_cache = {}
    for source_id, source in sources.items():
        with Session(engine()) as session:
            segment_cache[source_id] = list(
                session.scalars(
                    select(Segment)
                    .where(Segment.video_id == source["video_id"])
                    .order_by(Segment.start)
                ).all()
            )
        if not segment_cache[source_id]:
            raise RuntimeError(f"missing production transcript for {source_id}")

    for position, item in enumerate(manifest["questions"], 1):
        if item["question_id"] in completed:
            continue
        started = time.perf_counter()
        scope = classify_question_scope(item["question"])
        row = {
            "question_id": item["question_id"],
            "source_id": item["source_id"],
            "question": item["question"],
            "expected_supported_scope": item["supported_scope"],
            "expected_answerable": item["answerable_from_transcript"],
            "scope": {"supported": scope.supported, "category": scope.category},
            "generator_called": False,
        }
        if not scope.supported:
            row.update(
                answerable=False,
                answer=scope.response,
                evidence=[],
                citations=[],
                evidence_recall_at_5=None,
                usage={"provider": "none", "input_tokens": 0, "output_tokens": 0},
                retrieval_ms=0.0,
                generation_ms=0.0,
            )
        else:
            retrieval_started = time.perf_counter()
            chunks = chunk_segments(
                sources[item["source_id"]]["video_id"], segment_cache[item["source_id"]]
            )
            evidence = retrieve_evidence(item["question"], chunks)
            row["retrieval_ms"] = (time.perf_counter() - retrieval_started) * 1000
            row["evidence"] = [vars(unit) for unit in evidence]
            row["evidence_recall_at_5"] = (
                interval_hit(evidence, item["expected_evidence_intervals"])
                if item["answerable_from_transcript"]
                else None
            )
            if not evidence:
                row.update(
                    answerable=False,
                    answer="I couldn't find enough evidence in this video to answer that reliably.",
                    citations=[],
                    usage={"provider": "none", "input_tokens": 0, "output_tokens": 0},
                    generation_ms=0.0,
                )
            else:
                generation_started = time.perf_counter()
                row["generator_called"] = True
                try:
                    generated, usage = generator.generate(item["question"], evidence)
                    resolved = resolve_answer(generated, evidence, item["question"])
                    row.update(
                        generated=generated.model_dump(),
                        usage=usage,
                        answerable=resolved["answerable"],
                        answer=resolved["answer"],
                        citations=resolved["citations"],
                    )
                except Exception as error:  # persisted as a measured product failure
                    row.update(
                        error=f"{type(error).__name__}: {error}",
                        usage={"provider": "error", "input_tokens": 0, "output_tokens": 0},
                        answerable=False,
                        answer="",
                        citations=[],
                    )
                row["generation_ms"] = (time.perf_counter() - generation_started) * 1000
        row["total_ms"] = (time.perf_counter() - started) * 1000
        report["rows"].append(row)
        RAW.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{position:02d}/60 {item['question_id']} {scope.category}", flush=True)

    report["run_completed_utc"] = __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat()
    generated_rows = [row for row in report["rows"] if row["generator_called"]]
    report["performance"] = {
        "input_tokens": sum(row["usage"].get("input_tokens") or 0 for row in generated_rows),
        "output_tokens": sum(row["usage"].get("output_tokens") or 0 for row in generated_rows),
        "generation_median_ms": statistics.median(row["generation_ms"] for row in generated_rows),
        "generation_p95_ms": percentile([row["generation_ms"] for row in generated_rows], 0.95),
        "total_median_ms": statistics.median(row["total_ms"] for row in report["rows"]),
        "total_p95_ms": percentile([row["total_ms"] for row in report["rows"]], 0.95),
    }
    prices = manifest["frozen_configuration"]["pricing_usd_per_million_tokens"]
    report["performance"]["total_api_cost_usd"] = (
        report["performance"]["input_tokens"] * prices["input"]
        + report["performance"]["output_tokens"] * prices["output"]
    ) / 1_000_000
    report["performance"]["cost_per_generated_answer_usd"] = (
        report["performance"]["total_api_cost_usd"] / len(generated_rows)
    )
    RAW.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args().resume)
