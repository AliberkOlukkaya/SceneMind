"""Run the checksum-frozen semantic/hierarchical Q&A validation once."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import psutil
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Segment
from app.qa import (
    ABSTENTION,
    QA_INSTRUCTIONS,
    OpenAIAnswerGenerator,
    chunk_segments,
    resolve_answer,
    retrieve_evidence,
)
from app.semantic_qa import (
    MODEL_NAME,
    MODEL_REVISION,
    MiniLMEncoder,
    SemanticIndex,
    build_representations,
    hybrid_fine_candidates,
    select_hierarchical_evidence,
    semantic_retrieve,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ml/evaluation/semantic_hierarchical_qa_v1_manifest.json"
CHECKSUM = MANIFEST.with_suffix(".sha256")
RUN_ROOT = ROOT / "data/semantic-hierarchical-qa-v1"
RAW = RUN_ROOT / "validation-raw.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def overlaps(item, interval) -> bool:
    return item.start_seconds <= interval[1] and item.end_seconds >= interval[0]


def fact_hits(evidence, question) -> list[bool]:
    return [
        any(overlaps(item, interval) for item in evidence for interval in alternatives)
        for alternatives in question["minimum_sufficient_evidence"]
    ]


def serialized(evidence) -> list[dict]:
    return [
        {
            **vars(item),
            "segment_ids": list(item.segment_ids),
        }
        for item in evidence
    ]


def verify(manifest: dict) -> None:
    if sha256(MANIFEST) != CHECKSUM.read_text(encoding="ascii").split()[0]:
        raise RuntimeError("frozen manifest checksum mismatch")
    frozen = manifest["frozen_configuration"]
    if sha256(ROOT / "backend/app/semantic_qa.py") != frozen["semantic_implementation_sha256"]:
        raise RuntimeError("frozen semantic implementation mismatch")
    if hashlib.sha256(QA_INSTRUCTIONS.encode()).hexdigest() != frozen["generator_prompt_sha256"]:
        raise RuntimeError("frozen generator prompt mismatch")
    if (MODEL_NAME, MODEL_REVISION) != (
        frozen["embedding_model"],
        frozen["embedding_revision"],
    ):
        raise RuntimeError("frozen embedding model mismatch")
    if settings.qa_model != frozen["generator_model"]:
        raise RuntimeError("frozen generator model mismatch")


def generate(generator, question: str, evidence) -> dict:
    started = time.perf_counter()
    generated, usage = generator.generate(question, list(evidence))
    resolved = resolve_answer(generated, list(evidence), question)
    return {
        "generated": generated.model_dump(),
        "usage": usage,
        "answerable": resolved["answerable"],
        "answer": resolved["answer"],
        "citations": resolved["citations"],
        "generation_ms": (time.perf_counter() - started) * 1000,
    }


def main(resume: bool) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify(manifest)
    if RAW.exists() and not resume:
        raise RuntimeError("raw validation output exists; refusing a second frozen run")
    report = (
        json.loads(RAW.read_text(encoding="utf-8"))
        if RAW.exists()
        else {
            "schema_version": "1.0.0",
            "manifest_sha256": sha256(MANIFEST),
            "validation_run_count": 1,
            "started_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
            "qa_product_default_enabled": settings.qa_enabled,
            "rows": [],
        }
    )
    completed = {row["question_id"] for row in report["rows"]}
    sources = {source["source_id"]: source for source in manifest["sources"]}
    validation_sources = {
        question["source_id"] for question in manifest["validation"]["questions"]
    }
    database = create_engine(f"sqlite:///{RUN_ROOT / 'scenemind.db'}")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    encoder_started = time.perf_counter()
    encoder = MiniLMEncoder(ROOT / "data/models/hf")
    encoder_load_seconds = time.perf_counter() - encoder_started
    index_cache = {}
    index_stats = []
    peak_rss = process.memory_info().rss
    for source_id in validation_sources:
        source = sources[source_id]
        with Session(database) as session:
            segments = list(
                session.scalars(
                    select(Segment)
                    .where(Segment.video_id == source["video_id"])
                    .order_by(Segment.start)
                )
            )
        fine, context = build_representations(source["video_id"], segments)
        started = time.perf_counter()
        fine_index = SemanticIndex.build(fine, encoder)
        context_index = SemanticIndex.build(context, encoder)
        build_seconds = time.perf_counter() - started
        index_folder = RUN_ROOT / "validation-indexes" / source_id
        fine_index.save(index_folder / "fine")
        context_index.save(index_folder / "context")
        index_bytes = sum(path.stat().st_size for path in index_folder.rglob("*") if path.is_file())
        peak_rss = max(peak_rss, process.memory_info().rss)
        index_stats.append(
            {
                "source_id": source_id,
                "duration_seconds": source["duration_seconds"],
                "fine_units": len(fine),
                "context_units": len(context),
                "build_seconds": build_seconds,
                "index_bytes": index_bytes,
            }
        )
        index_cache[source_id] = (segments, fine, fine_index, context_index)
    report["index_performance"] = {
        "encoder_load_seconds": encoder_load_seconds,
        "sources": index_stats,
        "total_build_seconds": sum(item["build_seconds"] for item in index_stats),
        "total_index_bytes": sum(item["index_bytes"] for item in index_stats),
        "added_peak_rss_bytes": peak_rss - rss_before,
    }
    generator = OpenAIAnswerGenerator()
    if not generator.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured locally")

    questions = manifest["validation"]["questions"]
    for position, question in enumerate(questions, 1):
        if question["question_id"] in completed:
            continue
        segments, fine, fine_index, context_index = index_cache[question["source_id"]]
        video_id = sources[question["source_id"]]["video_id"]
        baseline_started = time.perf_counter()
        baseline_evidence = retrieve_evidence(
            question["question"], chunk_segments(video_id, segments), 5
        )
        baseline_retrieval_ms = (time.perf_counter() - baseline_started) * 1000
        semantic_started = time.perf_counter()
        fine_hits = semantic_retrieve(question["question"], fine_index, encoder, 10)
        context_hits = semantic_retrieve(question["question"], context_index, encoder, 5)
        semantic_ms = (time.perf_counter() - semantic_started) * 1000
        selection = select_hierarchical_evidence(
            question["question"], fine, fine_hits, context_hits
        )
        hybrid = hybrid_fine_candidates(question["question"], fine, fine_hits, context_hits)
        row = {
            "question_id": question["question_id"],
            "source_id": question["source_id"],
            "question": question["question"],
            "expected_answerable": question["answerable"],
            "baseline": {
                "evidence": serialized(baseline_evidence),
                "fact_hits": fact_hits(baseline_evidence, question),
                "retrieval_ms": baseline_retrieval_ms,
            },
            "candidate": {
                "evidence": serialized(selection.evidence),
                "fact_hits": fact_hits(selection.evidence, question),
                "semantic_recall": {
                    str(k): fact_hits([hit.unit for hit in fine_hits[:k]], question)
                    for k in (1, 3, 5, 10)
                },
                "hybrid_recall": {
                    str(k): fact_hits(hybrid[:k], question) for k in (1, 3, 5, 10)
                },
                "semantic_ms": semantic_ms,
                "selection_ms": selection.selection_ms,
                "evidence_units": len(selection.evidence),
                "evidence_characters": selection.total_characters,
                "estimated_tokens": selection.estimated_tokens,
                "temporal_span_seconds": selection.temporal_span_seconds,
            },
        }
        for name in ("baseline", "candidate"):
            evidence = baseline_evidence if name == "baseline" else selection.evidence
            if not evidence:
                row[name].update(
                    usage={"provider": "none", "input_tokens": 0, "output_tokens": 0},
                    answerable=False,
                    answer=ABSTENTION,
                    citations=[],
                    generation_ms=0.0,
                )
            else:
                try:
                    row[name].update(generate(generator, question["question"], evidence))
                except Exception as error:
                    row[name].update(
                        error=f"{type(error).__name__}: {error}",
                        usage={"provider": "error", "input_tokens": 0, "output_tokens": 0},
                        answerable=False,
                        answer="",
                        citations=[],
                        generation_ms=0.0,
                    )
            row[name]["total_ms"] = (
                row[name].get("retrieval_ms", semantic_ms + selection.selection_ms)
                + row[name]["generation_ms"]
            )
        report["rows"].append(row)
        RAW.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{position:02d}/{len(questions)} {question['question_id']}", flush=True)

    report["completed_utc"] = __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat()
    for name in ("baseline", "candidate"):
        rows = [row[name] for row in report["rows"]]
        report.setdefault("performance", {})[name] = {
            "input_tokens": sum(row["usage"].get("input_tokens") or 0 for row in rows),
            "output_tokens": sum(row["usage"].get("output_tokens") or 0 for row in rows),
            "generation_median_ms": statistics.median(row["generation_ms"] for row in rows),
            "generation_p95_ms": percentile([row["generation_ms"] for row in rows], 0.95),
            "total_median_ms": statistics.median(row["total_ms"] for row in rows),
            "total_p95_ms": percentile([row["total_ms"] for row in rows], 0.95),
        }
    prices = manifest["frozen_configuration"]["pricing_usd_per_million_tokens"]
    for values in report["performance"].values():
        values["cost_usd"] = (
            values["input_tokens"] * prices["input"]
            + values["output_tokens"] * prices["output"]
        ) / 1_000_000
        values["cost_per_answer_usd"] = values["cost_usd"] / len(report["rows"])
    RAW.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args().resume)
