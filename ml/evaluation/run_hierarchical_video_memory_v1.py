"""Run the checksum-frozen Hierarchical Video Memory V1 validation once."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import settings  # noqa: E402
from app.database import Segment  # noqa: E402
from app.qa import (  # noqa: E402
    ABSTENTION,
    QA_INSTRUCTIONS,
    OpenAIAnswerGenerator,
    chunk_segments,
    resolve_answer,
    retrieve_evidence,
)
from app.semantic_qa import (  # noqa: E402
    MODEL_NAME,
    MODEL_REVISION,
    MiniLMEncoder,
    SemanticIndex,
    build_representations,
    hybrid_fine_candidates,
    select_hierarchical_evidence,
    semantic_retrieve,
)
from app.video_memory import (  # noqa: E402
    SectionMemoryIndex,
    build_memory,
    configuration_fingerprint,
    retrieve_from_memory,
    transcript_fingerprint,
)

MANIFEST = ROOT / "ml/evaluation/hierarchical_video_memory_v1_manifest.json"
CHECKSUM = MANIFEST.with_suffix(".sha256")
RUN = ROOT / "data/hierarchical-video-memory-v1"
RAW = RUN / "validation-raw.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlaps(item, iv):
    return item.start_seconds <= iv[1] and item.end_seconds >= iv[0]


def fact_hits(items, q):
    return [
        any(overlaps(item, iv) for item in items for iv in alternatives)
        for alternatives in q["minimum_sufficient_evidence"]
    ]


def serialize(items):
    return [{**vars(x), "segment_ids": list(x.segment_ids)} for x in items]


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def verify(manifest):
    if sha(MANIFEST) != CHECKSUM.read_text(encoding="ascii").split()[0]:
        raise RuntimeError("frozen manifest checksum mismatch")
    frozen = manifest["frozen_configuration"]
    if sha(ROOT / "backend/app/video_memory.py") != frozen["implementation_sha256"]:
        raise RuntimeError("frozen memory implementation mismatch")
    if sha(ROOT / "backend/app/semantic_qa.py") != frozen["semantic_implementation_sha256"]:
        raise RuntimeError("frozen semantic implementation mismatch")
    if configuration_fingerprint() != frozen["configuration_fingerprint"]:
        raise RuntimeError("memory configuration mismatch")
    if (MODEL_NAME, MODEL_REVISION) != (frozen["embedding_model"], frozen["embedding_revision"]):
        raise RuntimeError("embedding mismatch")
    if hashlib.sha256(QA_INSTRUCTIONS.encode()).hexdigest() != frozen["generator_prompt_sha256"]:
        raise RuntimeError("generator prompt mismatch")
    if settings.qa_model != frozen["generator_model"]:
        raise RuntimeError("generator model mismatch")


def generate(generator, question, evidence):
    started = time.perf_counter()
    generated, usage = generator.generate(question, list(evidence))
    resolved = resolve_answer(generated, list(evidence), question)
    return {
        "generated": generated.model_dump(),
        "usage": usage,
        **resolved,
        "generation_ms": (time.perf_counter() - started) * 1000,
    }


def main(resume=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify(manifest)
    if RAW.exists() and not resume:
        raise RuntimeError("raw validation exists; refusing second frozen run")
    report = (
        json.loads(RAW.read_text())
        if RAW.exists()
        else {
            "schema_version": "1.0.0",
            "manifest_sha256": sha(MANIFEST),
            "validation_run_count": 1,
            "started_utc": datetime.now(UTC).isoformat(),
            "qa_product_default_enabled": settings.qa_enabled,
            "rows": [],
        }
    )
    completed = {x["question_id"] for x in report["rows"]}
    sources = {x["source_id"]: x for x in manifest["sources"]}
    questions = manifest["validation"]["questions"]
    engine = create_engine(f"sqlite:///{RUN / 'scenemind.db'}")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    encoder = MiniLMEncoder(ROOT / "data/models/hf")
    encoder_load = time.perf_counter() - started
    peak = process.memory_info().rss
    cache = {}
    stats = []
    for sid in sorted({q["source_id"] for q in questions}):
        source = sources[sid]
        with Session(engine) as session:
            segments = list(
                session.scalars(
                    select(Segment)
                    .where(Segment.video_id == source["video_id"])
                    .order_by(Segment.start)
                )
            )
        build_started = time.perf_counter()
        bundle = build_memory(source["video_id"], segments, encoder)
        build_seconds = time.perf_counter() - build_started
        memory_started = time.perf_counter()
        index = SectionMemoryIndex.build(bundle, encoder)
        memory_seconds = time.perf_counter() - memory_started
        index_dir = RUN / "validation-memory" / sid
        save_started = time.perf_counter()
        index.save(index_dir)
        persist_seconds = time.perf_counter() - save_started
        reload_started = time.perf_counter()
        reloaded = SectionMemoryIndex.load(
            index_dir,
            expected_transcript_fingerprint=transcript_fingerprint(source["video_id"], segments),
        )
        reload_seconds = time.perf_counter() - reload_started
        fine, context = build_representations(source["video_id"], segments)
        flat_fine = SemanticIndex.build(fine, encoder)
        flat_context = SemanticIndex.build(context, encoder)
        sizes = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file())
        peak = max(peak, process.memory_info().rss)
        durations = [x.end_seconds - x.start_seconds for x in bundle.sections]
        stats.append(
            {
                "source_id": sid,
                "duration_seconds": source["duration_seconds"],
                "fine_units": len(bundle.fine),
                "sections": len(bundle.sections),
                "mean_section_seconds": statistics.mean(durations),
                "median_section_seconds": statistics.median(durations),
                "min_section_seconds": min(durations),
                "max_section_seconds": max(durations),
                "section_construction_seconds": build_seconds,
                "memory_embedding_index_seconds": memory_seconds,
                "persist_seconds": persist_seconds,
                "restart_reload_seconds": reload_seconds,
                "persistent_bytes": sizes,
            }
        )
        cache[sid] = (segments, fine, context, flat_fine, flat_context, bundle, reloaded)
    report["preprocessing"] = {
        "encoder_load_seconds": encoder_load,
        "sources": stats,
        "total_seconds": sum(
            x["section_construction_seconds"]
            + x["memory_embedding_index_seconds"]
            + x["persist_seconds"]
            for x in stats
        ),
        "persistent_bytes": sum(x["persistent_bytes"] for x in stats),
        "added_peak_rss_bytes": peak - rss_before,
        "preprocessing_input_tokens": 0,
        "preprocessing_output_tokens": 0,
        "preprocessing_cost_usd": 0.0,
    }
    generator = OpenAIAnswerGenerator()
    if not generator.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured locally")
    for number, q in enumerate(questions, 1):
        if q["question_id"] in completed:
            continue
        segments, fine, context, flat_fine, flat_context, bundle, index = cache[q["source_id"]]
        video_id = sources[q["source_id"]]["video_id"]
        t = time.perf_counter()
        bm25 = retrieve_evidence(q["question"], chunk_segments(video_id, segments), 5)
        bm25_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        fine_hits = semantic_retrieve(q["question"], flat_fine, encoder, 10)
        context_hits = semantic_retrieve(q["question"], flat_context, encoder, 5)
        flat_sem_ms = (time.perf_counter() - t) * 1000
        flat_ranked = hybrid_fine_candidates(q["question"], fine, fine_hits, context_hits)
        flat = select_hierarchical_evidence(q["question"], fine, fine_hits, context_hits)
        hierarchical = retrieve_from_memory(q["question"], index, encoder)
        fine_by_id = {x.unit_id: x for x in bundle.fine}
        hier_ranked = [fine_by_id[x] for x in hierarchical.candidate_fine_ids]
        arms = {"bm25": bm25, "flat_semantic": flat.evidence, "hierarchical": hierarchical.evidence}
        row = {
            "question_id": q["question_id"],
            "source_id": q["source_id"],
            "question": q["question"],
            "expected_answerable": q["answerable"],
            "relevant_section_ids": q["relevant_section_ids"],
            "section_navigation": {
                "selected_ids": list(hierarchical.section_ids),
                "scores": list(hierarchical.section_scores),
                "recall_at_1": bool(
                    set(hierarchical.section_ids[:1]) & set(q["relevant_section_ids"])
                )
                if q["answerable"]
                else None,
                "recall_at_3": bool(
                    set(hierarchical.section_ids[:3]) & set(q["relevant_section_ids"])
                )
                if q["answerable"]
                else None,
                "section_retrieval_ms": hierarchical.section_retrieval_ms,
                "local_selection_ms": hierarchical.local_selection_ms,
            },
            "arms": {},
        }
        ranked = {"bm25": bm25, "flat_semantic": flat_ranked, "hierarchical": hier_ranked}
        times = {
            "bm25": bm25_ms,
            "flat_semantic": flat_sem_ms + flat.selection_ms,
            "hierarchical": hierarchical.retrieval_ms,
        }
        for name, evidence in arms.items():
            item = {
                "evidence": serialize(evidence),
                "fact_hits": fact_hits(evidence, q),
                "retrieval_ms": times[name],
                "recall": {str(k): fact_hits(ranked[name][:k], q) for k in (1, 3, 5)},
            }
            try:
                item.update(
                    generate(generator, q["question"], evidence)
                ) if evidence else item.update(
                    {
                        "answerable": False,
                        "answer": ABSTENTION,
                        "citations": [],
                        "usage": {"provider": "none", "input_tokens": 0, "output_tokens": 0},
                        "generation_ms": 0.0,
                    }
                )
            except Exception as exc:
                item.update(
                    {
                        "answerable": False,
                        "answer": ABSTENTION,
                        "citations": [],
                        "usage": {"provider": "error", "input_tokens": 0, "output_tokens": 0},
                        "generation_ms": 0.0,
                        "generation_error": type(exc).__name__ + ": " + str(exc),
                    }
                )
            row["arms"][name] = item
        report["rows"].append(row)
        RAW.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{number}/{len(questions)} {q['question_id']}", flush=True)
    report["completed_utc"] = datetime.now(UTC).isoformat()
    RAW.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args().resume)
