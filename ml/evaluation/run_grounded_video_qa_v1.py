"""Run the checksum-frozen Grounded Video Q&A V1 validation exactly once."""

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.qa import (  # noqa: E402
    ABSTENTION,
    OpenAIAnswerGenerator,
    chunk_segments,
    resolve_answer,
    retrieve_evidence,
)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def overlaps(chunk, intervals):
    return any(
        chunk.end_seconds >= start and chunk.start_seconds <= end for start, end in intervals
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = ROOT / "ml/evaluation/grounded_video_qa_v1_manifest.json"
    checksum_path = ROOT / "ml/evaluation/grounded_video_qa_v1_manifest.sha256"
    expected_checksum = checksum_path.read_text(encoding="ascii").split()[0]
    actual_checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_checksum != expected_checksum:
        raise SystemExit("Frozen manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    connection = sqlite3.connect(args.database)
    rows = connection.execute(
        "SELECT id, start, end, text FROM segments WHERE video_id=? ORDER BY start",
        (args.video_id,),
    ).fetchall()
    connection.close()
    if not rows:
        raise SystemExit("No validation transcript segments found")
    segments = [SimpleNamespace(id=row[0], start=row[1], end=row[2], text=row[3]) for row in rows]
    chunks = chunk_segments(args.video_id, segments)
    generator = OpenAIAnswerGenerator()
    results = []
    for question in manifest["validation"]["questions"]:
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        evidence = retrieve_evidence(question["question"], chunks)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        generation_started = time.perf_counter()
        if evidence:
            generated, usage = generator.generate(question["question"], evidence)
            answer = resolve_answer(generated, evidence)
        else:
            usage = {
                "provider": "none",
                "model": None,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            answer = {"answerable": False, "answer": ABSTENTION, "citations": []}
        generation_ms = (time.perf_counter() - generation_started) * 1000
        intervals = question["expected_intervals"]
        rank_hits = {
            str(k): bool(intervals) and any(overlaps(item, intervals) for item in evidence[:k])
            for k in (1, 3, 5)
        }
        results.append(
            {
                **question,
                "retrieved_evidence": [item.__dict__ for item in evidence],
                "evidence_recall_hit": rank_hits,
                "generated": answer,
                "usage": usage,
                "latency_ms": {
                    "retrieval": retrieval_ms,
                    "generation": generation_ms,
                    "total": (time.perf_counter() - total_started) * 1000,
                },
                "human_review": None,
            }
        )
        print(question["id"], answer["answerable"], round(results[-1]["latency_ms"]["total"], 1))

    answerable = [row for row in results if row["answerable"]]
    latency = {
        key: [row["latency_ms"][key] for row in results]
        for key in ("retrieval", "generation", "total")
    }
    output = {
        "schema_version": 1,
        "manifest_sha256": actual_checksum,
        "video_id": args.video_id,
        "segment_count": len(segments),
        "chunk_count": len(chunks),
        "provider": "openai",
        "results": results,
        "automatic_summary": {
            "evidence_recall_at_1": sum(row["evidence_recall_hit"]["1"] for row in answerable)
            / len(answerable),
            "evidence_recall_at_3": sum(row["evidence_recall_hit"]["3"] for row in answerable)
            / len(answerable),
            "evidence_recall_at_5": sum(row["evidence_recall_hit"]["5"] for row in answerable)
            / len(answerable),
            "latency_ms": {
                key: {"median": statistics.median(values), "p95": percentile(values, 0.95)}
                for key, values in latency.items()
            },
            "input_tokens": sum(row["usage"].get("input_tokens") or 0 for row in results),
            "output_tokens": sum(row["usage"].get("output_tokens") or 0 for row in results),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
