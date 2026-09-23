"""Run development or the checksum-frozen Q&A Abstention Safety V1 validation."""

import argparse
import hashlib
import json
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


def load_chunks(transcript_dir, source_id):
    payload = json.loads((transcript_dir / f"{source_id}.json").read_text(encoding="utf-8"))
    segments = [
        SimpleNamespace(id=index, start=row["start"], end=row["end"], text=row["text"])
        for index, row in enumerate(payload["segments"], 1)
    ]
    return segments, chunk_segments(source_id, segments)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation"), required=True)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = ROOT / "ml/evaluation/qa_abstention_safety_v1_manifest.json"
    checksum_path = ROOT / "ml/evaluation/qa_abstention_safety_v1_manifest.sha256"
    actual_checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if args.split == "validation":
        expected_checksum = checksum_path.read_text(encoding="ascii").split()[0]
        if actual_checksum != expected_checksum:
            raise SystemExit("Frozen manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_data = {
        source["source_id"]: load_chunks(args.transcript_dir, source["source_id"])
        for source in manifest[args.split]["sources"]
    }
    generator = OpenAIAnswerGenerator()
    results = []
    for question in manifest[args.split]["questions"]:
        segments, chunks = source_data[question["source_id"]]
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        evidence = retrieve_evidence(question["question"], chunks)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        generation_started = time.perf_counter()
        if evidence:
            generated, usage = generator.generate(question["question"], evidence)
            answer = resolve_answer(generated, evidence, question["question"])
            decision = generated.model_dump()
        else:
            usage = {
                "provider": "none",
                "model": None,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            answer = {"answerable": False, "answer": ABSTENTION, "citations": []}
            decision = None
        generation_ms = (time.perf_counter() - generation_started) * 1000
        intervals = question["expected_intervals"]
        results.append(
            {
                **question,
                "segment_count": len(segments),
                "chunk_count": len(chunks),
                "retrieved_evidence": [item.__dict__ for item in evidence],
                "evidence_recall_hit_at_5": bool(intervals)
                and any(overlaps(item, intervals) for item in evidence),
                "provider_decision": decision,
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
        "suite_id": "qa-abstention-safety-v1",
        "split": args.split,
        "manifest_sha256": actual_checksum,
        "source_count": len(source_data),
        "question_count": len(results),
        "provider": "openai",
        "results": results,
        "automatic_summary": {
            "evidence_recall_at_5": sum(row["evidence_recall_hit_at_5"] for row in answerable)
            / len(answerable),
            "correct_answerability_rate": sum(
                row["answerable"] == row["generated"]["answerable"] for row in results
            )
            / len(results),
            "correct_abstention_rate": sum(
                not row["generated"]["answerable"] for row in results if not row["answerable"]
            )
            / sum(not row["answerable"] for row in results),
            "false_answer_rate": sum(
                row["generated"]["answerable"] for row in results if not row["answerable"]
            )
            / sum(not row["answerable"] for row in results),
            "false_abstention_rate": sum(not row["generated"]["answerable"] for row in answerable)
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
