"""Run development or frozen validation for structured Q&A evidence V1."""

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
    expand_structured_evidence,
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


def overlaps(chunk, interval):
    return chunk.end_seconds >= interval[0] and chunk.start_seconds <= interval[1]


def interval_hits(evidence, intervals):
    return [any(overlaps(chunk, interval) for chunk in evidence) for interval in intervals]


def load_chunks(transcript_dir, source_id):
    payload = json.loads((transcript_dir / f"{source_id}.json").read_text(encoding="utf-8"))
    segments = [
        SimpleNamespace(id=index, start=row["start"], end=row["end"], text=row["text"])
        for index, row in enumerate(payload["segments"], 1)
    ]
    return segments, chunk_segments(source_id, segments)


def summarize(results):
    answerable = [row for row in results if row["answerable"]]
    structured = [row for row in answerable if row["question_type"] in {"LIST_COUNT", "TEMPORAL"}]
    lists = [row for row in answerable if row["question_type"] == "LIST_COUNT"]
    temporal = [row for row in answerable if row["question_type"] == "TEMPORAL"]
    latency_keys = ("retrieval", "expansion", "generation", "total")
    summary = {
        "base_evidence_recall": sum(row["base_any_hit"] for row in answerable) / len(answerable),
        "expanded_evidence_recall": sum(row["expanded_any_hit"] for row in answerable)
        / len(answerable),
        "structured_base_evidence_recall": sum(row["base_all_intervals_hit"] for row in structured)
        / len(structured),
        "structured_expanded_evidence_recall": sum(
            row["expanded_all_intervals_hit"] for row in structured
        )
        / len(structured),
        "list_evidence_completeness": sum(row["expanded_all_intervals_hit"] for row in lists)
        / len(lists),
        "temporal_anchor_recall": sum(row["temporal_anchor_hit"] for row in temporal)
        / len(temporal),
        "temporal_target_recall": sum(row["temporal_target_hit"] for row in temporal)
        / len(temporal),
        "temporal_pair_recall": sum(row["temporal_pair_hit"] for row in temporal) / len(temporal),
        "mean_initial_evidence_count": statistics.mean(
            row["evidence_budget"]["initial_count"] for row in results
        ),
        "mean_expanded_evidence_count": statistics.mean(
            row["evidence_budget"]["expanded_count"] for row in results
        ),
        "mean_added_characters": statistics.mean(
            row["evidence_budget"]["added_characters"] for row in results
        ),
        "input_tokens": sum(row["usage"].get("input_tokens") or 0 for row in results),
        "output_tokens": sum(row["usage"].get("output_tokens") or 0 for row in results),
        "latency_ms": {
            key: {
                "median": statistics.median(row["latency_ms"][key] for row in results),
                "p95": percentile([row["latency_ms"][key] for row in results], 0.95),
            }
            for key in latency_keys
        },
    }
    provider_rows = [row for row in results if row["provider_decision"] is not None]
    if provider_rows:
        negatives = [row for row in provider_rows if not row["answerable"]]
        summary.update(
            {
                "correct_answerability_rate": sum(
                    row["answerable"] == row["generated"]["answerable"] for row in provider_rows
                )
                / len(provider_rows),
                "correct_abstention_rate": sum(
                    not row["generated"]["answerable"] for row in negatives
                )
                / len(negatives),
                "false_answer_rate": sum(row["generated"]["answerable"] for row in negatives)
                / len(negatives),
                "false_abstention_rate": sum(
                    not row["generated"]["answerable"] for row in provider_rows if row["answerable"]
                )
                / sum(row["answerable"] for row in provider_rows),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation"), required=True)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--max-list-additions", type=int, default=3)
    parser.add_argument("--temporal-neighbor-count", type=int, default=2)
    args = parser.parse_args()

    manifest_path = ROOT / "ml/evaluation/qa_structured_question_evidence_v1_manifest.json"
    checksum_path = ROOT / "ml/evaluation/qa_structured_question_evidence_v1_manifest.sha256"
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
    generator = None if args.retrieval_only else OpenAIAnswerGenerator()
    results = []
    for question in manifest[args.split]["questions"]:
        segments, chunks = source_data[question["source_id"]]
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        base = retrieve_evidence(question["question"], chunks)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        selection = expand_structured_evidence(
            question["question"],
            chunks,
            base,
            segments=segments,
            max_list_additions=args.max_list_additions,
            temporal_neighbor_count=args.temporal_neighbor_count,
        )
        evidence = list(selection.expanded)
        generation_started = time.perf_counter()
        if generator is not None and evidence:
            generated, usage = generator.generate(question["question"], evidence)
            answer = resolve_answer(generated, evidence, question["question"])
            decision = generated.model_dump()
        else:
            usage = {"provider": "none", "model": None, "input_tokens": 0, "output_tokens": 0}
            answer = {"answerable": False, "answer": ABSTENTION, "citations": []}
            decision = None
        generation_ms = (time.perf_counter() - generation_started) * 1000
        intervals = question["expected_evidence_intervals"]
        base_hits = interval_hits(base, intervals)
        expanded_hits = interval_hits(evidence, intervals)
        anchor_interval = question["anchor_interval"]
        target_interval = question["target_interval"]
        anchor_hit = bool(anchor_interval) and any(
            item.role == "TEMPORAL_ANCHOR" and overlaps(item, anchor_interval) for item in evidence
        )
        target_hit = bool(target_interval) and any(
            item.role == "TEMPORAL_TARGET" and overlaps(item, target_interval) for item in evidence
        )
        results.append(
            {
                **question,
                "segment_count": len(segments),
                "chunk_count": len(chunks),
                "base_evidence": [item.__dict__ for item in base],
                "expanded_evidence": [item.__dict__ for item in evidence],
                "base_any_hit": bool(base_hits) and any(base_hits),
                "expanded_any_hit": bool(expanded_hits) and any(expanded_hits),
                "base_all_intervals_hit": bool(base_hits) and all(base_hits),
                "expanded_all_intervals_hit": bool(expanded_hits) and all(expanded_hits),
                "temporal_anchor_hit": anchor_hit,
                "temporal_target_hit": target_hit,
                "temporal_pair_hit": anchor_hit and target_hit,
                "provider_decision": decision,
                "generated": answer,
                "usage": usage,
                "latency_ms": {
                    "retrieval": retrieval_ms,
                    "expansion": selection.expansion_ms,
                    "generation": generation_ms,
                    "total": (time.perf_counter() - total_started) * 1000,
                },
                "evidence_budget": {
                    "initial_count": len(selection.base),
                    "expanded_count": len(selection.expanded),
                    "added_characters": selection.added_characters,
                    "temporal_span_seconds": selection.temporal_span_seconds,
                },
                "human_review": None,
            }
        )
        print(question["id"], len(base), len(evidence), answer["answerable"])

    output = {
        "schema_version": 1,
        "suite_id": "qa-structured-question-evidence-v1",
        "split": args.split,
        "manifest_sha256": actual_checksum,
        "retrieval_only": args.retrieval_only,
        "configuration": {
            "max_list_additions": args.max_list_additions,
            "temporal_neighbor_count": args.temporal_neighbor_count,
        },
        "source_count": len(source_data),
        "question_count": len(results),
        "provider": "none" if args.retrieval_only else "openai",
        "automatic_summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
