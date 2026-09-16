"""Run unchanged production retrieval with evaluation-only Hybrid traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import psutil

from ml.evaluation.hybrid_fusion_trace import trace_fusion

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).parent / "development/hybrid_fusion_queries_v1.json"
DEFAULT_OUTPUT = Path(__file__).parent / "reports/hybrid-fusion-diagnostics.json"
DATA_ROOT = ROOT / "data/hybrid-fusion-diagnostics"
SOURCE_RUNS = {
    "design-free-software-talk": ("design-product", "design-pipeline.json", 8028),
    "human-software-extensions-talk": ("human-product", "human-pipeline.json", 8029),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["acceptance_isolation"]["final_english_acceptance_v2_used"] is not False:
        raise ValueError("Final Acceptance V2 cannot be development evidence")
    sources = {source["source_id"]: source for source in manifest["sources"]}
    if set(sources) != set(SOURCE_RUNS):
        raise ValueError("unexpected development sources")
    source_catalog = json.loads(
        (ROOT / "ml/experiments/human_grounded_router/sources_v1.json").read_text(
            encoding="utf-8"
        )
    )["sources"]
    catalog = {source["source_id"]: source for source in source_catalog}
    for source_id, source in sources.items():
        media = ROOT / "data/human-grounded-router/media" / source["file"]
        if (
            not media.is_file()
            or media.stat().st_size != source["bytes"]
            or sha256(media) != source["sha256"]
        ):
            raise ValueError(f"development media missing or changed: {source_id}")
        if source["sha256"] != catalog[source_id]["sha256"]:
            raise ValueError(f"development source provenance changed: {source_id}")
    v2 = json.loads(
        (ROOT / "ml/evaluation/final_english_acceptance_v2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    protected = {query["text"].casefold() for query in v2["queries"]}
    protected_media_sha256 = v2["video"]["sha256"]
    if any(source["sha256"] == protected_media_sha256 for source in sources.values()):
        raise ValueError("development source duplicates protected V2 media")
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    for query in manifest["queries"]:
        if query["query_id"] in ids:
            raise ValueError("development query IDs must be unique")
        ids.add(query["query_id"])
        if query["source_id"] not in sources:
            raise ValueError("development query references an unknown source")
        if query["text"].casefold() in protected:
            raise ValueError("development query duplicates protected V2 text")
        if query["negative"] == bool(query["intervals"]):
            raise ValueError("positive queries need intervals and negatives need none")
        counts[query["evidence_category"]] += 1
    return {
        "manifest": manifest,
        "manifest_sha256": sha256(path),
        "query_distribution": dict(counts),
    }


def _stop_tree(process: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            child.kill()
        parent.kill()
        psutil.wait_procs([parent, *children], timeout=20)
    except psutil.NoSuchProcess:
        pass


def _overlaps(candidate: dict[str, Any], intervals: list[list[float]]) -> bool:
    start = candidate["timestamp"]
    end = candidate.get("end")
    end = start if end is None else end
    return any(start <= interval_end and end >= interval_start for interval_start, interval_end in intervals)


def _bucket_relevant(bucket: dict[str, Any], query: dict[str, Any]) -> bool:
    if query["negative"]:
        return False
    contributions = bucket["contributions"]
    speech = contributions.get("speech")
    visual = contributions.get("visual")
    speech_hit = speech is not None and _overlaps(speech, query["intervals"])
    visual_hit = visual is not None and _overlaps(visual, query["intervals"])
    category = query["evidence_category"]
    if category == "SPEECH":
        return speech_hit
    if category == "VISUAL":
        return visual_hit
    return speech_hit and visual_hit


def _first_rank(results: list[dict[str, Any]], intervals: list[list[float]]) -> int | None:
    return next((rank for rank, item in enumerate(results, 1) if _overlaps(item, intervals)), None)


def _run_source(
    source: dict[str, Any],
    queries: list[dict[str, Any]],
    product_name: str,
    pipeline_name: str,
    port: int,
) -> dict[str, Any]:
    product = DATA_ROOT / product_name
    pipeline = json.loads((DATA_ROOT / pipeline_name).read_text(encoding="utf-8"))
    video_id = pipeline["video"]["id"]
    python = ROOT / ".venv/Scripts/python.exe"
    environment = {
        **os.environ,
        "SCENEMIND_DATA_DIR": str((product / "videos").resolve()),
        "SCENEMIND_DATABASE_URL": f"sqlite:///{(product / 'scenemind.db').resolve().as_posix()}",
        "SCENEMIND_MODEL_CACHE": str((ROOT / "data/models").resolve()),
        "SCENEMIND_DURABLE_JOBS": "true",
    }
    log = (DATA_ROOT / f"{source['source_id']}-diagnostic-server.log").open(
        "w", encoding="utf-8"
    )
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [str(python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT / "backend",
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    base = f"http://127.0.0.1:{port}"
    rows = []
    try:
        with httpx.Client(timeout=180) as client:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if client.get(f"{base}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("diagnostic server did not start")
            for query in queries:
                modes = {}
                latencies = {}
                for mode, depth in (("speech", 50), ("visual", 50), ("hybrid", 5)):
                    started = time.perf_counter()
                    response = client.get(
                        f"{base}/videos/{video_id}/search",
                        params={"q": query["text"], "k": depth, "mode": mode},
                    )
                    latencies[mode] = (time.perf_counter() - started) * 1000
                    response.raise_for_status()
                    modes[mode] = response.json()
                    if modes[mode]["requested_mode"] != mode or modes[mode]["selected_route"] != mode:
                        raise AssertionError("explicit diagnostic mode was not preserved")
                trace = trace_fusion(modes["visual"]["results"], modes["speech"]["results"], 5)
                if trace["results"] != modes["hybrid"]["results"]:
                    raise AssertionError("trace changed or failed to reproduce production Hybrid")
                for bucket in trace["candidate_buckets"]:
                    bucket["matches_development_evidence"] = _bucket_relevant(bucket, query)
                speech_rank = None if query["negative"] else _first_rank(
                    modes["speech"]["results"], query["intervals"]
                )
                visual_rank = None if query["negative"] else _first_rank(
                    modes["visual"]["results"], query["intervals"]
                )
                hybrid_rank = next(
                    (bucket["post_fusion_rank"] for bucket in trace["candidate_buckets"] if bucket["matches_development_evidence"]),
                    None,
                )
                rows.append({
                    "query_id": query["query_id"],
                    "query": query["text"],
                    "evidence_category": query["evidence_category"],
                    "query_style": query["query_style"],
                    "negative": query["negative"],
                    "intervals": query["intervals"],
                    "requested_modes": {mode: modes[mode]["requested_mode"] for mode in modes},
                    "effective_modes": {mode: modes[mode]["selected_route"] for mode in modes},
                    "latency_ms": latencies,
                    "best_evidence_rank": {
                        "speech": speech_rank,
                        "visual": visual_rank,
                        "hybrid": hybrid_rank,
                    },
                    "top_5_evidence": {
                        "speech": speech_rank is not None and speech_rank <= 5,
                        "visual": visual_rank is not None and visual_rank <= 5,
                        "hybrid": hybrid_rank is not None and hybrid_rank <= 5,
                    },
                    "trace": trace,
                })
    finally:
        _stop_tree(process)
        log.close()
    return {"source": source, "pipeline": pipeline, "queries": rows}


def _aggregate(sources: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for source in sources for row in source["queries"]]
    positives = [row for row in rows if not row["negative"]]
    per_category = {}
    for category in ("SPEECH", "VISUAL", "MULTIMODAL"):
        group = [row for row in positives if row["evidence_category"] == category]
        per_category[category] = {
            "queries": len(group),
            "top_5_evidence_rate": {
                mode: sum(row["top_5_evidence"][mode] for row in group) / len(group)
                for mode in ("speech", "visual", "hybrid")
            },
        }
    demotions = {"speech": [], "visual": []}
    for row in positives:
        for modality in ("speech", "visual"):
            rank = row["best_evidence_rank"][modality]
            if rank is not None and rank <= 5:
                candidate = next(
                    (
                        bucket
                        for bucket in row["trace"]["candidate_buckets"]
                        if modality in bucket["contributions"]
                        and bucket["contributions"][modality]["retriever_rank"] == rank
                    ),
                    None,
                )
                if candidate and candidate["post_fusion_rank"] > 5:
                    demotions[modality].append({
                        "query_id": row["query_id"],
                        "retriever_rank": rank,
                        "post_fusion_rank": candidate["post_fusion_rank"],
                        "candidate_id": candidate["candidate_id"],
                    })
    top_slots = Counter()
    overlap_all = 0
    duplicate_speech = 0
    for row in rows:
        for bucket in row["trace"]["candidate_buckets"]:
            if bucket["candidate_overlap"]:
                overlap_all += 1
            duplicate_speech += len(bucket["deduplication"]["discarded_duplicate_speech_ranks"])
            if bucket["final_top_k"]:
                key = "shared" if bucket["candidate_overlap"] else bucket["originating_modalities"][0]
                top_slots[key] += 1
    recall_vs_ranking = Counter()
    for row in positives:
        category = row["evidence_category"].lower()
        expected = ("speech",) if category == "speech" else (("visual",) if category == "visual" else ("speech", "visual"))
        recall = all(row["best_evidence_rank"][mode] is not None for mode in expected)
        if row["top_5_evidence"]["hybrid"]:
            recall_vs_ranking["hybrid_top5_success"] += 1
        elif recall:
            recall_vs_ranking["ranking_or_exact_overlap_failure"] += 1
        else:
            recall_vs_ranking["candidate_recall_failure"] += 1
    latencies = {
        mode: [row["latency_ms"][mode] for row in rows]
        for mode in ("speech", "visual", "hybrid")
    }
    return {
        "queries": len(rows),
        "positive_queries": len(positives),
        "negative_queries": len(rows) - len(positives),
        "per_category": per_category,
        "strong_top5_candidate_demotions": demotions,
        "top5_slot_origin_counts": dict(top_slots),
        "overlapping_candidate_buckets": overlap_all,
        "discarded_duplicate_speech_segments": duplicate_speech,
        "recall_vs_ranking": dict(recall_vs_ranking),
        "latency_ms": {
            mode: {
                "median": statistics.median(values),
                "p95": sorted(values)[max(0, int(len(values) * 0.95 + 0.999999) - 1)],
                "maximum": max(values),
            }
            for mode, values in latencies.items()
        },
    }


def run(manifest_path: Path, output: Path) -> dict[str, Any]:
    validated = validate_manifest(manifest_path)
    manifest = validated["manifest"]
    by_source = {source["source_id"]: source for source in manifest["sources"]}
    source_results = []
    for source_id, (product, pipeline, port) in SOURCE_RUNS.items():
        queries = [query for query in manifest["queries"] if query["source_id"] == source_id]
        source_results.append(_run_source(by_source[source_id], queries, product, pipeline, port))
    report = {
        "schema_version": "1.0.0",
        "suite_id": manifest["suite_id"],
        "manifest_sha256": validated["manifest_sha256"],
        "query_distribution": validated["query_distribution"],
        "production_retrieval_modified": False,
        "acceptance_v2_used": False,
        "diagnostic_ranker": "app.hybrid.fuse",
        "sources": source_results,
        "summary": _aggregate(source_results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.manifest, args.output)
    print(json.dumps(result["summary"], indent=2))
