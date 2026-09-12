"""Run the frozen Natural Video Benchmark V2 against the current local API."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from ml.evaluation.metrics import retrieval_metrics
from ml.evaluation.natural_v2_report import accepted_results, aggregate, failures
from ml.evaluation.schema_v2 import load_benchmark

ROOT = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def run(manifest: Path, repeats: int = 3) -> dict:
    original = settings.model_dump()
    try:
        return _run(manifest.resolve(), repeats)
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


def _run(manifest: Path, repeats: int) -> dict:
    benchmark = load_benchmark(manifest)
    for video in benchmark.videos:
        if not video.path.is_file():
            raise FileNotFoundError(video.path)
        if _digest(video.path) != video.source.source_sha256:
            raise ValueError(f"prepared media checksum mismatch: {video.video_id}")

    settings.durable_jobs = False
    settings.auth_token = ""
    settings.calibration_path = None
    root = ROOT / "data" / "benchmarks" / str(uuid4())
    settings.data_dir = root / "videos"
    settings.database_url = f"sqlite:///{root / 'transcripts.db'}"
    report = {
        "schema_version": benchmark.schema_version,
        "benchmark_id": benchmark.benchmark_id,
        "annotation_status": benchmark.annotation_status,
        "annotation_frozen_at": benchmark.annotation_frozen_at,
        "manifest_sha256": _digest(manifest),
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["torch", "transformers", "faiss-cpu", "faster-whisper"]
        },
        "visual_model": settings.visual_model,
        "visual_revision": settings.visual_revision,
        "speech_model": settings.speech_model,
        "device": settings.model_device,
        "sampling_interval": settings.sampling_interval,
        "retrieval_depth": 50,
        "repeats": repeats,
        "videos": [],
        "queries": [],
    }
    with TestClient(app) as client:
        for video in benchmark.videos:
            _evaluate_video(client, video, repeats, report)
    report["analysis"] = aggregate(report)
    report["failures"] = failures(report, report["analysis"])
    return report


def _git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _evaluate_video(client: TestClient, video, repeats: int, report: dict) -> None:
    started = time.perf_counter()
    response = client.post(
        "/videos", params={"filename": video.path.name}, content=video.path.read_bytes()
    )
    response.raise_for_status()
    ingestion = time.perf_counter() - started
    api_video_id = response.json()["id"]
    record = client.get(f"/videos/{api_video_id}").json()
    if record["status"] != "ready":
        raise RuntimeError(record)
    duration = record["metadata"]["duration"]
    expected_duration = video.source.segment_end - video.source.segment_start
    if abs(duration - expected_duration) > max(1.0, 0.02 * expected_duration):
        raise ValueError(f"unexpected prepared duration for {video.video_id}: {duration}")

    timing = {
        "video_id": video.video_id,
        "file": str(video.path),
        "sha256": _digest(video.path),
        "split": video.split,
        "source_group": video.source_group,
        "duration_seconds": duration,
        "frames": len(record["frames"]),
        "ingestion_seconds": ingestion,
    }
    modes = {mode for query in video.queries for mode in query.evaluation_modes}
    for stage, needed in (("index", "visual" in modes or "hybrid" in modes),
                          ("transcript", "speech" in modes or "hybrid" in modes)):
        if needed:
            started = time.perf_counter()
            client.post(f"/videos/{api_video_id}/{stage}").raise_for_status()
            timing[f"{stage}_seconds_including_model_setup"] = time.perf_counter() - started
            status = client.get(f"/videos/{api_video_id}/{stage}").json()
            if status["status"] != "ready":
                raise RuntimeError(status)
    folder = settings.data_dir / api_video_id
    timing["storage_bytes"] = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    report["videos"].append(timing)

    for query in video.queries:
        for mode in query.evaluation_modes:
            durations = []
            payload = None
            for _ in range(repeats + 1):
                started = time.perf_counter()
                response = client.get(
                    f"/videos/{api_video_id}/search",
                    params={"q": query.query_text, "mode": mode, "k": 50},
                )
                response.raise_for_status()
                durations.append(time.perf_counter() - started)
                payload = response.json()
            results = payload["results"]
            metrics = {
                str(k): retrieval_metrics(
                    [result["timestamp"] for result in results[:k]], query.relevant_intervals, k
                )
                for k in (1, 3, 5)
            }
            report["queries"].append(
                {
                    "query_id": query.query_id,
                    "video_id": video.video_id,
                    "domain": video.domain,
                    "split": video.split,
                    "source_group": video.source_group,
                    "query": query.query_text,
                    "query_type": query.query_type.value,
                    "expected_presence": query.expected_presence,
                    "relevant_intervals": query.relevant_intervals,
                    "modality_requirement": query.modality_requirement,
                    "mode": mode,
                    "results": results,
                    "raw_metrics": metrics,
                    "first_search_seconds": durations[0],
                    "warm_median_seconds": statistics.median(durations[1:]),
                }
            )


def markdown(report: dict) -> str:
    analysis = report["analysis"]
    lines = [
        "# Natural Video Benchmark V2 results",
        "",
        f"Run `{report['utc']}` at commit `{report['git_commit']}` with frozen manifest ",
        f"`{report['manifest_sha256']}`. The visual cutoff `{analysis['threshold']:.6f}` was ",
        "fit only on calibration visual negatives. Held-out labels were not used for tuning.",
        "",
        "## Held-out metrics",
        "",
        "| Path | State | R@1 | R@3 | R@5 | MRR@5 | P@5 | Negative FAR@5 | False abstention@5 | Warm median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for path, values in analysis["by_path"].items():
        for state in ("raw", "calibrated"):
            at1, at3, at5 = values["1"][state], values["3"][state], values["5"][state]
            lines.append(
                f"| {path} | {state} | {_pct(at1['recall'])} | {_pct(at3['recall'])} | "
                f"{_pct(at5['recall'])} | {_pct(at5['mrr'])} | {_pct(at5['precision'])} | "
                f"{_pct(at5['negative_false_accept_rate'])} | "
                f"{_pct(at5['positive_false_abstention_rate'])} | "
                f"{at5['latency_median_seconds'] * 1000:.1f} ms |"
            )
    lines.extend(["", "## Held-out category breakdown", "",
                  "| Category | Positives | Negatives | Raw R@5 | Calibrated R@5 | Calibrated FAR@5 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for category, values in analysis["by_category"].items():
        raw, calibrated = values["5"]["raw"], values["5"]["calibrated"]
        lines.append(
            f"| {category} | {raw['positive_queries']} | {raw['negative_queries']} | "
            f"{_pct(raw['recall'])} | {_pct(calibrated['recall'])} | "
            f"{_pct(calibrated['negative_false_accept_rate'])} |"
        )
    lines.extend([
        "", "## Failure analysis", "",
        f"The calibrated held-out evaluation produced {len(report['failures'])} query-path failures at K=5.",
        "Each entry in the JSON report records the query, likely component, top timestamps, scores, transcript text, and hybrid evidence for review.",
        "", "| Likely component | Count |", "| --- | ---: |",
    ])
    counts = defaultdict(int)
    for failure in report["failures"]:
        counts[failure["likely_component"]] += 1
    lines.extend(f"| {cause} | {count} |" for cause, count in sorted(counts.items()))
    lines.extend([
        "", "## Scope", "",
        "This benchmark is a small diagnostic set of five openly licensed videos, with two calibration and three source-disjoint held-out videos. Repeated evaluation modes are separate query-path trials. The measurements characterize this pinned local configuration; they are not population accuracy estimates.",
        "",
    ])
    return "\n".join(lines)


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def write_csv(report: dict, path: Path) -> None:
    fields = [
        "query_id", "video_id", "split", "query_type", "mode", "expected_presence",
        "raw_recall_at_1", "raw_recall_at_3", "raw_recall_at_5", "raw_mrr_at_5",
        "calibrated_recall_at_5", "calibrated_mrr_at_5", "raw_returned_at_5",
        "calibrated_returned_at_5", "warm_median_seconds",
    ]
    threshold = report["analysis"]["threshold"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in report["queries"]:
            calibrated = {
                str(k): retrieval_metrics(
                    [
                        result["timestamp"]
                        for result in accepted_results(row, threshold)[:k]
                    ],
                    row["relevant_intervals"],
                    k,
                )
                for k in (1, 3, 5)
            }
            writer.writerow(
                {
                    **{field: row[field] for field in fields[:6]},
                    "raw_recall_at_1": row["raw_metrics"]["1"]["recall"],
                    "raw_recall_at_3": row["raw_metrics"]["3"]["recall"],
                    "raw_recall_at_5": row["raw_metrics"]["5"]["recall"],
                    "raw_mrr_at_5": row["raw_metrics"]["5"]["reciprocal_rank"],
                    "calibrated_recall_at_5": calibrated["5"]["recall"],
                    "calibrated_mrr_at_5": calibrated["5"]["reciprocal_rank"],
                    "raw_returned_at_5": row["raw_metrics"]["5"]["returned"],
                    "calibrated_returned_at_5": calibrated["5"]["returned"],
                    "warm_median_seconds": row["warm_median_seconds"],
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "ml/evaluation/natural_v2.json")
    parser.add_argument("--repeats", type=int, choices=range(1, 101), default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data/natural-v2-report.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "data/NATURAL_V2_RESULTS.md")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/natural-v2-report.csv")
    arguments = parser.parse_args()
    result = run(arguments.manifest, arguments.repeats)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.csv.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    arguments.markdown.write_text(markdown(result), encoding="utf-8")
    write_csv(result, arguments.csv)
    print(json.dumps(result["analysis"]["by_path"], indent=2))
