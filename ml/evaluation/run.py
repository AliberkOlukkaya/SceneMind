"""Reproducible local pipeline benchmark. Optional models are real, never mocked."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.config import settings
from app.main import app
from ml.evaluation.metrics import retrieval_metrics


class EvaluationQuery(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    mode: Literal["visual", "speech", "hybrid"] = "visual"
    intervals: list[tuple[float, float]]


class EvaluationVideo(BaseModel):
    path: Path
    queries: list[EvaluationQuery] = Field(min_length=1)


class EvaluationSet(BaseModel):
    name: str
    license: str
    videos: list[EvaluationVideo] = Field(min_length=1)


def run(manifest, k, repeats):
    dataset = EvaluationSet.model_validate_json(manifest.read_text(encoding="utf-8"))
    for video in dataset.videos:
        if not video.path.is_file():
            raise FileNotFoundError(video.path)
        for query in video.queries:
            retrieval_metrics([], query.intervals, k)
    root = Path("data/benchmarks") / str(uuid4())
    settings.data_dir = root / "videos"
    settings.database_url = f"sqlite:///{root / 'transcripts.db'}"
    report = {
        "dataset": dataset.name,
        "license": dataset.license,
        "utc": datetime.now(timezone.utc).isoformat(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["torch", "transformers", "faiss-cpu"]},
        "visual_model": settings.visual_model,
        "revision": settings.visual_revision,
        "device": settings.model_device,
        "sampling_interval": settings.sampling_interval,
        "k": k,
        "repeats": repeats,
        "videos": [],
        "queries": [],
    }
    with TestClient(app) as client:
        for video in dataset.videos:
            started = time.perf_counter()
            response = client.post(
                "/videos",
                params={"filename": video.path.name},
                content=video.path.read_bytes(),
            )
            response.raise_for_status()
            # TestClient waits for BackgroundTasks; this includes upload and processing.
            ingestion_seconds = time.perf_counter() - started
            video_id = response.json()["id"]
            record = client.get(f"/videos/{video_id}").json()
            if record["status"] != "ready":
                raise RuntimeError(record)
            timings = {
                "file": str(video.path),
                "ingestion_seconds": ingestion_seconds,
                "duration": record["metadata"]["duration"],
                "frames": len(record["frames"]),
            }
            for stage, needed in [
                ("index", any(q.mode != "speech" for q in video.queries)),
                ("transcript", any(q.mode != "visual" for q in video.queries)),
            ]:
                if needed:
                    started = time.perf_counter()
                    client.post(f"/videos/{video_id}/{stage}").raise_for_status()
                    timings[f"{stage}_seconds_including_model_setup"] = (
                        time.perf_counter() - started
                    )
                    status = client.get(f"/videos/{video_id}/{stage}").json()
                    if status["status"] != "ready":
                        raise RuntimeError(status)
            folder = settings.data_dir / video_id
            timings["storage_bytes"] = sum(
                path.stat().st_size for path in folder.rglob("*") if path.is_file()
            )
            vector_path = folder / "embeddings.npy"
            timings["vector_bytes"] = vector_path.stat().st_size if vector_path.exists() else 0
            report["videos"].append(timings)
            for query in video.queries:
                durations = []
                for _ in range(repeats + 1):
                    started = time.perf_counter()
                    response = client.get(
                        f"/videos/{video_id}/search",
                        params={"q": query.text, "mode": query.mode, "k": k},
                    )
                    response.raise_for_status()
                    durations.append(time.perf_counter() - started)
                timestamps = [r["timestamp"] for r in response.json()["results"]]
                report["queries"].append(
                    {
                        "query": query.text,
                        "mode": query.mode,
                        "intervals": query.intervals,
                        "timestamps": timestamps,
                        **retrieval_metrics(timestamps, query.intervals, k),
                        "first_search_seconds": durations[0],
                        "warm_median_seconds": statistics.median(durations[1:]),
                    }
                )
    positives = [q for q in report["queries"] if q["intervals"]]
    negatives = [q for q in report["queries"] if not q["intervals"]]
    report["summary"] = {
        "positive_queries": len(positives),
        "negative_queries": len(negatives),
        "recall_at_k": statistics.mean(q["recall"] for q in positives) if positives else None,
        "precision_at_k": statistics.mean(q["precision"] for q in positives) if positives else None,
        "mrr_at_k": statistics.mean(q["reciprocal_rank"] for q in positives) if positives else None,
        "negative_return_rate": statistics.mean(q["returned"] > 0 for q in negatives)
        if negatives
        else None,
        "warm_median_seconds": statistics.median(
            q["warm_median_seconds"] for q in report["queries"]
        ),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("ml/evaluation/synthetic.json"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--k", type=int, choices=range(1, 51), default=1)
    parser.add_argument("--repeats", type=int, choices=range(1, 101), default=5)
    parser.add_argument("--output", type=Path, default=Path("data/benchmark.json"))
    args = parser.parse_args()
    if args.synthetic:
        subprocess.run([sys.executable, "scripts/create_fixture.py"], check=True)
    report = run(args.manifest, args.k, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
