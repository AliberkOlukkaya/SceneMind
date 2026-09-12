"""Run frozen CLIP-top-five, UForm ONNX and tiny learned-scorer evaluation."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import psutil
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from ml.evaluation.schema_v2 import load_benchmark
from ml.experiments.image_text_verifier.schema import load_calibration
from ml.experiments.lightweight_pair_scorer.external import UFormOnnxScorer, UFormSpec
from ml.experiments.lightweight_pair_scorer.learned import fit, inference_latency, score
from ml.experiments.lightweight_pair_scorer.ranking import fit_threshold, rerank, slices, summarize
from ml.experiments.lightweight_pair_scorer.schema import (
    LearnedScorerArtifact,
    assert_all_disjoint,
    load_development,
)

ROOT = Path(__file__).resolve().parents[3]
NATURAL_MANIFEST = ROOT / "ml/evaluation/natural_v2.json"
PREVIOUS_MANIFEST = ROOT / "ml/experiments/image_text_verifier/calibration_v1.json"
DEVELOPMENT_MANIFEST = Path(__file__).with_name("calibration_v2.json")
BASELINE_REPORT = ROOT / "ml/evaluation/reports/natural-v2.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def current_rss() -> int:
    return psutil.Process().memory_info().rss


def percentile(values: list[float], value: int) -> float:
    return float(np.percentile(values, value))


def run(output: Path, artifact_path: Path, threads: int = 4) -> dict:
    original = settings.model_dump()
    try:
        return _run(output, artifact_path, threads)
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


def _run(output: Path, artifact_path: Path, threads: int) -> dict:
    natural = load_benchmark(NATURAL_MANIFEST)
    previous = load_calibration(PREVIOUS_MANIFEST)
    development = load_development(DEVELOPMENT_MANIFEST)
    if digest(NATURAL_MANIFEST) != development.parent_manifest_sha256:
        raise ValueError("natural benchmark parent hash changed")
    if digest(PREVIOUS_MANIFEST) != development.parent_calibration_manifest_sha256:
        raise ValueError("previous calibration parent hash changed")
    assert_all_disjoint(development, previous, natural.videos)
    videos = [video for video in natural.videos if video.split == "calibration"]
    videos.extend(previous.videos)
    videos.extend(development.videos)
    videos.extend(video for video in natural.videos if video.split == "heldout")
    for video in videos:
        if not video.path.is_file() or digest(video.path) != video.source.source_sha256:
            raise ValueError(f"missing or invalid media: {video.video_id}")

    settings.durable_jobs = False
    settings.auth_token = ""
    settings.calibration_path = None
    run_root = ROOT / "data/benchmarks" / str(uuid4())
    settings.data_dir = run_root / "videos"
    settings.database_url = f"sqlite:///{run_root / 'transcripts.db'}"
    report = {
        "experiment_schema_version": "1.0.0",
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "natural_manifest_sha256": digest(NATURAL_MANIFEST),
        "previous_calibration_manifest_sha256": digest(PREVIOUS_MANIFEST),
        "development_manifest_sha256": digest(DEVELOPMENT_MANIFEST),
        "candidate_model": settings.visual_model,
        "candidate_revision": settings.visual_revision,
        "candidate_depth": 5,
        "sampling_interval": settings.sampling_interval,
        "environment": {
            "platform": platform.platform(),
            "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("torch", "transformers", "faiss-cpu", "onnxruntime", "uform")
            },
            "onnx_intra_op_threads": threads,
            "onnx_inter_op_threads": 1,
        },
        "rows": [],
        "videos": [],
    }
    scorer = None
    cold_seconds = None
    scorer_memory_before = None
    scorer_memory_after = None
    spec = UFormSpec()
    with TestClient(app) as client:
        for video in videos:
            api_id, timing = ingest(client, video)
            report["videos"].append(timing)
            split = "heldout" if getattr(video, "split", "calibration") == "heldout" else "calibration"
            for query in video.queries:
                if "visual" not in query.evaluation_modes:
                    continue
                response = client.get(
                    f"/videos/{api_id}/search",
                    params={"q": query.query_text, "mode": "visual", "k": 5},
                )
                response.raise_for_status()
                candidates = response.json()["results"]
                paths = [
                    settings.data_dir / api_id / "frames" / item["thumbnail"].rsplit("/", 1)[1]
                    for item in candidates
                ]
                if scorer is None:
                    scorer_memory_before = current_rss()
                    scorer = UFormOnnxScorer(spec, settings.model_cache, threads)
                    scorer_memory_after = current_rss()
                    _, cold_seconds = scorer.timed_score(query.query_text, paths)
                    scorer.warm(query.query_text, paths)
                pair_scores, elapsed = scorer.timed_score(query.query_text, paths)
                report["rows"].append(
                    {
                        "query_id": query.query_id,
                        "video_id": video.video_id,
                        "split": split,
                        "query": query.query_text,
                        "query_type": query.query_type.value,
                        "expected_presence": query.expected_presence,
                        "relevant_intervals": query.relevant_intervals,
                        "clip_candidates": candidates,
                        "uform_reranked": rerank(candidates, pair_scores, "uform_score"),
                        "uform_seconds": elapsed,
                    }
                )

    calibration_rows = [row for row in report["rows"] if row["split"] == "calibration"]
    heldout_rows = [row for row in report["rows"] if row["split"] == "heldout"]
    uform_threshold = fit_threshold(calibration_rows, "uform_reranked", "uform_score")
    model = fit(calibration_rows)
    for row in report["rows"]:
        row["learned_reranked"] = rerank(
            row["clip_candidates"], score(row["clip_candidates"], model), "learned_score"
        )
    learned_threshold = fit_threshold(calibration_rows, "learned_reranked", "learned_score")
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    clip_threshold = baseline["analysis"]["threshold"]
    systems = {
        "clip": ("clip_candidates", None, None),
        "clip_calibrated": ("clip_candidates", clip_threshold, "score"),
        "uform_reranked": ("uform_reranked", None, None),
        "uform_calibrated": ("uform_reranked", uform_threshold, "uform_score"),
        "learned_reranked": ("learned_reranked", None, None),
        "learned_calibrated": ("learned_reranked", learned_threshold, "learned_score"),
    }
    report["calibration"] = {
        "rows": len(calibration_rows),
        "positive_queries": sum(row["expected_presence"] for row in calibration_rows),
        "negative_queries": sum(not row["expected_presence"] for row in calibration_rows),
        "threshold_rule": "lowest score allowing at most 10% calibration negative-query accepts",
        "uform_threshold": uform_threshold,
        "learned_threshold": learned_threshold,
        "learned_model": model,
        "metrics": {
            name: summarize(calibration_rows, field, threshold, score_key)
            for name, (field, threshold, score_key) in systems.items()
        },
    }
    report["heldout"] = {
        "metrics": {
            name: summarize(heldout_rows, field, threshold, score_key)
            for name, (field, threshold, score_key) in systems.items()
        },
        "slices": slices(heldout_rows, systems),
    }
    timings = [row["uform_seconds"] for row in report["rows"]]
    learned_timing = inference_latency(report["rows"][0]["clip_candidates"], model)
    report["uform"] = {
        **spec.as_dict(),
        "runtime": "ONNX Runtime CPU",
        "providers": scorer.providers,
        "model_files": scorer.model_files,
        "load_seconds": scorer.load_seconds,
        "cold_top5_seconds": cold_seconds,
        "warm_top5_median_seconds": float(np.median(timings)),
        "warm_top5_p95_seconds": percentile(timings, 95),
        "warm_per_pair_median_seconds": float(np.median(timings)) / 5,
        "measured_queries": len(timings),
        "working_set_before_load_bytes": scorer_memory_before,
        "working_set_after_load_bytes": scorer_memory_after,
        "added_working_set_bytes": scorer_memory_after - scorer_memory_before,
    }
    report["learned_scorer"] = {**model, **learned_timing, "runtime": "NumPy CPU float64"}
    report["gate"] = {
        name: gate(report, name) for name in ("uform_calibrated", "learned_calibrated")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifact = LearnedScorerArtifact(
        schema_version="1.0.0",
        scorer_type="logistic-regression",
        **{key: model[key] for key in (
            "feature_schema", "feature_mean", "feature_scale", "weights", "intercept",
            "regularization", "training_seconds", "trainable_parameters"
        )},
        threshold=learned_threshold,
        training_dataset=development.dataset_id,
        training_manifest_sha256=report["development_manifest_sha256"],
        parent_calibration_manifest_sha256=report["previous_calibration_manifest_sha256"],
        parent_benchmark_manifest_sha256=report["natural_manifest_sha256"],
        candidate_model_id=settings.visual_model,
        candidate_model_revision=settings.visual_revision,
        runtime="NumPy CPU float64",
        random_seed=0,
        created_at=report["utc"],
        code_version=report["git_commit"],
    )
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return report


def gate(report: dict, name: str) -> dict:
    result = report["heldout"]["metrics"][name]["5"]
    clip = report["heldout"]["metrics"]["clip"]["5"]
    reranked_name = name.replace("calibrated", "reranked")
    reranked = report["heldout"]["metrics"][reranked_name]["5"]
    if name.startswith("uform"):
        latency = report["uform"]["warm_top5_median_seconds"] if "uform" in report else 0
        memory = report["uform"]["added_working_set_bytes"] if "uform" in report else 0
    else:
        latency = report["learned_scorer"]["median_seconds"] if "learned_scorer" in report else 0
        memory = 0
    values = {
        "negative_far_at_5": result["negative_false_accept_rate"],
        "positive_false_abstention": result["positive_false_abstention_rate"],
        "candidate_recall_at_5_loss": clip["recall"] - reranked["recall"],
        "added_warm_median_seconds": latency,
        "added_working_set_bytes": memory,
    }
    values["passed"] = (
        values["negative_far_at_5"] <= 0.1
        and values["positive_false_abstention"] <= 0.2
        and values["candidate_recall_at_5_loss"] <= 0.02
        and values["added_warm_median_seconds"] <= 0.25
        and values["added_working_set_bytes"] <= 750_000_000
    )
    return values


def ingest(client: TestClient, video) -> tuple[str, dict]:
    started = time.perf_counter()
    response = client.post(
        "/videos", params={"filename": video.path.name}, content=video.path.read_bytes()
    )
    response.raise_for_status()
    api_id = response.json()["id"]
    record = client.get(f"/videos/{api_id}").json()
    if record["status"] != "ready":
        raise RuntimeError(record)
    index_started = time.perf_counter()
    client.post(f"/videos/{api_id}/index").raise_for_status()
    status = client.get(f"/videos/{api_id}/index").json()
    if status["status"] != "ready":
        raise RuntimeError(status)
    return api_id, {
        "video_id": video.video_id,
        "sha256": digest(video.path),
        "duration_seconds": record["metadata"]["duration"],
        "frames": len(record["frames"]),
        "ingestion_seconds": index_started - started,
        "index_seconds": time.perf_counter() - index_started,
    }


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/lightweight-pair-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "data/lightweight-learned-scorer.json")
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.artifact, arguments.threads)
    print(json.dumps({"gate": result["gate"], "heldout": result["heldout"]}, indent=2))
