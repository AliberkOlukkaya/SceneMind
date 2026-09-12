"""Run isolated CLIP-top-five plus BLIP-ITM evaluation on frozen splits."""

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from ml.evaluation.schema_v2 import load_benchmark
from ml.experiments.image_text_verifier.ranking import analyze, fit_threshold, rerank, summarize
from ml.experiments.image_text_verifier.schema import (
    CalibrationArtifact,
    Preprocessing,
    assert_disjoint,
    load_calibration,
)
from ml.experiments.image_text_verifier.verifier import BlipVerifier, VerifierSpec, median_latency

ROOT = Path(__file__).resolve().parents[3]
NATURAL_MANIFEST = ROOT / "ml/evaluation/natural_v2.json"
CALIBRATION_MANIFEST = Path(__file__).with_name("calibration_v1.json")
BASELINE_REPORT = ROOT / "ml/evaluation/reports/natural-v2.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def run(output: Path, artifact_path: Path) -> dict:
    original = settings.model_dump()
    try:
        return _run(output, artifact_path)
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


def _run(output: Path, artifact_path: Path) -> dict:
    natural = load_benchmark(NATURAL_MANIFEST)
    expansion = load_calibration(CALIBRATION_MANIFEST)
    if digest(NATURAL_MANIFEST) != expansion.parent_manifest_sha256:
        raise ValueError("calibration parent manifest hash changed")
    assert_disjoint(expansion, natural.videos)
    videos = [video for video in natural.videos if video.split == "calibration"]
    videos.extend(expansion.videos)
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
    baseline_memory = peak_rss_bytes()
    spec = VerifierSpec()
    verifier = BlipVerifier(spec, settings.model_cache, settings.model_device)
    after_load_memory = peak_rss_bytes()
    report = {
        "experiment_schema_version": "1.0.0",
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "natural_manifest_sha256": digest(NATURAL_MANIFEST),
        "calibration_manifest_sha256": digest(CALIBRATION_MANIFEST),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "faiss-cpu")},
        "candidate_model": settings.visual_model,
        "candidate_revision": settings.visual_revision,
        "candidate_depth": 5,
        "sampling_interval": settings.sampling_interval,
        "verifier": {
            **spec.__dict__,
            "device": settings.model_device,
            "parameter_count": verifier.parameter_count,
            "load_seconds": verifier.load_seconds,
        },
        "memory": {
            "peak_rss_before_model_bytes": baseline_memory,
            "peak_rss_after_model_load_bytes": after_load_memory,
        },
        "rows": [],
        "videos": [],
    }
    warmed = False
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
                if not warmed:
                    verifier.warm(paths)
                    warmed = True
                scores, elapsed = verifier.score(query.query_text, paths)
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
                        "reranked": rerank(candidates, scores),
                        "verifier_seconds": elapsed,
                    }
                )
    calibration_rows = [row for row in report["rows"] if row["split"] == "calibration"]
    threshold = fit_threshold(calibration_rows)
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    clip_threshold = baseline["analysis"]["threshold"]
    report["calibration"] = {
        "threshold": threshold,
        "selection_rule": "lowest threshold allowing at most 10% calibration negative query accepts",
        "rows": len(calibration_rows),
        "positive_queries": sum(row["expected_presence"] for row in calibration_rows),
        "negative_queries": sum(not row["expected_presence"] for row in calibration_rows),
        "metrics": {
            "clip": summarize(calibration_rows, "clip_candidates"),
            "verifier_reranked": summarize(calibration_rows, "reranked"),
            "verifier_calibrated": summarize(calibration_rows, "reranked", threshold),
        },
    }
    report["heldout"] = analyze(report["rows"], threshold, clip_threshold)
    heldout_clip = report["heldout"]["clip"]["5"]
    heldout_reranked = report["heldout"]["verifier_reranked"]["5"]
    heldout_calibrated = report["heldout"]["verifier_calibrated"]["5"]
    latency = median_latency(report["rows"])
    report["memory"]["peak_rss_end_bytes"] = peak_rss_bytes()
    report["resource_metrics"] = {"added_warm_median_seconds": latency}
    report["gate"] = {
        "negative_far_at_5": heldout_calibrated["negative_false_accept_rate"],
        "negative_far_limit": 0.1,
        "positive_false_abstention": heldout_calibrated["positive_false_abstention_rate"],
        "positive_false_abstention_limit": 0.2,
        "candidate_recall_at_5_loss": heldout_clip["recall"] - heldout_reranked["recall"],
        "candidate_recall_loss_limit": 0.02,
        "added_warm_median_seconds": latency,
        "added_warm_median_limit_seconds": 0.25,
    }
    report["gate"]["passed"] = (
        report["gate"]["negative_far_at_5"] <= report["gate"]["negative_far_limit"]
        and report["gate"]["positive_false_abstention"]
        <= report["gate"]["positive_false_abstention_limit"]
        and report["gate"]["candidate_recall_at_5_loss"]
        <= report["gate"]["candidate_recall_loss_limit"]
        and latency <= report["gate"]["added_warm_median_limit_seconds"]
    )
    artifact = CalibrationArtifact(
        schema_version="1.0.0",
        model_id=spec.model_id,
        model_revision=spec.revision,
        model_license=spec.license,
        preprocessing=Preprocessing(
            image_size=spec.image_size,
            max_text_tokens=spec.max_text_tokens,
            processor="BlipProcessor pinned with model revision",
        ),
        match_label_index=spec.match_label_index,
        threshold=threshold,
        threshold_rule=report["calibration"]["selection_rule"],
        calibration_dataset=expansion.dataset_id,
        calibration_manifest_sha256=report["calibration_manifest_sha256"],
        parent_benchmark=natural.benchmark_id,
        benchmark_schema_version=natural.schema_version,
        created_at=report["utc"],
        promotable=report["gate"]["passed"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return report


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


def peak_rss_bytes() -> int | None:
    if platform.system() != "Windows":
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except (ImportError, AttributeError):
            return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32
    process = ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    process.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    process.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess()
    if not process.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/verifier-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "data/verifier-calibration.json")
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.artifact)
    print(json.dumps({"gate": result["gate"], "heldout": result["heldout"]}, indent=2))
