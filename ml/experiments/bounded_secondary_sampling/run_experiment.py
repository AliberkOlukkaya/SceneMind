"""Run the frozen bounded-secondary sampling experiment with real local models."""

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import psutil

from ml.evaluation.metrics import retrieval_metrics
from ml.experiments.bounded_secondary_sampling.cache import SecondaryFrameCache
from ml.experiments.bounded_secondary_sampling.fusion import (
    abstain,
    fit_abstention_threshold,
    fuse,
)
from ml.experiments.bounded_secondary_sampling.policy import (
    CoarseCandidate,
    diversify,
    local_windows,
)
from ml.experiments.object_detector_branch.detector import YoloXNanoDetector
from ml.experiments.object_detector_branch.evaluate import source_paths
from ml.experiments.object_detector_branch.mapping import map_query
from ml.experiments.object_detector_branch.ranking import object_features, summarize

ROOT = Path(__file__).resolve().parents[3]
PARENT_REPORT = ROOT / "ml/evaluation/reports/lightweight-pair-scorer-v1.json"
NATURAL_MANIFEST = ROOT / "ml/evaluation/natural_v2.json"
MODEL_PATH = ROOT / "data/models/yolox_nano-640-0.1.1rc0.onnx"
MODEL_SHA256 = "a78d8834e54f709b15269c31e4ab4970c8faba014e895814d964bacf0ba4197f"
SMALL_OBJECT_IDS = {"street-object-bicycle", "throw-object-ball", "throw-compositional-holding"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def percentile(values: list[float], value: int) -> float:
    return float(np.percentile(values, value)) if values else 0.0


def extract_seek(source: Path, timestamp: float, output: Path) -> None:
    completed = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-ss", str(timestamp), "-i", str(source), "-frames:v", "1",
            "-vf", "scale=480:-2", "-q:v", "3", "-y", str(output),
        ],
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace"))


def prepare_global_two_second(source: Path, duration: float, folder: Path) -> dict:
    record_path = folder / "timestamps.json"
    if record_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "info",
            "-i", str(source), "-an", "-vf",
            "select='isnan(prev_selected_t)+gte(t-prev_selected_t,2)',showinfo,scale=480:-2",
            "-fps_mode", "vfr", "-q:v", "3", str(folder / "%06d.jpg"),
        ],
        check=True, capture_output=True,
    )
    timestamps = re.findall(
        rb"\bn:\s*\d+.*?\bpts_time:([\d.eE+-]+)", completed.stderr
    )
    frames = sorted(folder.glob("*.jpg"))
    if not frames or len(timestamps) < len(frames):
        raise RuntimeError("global 2-second extraction did not emit timestamps")
    record = {
        "duration": duration,
        "extraction_seconds": time.perf_counter() - started,
        "frames": [
            {"timestamp": round(float(timestamps[index]), 3), "file": path.name}
            for index, path in enumerate(frames)
        ],
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def coarse_assets(parent: dict) -> dict:
    assets = {}
    for row in parent["rows"]:
        if row["video_id"] in assets:
            continue
        identifier = row["clip_candidates"][0]["thumbnail"].split("/")[2]
        matches = list((ROOT / "data/benchmarks").glob(f"*/videos/{identifier}"))
        if len(matches) != 1:
            raise ValueError(f"could not resolve frozen coarse assets: {row['video_id']}")
        folder = matches[0]
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assets[row["video_id"]] = {
            "folder": folder, "frames": manifest["frames"],
            "vectors": np.load(folder / "embeddings.npy", allow_pickle=False),
        }
    return assets


def ranked_candidates(frames: list[dict], vectors: np.ndarray, text: np.ndarray) -> list[dict]:
    scores = vectors @ text[0]
    order = np.argsort(-scores)
    return [
        {"timestamp": frames[index]["timestamp"], "score": float(scores[index])}
        for index in order
    ]


def choose_coarse(row: dict, top_k: int, spacing: float) -> list[CoarseCandidate]:
    candidates = [CoarseCandidate(item["timestamp"], item["score"]) for item in row["coarse"]]
    return diversify(candidates, min(top_k, len(candidates)), spacing)


def local_pool(row: dict, frames: list[dict], radius: float, top_k: int, spacing: float):
    coarse = choose_coarse(row, top_k, spacing)
    windows = local_windows(coarse, radius, row["duration"])
    selected = [
        (index, frame) for index, frame in enumerate(frames)
        if any(start <= frame["timestamp"] <= end for start, end in windows)
    ]
    return coarse, selected


def pool_candidates(
    row: dict, frames: list[dict], vectors: np.ndarray,
    radius: float, top_k: int, spacing: float,
) -> tuple[list[dict], list[dict]]:
    coarse, selected = local_pool(row, frames, radius, top_k, spacing)
    if not selected:
        return [], []
    indices = [index for index, _ in selected]
    scores = vectors[indices] @ row["text_vector"][0]
    ranked = sorted(
        (
            {
                "timestamp": frame["timestamp"], "secondary_clip_score": float(score),
                "score": float(score), "modality": "visual",
                "coarse_support_score": max(
                    item.score for item in coarse
                    if abs(item.timestamp - frame["timestamp"]) <= radius
                ),
            }
            for (_, frame), score in zip(selected, scores)
        ),
        key=lambda item: -item["secondary_clip_score"],
    )
    plain = sorted(
        ranked,
        key=lambda item: (
            -item["coarse_support_score"],
            min(abs(item["timestamp"] - coarse_row.timestamp) for coarse_row in coarse),
        ),
    )
    return plain[:5], ranked[:5]


def slices(rows: list[dict]) -> dict[str, list[dict]]:
    result = {name: [] for name in (
        "OBJECT", "SMALL_OBJECT", "RELATIONSHIP", "COMPOSITIONAL", "SCENE",
        "ACTION_TEMPORAL", "NEGATIVE",
    )}
    for row in rows:
        if row["query_type"] in result:
            result[row["query_type"]].append(row)
        if row["query_id"] in SMALL_OBJECT_IDS:
            result["SMALL_OBJECT"].append(row)
        if row["mapping"].relationship_required:
            result["RELATIONSHIP"].append(row)
    return result


def metrics(rows: list[dict], field: str) -> dict:
    return summarize(rows, field)


def run(output: Path, run_root: Path) -> dict:
    from app.encoder import encoder

    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    if digest(NATURAL_MANIFEST) != parent["natural_manifest_sha256"]:
        raise ValueError("frozen held-out manifest changed")
    if digest(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("YOLOX-Nano 640 artifact checksum mismatch")
    sources = source_paths()
    assets = coarse_assets(parent)
    durations = {row["video_id"]: row["duration_seconds"] for row in parent["videos"]}
    clip = encoder()
    rows = []
    for source_row in parent["rows"]:
        text_vector = clip.text(source_row["query"])
        asset = assets[source_row["video_id"]]
        rows.append({
            **{key: source_row[key] for key in (
                "query_id", "video_id", "split", "query", "query_type",
                "expected_presence", "relevant_intervals",
            )},
            "duration": durations[source_row["video_id"]],
            "mapping": map_query(source_row["query"], source_row["query_type"]),
            "text_vector": text_vector,
            "coarse": ranked_candidates(asset["frames"], asset["vectors"], text_vector),
            "global_5s": source_row["clip_candidates"],
        })

    pool = {}
    pool_resources = {}
    for video_id, source in sources.items():
        folder = run_root / "global-2s" / video_id
        record = prepare_global_two_second(source, durations[video_id], folder)
        paths = [folder / frame["file"] for frame in record["frames"]]
        started = time.perf_counter()
        vectors = clip.images(paths)
        pool[video_id] = (record["frames"], vectors)
        pool_resources[video_id] = {
            "frames": len(paths), "jpeg_bytes": sum(path.stat().st_size for path in paths),
            "embedding_bytes": int(vectors.nbytes),
            "extraction_seconds": record["extraction_seconds"],
            "clip_index_seconds": time.perf_counter() - started,
        }
    for row in rows:
        frames, vectors = pool[row["video_id"]]
        scores = vectors @ row["text_vector"][0]
        order = np.argsort(-scores)[:5]
        row["global_2s"] = [
            {"timestamp": frames[index]["timestamp"], "score": float(scores[index]), "modality": "visual"}
            for index in order
        ]

    policy_trials = []
    for radius in (2.0, 4.0, 6.0):
        for top_k in (5, 10, 20):
            for spacing in (0.0, 6.0):
                counts = []
                trial_rows = []
                for row in rows:
                    trial = dict(row)
                    if row["mapping"].detector_enabled:
                        frames, vectors = pool[row["video_id"]]
                        plain, clip_candidates = pool_candidates(
                            row, frames, vectors, radius, top_k, spacing
                        )
                        trial["bounded_plain"] = plain
                        trial["bounded_clip"] = clip_candidates
                        _, selected = local_pool(row, frames, radius, top_k, spacing)
                        counts.append(len(selected))
                    else:
                        trial["bounded_plain"] = row["global_5s"]
                        trial["bounded_clip"] = row["global_5s"]
                    trial_rows.append(trial)
                calibration = [row for row in trial_rows if row["split"] == "calibration"]
                routed = [row for row in calibration if row["mapping"].detector_enabled]
                heldout_trial = [row for row in trial_rows if row["split"] == "heldout"]
                heldout_routed = [
                    row for row in heldout_trial if row["mapping"].detector_enabled
                ]
                result = metrics(routed, "bounded_clip")
                policy_trials.append({
                    "radius_seconds": radius, "top_k": top_k, "spacing_seconds": spacing,
                    "calibration_routed_metrics": result,
                    "heldout_routed_metrics": metrics(heldout_routed, "bounded_clip"),
                    "average_secondary_frames": statistics.mean(counts),
                })
    winner = max(
        policy_trials,
        key=lambda item: (
            item["calibration_routed_metrics"]["5"]["recall"],
            item["calibration_routed_metrics"]["1"]["recall"],
            item["calibration_routed_metrics"]["5"]["mrr"],
            -item["average_secondary_frames"],
        ),
    )
    radius, top_k, spacing = (
        winner["radius_seconds"], winner["top_k"], winner["spacing_seconds"]
    )

    cache = SecondaryFrameCache(run_root / "cache", 512 * 1024 * 1024, extract_seek)
    vector_cache = {}
    detector_cache = {}
    process = psutil.Process()
    memory_baseline = process.memory_info().rss
    memory_samples = [memory_baseline]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.005):
            memory_samples.append(process.memory_info().rss)

    watcher = threading.Thread(target=sample_memory, daemon=True)
    watcher.start()
    detector = YoloXNanoDetector(
        MODEL_PATH, threads=4, confidence_floor=0.01,
        nms_threshold=0.45, input_size=(640, 640),
    )
    query_runtime = []
    for row in rows:
        if not row["mapping"].detector_enabled:
            row["bounded_plain"] = row["global_5s"]
            row["bounded_clip"] = row["global_5s"]
            row["detector_fused"] = [
                {**item, "secondary_clip_score": item["score"], "combined_score": item["score"]}
                for item in row["global_5s"]
            ]
            continue
        coarse = choose_coarse(row, top_k, spacing)
        windows = local_windows(coarse, radius, row["duration"])
        timestamps = [
            round(index * 2.0, 3)
            for index in range(int(row["duration"] // 2.0) + 1)
            if any(start <= index * 2.0 <= end for start, end in windows)
        ]
        decode_seconds = 0.0
        paths = []
        hits = 0
        for timestamp in timestamps:
            path, hit, elapsed = cache.get(
                row["video_id"], timestamp, sources[row["video_id"]]
            )
            paths.append(path)
            hits += int(hit)
            decode_seconds += elapsed
        missing = [
            (timestamp, path) for timestamp, path in zip(timestamps, paths)
            if (row["video_id"], timestamp) not in vector_cache
        ]
        clip_started = time.perf_counter()
        if missing:
            vectors = clip.images([path for _, path in missing])
            for (timestamp, _), vector in zip(missing, vectors):
                vector_cache[(row["video_id"], timestamp)] = vector
        clip_seconds = time.perf_counter() - clip_started
        rank_started = time.perf_counter()
        scores = np.array([
            vector_cache[(row["video_id"], timestamp)] @ row["text_vector"][0]
            for timestamp in timestamps
        ])
        order = np.argsort(-scores)
        all_clip = [
            {
                "timestamp": timestamps[index], "score": float(scores[index]),
                "secondary_clip_score": float(scores[index]), "modality": "visual",
                "coarse_support_score": max(
                    item.score for item in coarse
                    if abs(item.timestamp - timestamps[index]) <= radius
                ),
            }
            for index in order
        ]
        row["bounded_clip"] = all_clip[:5]
        row["secondary_timestamps"] = timestamps
        row["bounded_plain"] = sorted(
            all_clip,
            key=lambda item: (
                -item["coarse_support_score"],
                min(abs(item["timestamp"] - candidate.timestamp) for candidate in coarse),
            ),
        )[:5]
        rank_seconds = time.perf_counter() - rank_started
        detector_seconds = 0.0
        enriched = []
        for candidate in row["bounded_clip"]:
            identity = (row["video_id"], candidate["timestamp"])
            if identity not in detector_cache:
                path = cache.path_for(*identity)
                started = time.perf_counter()
                detector_cache[identity] = detector.detect_path(
                    f"{identity[0]}:{identity[1]:.3f}", identity[1], path
                )
                detector_seconds += time.perf_counter() - started
            enriched.append({
                **candidate,
                **object_features(row["mapping"], detector_cache[identity]),
            })
        row["detector_candidates"] = enriched
        query_runtime.append({
            "query_id": row["query_id"], "split": row["split"],
            "secondary_frames": len(timestamps), "cache_hits": hits,
            "decode_seconds": decode_seconds, "secondary_clip_seconds": clip_seconds,
            "ranking_seconds": rank_seconds, "detector_seconds": detector_seconds,
            "total_added_seconds": decode_seconds + clip_seconds + rank_seconds + detector_seconds,
        })
    stop.set()
    watcher.join()
    memory_samples.append(process.memory_info().rss)

    calibration = [row for row in rows if row["split"] == "calibration"]
    weight_trials = []
    for weight in (0.0, 0.05, 0.1, 0.2):
        for row in rows:
            candidates = row.get("detector_candidates")
            if candidates is None:
                candidates = row["detector_fused"]
            row["trial_fused"] = fuse(candidates, weight if row["mapping"].detector_enabled else 0)
        threshold = fit_abstention_threshold(calibration, "trial_fused")
        for row in calibration:
            row["trial_results"] = abstain(row["trial_fused"], threshold)
        result = metrics(calibration, "trial_results")
        weight_trials.append({"weight": weight, "threshold": threshold, "metrics": result})
    selected_weight = max(
        weight_trials,
        key=lambda item: (
            item["metrics"]["5"]["recall"], item["metrics"]["1"]["recall"],
            item["metrics"]["5"]["mrr"], -item["weight"],
        ),
    )
    for row in rows:
        candidates = row.get("detector_candidates")
        if candidates is None:
            candidates = row["detector_fused"]
        row["fused_candidates"] = fuse(
            candidates, selected_weight["weight"] if row["mapping"].detector_enabled else 0
        )
        row["final_results"] = abstain(
            row["fused_candidates"], selected_weight["threshold"]
        )

    heldout = [row for row in rows if row["split"] == "heldout"]
    heldout_slices = slices(heldout)
    runtime_values = [row["total_added_seconds"] for row in query_runtime]
    candidate_loss = (
        metrics(heldout, "bounded_clip")["5"]["recall"]
        - metrics(heldout, "fused_candidates")["5"]["recall"]
    )
    small = metrics(heldout_slices["SMALL_OBJECT"], "final_results")["5"]
    secondary_positive = [
        retrieval_metrics(
            row.get("secondary_timestamps", []), row["relevant_intervals"],
            max(1, len(row.get("secondary_timestamps", []))),
        )["recall"]
        for row in heldout
        if row["mapping"].detector_enabled and row["expected_presence"]
    ]
    gate = {
        "small_object_positive_false_abstention": small["positive_false_abstention_rate"],
        "negative_false_accept_rate": metrics(heldout, "final_results")["5"]["negative_false_accept_rate"],
        "candidate_recall_at_5_loss": candidate_loss,
        "added_warm_median_seconds": statistics.median(runtime_values),
        "added_p95_seconds": percentile(runtime_values, 95),
        "added_peak_rss_bytes": max(memory_samples) - memory_baseline,
    }
    gate["passed"] = (
        gate["small_object_positive_false_abstention"] is not None
        and gate["small_object_positive_false_abstention"] <= 0.2
        and gate["negative_false_accept_rate"] is not None
        and gate["negative_false_accept_rate"] <= 0.1
        and gate["candidate_recall_at_5_loss"] <= 0.02
        and gate["added_warm_median_seconds"] <= 0.4
        and gate["added_peak_rss_bytes"] <= 500 * 1024 * 1024
    )
    fields = ["global_5s", "global_2s", "bounded_plain", "bounded_clip", "final_results"]
    report = {
        "schema_version": "1.0.0", "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "production_changed": False,
        "heldout_tuning": False, "parent_report_sha256": digest(PARENT_REPORT),
        "natural_manifest_sha256": digest(NATURAL_MANIFEST),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "policy_selection": {
            "split": "calibration", "trials": policy_trials, "winner": winner,
            "selection_rule": "maximize routed calibration R@5, then R@1/MRR, then minimize secondary frames",
        },
        "fusion_selection": {
            "split": "calibration", "trials": weight_trials,
            "selected_weight": selected_weight["weight"],
            "selected_threshold": selected_weight["threshold"],
        },
        "quality": {
            "calibration": {field: metrics(calibration, field) for field in fields},
            "heldout": {field: metrics(heldout, field) for field in fields},
            "heldout_slices": {
                name: {field: metrics(slice_rows, field) for field in fields}
                for name, slice_rows in heldout_slices.items()
            },
            "heldout_routed_secondary_frame_recall": statistics.mean(secondary_positive),
        },
        "resources": {
            "global_2s": pool_resources,
            "bounded_cache": cache.stats(),
            "routed_queries": len(query_runtime),
            "average_secondary_frames_per_query": statistics.mean(
                row["secondary_frames"] for row in query_runtime
            ),
            "duplicate_frame_requests": sum(row["cache_hits"] for row in query_runtime),
            "query_runtime": query_runtime,
            "added_median_seconds": statistics.median(runtime_values),
            "added_p95_seconds": percentile(runtime_values, 95),
            "added_peak_rss_bytes": max(memory_samples) - memory_baseline,
        },
        "detector": {
            "model": "YOLOX-Nano", "revision": "e1052df71842031413f6030723c3607b839c80ce",
            "input_size": [640, 640], "model_sha256": MODEL_SHA256,
            "unique_frames": len(detector_cache), "session_reused": True,
        },
        "gate": gate,
        "second_frozen_run": {
            "required": gate["passed"], "executed": False,
            "reason": "Run only if every critical gate passes.",
        },
        "rows": [
            {
                **{key: row[key] for key in (
                    "query_id", "video_id", "split", "query", "query_type",
                    "expected_presence", "relevant_intervals",
                )},
                "mapping": row["mapping"].model_dump(mode="json"),
                "secondary_timestamps": row.get("secondary_timestamps"),
                "detector_candidates": row.get("detector_candidates"),
                "fused_candidates": row["fused_candidates"],
                **{field: row[field] for field in fields},
            }
            for row in rows
        ],
        "detector_evidence": {
            f"{video_id}:{timestamp:.3f}": [
                item.model_dump(mode="json") for item in evidence
            ]
            for (video_id, timestamp), evidence in detector_cache.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/bounded-secondary/report.json")
    parser.add_argument("--run-root", type=Path, default=ROOT / "data/bounded-secondary")
    args = parser.parse_args()
    result = run(args.output, args.run_root)
    print(json.dumps({"winner": result["policy_selection"]["winner"], "gate": result["gate"]}, indent=2))
