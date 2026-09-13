"""Measure isolated YOLOX-Nano top-five CPU latency and added peak RSS."""

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from ml.experiments.object_detector_branch.detector import YoloXNanoDetector


def percentile(values: list[float], percentile_value: int) -> float:
    return float(np.percentile(values, percentile_value))


def benchmark(model: Path, frames: list[Path], loops: int, threads: int) -> dict:
    if len(frames) != 5:
        raise ValueError("runtime benchmark requires exactly five candidate frames")
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.002):
            samples.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    load_started = time.perf_counter()
    detector = YoloXNanoDetector(model, threads=threads)
    load_seconds = time.perf_counter() - load_started

    def pass_once() -> float:
        started = time.perf_counter()
        for index, frame in enumerate(frames):
            detector.detect_path(str(index), float(index), frame)
        return time.perf_counter() - started

    cold_seconds = pass_once()
    warm = [pass_once() for _ in range(loops)]
    stop.set()
    monitor.join()
    samples.append(process.memory_info().rss)
    return {
        "runtime": "ONNX Runtime CPUExecutionProvider",
        "providers": detector.providers,
        "threads": threads,
        "frames_per_query": 5,
        "warmup_passes": 1,
        "measured_passes": loops,
        "load_seconds": load_seconds,
        "cold_top5_seconds": cold_seconds,
        "warm_top5_median_seconds": statistics.median(warm),
        "warm_top5_p95_seconds": percentile(warm, 95),
        "warm_top5_samples_seconds": warm,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": max(samples),
        "added_peak_rss_bytes": max(samples) - baseline,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frame", type=Path, action="append", required=True)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    print(json.dumps(benchmark(arguments.model, arguments.frame, arguments.loops, arguments.threads)))
