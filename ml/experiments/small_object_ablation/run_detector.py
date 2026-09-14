"""Run one detector configuration on the same frozen human-visible frames."""

import argparse
import hashlib
import json
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from .detectors import RTDetrR18Adapter, YoloXAdapter
from .metrics import conditional_detector_summary
from .schema import VisibilityManifest

ROOT = Path(__file__).resolve().parents[3]
RTDETR_REVISION = "6401be7fee8b49ee00b42fcc4e0064bba8061777"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_path(source_id: str, timestamp: float) -> Path:
    folder = ROOT / "data/small-object-ablation/sampling/1s" / source_id
    timestamps = json.loads((folder / "timestamps.json").read_text(encoding="utf-8"))
    item = next(row for row in timestamps["frames"] if row["timestamp"] == timestamp)
    return folder / item["file"]


def run(config: str, manifest_path: Path, loops: int = 1) -> dict:
    manifest = VisibilityManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    process = psutil.Process()
    baseline = process.memory_info().rss
    memory = [baseline]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.005):
            memory.append(process.memory_info().rss)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    started = time.perf_counter()
    if config.startswith("yolox-"):
        size = int(config.split("-")[1])
        suffix = "" if size == 416 else f"-{size}"
        model_path = ROOT / f"data/models/yolox_nano{suffix}-0.1.1rc0.onnx"
        detector = YoloXAdapter(model_path, size)
        model = {
            "id": "YOLOX-Nano", "revision": "e1052df71842031413f6030723c3607b839c80ce",
            "license": "Apache-2.0", "input_size": [size, size],
            "artifact_sha256": sha256(model_path), "artifact_bytes": model_path.stat().st_size,
        }
    elif config == "rtdetr-r18-640":
        detector = RTDetrR18Adapter(ROOT / "data/models/hf/hub", RTDETR_REVISION)
        snapshot = ROOT / "data/models/models--PekingU--rtdetr_r18vd/snapshots" / RTDETR_REVISION
        weight = snapshot / "model.safetensors"
        if not weight.exists():
            weight = next((ROOT / "data/models/hf/hub/models--PekingU--rtdetr_r18vd/snapshots" / RTDETR_REVISION).glob("model.safetensors"))
        model = {
            "id": "RT-DETR-R18", "revision": RTDETR_REVISION, "license": "Apache-2.0",
            "input_size": [640, 640], "artifact_sha256": sha256(weight),
            "artifact_bytes": weight.stat().st_size,
        }
    else:
        raise ValueError(f"unknown detector configuration: {config}")
    load_seconds = time.perf_counter() - started
    rows = []
    latencies = []
    visible = [
        (event, observation)
        for event in manifest.events
        for observation in event.observations
        if observation.visibility == "visible"
    ]
    for loop in range(loops + 1):
        for event, observation in visible:
            path = frame_path(event.source_id, observation.timestamp)
            call_started = time.perf_counter()
            detections = detector.detect(path, event.target_class)
            elapsed = time.perf_counter() - call_started
            if loop == 0:
                continue
            latencies.append(elapsed)
            if loop > 1:
                continue
            best = max(detections, key=lambda row: row["confidence"], default=None)
            rows.append({
                "event_id": event.event_id, "split": event.split,
                "timestamp": observation.timestamp, "visibility": observation.visibility,
                "target_class": event.target_class, "detected": best is not None,
                "confidence": best["confidence"] if best else 0.0,
                "bbox_area_ratio": best["bbox_area_ratio"] if best else 0.0,
                "bbox": best["bbox"] if best else None,
            })
    stop.set()
    watcher.join()
    memory.append(process.memory_info().rss)
    return {
        "schema_version": "1.0.0", "configuration": config, "model": model,
        "protocol": {"threads": 4, "confidence_floor": 0.01, "warmup_passes": 1,
                     "measured_passes": loops, "visibility_denominator": "visible only"},
        "summary": conditional_detector_summary(rows),
        "split_summary": {
            split: conditional_detector_summary([row for row in rows if row["split"] == split])
            for split in ("calibration", "heldout")
        },
        "runtime": {
            "load_seconds": load_seconds,
            "per_frame_median_seconds": statistics.median(latencies),
            "per_frame_p95_seconds": float(np.percentile(latencies, 95)),
            "baseline_rss_bytes": baseline, "peak_rss_bytes": max(memory),
            "added_peak_rss_bytes": max(memory) - baseline,
        },
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["yolox-416", "yolox-640", "yolox-768", "rtdetr-r18-640"])
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("visibility_v1.json"))
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.manifest, args.loops)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
