"""Compare relevant ONNX CPU thread settings on one real top-five frame batch."""

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil

from ml.experiments.lightweight_pair_scorer.external import UFormOnnxScorer, UFormSpec


def run(images: list[Path], output: Path, repeats: int = 20) -> dict:
    if len(images) != 5 or any(not path.is_file() for path in images):
        raise ValueError("runtime benchmark requires exactly five existing images")
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "model": UFormSpec().as_dict(),
        "runtime": "ONNX Runtime CPUExecutionProvider",
        "environment": {
            "platform": platform.platform(),
            "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "python": platform.python_version(),
            "onnxruntime": version("onnxruntime"),
            "uform": version("uform"),
        },
        "batch_size": 5,
        "repeats": repeats,
        "options": [],
    }
    for threads in (1, 2, 4, 8, 0):
        scorer = UFormOnnxScorer(UFormSpec(), "data/models", threads)
        scorer.warm("a dog outdoors", images)
        timings = [scorer.timed_score("a dog outdoors", images)[1] for _ in range(repeats)]
        report["options"].append(
            {
                "intra_op_threads": threads,
                "inter_op_threads": 1,
                "median_seconds": float(np.median(timings)),
                "p95_seconds": float(np.percentile(timings, 95)),
                "load_seconds": scorer.load_seconds,
            }
        )
    report["selected_intra_op_threads"] = min(
        report["options"], key=lambda item: item["median_seconds"]
    )["intra_op_threads"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs=5, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/uform-runtime-options.json"))
    parser.add_argument("--repeats", type=int, default=20)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.images, arguments.output, arguments.repeats), indent=2))
