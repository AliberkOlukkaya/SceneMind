"""Create an ignored long-duration infrastructure fixture from a local video."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or frames <= 0:
            raise ValueError("Source video cannot be inspected")
        return frames / fps
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minutes", type=float, default=45, choices=range(30, 61))
    args = parser.parse_args()
    source_duration = duration(args.source)
    target_seconds = args.minutes * 60
    loops = max(0, math.ceil(target_seconds / source_duration) - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-stream_loop",
            str(loops),
            "-i",
            str(args.source),
            "-t",
            str(target_seconds),
            "-map",
            "0",
            "-c",
            "copy",
            str(args.output),
        ],
        check=True,
    )
    print(json.dumps({
        "kind": "INGESTION_STRESS_TEST_ONLY",
        "source": str(args.source),
        "output": str(args.output),
        "target_seconds": target_seconds,
        "measured_seconds": duration(args.output),
        "bytes": args.output.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
