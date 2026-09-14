"""Verify licensed sources and reproduce production-style sampled JPEGs."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg

from .schema import VisibilityManifest

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(manifest_path: Path, output: Path) -> None:
    manifest = VisibilityManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    for interval in (5, 2, 1):
        for source in manifest.sources:
            video = ROOT / "data/small-object-ablation/source" / source.filename
            if video.stat().st_size != source.bytes or sha256(video) != source.sha256:
                raise ValueError(f"source checksum mismatch: {source.source_id}")
            folder = output / f"{interval}s" / source.source_id
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob("*.jpg"):
                old.unlink()
            started = time.perf_counter()
            result = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "info",
                    "-i", str(video), "-an", "-vf",
                    f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',showinfo,scale=480:-2",
                    "-fps_mode", "vfr", "-frames:v",
                    str(math.ceil(source.duration_seconds / interval)), "-q:v", "3",
                    str(folder / "%06d.jpg"),
                ],
                check=True, capture_output=True, timeout=300,
            )
            elapsed = time.perf_counter() - started
            timestamps = re.findall(
                r"\bn:\s*\d+.*?\bpts_time:([\d.eE+-]+)",
                result.stderr.decode("utf-8", errors="replace"),
            )
            frames = sorted(folder.glob("*.jpg"))
            if len(timestamps) < len(frames) or not frames:
                raise ValueError(f"could not determine timestamps: {source.source_id}")
            record = {
                "video": source.filename, "interval": interval,
                "duration": source.duration_seconds, "extraction_seconds": elapsed,
                "frames": [
                    {"file": path.name, "timestamp": round(float(timestamps[index]), 3)}
                    for index, path in enumerate(frames)
                ],
            }
            (folder / "timestamps.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("visibility_v1.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/small-object-ablation/sampling")
    args = parser.parse_args()
    prepare(args.manifest, args.output)
