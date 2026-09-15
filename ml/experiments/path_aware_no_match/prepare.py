"""Prepare ignored five-second frames, transcripts, and vectors for held-out sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from ml.experiments.path_aware_no_match.build_manifest import MANIFEST, ROOT

DATA_ROOT = ROOT / "data/path-aware-no-match"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root: Path = DATA_ROOT, verify_only: bool = False) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    for source in manifest["sources"]:
        media = ROOT / source["path"]
        if not media.is_file() or sha256(media) != source["sha256"]:
            raise ValueError(f"missing or changed source media: {source['source_id']}")
        if source["asset_set"] == "query-routing":
            continue
        frames = root / f"{source['source_id']}-frames"
        timestamps_path = root / f"{source['source_id']}.timestamps.json"
        transcript_path = root / f"{source['source_id']}.transcript.json"
        vector_path = root / f"{source['source_id']}.production-480.npy"
        if verify_only:
            for artifact in (timestamps_path, transcript_path, vector_path):
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
            if not list(frames.glob("*.jpg")):
                raise FileNotFoundError(frames)
            continue
        frames.mkdir(exist_ok=True)
        if not timestamps_path.exists() or not list(frames.glob("*.jpg")):
            result = subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "info",
                "-i", str(media), "-an", "-vf",
                "select='isnan(prev_selected_t)+gte(t-prev_selected_t,5)',showinfo,scale=480:-2",
                "-fps_mode", "vfr", "-frames:v", str(math.ceil(source["duration_seconds"] / 5)),
                "-q:v", "3", str(frames / "%06d.jpg"),
            ], check=True, capture_output=True)
            timestamps = [round(float(value), 3) for value in re.findall(
                rb"\bn:\s*\d+.*?\bpts_time:([\d.eE+-]+)", result.stderr
            )]
            images = sorted(frames.glob("*.jpg"))
            if not images or len(timestamps) < len(images):
                raise ValueError(f"frame timestamp extraction failed: {source['source_id']}")
            timestamps_path.write_text(json.dumps(timestamps[:len(images)], indent=2),
                                       encoding="utf-8")
        if not transcript_path.exists():
            from app.speech import infer_audio

            audio = root / f"{source['source_id']}.wav"
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error",
                "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", "-c:a",
                "pcm_s16le", str(audio),
            ], check=True, capture_output=True)
            started = time.perf_counter()
            segments, language = infer_audio(audio)
            elapsed = time.perf_counter() - started
            audio.unlink(missing_ok=True)
            transcript_path.write_text(json.dumps({"language": language, "seconds": elapsed,
                                                   "segments": segments}, indent=2),
                                       encoding="utf-8")
        if not vector_path.exists():
            from app.encoder import encoder

            vectors = encoder().images(sorted(frames.glob("*.jpg")))
            with vector_path.open("wb") as output:
                np.save(output, vectors, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    prepare(arguments.root, arguments.verify_only)
