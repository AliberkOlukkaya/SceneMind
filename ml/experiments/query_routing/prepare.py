"""Download and checksum-verify ignored query-routing calibration media."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import urllib.request
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "ml/experiments/query_routing/calibration_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root: Path, verify_only: bool = False) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    for source in manifest["sources"]:
        target = root / source["file"]
        if not target.exists():
            if verify_only:
                raise FileNotFoundError(target)
            request = urllib.request.Request(
                source["download_url"],
                headers={"User-Agent": "SceneMind local evaluation/1.0"},
            )
            with urllib.request.urlopen(request) as response, target.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if sha256(target) != source["sha256"]:
            raise ValueError(f"checksum mismatch: {source['source_id']}")
        if verify_only:
            continue
        frames = root / f"{target.stem}-frames"
        frames.mkdir(exist_ok=True)
        result = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "info",
            "-i", str(target), "-an", "-vf",
            "select='isnan(prev_selected_t)+gte(t-prev_selected_t,5)',showinfo,scale=480:-2",
            "-fps_mode", "vfr", "-frames:v", str(math.ceil(source["duration_seconds"] / 5)),
            "-q:v", "3", str(frames / "%06d.jpg"),
        ], check=True, capture_output=True)
        timestamps = [round(float(value), 3) for value in re.findall(
            rb"\bn:\s*\d+.*?\bpts_time:([\d.eE+-]+)", result.stderr
        )]
        images = sorted(frames.glob("*.jpg"))
        if len(timestamps) < len(images) or not images:
            raise ValueError(f"frame timestamp extraction failed: {source['source_id']}")
        (root / f"{target.stem}.timestamps.json").write_text(
            json.dumps(timestamps[:len(images)], indent=2), encoding="utf-8"
        )
        transcript = root / f"{target.stem}.transcript.json"
        if not transcript.exists():
            from app.speech import infer_audio

            audio = root / f"{target.stem}.wav"
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error",
                "-i", str(target), "-vn", "-ac", "1", "-ar", "16000", "-c:a",
                "pcm_s16le", str(audio),
            ], check=True, capture_output=True)
            started = __import__("time").perf_counter()
            segments, language = infer_audio(audio)
            elapsed = __import__("time").perf_counter() - started
            audio.unlink(missing_ok=True)
            transcript.write_text(json.dumps({"language": language, "seconds": elapsed,
                                              "segments": segments}, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/query-routing/sources")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    prepare(args.root, args.verify_only)
