"""Verify ignored media and create review frames, contact sheets, and transcripts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
CATALOG = Path(__file__).with_name("sources_v1.json")
DATA = ROOT / "data/human-grounded-router"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_frames(source: dict) -> list[dict]:
    media = DATA / "media" / source["file"]
    target = DATA / "frames" / source["source_id"]
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    result = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "info",
        "-i", str(media), "-an", "-vf",
        "select='isnan(prev_selected_t)+gte(t-prev_selected_t,5)',showinfo,scale=480:-2",
        "-fps_mode", "vfr", "-frames:v", str(math.ceil(source["duration_seconds"] / 5) + 1),
        "-q:v", "3", str(target / "%06d.jpg"),
    ], check=True, capture_output=True)
    timestamps = [float(value) for value in re.findall(
        rb"\bn:\s*\d+.*?\bpts_time:([\d.eE+-]+)", result.stderr
    )]
    images = sorted(target.glob("*.jpg"))
    if len(images) != len(timestamps) or not images:
        raise ValueError(f"frame extraction mismatch for {source['source_id']}")
    return [{"file": image.name, "timestamp": round(timestamp, 3)}
            for image, timestamp in zip(images, timestamps)]


def make_contact_sheets(source: dict, frames: list[dict]) -> list[str]:
    frame_root = DATA / "frames" / source["source_id"]
    sheet_root = DATA / "contact-sheets" / source["source_id"]
    shutil.rmtree(sheet_root, ignore_errors=True)
    sheet_root.mkdir(parents=True)
    outputs = []
    for page, start in enumerate(range(0, len(frames), 30), 1):
        subset = frames[start:start + 30]
        canvas = Image.new("RGB", (1500, 6 * 190), "white")
        draw = ImageDraw.Draw(canvas)
        for index, frame in enumerate(subset):
            image = Image.open(frame_root / frame["file"]).convert("RGB")
            image.thumbnail((290, 150))
            x, y = (index % 5) * 300, (index // 5) * 190
            canvas.paste(image, (x, y))
            minutes, seconds = divmod(frame["timestamp"], 60)
            draw.text((x + 4, y + 154), f"{int(minutes):02d}:{seconds:04.1f}", fill="black")
        output = sheet_root / f"sheet-{page:02d}.jpg"
        canvas.save(output, quality=88)
        outputs.append(str(output.relative_to(DATA)))
    return outputs


def transcribe(source: dict) -> dict:
    output = DATA / "transcripts" / f"{source['source_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    media = DATA / "media" / source["file"]
    audio = DATA / "transcripts" / f"{source['source_id']}.wav"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error", "-i", str(media),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio),
    ], check=True, capture_output=True)
    try:
        from app.speech import infer_audio
        segments, language = infer_audio(audio)
    finally:
        audio.unlink(missing_ok=True)
    value = {"source_id": source["source_id"], "language": language, "segments": segments}
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value


def prepare() -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    inventory = []
    for source in catalog["sources"]:
        media = DATA / "media" / source["file"]
        if not media.exists() or media.stat().st_size != source["bytes"] or sha256(media) != source["sha256"]:
            raise ValueError(f"media checksum mismatch: {source['source_id']}")
        frames = extract_frames(source)
        sheets = make_contact_sheets(source, frames)
        transcript = transcribe(source)
        inventory.append({"source_id": source["source_id"], "frames": len(frames),
                          "contact_sheets": sheets, "language": transcript["language"],
                          "segments": len(transcript["segments"])})
    result = {"sources": inventory}
    (DATA / "review-inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
