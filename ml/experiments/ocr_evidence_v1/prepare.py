"""Extract production-equivalent and diagnostic OCR frames from ignored local media."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "ocr-evidence-v1"


def extract(video: Path, destination: Path, interval: int, width: int = 480) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    for old in destination.glob("*.jpg"):
        old.unlink()
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin", "-y", "-v", "info", "-i", str(video), "-an",
        "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',showinfo,scale={width}:-2",
        "-fps_mode", "vfr", "-q:v", "3", str(destination / "%06d.jpg"),
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    timestamps = re.findall(
        r"\bn:\s*\d+.*?\bpts_time:([\d.eE+-]+)",
        result.stderr.decode("utf-8", errors="replace"),
    )
    images = sorted(destination.glob("*.jpg"))
    if len(timestamps) < len(images):
        raise RuntimeError("frame timestamp extraction failed")
    return [
        {"frame_id": path.stem, "timestamp": round(float(timestamps[index]), 3), "path": str(path.relative_to(DATA)).replace("\\", "/")}
        for index, path in enumerate(images)
    ]


def contact_sheet(source_id: str, frames: list[dict], output: Path) -> None:
    columns, label_height = 4, 28
    opened = [Image.open(DATA / row["path"]).convert("RGB") for row in frames]
    if not opened:
        return
    width, height = opened[0].size
    rows = math.ceil(len(opened) / columns)
    sheet = Image.new("RGB", (width * columns, (height + label_height) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, (image, row) in enumerate(zip(opened, frames)):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + height + 4), f"{source_id} {row['frame_id']}  {row['timestamp']:.3f}s", fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def main() -> None:
    sources = json.loads((DATA / "source_manifest.json").read_text(encoding="utf-8"))
    prepared = []
    for source in sources:
        source_id = source["source_id"]
        video = Path(source["local_path"])
        if not video.exists() or video.stat().st_size < 10_000:
            print(f"skip missing source: {source_id}")
            continue
        five = extract(video, DATA / "frames-5s" / source_id, 5)
        one = extract(video, DATA / "frames-1s" / source_id, 1)
        contact_sheet(source_id, five, DATA / "contact-sheets" / f"{source_id}-5s.jpg")
        contact_sheet(source_id, one, DATA / "contact-sheets" / f"{source_id}-1s.jpg")
        prepared.append({**source, "production_frames": five, "diagnostic_frames": one})
    (DATA / "prepared_sources.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
