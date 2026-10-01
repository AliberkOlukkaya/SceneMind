"""Local-only source acquisition and timestamped contact sheets for manual review.

No model inference or automatic ground truth. Metadata must be separately
reviewed for licensing and provenance before promotion to a frozen manifest.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluation.temporal_retrieval import sha256  # noqa: E402

DATA = ROOT / "data/temporal-v1"
DEVELOPMENT = {
    "one-week": "File:One Week (1920).webm",
    "pottery": "File:Der Bartmannskrug.webm",
    "parade": "File:Youth Parade for an Unlimited Period of Use of Tovarna Rog.webm",
    "cycling": "File:Zesdaagse Weeknummer 69-51 - Open Beelden - 12443.ogv",
    "solar-oven": "File:How to Make Solar Oven S’mores (SVS14643).webm",
}


def acquire_development():
    research = json.loads((DATA / "development-source-research.json").read_text(encoding="utf8"))
    pages = {v["title"]: v for v in research["query"]["pages"].values()}
    session = requests.Session()
    session.headers["User-Agent"] = "SceneMindResearch/1.0 local educational video evaluation"
    sources = []
    for source_id, title in DEVELOPMENT.items():
        page = pages[title]
        info = page["videoinfo"][0]
        meta = info["extmetadata"]
        choices = [d for d in info["derivatives"] if d.get("transcodekey") == "480p.vp9.webm"]
        selected = choices[0]["src"] if choices else info["url"]
        extension = Path(urlsplit(selected).path).suffix
        path = DATA / "media" / f"{source_id}{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Downloading {source_id}", flush=True)
            with session.get(selected, stream=True, timeout=90) as response:
                response.raise_for_status()
                temporary = path.with_suffix(".download")
                with temporary.open("wb") as output:
                    for block in response.iter_content(1024 * 1024):
                        output.write(block)
                temporary.replace(path)
        capture = cv2.VideoCapture(str(path))
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or count <= 0:
            raise RuntimeError(f"Invalid downloaded media: {source_id}")
        capture.release()
        source = dict(source_id=source_id, title=title, source_url=info["descriptionurl"],
                      commons_page_id=page["pageid"], media_url=selected,
                      attribution=meta["Artist"]["value"], credit=meta.get("Credit", {}).get("value"),
                      license=meta["LicenseShortName"]["value"],
                      license_url=meta.get("LicenseUrl", {}).get("value"),
                      media_path=str(path.relative_to(ROOT)).replace("\\", "/"),
                      media_sha256=sha256(path), duration_seconds=count / fps,
                      fps=fps, bytes=path.stat().st_size)
        sources.append(source)
        print(json.dumps({k: source[k] for k in ("source_id", "duration_seconds", "bytes")}), flush=True)
        (DATA / "development-sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf8")


def sheets(source_id, step, start, end):
    sources = json.loads((DATA / "development-sources.json").read_text(encoding="utf8"))
    source = next(s for s in sources if s["source_id"] == source_id)
    end = min(end if end is not None else source["duration_seconds"], source["duration_seconds"])
    if step <= 0 or start < 0 or end <= start:
        raise ValueError("Invalid review range")
    capture = cv2.VideoCapture(str(ROOT / source["media_path"]))
    timestamps = [start + i * step for i in range(math.ceil((end - start) / step))]
    directory = DATA / "review" / source_id
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for offset in range(0, len(timestamps), 24):
            sheet = Image.new("RGB", (4 * 320, 6 * 204), "#151515")
            draw = ImageDraw.Draw(sheet)
            for j, timestamp in enumerate(timestamps[offset:offset + 24]):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Decode failure: {source_id} {timestamp}")
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                rgb.thumbnail((320, 180))
                x, y = (j % 4) * 320, (j // 4) * 204
                sheet.paste(rgb, (x + (320 - rgb.width) // 2, y))
                draw.text((x + 4, y + 182), f"{source_id} {timestamp:.2f}s", fill="white")
            output = directory / f"{timestamps[offset]:08.2f}-step{step:g}.jpg"
            sheet.save(output, quality=90)
            print(output.relative_to(ROOT), flush=True)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["acquire-development", "sheets"])
    parser.add_argument("--source")
    parser.add_argument("--step", type=float, default=5)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    if args.command == "acquire-development":
        acquire_development()
    else:
        sheets(args.source, args.step, args.start, args.end)


if __name__ == "__main__":
    main()
