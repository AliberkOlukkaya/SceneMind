"""Prepare the licensed animated pilot; runtime media stays under data/."""

import hashlib
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import imageio_ffmpeg


def main():
    manifest = json.loads(Path("ml/evaluation/bunny.json").read_text())
    root = Path("data/benchmark-source")
    root.mkdir(parents=True, exist_ok=True)
    source = root / "bunny.mp4"
    if not source.exists():
        archive = root / "bunny.zip"
        urllib.request.urlretrieve(
            "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4.zip",
            archive,
        )
        with zipfile.ZipFile(archive) as bundle:
            source.write_bytes(bundle.read("BigBuckBunny_320x180.mp4"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Source checksum mismatch")
    for name, start in [("calibration", 30), ("heldout", 180)]:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-ss",
                str(start),
                "-t",
                "40",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(root / f"{name}.mp4"),
            ],
            check=True,
            timeout=120,
        )
    print("Verified source; prepared two disjoint 40-second clips.")


if __name__ == "__main__":
    main()
