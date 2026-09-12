"""Download and checksum-verify the frozen calibration expansion."""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.lightweight_pair_scorer.schema import load_development  # noqa: E402


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def prepare(manifest: Path, verify_only: bool = False) -> None:
    development = load_development(manifest)
    for video in development.videos:
        destination = video.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if verify_only:
                raise FileNotFoundError(destination)
            temporary = destination.with_suffix(destination.suffix + ".part")
            request = urllib.request.Request(
                str(video.source.download_url),
                headers={
                    "User-Agent": (
                        "SceneMind/1.0 (https://github.com/AliberkOlukkaya/SceneMind; "
                        "reproducible benchmark)"
                    )
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            temporary.replace(destination)
        if (
            digest(destination, "sha1") != video.source.source_sha1
            or digest(destination, "sha256") != video.source.source_sha256
        ):
            raise ValueError(f"checksum mismatch: {video.video_id}")
        print(f"verified {video.video_id}: {video.source.source_sha256}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("calibration_v2.json")
    )
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    prepare(arguments.manifest, arguments.verify_only)
