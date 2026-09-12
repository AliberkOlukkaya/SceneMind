"""Download and verify the frozen Wikimedia Commons natural-video benchmark."""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation.schema_v2 import load_benchmark  # noqa: E402


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def prepare(manifest: Path, verify_only: bool = False):
    benchmark = load_benchmark(manifest)
    for video in benchmark.videos:
        destination = video.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if verify_only:
                raise FileNotFoundError(destination)
            temporary = destination.with_suffix(destination.suffix + ".part")
            request = urllib.request.Request(
                str(video.source.download_url),
                headers={"User-Agent": "SceneMindBenchmark/2.0 (github.com/AliberkOlukkaya/SceneMind)"},
            )
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            temporary.replace(destination)
        actual_sha1 = digest(destination, "sha1")
        actual_sha256 = digest(destination, "sha256")
        if actual_sha1 != video.source.source_sha1 or actual_sha256 != video.source.source_sha256:
            raise ValueError(f"checksum mismatch for {video.video_id}")
        print(f"verified {video.video_id}: sha256={actual_sha256}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("ml/evaluation/natural_v2.json"))
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    prepare(arguments.manifest, arguments.verify_only)
