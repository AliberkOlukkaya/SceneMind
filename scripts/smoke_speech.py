"""Real local inference smoke test. Run with a short WAV containing 'learning rate'."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

parser = argparse.ArgumentParser()
parser.add_argument("wav", type=Path)
args = parser.parse_args()
settings.data_dir = Path("data/speech-smoke/videos")
settings.database_url = "sqlite:///data/speech-smoke/results.db"
video = Path("data/speech-smoke/reference.mp4")
video.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x240:r=10",
        "-i",
        str(args.wav),
        "-shortest",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(video),
    ],
    check=True,
)
with TestClient(app) as client:
    upload = client.post("/videos?filename=reference.mp4", content=video.read_bytes())
    upload.raise_for_status()
    video_id = upload.json()["id"]
    started = time.perf_counter()
    client.post(f"/videos/{video_id}/transcript").raise_for_status()
    result = client.get(f"/videos/{video_id}/transcript", params={"q": "learning rate"}).json()
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(result, indent=2))
    assert result["status"] == "ready" and result["segments"], result
