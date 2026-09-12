"""Synthetic visual smoke test; does not measure real-video semantic quality."""

import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

settings.data_dir = Path("data/visual-smoke/videos")
settings.database_url = "sqlite:///data/visual-smoke/results.db"
video = Path("data/visual-smoke/colors.mp4")
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
        "color=red:s=320x240:r=10:d=5",
        "-f",
        "lavfi",
        "-i",
        "color=blue:s=320x240:r=10:d=5",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0",
        "-c:v",
        "libx264",
        str(video),
    ],
    check=True,
)
with TestClient(app) as client:
    upload = client.post("/videos?filename=colors.mp4", content=video.read_bytes())
    upload.raise_for_status()
    video_id = upload.json()["id"]
    started = time.perf_counter()
    client.post(f"/videos/{video_id}/index").raise_for_status()
    print("Index seconds:", round(time.perf_counter() - started, 3))
    for query, expected in [("a red screen", 0), ("a blue screen", 5)]:
        response = client.get(f"/videos/{video_id}/search", params={"q": query, "k": 1})
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, indent=2))
        assert result["results"][0]["timestamp"] == expected
