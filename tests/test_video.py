import subprocess

import imageio_ffmpeg
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    return TestClient(app)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:r=10:d=6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


def test_real_video_ingestion(client, video):
    response = client.post("/videos?filename=sample.mp4", content=video)
    assert response.status_code == 202
    video_id = response.json()["id"]
    record = client.get(f"/videos/{video_id}").json()
    assert record["status"] == "ready"
    assert record["metadata"]["width"] == 320
    assert record["metadata"]["duration"] == pytest.approx(6, abs=0.2)
    assert [frame["timestamp"] for frame in record["frames"]] == [0, 5]
    assert client.get(record["frames"][0]["thumbnail"]).headers["content-type"] == "image/jpeg"
    assert len(client.get("/videos").json()) == 1
    media = client.get(f"/videos/{video_id}/media", headers={"Range": "bytes=0-9"})
    assert media.status_code == 206
    assert media.content == video[:10]


@pytest.mark.parametrize(
    "filename,body,code",
    [
        ("bad.exe", b"abc", 415),
        ("empty.mp4", b"", 400),
    ],
)
def test_invalid_upload(client, filename, body, code):
    assert client.post(f"/videos?filename={filename}", content=body).status_code == code
    assert client.get("/videos").json() == []


def test_corrupt_video_reports_failure(client):
    response = client.post("/videos?filename=fake.mp4", content=b"not a video")
    record = client.get(f"/videos/{response.json()['id']}").json()
    assert record["status"] == "failed"
    assert "error" in record
    # A failed job releases the ingestion slot.
    assert client.post("/videos?filename=empty.mp4", content=b"").status_code == 400


def test_upload_limit_cleans_partial_files(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 2)
    assert client.post("/videos?filename=big.mp4", content=b"123").status_code == 413
    assert list(settings.data_dir.iterdir()) == []


def test_missing_and_unsafe_ids(client):
    assert client.get("/videos/not-a-uuid").status_code == 404
    assert client.get("/videos/00000000-0000-0000-0000-000000000000").status_code == 404


def test_busy_pipeline(client):
    from app.video import ingestion_lock

    ingestion_lock.acquire()
    try:
        assert client.post("/videos?filename=test.mp4", content=b"123").status_code == 429
    finally:
        ingestion_lock.release()


def test_timeout_is_reported(client, video, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 1)

    monkeypatch.setattr("app.video.subprocess.run", timeout)
    response = client.post("/videos?filename=test.mp4", content=video)
    record = client.get(f"/videos/{response.json()['id']}").json()
    assert record["status"] == "failed"
    assert record["frames"] == []


def test_restart_marks_abandoned_job(client):
    from app.video import read_manifest, recover_interrupted, save_manifest

    folder = settings.data_dir / "00000000-0000-0000-0000-000000000000"
    folder.mkdir(parents=True)
    save_manifest(folder, {"status": "processing"})
    recover_interrupted()
    assert read_manifest(folder)["status"] == "failed"
