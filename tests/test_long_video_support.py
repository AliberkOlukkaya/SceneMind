import asyncio
import subprocess
from types import SimpleNamespace

import imageio_ffmpeg
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.config import settings
from app.main import app
from app.video import ingestion_lock, inspect_video, process_video, read_manifest, upload
from app.worker import cleanup_job_artifacts, timeout_for

OLD_UPLOAD_LIMIT = 250 * 1024 * 1024


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
            "color=c=red:s=160x120:r=5:d=6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    return TestClient(app)


class DeferredBackground:
    def __init__(self):
        self.tasks = []

    def add_task(self, function, *args, **kwargs):
        self.tasks.append((function, args, kwargs))


def streaming_request(chunks, content_length=None):
    messages = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]

    async def receive():
        return messages.pop(0)

    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return Request({"type": "http", "method": "POST", "headers": headers}, receive)


def test_long_video_defaults_are_configurable():
    assert settings.max_upload_bytes >= 1024 * 1024 * 1024
    assert settings.max_duration >= 3600
    assert settings.processing_timeout >= 1800
    assert settings.speech_job_timeout > settings.ingest_job_timeout


def test_limit_endpoint_reflects_runtime_configuration(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 777)
    monkeypatch.setattr(settings, "max_duration", 888)
    assert client.get("/videos/limits").json() == {
        "max_upload_bytes": 777,
        "max_duration_seconds": 888,
    }


def test_streamed_upload_accepts_declared_size_above_old_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "min_free_disk_bytes", 0)
    monkeypatch.setattr(settings, "durable_jobs", False)
    background = DeferredBackground()
    request = streaming_request([b"one", b"two", b"three"], OLD_UPLOAD_LIMIT + 1)

    record = asyncio.run(upload(request, background, filename="large.webm"))
    try:
        assert record["bytes"] == 11
        assert record["status"] == "queued"
        assert len(background.tasks) == 1
        assert (settings.data_dir / record["id"] / "source.webm").read_bytes() == b"onetwothree"
    finally:
        ingestion_lock.release()


def test_declared_upload_over_configured_limit_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    request = streaming_request([b"small"], 11)
    with pytest.raises(HTTPException) as failure:
        asyncio.run(upload(request, DeferredBackground(), filename="large.webm"))
    assert failure.value.status_code == 413


def test_stream_limit_removes_partial_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "max_upload_bytes", 5)
    monkeypatch.setattr(settings, "min_free_disk_bytes", 0)
    request = streaming_request([b"123", b"456"])
    with pytest.raises(HTTPException) as failure:
        asyncio.run(upload(request, DeferredBackground(), filename="large.webm"))
    assert failure.value.status_code == 413
    assert list(settings.data_dir.iterdir()) == []


def test_disk_reserve_rejects_before_folder_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "min_free_disk_bytes", 100)
    monkeypatch.setattr("app.video.shutil.disk_usage", lambda _: SimpleNamespace(free=99))
    request = streaming_request([b"123"])
    with pytest.raises(HTTPException) as failure:
        asyncio.run(upload(request, DeferredBackground(), filename="large.webm"))
    assert failure.value.status_code == 507
    assert list(settings.data_dir.iterdir()) == []


def test_disk_guard_reserves_processing_headroom(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "max_upload_bytes", 1000)
    monkeypatch.setattr(settings, "min_free_disk_bytes", 100)
    monkeypatch.setattr(settings, "processing_disk_headroom_ratio", 0.25)
    monkeypatch.setattr("app.video.shutil.disk_usage", lambda _: SimpleNamespace(free=1349))
    request = streaming_request([b"x"], 1000)
    with pytest.raises(HTTPException) as failure:
        asyncio.run(upload(request, DeferredBackground(), filename="large.webm"))
    assert failure.value.status_code == 507


def test_duration_limit_remains_enforced(video, tmp_path, monkeypatch):
    path = tmp_path / "six-seconds.mp4"
    path.write_bytes(video)
    monkeypatch.setattr(settings, "max_duration", 5)
    with pytest.raises(ValueError, match="duration"):
        inspect_video(path)


def test_failed_frame_extraction_cleans_staging_and_completed_frames(tmp_path, monkeypatch):
    folder = tmp_path / "video"
    folder.mkdir()
    (folder / "source.webm").write_bytes(b"source")
    old_frames = folder / "frames"
    old_frames.mkdir()
    (old_frames / "000001.jpg").write_bytes(b"old")
    record = {
        "id": "00000000-0000-0000-0000-000000000000",
        "source": "source.webm",
        "sampling_interval": 5,
        "frames": [{"timestamp": 0}],
        "status": "queued",
    }
    monkeypatch.setattr(
        "app.video.inspect_video",
        lambda _: {"duration": 10, "fps": 1, "width": 10, "height": 10, "codec": "test"},
    )

    def fail(*args, **kwargs):
        staged = folder / "frames.tmp"
        (staged / "000001.jpg").write_bytes(b"partial")
        raise subprocess.TimeoutExpired("ffmpeg", 1)

    monkeypatch.setattr("app.video.subprocess.run", fail)
    ingestion_lock.acquire()
    process_video(folder, record)

    failed = read_manifest(folder)
    assert failed["status"] == "failed"
    assert failed["stage"] == "failed"
    assert failed["frames"] == []
    assert not (folder / "frames.tmp").exists()
    assert not (folder / "frames").exists()


def test_stage_specific_worker_deadlines(monkeypatch):
    monkeypatch.setattr(settings, "ingest_job_timeout", 101)
    monkeypatch.setattr(settings, "speech_job_timeout", 202)
    monkeypatch.setattr(settings, "visual_job_timeout", 303)
    monkeypatch.setattr(settings, "job_timeout", 404)
    assert timeout_for("ingest") == 101
    assert timeout_for("speech") == 202
    assert timeout_for("visual") == 303
    assert timeout_for("unknown") == 404


def test_supervisor_cleans_only_partial_stage_artifacts(tmp_path, monkeypatch):
    video_id = "00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    folder = settings.data_dir / video_id
    staged = folder / "frames.tmp"
    completed = folder / "frames"
    staged.mkdir(parents=True)
    completed.mkdir()
    (staged / "partial.jpg").write_bytes(b"partial")
    (completed / "complete.jpg").write_bytes(b"complete")
    cleanup_job_artifacts({"video_id": video_id, "kind": "ingest"})
    assert not staged.exists()
    assert (completed / "complete.jpg").is_file()
