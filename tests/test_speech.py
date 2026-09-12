import subprocess
import sys
from types import SimpleNamespace

import imageio_ffmpeg
import pytest
from app.config import settings
from app.database import Transcript, engine, migrate
from app.main import app
from app.speech import recover_speech
from app.video import save_manifest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

VIDEO_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def speech_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'speech.db'}")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace())
    migrate()
    folder = settings.data_dir / VIDEO_ID
    folder.mkdir(parents=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            str(folder / "source.wav"),
        ],
        check=True,
    )
    save_manifest(
        folder,
        {
            "status": "ready",
            "source": "source.wav",
            "metadata": {"duration": 3},
            "frames": [{"timestamp": 0, "thumbnail": "frame.jpg"}],
        },
    )
    return TestClient(app)


def test_transcript_persistence_and_literal_search(speech_client, monkeypatch):
    def inference(path):
        assert path.is_file()
        return [{"start": 0.2, "end": 2.8, "text": "Learning rate is 10% today."}], "en"

    monkeypatch.setattr("app.speech.infer_audio", inference)
    route = f"/videos/{VIDEO_ID}/transcript"
    assert speech_client.get(route).json()["status"] == "not_started"
    assert speech_client.post(route).status_code == 202
    data = speech_client.get(route, params={"q": "LEARNING"}).json()
    assert data["status"] == "ready"
    assert data["segments"][0]["start"] == 0.2
    assert len(speech_client.get(route, params={"q": "%"}).json()["segments"]) == 1
    assert speech_client.get(route, params={"q": "missing"}).json()["segments"] == []
    assert not (settings.data_dir / VIDEO_ID / "audio.wav").exists()
    # Retrying replaces segments rather than duplicating them.
    speech_client.post(route)
    assert len(speech_client.get(route).json()["segments"]) == 1
    for mode in ("speech", "hybrid"):
        results = speech_client.get(
            f"/videos/{VIDEO_ID}/search", params={"q": "learning", "mode": mode}
        ).json()
        assert results["results"][0]["text"] == "Learning rate is 10% today."
    assert results["modalities_used"] == ["speech"]


def test_model_failure_and_restart(speech_client, monkeypatch):
    def failure(path):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("app.speech.infer_audio", failure)
    route = f"/videos/{VIDEO_ID}/transcript"
    speech_client.post(route)
    assert speech_client.get(route).json()["status"] == "failed"
    with Session(engine()) as session, session.begin():
        session.get(Transcript, VIDEO_ID).status = "processing"
    recover_speech()
    assert "interrupted" in speech_client.get(route).json()["error"]


def test_invalid_model_timestamps(speech_client, monkeypatch):
    monkeypatch.setattr(
        "app.speech.infer_audio",
        lambda path: ([{"start": -1, "end": 2, "text": "bad"}], "en"),
    )
    route = f"/videos/{VIDEO_ID}/transcript"
    speech_client.post(route)
    assert speech_client.get(route).json()["status"] == "failed"
