import json
import os
import subprocess

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import migrate
from app.jobs import claim, enqueue, finish
from app.main import app
from app.url_ingest import (
    DirectMediaProvider,
    URLIngestError,
    YouTubeProvider,
    acquire_to_local,
    select_provider,
    validate_public_url,
)
from app.video import save_manifest
from app.worker import execute

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def public_dns(monkeypatch):
    def resolve(hostname, _port):
        if hostname in {
            "127.0.0.1",
            "::1",
            "10.0.0.1",
            "168.63.129.16",
            "169.254.169.254",
            "fc00::1",
        }:
            return {hostname}
        return {PUBLIC_IP}

    monkeypatch.setattr("app.url_ingest._resolved_addresses", resolve)


@pytest.fixture
def durable_client(tmp_path, monkeypatch, public_dns):
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'url-jobs.db'}")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "videos")
    monkeypatch.setattr(settings, "durable_jobs", True)
    migrate()
    return TestClient(app)


@pytest.mark.parametrize("url", ["file:///tmp/a.mp4", "ftp://example.com/a.mp4"])
def test_only_http_schemes_are_supported(url):
    with pytest.raises(URLIngestError, match="HTTP and HTTPS"):
        validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/a.mp4",
        "http://127.0.0.1/a.mp4",
        "http://10.0.0.1/a.mp4",
        "http://168.63.129.16/a.mp4",
        "http://169.254.169.254/latest/meta-data/a.mp4",
        "http://[::1]/a.mp4",
        "http://[fc00::1]/a.mp4",
    ],
)
def test_local_private_metadata_and_ipv6_destinations_are_rejected(url, public_dns):
    with pytest.raises(URLIngestError, match="Local|private|non-public|Cloud metadata"):
        validate_public_url(url)


def test_embedded_credentials_are_rejected(public_dns):
    with pytest.raises(URLIngestError, match="usernames or passwords"):
        validate_public_url("https://user:secret@example.com/video.mp4")


def test_dns_address_change_fails_closed(public_dns):
    with pytest.raises(URLIngestError, match="destination changed"):
        validate_public_url("https://example.com/video.mp4", expected_addresses={"8.8.8.8"})


def test_provider_selection_is_bounded(public_dns):
    provider, normalized = select_provider("https://example.com/media/video.webm#fragment")
    assert isinstance(provider, DirectMediaProvider)
    assert normalized == "https://example.com/media/video.webm"
    provider, _ = select_provider("https://youtu.be/abcdefghijk")
    assert isinstance(provider, YouTubeProvider)
    with pytest.raises(URLIngestError, match="not a supported"):
        select_provider("https://example.com/page")


def _mock_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "app.url_ingest._direct_http_client",
        lambda _timeout: httpx.Client(transport=transport, follow_redirects=False),
    )


def test_direct_media_streams_and_validates_redirects(tmp_path, monkeypatch, public_dns):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"Location": "https://cdn.example/video.mp4"})
        return httpx.Response(200, headers={"Content-Length": "6"}, content=b"abcdef")

    _mock_client(monkeypatch, handler)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "min_free_disk_bytes", 0)
    metadata = {}
    path = DirectMediaProvider().acquire("https://example.com/video.mp4", tmp_path, metadata)
    assert path.read_bytes() == b"abcdef"
    assert calls == ["https://example.com/video.mp4", "https://cdn.example/video.mp4"]
    assert metadata["bytes"] == 6
    assert not list(tmp_path.glob("*.part"))


def test_redirect_to_private_destination_is_rejected_and_cleaned(tmp_path, monkeypatch, public_dns):
    _mock_client(
        monkeypatch,
        lambda _request: httpx.Response(302, headers={"Location": "http://127.0.0.1/a.mp4"}),
    )
    with pytest.raises(URLIngestError, match="private|non-public"):
        DirectMediaProvider().acquire("https://example.com/video.mp4", tmp_path, {})
    assert list(tmp_path.iterdir()) == []


def test_declared_and_streamed_size_limits_cleanup(tmp_path, monkeypatch, public_dns):
    monkeypatch.setattr(settings, "max_upload_bytes", 4)
    for response in (
        httpx.Response(200, headers={"Content-Length": "5"}, content=b"12345"),
        httpx.Response(200, content=b"12345"),
    ):
        _mock_client(monkeypatch, lambda _request, value=response: value)
        with pytest.raises(URLIngestError, match="size limit"):
            DirectMediaProvider().acquire("https://example.com/video.mp4", tmp_path, {})
        assert list(tmp_path.iterdir()) == []


def test_timeout_is_retryable_and_partial_file_is_cleaned(tmp_path, monkeypatch, public_dns):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    _mock_client(monkeypatch, handler)
    with pytest.raises(URLIngestError) as caught:
        DirectMediaProvider().acquire("https://example.com/video.mp4", tmp_path, {})
    assert caught.value.retryable is True
    assert list(tmp_path.iterdir()) == []


def test_youtube_metadata_is_structured_and_duration_checked(monkeypatch, public_dns):
    provider = YouTubeProvider()
    payload = {
        "id": "abcdefghijk",
        "title": "Public test video",
        "duration": 30,
        "thumbnail": "https://i.ytimg.com/example.jpg",
        "uploader": "Test channel",
        "webpage_url": "https://www.youtube.com/watch?v=abcdefghijk",
    }
    monkeypatch.setattr(
        provider,
        "_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(payload), ""),
    )
    metadata = provider.inspect("https://youtu.be/abcdefghijk")
    assert metadata["external_id"] == "abcdefghijk"
    assert metadata["title"] == "Public test video"
    monkeypatch.setattr(settings, "max_duration", 10)
    with pytest.raises(URLIngestError, match="duration limit"):
        provider.inspect("https://youtu.be/abcdefghijk")


def test_private_youtube_error_is_non_retryable(monkeypatch):
    provider = YouTubeProvider()

    def blocked(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["yt-dlp"], stderr="Private video; login required")

    monkeypatch.setattr("app.url_ingest.subprocess.run", blocked)
    with pytest.raises(URLIngestError, match="private") as caught:
        provider.inspect("https://youtu.be/abcdefghijk")
    assert caught.value.retryable is False


def test_duplicate_provider_id_returns_existing_video(durable_client):
    provider = DirectMediaProvider()
    url = "https://example.com/video.mp4"
    folder = settings.data_dir / "00000000-0000-0000-0000-000000000001"
    folder.mkdir(parents=True)
    save_manifest(
        folder,
        {
            "id": folder.name,
            "filename": "existing.mp4",
            "status": "ready",
            "frames": [],
            "source_provider": "direct",
            "source_external_id": provider.external_id(url),
        },
    )
    response = durable_client.post("/videos/import-url", json={"url": url})
    assert response.status_code == 200
    assert response.json()["id"] == folder.name
    assert durable_client.get("/jobs").json() == []


def test_url_import_creates_durable_acquire_job(durable_client):
    response = durable_client.post(
        "/videos/import-url", json={"url": "https://example.com/new-video.webm"}
    )
    assert response.status_code == 202
    record = durable_client.get(f"/videos/{response.json()['id']}").json()
    assert record["source_type"] == "url"
    assert record["source_provider"] == "direct"
    assert record["job_status"] == "queued"
    assert record["stage"] == "fetching"


def test_invalid_downloaded_media_is_cleaned(tmp_path, monkeypatch, public_dns):
    video_id = "00000000-0000-0000-0000-000000000002"
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    folder = tmp_path / video_id
    folder.mkdir()
    provider = DirectMediaProvider()
    url = "https://example.com/bad.mp4"
    save_manifest(
        folder,
        {
            "id": video_id,
            "source_provider": "direct",
            "source_url": url,
            "source_external_id": provider.external_id(url),
        },
    )
    monkeypatch.setattr(
        provider, "inspect", lambda _url: {"external_id": provider.external_id(url)}
    )
    monkeypatch.setattr(
        provider,
        "acquire",
        lambda _url, target, _metadata: target / "source.mp4",
    )
    (folder / "source.mp4").write_bytes(b"invalid")
    monkeypatch.setattr("app.url_ingest.PROVIDERS", (provider,))
    monkeypatch.setattr(
        "app.video.inspect_video", lambda _path: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(URLIngestError, match="invalid") as caught:
        acquire_to_local(video_id)
    assert caught.value.retryable is False
    assert not (folder / "source.mp4").exists()


def test_acquire_job_hands_off_to_normal_ingest(durable_client, monkeypatch):
    video_id = "00000000-0000-0000-0000-000000000003"
    folder = settings.data_dir / video_id
    folder.mkdir(parents=True)
    save_manifest(folder, {"id": video_id})
    monkeypatch.setattr("app.url_ingest.acquire_to_local", lambda _video_id: None)
    result = execute({"video_id": video_id, "kind": "acquire"})
    assert result["next_kind"] == "ingest"
    enqueue(video_id, result["next_kind"])
    job = claim()
    assert job["kind"] == "ingest"


def test_non_retryable_failure_is_not_requeued_or_manually_retried(durable_client):
    job_id = enqueue("00000000-0000-0000-0000-000000000004", "acquire")
    job = claim()
    finish(job, "Unsupported media", retryable=False)
    assert claim() is None
    response = durable_client.post(f"/jobs/{job_id}/retry")
    assert response.status_code == 409
    assert "cannot be retried" in response.json()["detail"]


def test_existing_upload_records_upload_provenance(durable_client):
    response = durable_client.post("/videos?filename=fixture.mp4", content=b"invalid but queued")
    assert response.status_code == 202
    assert response.json()["source_type"] == "upload"


@pytest.mark.skipif(
    not os.getenv("SCENEMIND_REAL_URL"),
    reason="opt-in: set SCENEMIND_REAL_URL to a permitted public direct video",
)
def test_opt_in_real_direct_provider(tmp_path, monkeypatch):
    from app.video import inspect_video

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "min_free_disk_bytes", 0)
    provider, url = select_provider(os.environ["SCENEMIND_REAL_URL"])
    assert isinstance(provider, DirectMediaProvider)
    metadata = provider.inspect(url)
    path = provider.acquire(url, tmp_path, metadata)
    assert inspect_video(path)["duration"] > 0
    assert not list(tmp_path.glob("*.part"))
