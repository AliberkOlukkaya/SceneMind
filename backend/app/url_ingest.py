"""Secure URL acquisition that hands local media to the existing ingest pipeline."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import socket
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from uuid import uuid4

import httpx
import imageio_ffmpeg
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter(prefix="/videos", tags=["URL ingestion"])
provider_lock = Lock()
SUPPORTED_MEDIA = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
REDIRECT_CODES = {301, 302, 303, 307, 308}
BLOCKED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.aws.internal",
}
BLOCKED_IPS = {"168.63.129.16"}  # Azure platform virtual IP; never fetch as user media.


class ImportRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


class URLIngestError(Exception):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.public_message = message
        self.retryable = retryable


def _resolved_addresses(hostname: str, port: int) -> set[str]:
    try:
        records = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise URLIngestError("The URL hostname could not be resolved.", retryable=True) from error
    addresses = {record[4][0].split("%", 1)[0] for record in records}
    if not addresses:
        raise URLIngestError("The URL hostname could not be resolved.", retryable=True)
    return addresses


def validate_public_url(
    url: str, *, expected_addresses: set[str] | None = None
) -> tuple[str, set[str]]:
    """Normalize an HTTP(S) URL and reject destinations unsafe for server-side fetches."""
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as error:
        raise URLIngestError("Enter a valid video URL.") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise URLIngestError("Only HTTP and HTTPS video URLs are supported.")
    if parsed.username is not None or parsed.password is not None:
        raise URLIngestError("URLs containing usernames or passwords are not supported.")
    if not parsed.hostname:
        raise URLIngestError("Enter a URL with a valid hostname.")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in BLOCKED_HOSTS or hostname.endswith(".localhost"):
        raise URLIngestError("Local and private network URLs are not allowed.")
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as error:
        raise URLIngestError("The URL hostname is invalid.") from error
    port = port or (443 if parsed.scheme.lower() == "https" else 80)
    addresses = _resolved_addresses(hostname, port)
    for address in addresses:
        if address in BLOCKED_IPS:
            raise URLIngestError("Cloud metadata and platform service URLs are not allowed.")
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as error:
            raise URLIngestError("The URL resolved to an invalid network address.") from error
        if not ip.is_global:
            raise URLIngestError("Local, private, reserved, and non-public URLs are not allowed.")
    if expected_addresses is not None and addresses != expected_addresses:
        raise URLIngestError("The URL destination changed during validation.")
    host = f"[{hostname}]" if ":" in hostname else hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    normalized = urlunsplit((parsed.scheme.lower(), host, parsed.path or "/", parsed.query, ""))
    return normalized, addresses


def _safe_name(value: str, fallback: str) -> str:
    value = Path(unquote(value)).name
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" ._")
    return (value[:160] or fallback)[:200]


def _reserve_disk(incoming_bytes: int = 0) -> None:
    from app.video import ensure_disk_space

    try:
        ensure_disk_space(settings.data_dir, incoming_bytes)
    except HTTPException as error:
        raise URLIngestError("There is not enough disk space to import this video.") from error


class URLIngestProvider(ABC):
    name: str

    @abstractmethod
    def supports(self, url: str) -> bool: ...

    @abstractmethod
    def external_id(self, url: str) -> str: ...

    @abstractmethod
    def inspect(self, url: str) -> dict[str, Any]: ...

    @abstractmethod
    def acquire(self, url: str, folder: Path, metadata: dict[str, Any]) -> Path: ...

    def cleanup(self, folder: Path) -> None:
        for path in folder.glob("source*"):
            if path.is_file():
                path.unlink(missing_ok=True)


class DirectMediaProvider(URLIngestProvider):
    name = "direct"

    def supports(self, url: str) -> bool:
        return Path(urlsplit(url).path).suffix.lower() in SUPPORTED_MEDIA

    def external_id(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def inspect(self, url: str) -> dict[str, Any]:
        return {
            "source_url": url,
            "external_id": self.external_id(url),
            "title": _safe_name(Path(urlsplit(url).path).name, "Imported video"),
        }

    def acquire(self, url: str, folder: Path, metadata: dict[str, Any]) -> Path:
        suffix = Path(urlsplit(url).path).suffix.lower()
        temporary = folder / f"source{suffix}.part"
        final = folder / f"source{suffix}"
        current = url
        try:
            timeout = httpx.Timeout(
                settings.url_read_timeout,
                connect=settings.url_connect_timeout,
                write=settings.url_read_timeout,
                pool=settings.url_connect_timeout,
            )
            with _direct_http_client(timeout) as client:
                for redirect_count in range(settings.url_redirect_limit + 1):
                    current, addresses = validate_public_url(current)
                    try:
                        with client.stream(
                            "GET",
                            current,
                            headers={
                                "Accept-Encoding": "identity",
                                "User-Agent": (
                                    "SceneMind/1.0 "
                                    "(https://github.com/AliberkOlukkaya/SceneMind; URL ingestion)"
                                ),
                            },
                        ) as response:
                            validate_public_url(current, expected_addresses=addresses)
                            if response.status_code in REDIRECT_CODES:
                                location = response.headers.get("location")
                                if not location:
                                    raise URLIngestError(
                                        "The media URL returned an invalid redirect."
                                    )
                                if redirect_count >= settings.url_redirect_limit:
                                    raise URLIngestError(
                                        "The media URL exceeded the redirect limit."
                                    )
                                current = urljoin(current, location)
                                continue
                            if response.status_code >= 400:
                                retryable = (
                                    response.status_code in {408, 425, 429}
                                    or response.status_code >= 500
                                )
                                raise URLIngestError(
                                    "The remote video is temporarily unavailable."
                                    if retryable
                                    else "The remote video could not be accessed.",
                                    retryable=retryable,
                                )
                            declared = response.headers.get("content-length")
                            if declared:
                                try:
                                    declared_size = int(declared)
                                except ValueError:
                                    declared_size = 0
                                if declared_size > settings.max_upload_bytes:
                                    raise URLIngestError("The remote video exceeds the size limit.")
                                _reserve_disk(
                                    declared_size
                                    + int(declared_size * settings.processing_disk_headroom_ratio),
                                )
                            size = 0
                            next_disk_check = 64 * 1024 * 1024
                            with temporary.open("wb") as output:
                                for chunk in response.iter_bytes(1024 * 1024):
                                    size += len(chunk)
                                    if size > settings.max_upload_bytes:
                                        raise URLIngestError(
                                            "The remote video exceeds the size limit."
                                        )
                                    output.write(chunk)
                                    if size >= next_disk_check:
                                        _reserve_disk()
                                        next_disk_check = size + 64 * 1024 * 1024
                            if not size:
                                raise URLIngestError("The remote video was empty.")
                            _reserve_disk(int(size * settings.processing_disk_headroom_ratio))
                            temporary.replace(final)
                            metadata.update(bytes=size, final_url=current)
                            return final
                    except httpx.TimeoutException as error:
                        raise URLIngestError(
                            "The remote video request timed out.", retryable=True
                        ) from error
                    except httpx.NetworkError as error:
                        raise URLIngestError(
                            "The remote video could not be downloaded due to a network error.",
                            retryable=True,
                        ) from error
        finally:
            temporary.unlink(missing_ok=True)
        raise URLIngestError("The remote video could not be downloaded.")


def _direct_http_client(timeout: httpx.Timeout) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=False)


class YouTubeProvider(URLIngestProvider):
    name = "youtube"
    hosts = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}

    def supports(self, url: str) -> bool:
        return (urlsplit(url).hostname or "").rstrip(".").lower() in self.hosts

    def external_id(self, url: str) -> str:
        parsed = urlsplit(url)
        if parsed.hostname == "youtu.be":
            candidate = parsed.path.strip("/").split("/", 1)[0]
        elif parsed.path.startswith("/shorts/"):
            candidate = parsed.path.split("/", 3)[2]
        else:
            from urllib.parse import parse_qs

            candidate = parse_qs(parsed.query).get("v", [""])[0]
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", candidate):
            raise URLIngestError("Enter a supported public YouTube video URL.")
        return candidate

    def _command(self, *arguments: str, timeout: int) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [sys.executable, "-m", "yt_dlp", *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise URLIngestError("The YouTube request timed out.", retryable=True) from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or "").casefold()
            blocked = any(
                word in detail for word in ("private", "members-only", "login", "unavailable")
            )
            raise URLIngestError(
                "This YouTube video is private, blocked, or unavailable."
                if blocked
                else "YouTube could not provide this video.",
                retryable=not blocked,
            ) from error

    def _download(self, arguments: list[str], folder: Path) -> None:
        command = [sys.executable, "-m", "yt_dlp", *arguments]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + settings.url_acquire_timeout
        try:
            while process.poll() is None:
                size = sum(path.stat().st_size for path in folder.glob("source*") if path.is_file())
                if size > settings.max_upload_bytes:
                    process.kill()
                    process.communicate()
                    raise URLIngestError("The remote video exceeds the size limit.")
                if time.monotonic() >= deadline:
                    process.kill()
                    process.communicate()
                    raise URLIngestError("The YouTube request timed out.", retryable=True)
                time.sleep(0.1)
            stdout, stderr = process.communicate()
            if process.returncode:
                detail = (stderr or stdout or "").casefold()
                blocked = any(
                    word in detail for word in ("private", "members-only", "login", "unavailable")
                )
                raise URLIngestError(
                    "This YouTube video is private, blocked, or unavailable."
                    if blocked
                    else "YouTube could not provide this video.",
                    retryable=not blocked,
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def inspect(self, url: str) -> dict[str, Any]:
        result = self._command(
            "--dump-single-json",
            "--skip-download",
            "--no-playlist",
            "--no-warnings",
            "--socket-timeout",
            str(int(settings.url_connect_timeout)),
            url,
            timeout=min(90, settings.url_acquire_timeout),
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise URLIngestError(
                "YouTube returned invalid video metadata.", retryable=True
            ) from error
        duration = payload.get("duration")
        if duration is not None and float(duration) > settings.max_duration:
            raise URLIngestError("The remote video exceeds the duration limit.")
        approximate_size = payload.get("filesize") or payload.get("filesize_approx")
        if approximate_size is not None and int(approximate_size) > settings.max_upload_bytes:
            raise URLIngestError("The remote video exceeds the size limit.")
        video_id = str(payload.get("id") or self.external_id(url))
        return {
            "source_url": payload.get("webpage_url") or url,
            "external_id": video_id,
            "title": str(payload.get("title") or f"YouTube {video_id}")[:200],
            "duration": duration,
            "thumbnail_url": payload.get("thumbnail"),
            "uploader": str(payload.get("uploader") or "")[:200] or None,
            "estimated_bytes": int(approximate_size) if approximate_size is not None else None,
        }

    def acquire(self, url: str, folder: Path, metadata: dict[str, Any]) -> Path:
        self.cleanup(folder)
        estimated = metadata.get("estimated_bytes") or 0
        _reserve_disk(
            estimated + int(estimated * settings.processing_disk_headroom_ratio),
        )
        try:
            self._download(
                [
                    "--no-playlist",
                    "--no-warnings",
                    "--no-progress",
                    "--socket-timeout",
                    str(int(settings.url_connect_timeout)),
                    "--max-filesize",
                    str(settings.max_upload_bytes),
                    "--format",
                    "bv*[height<=720]+ba/b[height<=720]/b",
                    "--merge-output-format",
                    "mp4",
                    "--ffmpeg-location",
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "--paths",
                    str(folder),
                    "--output",
                    "source.%(ext)s",
                    url,
                ],
                folder,
            )
        except Exception:
            self.cleanup(folder)
            raise
        candidates = [
            path
            for path in folder.glob("source.*")
            if path.suffix.lower() in SUPPORTED_MEDIA and path.is_file()
        ]
        if not candidates:
            self.cleanup(folder)
            raise URLIngestError("YouTube did not return a supported video file.")
        selected = max(candidates, key=lambda path: path.stat().st_size)
        for path in candidates:
            if path != selected:
                path.unlink(missing_ok=True)
        size = selected.stat().st_size
        if size > settings.max_upload_bytes:
            self.cleanup(folder)
            raise URLIngestError("The remote video exceeds the size limit.")
        metadata["bytes"] = size
        return selected


PROVIDERS: tuple[URLIngestProvider, ...] = (YouTubeProvider(), DirectMediaProvider())


def select_provider(url: str) -> tuple[URLIngestProvider, str]:
    normalized, _ = validate_public_url(url)
    for provider in PROVIDERS:
        if provider.supports(normalized):
            return provider, normalized
    raise URLIngestError("This URL is not a supported direct video or YouTube URL.")


def _find_duplicate(provider: URLIngestProvider, external_id: str) -> dict[str, Any] | None:
    from app.video import read_manifest

    for manifest in settings.data_dir.glob("*/manifest.json"):
        record = read_manifest(manifest.parent)
        if (
            record.get("source_provider") == provider.name
            and record.get("source_external_id") == external_id
        ):
            return record
    return None


def acquire_to_local(video_id: str) -> None:
    """Acquire and validate remote media; durable callers enqueue normal ingest next."""
    from app.video import folder_for, inspect_video, read_manifest, save_manifest

    folder = folder_for(video_id)
    record = read_manifest(folder)
    provider = next((item for item in PROVIDERS if item.name == record["source_provider"]), None)
    if provider is None:
        raise URLIngestError("The URL provider is no longer available.")
    validated_url, _ = validate_public_url(record["source_url"])
    record["source_url"] = validated_url
    record.update(status="processing", stage="fetching")
    save_manifest(folder, record)
    metadata = provider.inspect(record["source_url"])
    if metadata.get("external_id") != record["source_external_id"]:
        raise URLIngestError("The remote provider returned a different video.")
    path = provider.acquire(record["source_url"], folder, metadata)
    try:
        record.update(status="processing", stage="validating")
        save_manifest(folder, record)
        media = inspect_video(path)
        size = path.stat().st_size
        if size > settings.max_upload_bytes:
            raise URLIngestError("The remote video exceeds the size limit.")
        filename = _safe_name(
            f"{metadata.get('title') or 'Imported video'}{path.suffix}", f"imported{path.suffix}"
        )
        record.update(
            filename=filename,
            source=path.name,
            bytes=size,
            metadata=media,
            provider_metadata={
                key: metadata.get(key)
                for key in ("title", "duration", "thumbnail_url", "uploader", "final_url")
                if metadata.get(key) is not None
            },
            stage="preparing_video",
        )
        save_manifest(folder, record)
    except ValueError as error:
        provider.cleanup(folder)
        raise URLIngestError(
            "The downloaded media is invalid or exceeds SceneMind limits."
        ) from error
    except Exception:
        provider.cleanup(folder)
        raise


def acquire_and_process_inline(video_id: str) -> None:
    from app.video import folder_for, ingestion_lock, process_video, read_manifest, save_manifest

    folder = folder_for(video_id)
    try:
        acquire_to_local(video_id)
        ingestion_lock.acquire()
        process_video(folder, read_manifest(folder))
    except URLIngestError as error:
        record = read_manifest(folder)
        record.update(status="failed", stage="failed", error=error.public_message)
        save_manifest(folder, record)


@router.post("/import-url", status_code=202)
def import_url(payload: ImportRequest, background: BackgroundTasks, response: Response):
    try:
        provider, normalized = select_provider(payload.url)
        external_id = provider.external_id(normalized)
    except URLIngestError as error:
        raise HTTPException(400, error.public_message) from None

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with provider_lock:
        duplicate = _find_duplicate(provider, external_id)
        if duplicate is not None:
            response.status_code = 200
            return duplicate
        folder = settings.data_dir / str(uuid4())
        folder.mkdir(parents=True)
        record = {
            "id": folder.name,
            "filename": "Importing video",
            "bytes": 0,
            "status": "queued",
            "stage": "fetching",
            "frames": [],
            "sampling_interval": settings.sampling_interval,
            "source_type": "url",
            "source_provider": provider.name,
            "source_url": normalized,
            "source_external_id": external_id,
        }
        from app.video import save_manifest

        save_manifest(folder, record)
    try:
        if settings.durable_jobs:
            from app.jobs import enqueue

            enqueue(folder.name, "acquire")
        else:
            background.add_task(acquire_and_process_inline, folder.name)
        return record
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
