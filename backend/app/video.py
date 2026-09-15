"""Local ingestion. Manifests are atomic; each video owns a UUID directory."""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

import cv2
import imageio_ffmpeg
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter(prefix="/videos", tags=["videos"])
ingestion_lock = Lock()
SUPPORTED = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
DISK_CHECK_INTERVAL = 64 * 1024 * 1024


def ensure_disk_space(path: Path, incoming_bytes: int = 0) -> None:
    """Keep a configured reserve while accepting bytes directly into local storage."""
    path.mkdir(parents=True, exist_ok=True)
    required = settings.min_free_disk_bytes + max(0, incoming_bytes)
    if shutil.disk_usage(path).free < required:
        raise HTTPException(507, "Insufficient disk space for this video.")


def recover_interrupted() -> None:
    """Single-process development server: unfinished jobs cannot survive a restart."""
    for path in settings.data_dir.glob("*/manifest.json"):
        record = read_manifest(path.parent)
        if record["status"] in {"queued", "processing"}:
            record.update(status="failed", error="Processing interrupted. Upload the video again.")
            save_manifest(path.parent, record)


def folder_for(video_id: str) -> Path:
    try:
        canonical = str(UUID(video_id))
    except ValueError:
        raise HTTPException(404, "Video not found") from None
    return settings.data_dir / canonical


def save_manifest(folder: Path, record: dict) -> None:
    temporary = folder / "manifest.tmp"
    temporary.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    temporary.replace(folder / "manifest.json")


def read_manifest(folder: Path) -> dict:
    try:
        return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(404, "Video not found") from None


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or not all(
            math.isfinite(x) and x > 0 for x in (fps, count, width, height)
        ):
            raise ValueError("The file does not contain a supported video stream.")
        duration = count / fps
        if duration > settings.max_duration or width * height > 3840 * 2160:
            raise ValueError("Video exceeds the duration or 4K resolution limit.")
        codec = int(capture.get(cv2.CAP_PROP_FOURCC))
        return {
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "codec": "".join(chr((codec >> (8 * i)) & 255) for i in range(4)),
        }
    finally:
        capture.release()


def process_video(folder: Path, record: dict) -> None:
    staged_frames = folder / "frames.tmp"
    try:
        record["status"] = "processing"
        record["stage"] = "preparing_video"
        record["frames"] = []
        record.pop("error", None)
        save_manifest(folder, record)
        source = folder / record["source"]
        record["metadata"] = inspect_video(source)
        interval = record["sampling_interval"]
        shutil.rmtree(staged_frames, ignore_errors=True)
        staged_frames.mkdir()
        # Select first frame, then the first frame at least interval seconds later.
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-nostdin",
                "-y",
                "-v",
                "info",
                "-i",
                str(source),
                "-an",
                "-vf",
                f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',showinfo,scale=480:-2",
                "-fps_mode",
                "vfr",
                "-frames:v",
                str(math.ceil(record["metadata"]["duration"] / interval)),
                "-q:v",
                "3",
                str(staged_frames / "%06d.jpg"),
            ],
            check=True,
            capture_output=True,
            timeout=settings.processing_timeout,
        )
        timestamps = re.findall(
            r"\bn:\s*\d+.*?\bpts_time:([\d.eE+-]+)", result.stderr.decode("utf-8", errors="replace")
        )
        images = sorted(staged_frames.glob("*.jpg"))
        if len(timestamps) < len(images):
            raise ValueError("Could not determine frame timestamps.")
        record["frames"] = [
            {
                "timestamp": round(float(timestamps[index]), 3),
                "thumbnail": f"/videos/{record['id']}/frames/{path.name}",
            }
            for index, path in enumerate(images)
        ]
        if not record["frames"]:
            raise ValueError("No frames could be decoded from this video.")
        frames = folder / "frames"
        shutil.rmtree(frames, ignore_errors=True)
        staged_frames.replace(frames)
        record["status"] = "ready"
        record["stage"] = "ready"
    except (ValueError, OSError, subprocess.SubprocessError, cv2.error) as error:
        record["status"] = "failed"
        record["error"] = (
            str(error) if isinstance(error, ValueError) else "Video processing failed."
        )
        record["stage"] = "failed"
        record["frames"] = []
        shutil.rmtree(staged_frames, ignore_errors=True)
        shutil.rmtree(folder / "frames", ignore_errors=True)
    finally:
        try:
            save_manifest(folder, record)
        finally:
            ingestion_lock.release()


@router.post("", status_code=202)
async def upload(
    request: Request,
    background: BackgroundTasks,
    filename: str = Query(min_length=1, max_length=200),
):
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(415, "Supported formats: MP4, MOV, WebM, MKV, AVI")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length header.") from None
        if declared_size > settings.max_upload_bytes:
            raise HTTPException(413, "Video exceeds the upload size limit.")
        ensure_disk_space(
            settings.data_dir,
            declared_size + int(declared_size * settings.processing_disk_headroom_ratio),
        )
    else:
        ensure_disk_space(settings.data_dir)
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(429, "Another video is being ingested. Try again shortly.")
    folder = settings.data_dir / str(uuid4())
    handed_off = False
    try:
        folder.mkdir(parents=True)
        size = 0
        next_disk_check = DISK_CHECK_INTERVAL
        source = f"source{suffix}"
        with (folder / source).open("wb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, "Video exceeds the upload size limit.")
                output.write(chunk)
                if size >= next_disk_check:
                    ensure_disk_space(settings.data_dir)
                    next_disk_check = size + DISK_CHECK_INTERVAL
        ensure_disk_space(
            settings.data_dir, int(size * settings.processing_disk_headroom_ratio)
        )
        if not size:
            raise HTTPException(400, "The video is empty.")
        record = {
            "id": folder.name,
            "filename": filename,
            "source": source,
            "bytes": size,
            "status": "queued",
            "stage": "preparing_video",
            "frames": [],
            "sampling_interval": settings.sampling_interval,
        }
        save_manifest(folder, record)
        if settings.durable_jobs:
            from app.jobs import enqueue

            enqueue(folder.name, "ingest")
            ingestion_lock.release()
        else:
            background.add_task(process_video, folder, record.copy())
        handed_off = True
        return record
    finally:
        if not handed_off:
            shutil.rmtree(folder, ignore_errors=True)
            ingestion_lock.release()


@router.get("/limits")
def limits():
    return {
        "max_upload_bytes": settings.max_upload_bytes,
        "max_duration_seconds": settings.max_duration,
    }


@router.get("")
def list_videos():
    return [
        get_video(path.parent.name)
        for path in sorted(
            settings.data_dir.glob("*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    ]


@router.get("/{video_id}")
def get_video(video_id: str):
    from app.jobs import pending

    record = read_manifest(folder_for(video_id))
    return {**record, **(pending(video_id, "ingest") or {})}


@router.get("/{video_id}/media")
def media(video_id: str):
    folder = folder_for(video_id)
    record = read_manifest(folder)
    return FileResponse(
        folder / record["source"], filename=record["filename"], content_disposition_type="inline"
    )


@router.get("/{video_id}/frames/{name}")
def thumbnail(video_id: str, name: str):
    folder = folder_for(video_id)
    if len(name) != 10 or not name[:6].isdigit() or name[6:] != ".jpg":
        raise HTTPException(404, "Frame not found")
    path = folder / "frames" / name
    if not path.is_file():
        raise HTTPException(404, "Frame not found")
    return FileResponse(path, media_type="image/jpeg")
