"""On-demand local speech inference, separate from the video ingestion stage."""

import logging
import math
import subprocess
from functools import lru_cache
from threading import Lock

import imageio_ffmpeg
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Segment, Transcript, engine
from app.video import folder_for, read_manifest

router = APIRouter(prefix="/videos", tags=["speech"])
speech_lock = Lock()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def speech_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.speech_model,
        device=settings.model_device,
        compute_type=settings.speech_compute_type,
        download_root=settings.model_cache,
        cpu_threads=4,
    )


def infer_audio(path):
    segments, info = speech_model().transcribe(str(path), beam_size=5, vad_filter=True)
    # faster-whisper is lazy: consume the generator before releasing the worker slot.
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments if s.text.strip()
    ], info.language


def normalize_segments(segments, duration):
    """Validate model timestamps and keep them inside the playable video timeline."""
    normalized = []
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start <= end
            and start <= duration + 1
        ):
            raise ValueError("Invalid transcription timestamps")
        if start >= duration:
            continue
        normalized.append({**segment, "end": min(end, duration)})
    return normalized


def recover_speech():
    with Session(engine()) as session, session.begin():
        session.execute(
            update(Transcript)
            .where(Transcript.status == "processing")
            .values(status="failed", error="Transcription interrupted. Retry transcription.")
        )


def transcribe(video_id: str):
    folder = folder_for(video_id)
    audio = folder / "audio.wav"
    try:
        record = read_manifest(folder)
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(folder / record["source"]),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio),
            ],
            check=True,
            capture_output=True,
            timeout=settings.processing_timeout,
        )
        segments, language = infer_audio(audio)
        duration = record["metadata"]["duration"]
        segments = normalize_segments(segments, duration)
        with Session(engine()) as session, session.begin():
            session.execute(delete(Segment).where(Segment.video_id == video_id))
            session.add_all(Segment(video_id=video_id, **segment) for segment in segments)
            job = session.get(Transcript, video_id)
            job.status, job.language, job.error = "ready", language, None
    except Exception:
        logger.exception("Transcription failed for video %s", video_id)
        with Session(engine()) as session, session.begin():
            job = session.get(Transcript, video_id)
            job.status = "failed"
            job.error = "Transcription failed. Check audio, model availability and server logs."
    finally:
        try:
            audio.unlink(missing_ok=True)
        finally:
            speech_lock.release()


@router.post("/{video_id}/transcript", status_code=202)
def start_transcript(video_id: str, background: BackgroundTasks):
    folder = folder_for(video_id)
    video_id = folder.name
    if read_manifest(folder)["status"] != "ready":
        raise HTTPException(409, "Wait for video processing to finish.")
    if settings.durable_jobs:
        from app.jobs import enqueue

        job_id = enqueue(folder.name, "speech")
        return {"status": "queued", "job_id": job_id}
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        raise HTTPException(
            503, 'Install the backend speech extra: pip install -e "backend[speech]"'
        )
    if not speech_lock.acquire(blocking=False):
        raise HTTPException(429, "Another transcription is running.")
    try:
        with Session(engine()) as session, session.begin():
            session.merge(
                Transcript(
                    video_id=video_id,
                    status="processing",
                    model=settings.speech_model,
                    error=None,
                    language=None,
                )
            )
    except Exception:
        speech_lock.release()
        raise
    background.add_task(transcribe, video_id)
    return {"status": "processing", "video_id": video_id}


@router.get("/{video_id}/transcript")
def get_transcript(video_id: str, q: str = Query(default="", max_length=500)):
    folder = folder_for(video_id)
    read_manifest(folder)
    from app.jobs import pending

    queued = pending(folder.name, "speech")
    if queued:
        return {**queued, "segments": []}
    with Session(engine()) as session:
        job = session.get(Transcript, folder.name)
        if job is None:
            return {"status": "not_started", "segments": []}
        statement = select(Segment).where(Segment.video_id == folder.name).order_by(Segment.start)
        if q.strip():
            statement = statement.where(Segment.text.icontains(q.strip(), autoescape=True))
        segments = session.scalars(statement).all() if job.status == "ready" else []
        return {
            "status": job.status,
            "model": job.model,
            "language": job.language,
            "error": job.error,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
        }
