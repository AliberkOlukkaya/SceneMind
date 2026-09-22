"""Run: python -m app.worker. Persistent inference child; bounded job deadline."""

import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from threading import Thread

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Transcript, engine, migrate
from app.jobs import claim, finish, jobs


@contextmanager
def worker_lock(name=".worker.lock"):

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with (settings.data_dir / name).open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another SceneMind worker owns this data directory") from error
        yield


def execute(job):
    from app import speech, url_ingest, video, visual

    folder = video.folder_for(job["video_id"])
    if job["kind"] == "acquire":
        try:
            url_ingest.acquire_to_local(folder.name)
        except url_ingest.URLIngestError as error:
            return {"error": error.public_message, "retryable": error.retryable}
        return {"error": None, "retryable": True, "next_kind": "ingest"}
    elif job["kind"] == "ingest":
        video.ingestion_lock.acquire()
        video.process_video(folder, video.read_manifest(folder))
        state = video.read_manifest(folder)
    elif job["kind"] == "visual":
        visual.visual_lock.acquire()
        visual.save_status(folder, "processing")
        visual.build_index(folder.name)
        state = visual.index_status(folder, include_jobs=False)
    else:
        with Session(engine()) as session, session.begin():
            session.merge(
                Transcript(video_id=folder.name, status="processing", model=settings.speech_model)
            )
        speech.speech_lock.acquire()
        speech.transcribe(folder.name)
        with Session(engine()) as session:
            transcript = session.get(Transcript, folder.name)
            state = {"status": transcript.status, "error": transcript.error}
    return None if state["status"] == "ready" else state.get("error", "Processing failed")


def stop_process(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            process.kill()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join(5)


def await_result(parent, process, timeout):
    if parent.poll(timeout):
        try:
            return parent.recv(), process
        except EOFError:
            stop_process(process)
            return "Inference process exited", None
    stop_process(process)
    return "Job exceeded inference deadline", None


def timeout_for(kind):
    return {
        "acquire": settings.acquire_job_timeout,
        "ingest": settings.ingest_job_timeout,
        "speech": settings.speech_job_timeout,
        "visual": settings.visual_job_timeout,
    }.get(kind, settings.job_timeout)


def cleanup_job_artifacts(job):
    """Remove only disposable partial files after a killed or failed worker job."""
    from app.video import folder_for

    folder = folder_for(job["video_id"])
    targets = {
        "acquire": tuple(folder.glob("source*")) + tuple(folder.glob("*.ytdl")),
        "ingest": (folder / "frames.tmp", folder / "manifest.tmp"),
        "speech": (folder / "audio.wav",),
        "visual": (folder / "embeddings.tmp", folder / "index.tmp"),
    }.get(job["kind"], ())
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)


def child_loop(pipe, config):
    if os.name != "nt":
        os.setsid()

    def watch_parent():
        while mp.parent_process().is_alive():
            time.sleep(1)
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(os.getpid()), "/T", "/F"], capture_output=True, timeout=15
            )
        else:
            os.killpg(os.getpid(), signal.SIGKILL)
        os._exit(1)

    Thread(target=watch_parent, daemon=True).start()
    for key, value in config.items():
        setattr(settings, key, value)
    while True:
        job = pipe.recv()
        try:
            with worker_lock(".execution.lock"):
                pipe.send(execute(job))
        except Exception:
            import logging

            logging.exception("Worker job failed")
            pipe.send("Worker failed; inspect local logs.")


def main():
    context = mp.get_context("spawn")
    process = None
    with worker_lock():
        migrate()
        # Exclusive ownership guarantees the old child is no longer a writer after a clean
        # supervisor restart. On host/process-tree crash, replay is idempotent.
        with engine().connect() as conn:
            abandoned = [
                dict(r)
                for r in conn.execute(select(jobs).where(jobs.c.status == "running")).mappings()
            ]
        for job in abandoned:
            finish(job, "Worker interrupted")
        try:
            while True:
                job = claim()
                if job is None:
                    time.sleep(0.5)
                    continue
                if process is None:
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=child_loop, args=(child, settings.model_dump()), daemon=True
                    )
                    process.start()
                    child.close()
                try:
                    parent.send(job)
                except (BrokenPipeError, EOFError, OSError):
                    stop_process(process)
                    process = None
                    parent.close()
                    finish(job, "Inference process exited before dispatch")
                    continue
                result, process = await_result(parent, process, timeout_for(job["kind"]))
                if process is None:
                    parent.close()
                retryable = True
                next_kind = None
                if isinstance(result, dict):
                    error = result.get("error")
                    retryable = result.get("retryable", True)
                    next_kind = result.get("next_kind")
                else:
                    error = result
                if error:
                    cleanup_job_artifacts(job)
                finish(job, error, retryable=retryable)
                if not error and next_kind:
                    from app.jobs import enqueue

                    enqueue(job["video_id"], next_kind)
        finally:
            if process is not None:
                stop_process(process)


if __name__ == "__main__":
    main()
