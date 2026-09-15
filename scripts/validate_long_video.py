"""Measure streamed HTTP upload and durable ingest/speech/index processing."""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import psutil


class PeakSampler:
    def __init__(self, roots):
        self.roots = roots
        self.peak = 0
        self.running = False

    def sample(self):
        total = 0
        seen = set()
        for root in self.roots():
            try:
                processes = [root, *root.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for process in processes:
                if process.pid in seen:
                    continue
                seen.add(process.pid)
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        self.peak = max(self.peak, total)

    def __enter__(self):
        self.running = True

        def run():
            while self.running:
                self.sample()
                time.sleep(0.05)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.running = False
        self.thread.join()
        self.sample()


def process_tree_rss(root):
    try:
        return sum(
            process.memory_info().rss
            for process in [root, *root.children(recursive=True)]
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def file_chunks(path: Path, chunk_size: int = 1024 * 1024):
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            yield chunk


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def stop_tree(process: subprocess.Popen) -> None:
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in reversed(children):
            child.kill()
        root.kill()
        psutil.wait_procs([root, *children], timeout=20)
    except psutil.NoSuchProcess:
        pass


def wait_health(client: httpx.Client, base_url: str, processes, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("Server or worker exited during startup")
        try:
            if client.get(f"{base_url}/health").status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError("SceneMind server did not become healthy")


def wait_stage(client, base_url, path, timeout, processes):
    start = time.perf_counter()
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("Server or worker exited during processing")
        response = client.get(f"{base_url}{path}")
        response.raise_for_status()
        last = response.json()
        if last["status"] == "ready":
            return time.perf_counter() - start, last
        if last["status"] == "failed":
            raise RuntimeError(last.get("error", "Processing failed"))
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {path}: {last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--skip-speech", action="store_true")
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)

    repo = Path(__file__).resolve().parents[1]
    python = repo / ".venv" / "Scripts" / "python.exe"
    data_dir = args.root / "videos"
    database = (args.root / "scenemind.db").as_posix()
    environment = {
        **os.environ,
        "SCENEMIND_DATA_DIR": str(data_dir),
        "SCENEMIND_DATABASE_URL": f"sqlite:///{database}",
        "SCENEMIND_MODEL_CACHE": str((repo / "data" / "models").resolve()),
        "SCENEMIND_DURABLE_JOBS": "true",
        "SCENEMIND_MAX_UPLOAD_BYTES": str(1024 * 1024 * 1024),
        "SCENEMIND_MAX_DURATION": "3600",
        "SCENEMIND_PROCESSING_TIMEOUT": "1800",
        "SCENEMIND_INGEST_JOB_TIMEOUT": "1800",
        "SCENEMIND_SPEECH_JOB_TIMEOUT": "7200",
        "SCENEMIND_VISUAL_JOB_TIMEOUT": "3600",
    }
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    server_log = (args.root / "server.log").open("w", encoding="utf-8")
    worker_log = (args.root / "worker.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [str(python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(args.port)],
        cwd=repo / "backend", env=environment, stdout=server_log, stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )
    worker = subprocess.Popen(
        [str(python), "-m", "app.worker"], cwd=repo / "backend", env=environment,
        stdout=worker_log, stderr=subprocess.STDOUT, creationflags=creation_flags,
    )
    processes = [server, worker]
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        with httpx.Client(timeout=httpx.Timeout(120, write=1800)) as client:
            wait_health(client, base_url, processes)
            server_process = psutil.Process(server.pid)
            worker_process = psutil.Process(worker.pid)
            baseline_server_rss = process_tree_rss(server_process)
            processing_memory = PeakSampler(lambda: [worker_process])
            processing_memory.__enter__()
            upload_started = time.perf_counter()
            with PeakSampler(lambda: [server_process]) as upload_memory:
                response = client.post(
                    f"{base_url}/videos",
                    params={"filename": args.source.name},
                    content=file_chunks(args.source),
                    headers={"Content-Type": "application/octet-stream"},
                )
            upload_seconds = time.perf_counter() - upload_started
            response.raise_for_status()
            video_id = response.json()["id"]
            ingest_seconds, record = wait_stage(
                client, base_url, f"/videos/{video_id}", 1860, processes
            )
            speech_seconds = None
            transcript = {"status": "not_started", "segments": [], "language": None}
            if not args.skip_speech:
                response = client.post(f"{base_url}/videos/{video_id}/transcript")
                response.raise_for_status()
                speech_seconds, transcript = wait_stage(
                    client, base_url, f"/videos/{video_id}/transcript", 7260, processes
                )
            response = client.post(f"{base_url}/videos/{video_id}/index")
            response.raise_for_status()
            visual_seconds, index = wait_stage(
                client, base_url, f"/videos/{video_id}/index", 3660, processes
            )
            processing_memory.__exit__()
            segments = transcript.get("segments", [])
            continuity_valid = all(
                0 <= segment["start"] <= segment["end"] <= record["metadata"]["duration"] + 1
                for segment in segments
            ) and all(
                left["start"] <= right["start"] for left, right in zip(segments, segments[1:])
            )
            folder = data_dir / video_id
            parts = {
                "source_bytes": (folder / record["source"]).stat().st_size,
                "frames_bytes": tree_bytes(folder / "frames"),
                "embeddings_bytes": (folder / "embeddings.npy").stat().st_size,
                "database_bytes": Path(args.root / "scenemind.db").stat().st_size,
                "logs_bytes": (args.root / "server.log").stat().st_size
                + (args.root / "worker.log").stat().st_size,
                "temporary_bytes": sum(
                    item.stat().st_size for item in folder.rglob("*")
                    if item.is_file() and (item.suffix == ".tmp" or item.name == "audio.wav")
                ),
            }
            result = {
                "schema_version": "1.0.0",
                "kind": "LONG_VIDEO_INFRASTRUCTURE_VALIDATION",
                "source_name": args.source.name,
                "source_bytes": args.source.stat().st_size,
                "upload": {
                    "status_code": 202,
                    "seconds": upload_seconds,
                    "chunk_bytes": 1024 * 1024,
                    "baseline_server_rss_bytes": baseline_server_rss,
                    "peak_server_rss_bytes": upload_memory.peak,
                    "added_peak_server_rss_bytes": max(0, upload_memory.peak - baseline_server_rss),
                },
                "video": {
                    "id": video_id,
                    "status": record["status"],
                    "duration_seconds": record["metadata"]["duration"],
                    "frames": len(record["frames"]),
                },
                "processing": {
                    "stage_seconds": {
                        "ingest": ingest_seconds,
                        "speech": speech_seconds,
                        "visual": visual_seconds,
                    },
                    "total_seconds": ingest_seconds + (speech_seconds or 0) + visual_seconds,
                    "peak_worker_tree_rss_bytes": processing_memory.peak,
                    "steady_worker_tree_rss_bytes": process_tree_rss(worker_process),
                },
                "speech": {
                    "status": transcript["status"],
                    "language": transcript.get("language"),
                    "segments": len(segments),
                    "timestamp_continuity_valid": continuity_valid,
                },
                "visual": {"status": index["status"]},
                "storage": {**parts, "total_bytes": tree_bytes(args.root)},
                "cleanup": {"temporary_bytes_after_completion": parts["temporary_bytes"]},
            }
    finally:
        for process in reversed(processes):
            stop_tree(process)
        server_log.close()
        worker_log.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
