"""Run one checksum-bound temporal development/validation comparison arm.

Each arm uses a separate process. No model or media is loaded on import.
Annotations and all-occurrence audit must pass before any retrieval executes.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from ml.evaluation.temporal_protocol import verify_freeze  # noqa: E402
from ml.evaluation.temporal_retrieval import (  # noqa: E402
    XClipEncoder,
    aggregate,
    decode_window,
    first_rank,
    save_index,
    search,
    windows,
)


class Peak:
    def __init__(self):
        self.bytes = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                self.bytes = max(self.bytes, sum(p.memory_info().rss for p in
                                                [process, *process.children(recursive=True)]))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self.stop.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def persist(path, report):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def summarize(rows):
    groups = {"overall": rows}
    for key in ("category", "video_id", "difficulty"):
        for value in sorted({q[key] for q in rows}):
            groups[f"{key}:{value}"] = [q for q in rows if q[key] == value]
    groups["combined_temporal"] = [q for q in rows if q["category"] in
                                   {"HUMAN_ACTION", "OBJECT_INTERACTION", "TEMPORAL_EVENT"}]
    groups["combined_static"] = [q for q in rows if q["category"] in {"STATIC_OBJECT", "SCENE"}]
    return {key: aggregate([q["first_rank"] for q in subset]) for key, subset in groups.items()}


def candidate(manifest, work, report, save):
    import cv2

    start = time.perf_counter()
    encoder = XClipEncoder(ROOT / "data/models")
    report["model_load_seconds"] = time.perf_counter() - start
    config = report["temporal_config"]
    for source in manifest["sources"]:
        sid = source["source_id"]
        print(f"CANDIDATE {sid}", flush=True)
        clips = windows(source["duration_seconds"], config["length"], config["stride"])
        identity = {"media_sha256": source["media_sha256"], "config": config}
        index_path = work / f"{sid}.npz"
        vectors, patches, actual = [], [], []
        preprocess = embedding = 0
        capture = cv2.VideoCapture(str(ROOT / source["media_path"]))
        started = time.perf_counter()
        with Peak() as peak:
            try:
                for i, window in enumerate(clips):
                    stamp = time.perf_counter()
                    frames, positions = decode_window(capture, window)
                    preprocess += time.perf_counter() - stamp
                    stamp = time.perf_counter()
                    vector, patch = encoder.video(frames)
                    embedding += time.perf_counter() - stamp
                    vectors.append(vector)
                    patches.append(patch)
                    actual.append(positions)
                    if i % 100 == 0:
                        print(f"  {i}/{len(clips)} clips", flush=True)
            finally:
                capture.release()
            save_index(index_path, clips, np.concatenate(vectors), np.concatenate(patches),
                       identity=identity, actual_samples=actual)
        report["videos"][sid] = dict(duration_seconds=source["duration_seconds"],
                                     preprocess_seconds=preprocess,
                                     embedding_seconds=embedding,
                                     indexing_seconds=time.perf_counter() - started,
                                     temporal_clips=len(clips), processed_frames=len(clips) * 8,
                                     index_bytes=index_path.stat().st_size,
                                     indexing_peak_rss_bytes=peak.bytes)
        del vectors, patches, actual
        save()
        for query in (q for q in manifest["queries"] if q["video_id"] == sid):
            started = time.perf_counter()
            with Peak() as peak:
                results = search(index_path, query["query"], encoder, identity=identity, k=10)
            row = {**query, "results": results, "latency_seconds": time.perf_counter() - started,
                   "query_peak_rss_bytes": peak.bytes, "first_rank": first_rank(results, query["intervals"])}
            report["queries"].append(row)
            save()


def baseline(manifest, work, report, save):
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.encoder import encoder
    from app.main import app

    # Process-local storage instrumentation, no config files or algorithms changed.
    settings.data_dir = work / "videos"
    settings.database_url = f"sqlite:///{(work / 'runtime.db').as_posix()}"
    settings.durable_jobs = False
    settings.calibration_path = None
    if settings.sampling_interval != 5 or settings.qa_enabled:
        raise ValueError("Protected production configuration differs")
    if (settings.visual_model, settings.visual_revision) != (
            "openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"):
        raise ValueError("Production CLIP revision mismatch")
    started = time.perf_counter()
    encoder()
    report["model_load_seconds"] = time.perf_counter() - started
    headers = {"Authorization": f"Bearer {settings.auth_token}"} if settings.auth_token else {}
    with TestClient(app, headers=headers) as client:
        for source in manifest["sources"]:
            sid = source["source_id"]
            print(f"BASELINE {sid}", flush=True)
            path = ROOT / source["media_path"]
            # OGV is not a production upload extension. A remux, if needed, must
            # be prepared and hashed as shared evaluation media BEFORE freeze.
            with Peak() as peak:
                started = time.perf_counter()
                response = client.post("/videos", params={"filename": path.name}, content=path.read_bytes())
                response.raise_for_status()
                video_id = response.json()["id"]
                ingest = time.perf_counter() - started
                record = client.get(f"/videos/{video_id}").json()
                if record["status"] != "ready":
                    raise RuntimeError("Production ingest failed")
                started = time.perf_counter()
                response = client.post(f"/videos/{video_id}/index")
                response.raise_for_status()
                state = client.get(f"/videos/{video_id}/index").json()
                if state["status"] != "ready":
                    raise RuntimeError("Production visual indexing failed")
                embedding = time.perf_counter() - started
            folder = settings.data_dir / video_id
            report["videos"][sid] = dict(duration_seconds=source["duration_seconds"],
                                         preprocess_seconds=ingest, embedding_seconds=embedding,
                                         indexing_seconds=ingest + embedding,
                                         frames=len(record["frames"]),
                                         index_bytes=(folder / "embeddings.npy").stat().st_size,
                                         indexing_peak_rss_bytes=peak.bytes, video_id=video_id)
            save()
            for query in (q for q in manifest["queries"] if q["video_id"] == sid):
                started = time.perf_counter()
                response = client.get(f"/videos/{video_id}/search/visual", params={"q": query["query"], "k": 10})
                response.raise_for_status()
                results = response.json()["results"]
                report["queries"].append({**query, "results": results,
                                           "latency_seconds": time.perf_counter() - started,
                                           "first_rank": first_rank(results, query["intervals"])})
                save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--arm", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--length", type=int, choices=[4, 8], default=4)
    args = parser.parse_args()
    manifest = verify_freeze(ROOT, args.manifest, args.sha256)
    config = {"length": args.length, "stride": args.length // 2, "frames": 8}
    if manifest["split"] == "validation" and config != manifest["selected_config"]:
        raise SystemExit("Only development-selected frozen configuration allowed")
    name = args.arm if args.arm == "baseline" else f"candidate-{args.length}"
    work = ROOT / "data/temporal-v1" / manifest["split"] / name
    work.mkdir(parents=True, exist_ok=True)
    # Exclusive marker persists even after interruption; no silent selective reruns.
    with (work / "started.json").open("x", encoding="utf8") as marker:
        json.dump({"manifest_sha256": args.sha256}, marker)
    output = work / "run.json"
    report = dict(status="running", arm=args.arm, manifest_sha256=args.sha256,
                  temporal_config=config if args.arm == "candidate" else None, videos={}, queries=[])

    def save():
        persist(output, report)

    save()
    try:
        with Peak() as peak:
            (candidate if args.arm == "candidate" else baseline)(manifest, work, report, save)
        report.update(status="complete", process_tree_peak_rss_bytes=peak.bytes,
                      metrics=summarize(report["queries"]))
        latencies = [q["latency_seconds"] for q in report["queries"]]
        report["query_latency_seconds"] = dict(median=float(np.median(latencies)),
                                               p95=float(np.percentile(latencies, 95)))
        save()
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        save()
        raise
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
