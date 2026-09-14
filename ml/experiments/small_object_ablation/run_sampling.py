"""Measure frozen 5s/2s/1s extraction artifacts and CLIP candidate behavior."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .schema import VisibilityManifest

ROOT = Path(__file__).resolve().parents[3]


def percentile(values: list[float], value: int) -> float:
    return float(np.percentile(values, value))


def run(manifest_path: Path, search_loops: int = 10) -> dict:
    from app.encoder import encoder, rank_vectors

    manifest = VisibilityManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    model = encoder()
    by_source = {source.source_id: source for source in manifest.sources}
    result = {}
    for interval in (5, 2, 1):
        all_paths = []
        source_rows = {}
        extraction_seconds = 0.0
        jpeg_bytes = 0
        for source_id in by_source:
            folder = ROOT / f"data/small-object-ablation/sampling/{interval}s" / source_id
            record = json.loads((folder / "timestamps.json").read_text(encoding="utf-8"))
            paths = [folder / row["file"] for row in record["frames"]]
            source_rows[source_id] = (record["frames"], paths)
            all_paths.extend(paths)
            extraction_seconds += record["extraction_seconds"]
            jpeg_bytes += sum(path.stat().st_size for path in paths)
        started = time.perf_counter()
        vectors = model.images(all_paths)
        clip_index_seconds = time.perf_counter() - started
        offset = 0
        source_vectors = {}
        for source_id, (_, paths) in source_rows.items():
            source_vectors[source_id] = vectors[offset:offset + len(paths)]
            offset += len(paths)
        candidate_hits = []
        text_vectors = {}
        for event in manifest.events:
            frames, _ = source_rows[event.source_id]
            text_vectors[event.event_id] = model.text(event.query)
            _, indices = rank_vectors(source_vectors[event.source_id], text_vectors[event.event_id], 5)
            visible = {
                row.timestamp for row in event.observations
                if row.visibility in {"visible", "partially_cropped"}
            }
            top5 = [frames[index]["timestamp"] for index in indices]
            candidate_hits.append({
                "event_id": event.event_id, "split": event.split, "top5_timestamps": top5,
                "hit": any(timestamp in visible for timestamp in top5),
            })
        search_times = []
        for _ in range(search_loops):
            for event in manifest.events:
                started = time.perf_counter()
                query = model.text(event.query)
                rank_vectors(source_vectors[event.source_id], query, 5)
                search_times.append(time.perf_counter() - started)
        result[str(interval)] = {
            "frames_indexed": len(all_paths), "jpeg_bytes": jpeg_bytes,
            "embedding_bytes": int(vectors.nbytes),
            "extraction_seconds": extraction_seconds,
            "clip_index_seconds": clip_index_seconds,
            "clip_search_median_seconds": statistics.median(search_times),
            "clip_search_p95_seconds": percentile(search_times, 95),
            "small_object_candidate_recall_at_5": sum(row["hit"] for row in candidate_hits) / len(candidate_hits),
            "candidate_rows": candidate_hits,
        }
    baseline = result["5"]
    for row in result.values():
        row["frame_growth_vs_5s"] = row["frames_indexed"] / baseline["frames_indexed"]
        row["jpeg_growth_vs_5s"] = row["jpeg_bytes"] / baseline["jpeg_bytes"]
        row["embedding_growth_vs_5s"] = row["embedding_bytes"] / baseline["embedding_bytes"]
        row["extraction_growth_vs_5s"] = row["extraction_seconds"] / baseline["extraction_seconds"]
        row["clip_index_growth_vs_5s"] = row["clip_index_seconds"] / baseline["clip_index_seconds"]
    return {"schema_version": "1.0.0", "search_loops": search_loops, "intervals": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("visibility_v1.json"))
    parser.add_argument("--search-loops", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.manifest, args.search_loops)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
