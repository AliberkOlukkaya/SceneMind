"""Apply a frozen bounded policy to actual human-reviewed small-object frames."""

import argparse
import json
from pathlib import Path

import numpy as np

from ml.experiments.bounded_secondary_sampling.policy import (
    CoarseCandidate,
    diversify,
    local_windows,
)
from ml.experiments.small_object_ablation.schema import VisibilityManifest

ROOT = Path(__file__).resolve().parents[3]


def load_frames(source_id: str, interval: int):
    folder = ROOT / f"data/small-object-ablation/sampling/{interval}s" / source_id
    record = json.loads((folder / "timestamps.json").read_text(encoding="utf-8"))
    return record, [folder / row["file"] for row in record["frames"]]


def run(manifest_path: Path, radius: float, top_k: int, spacing: float) -> dict:
    from app.encoder import encoder

    manifest = VisibilityManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    clip = encoder()
    rows = []
    for event in manifest.events:
        coarse_record, coarse_paths = load_frames(event.source_id, 5)
        dense_record, dense_paths = load_frames(event.source_id, 2)
        coarse_vectors = clip.images(coarse_paths)
        dense_vectors = clip.images(dense_paths)
        text = clip.text(event.query)[0]
        coarse_scores = coarse_vectors @ text
        order = np.argsort(-coarse_scores)
        candidates = [
            CoarseCandidate(
                coarse_record["frames"][index]["timestamp"], float(coarse_scores[index])
            )
            for index in order
        ]
        selected_coarse = diversify(candidates, min(top_k, len(candidates)), spacing)
        windows = local_windows(selected_coarse, radius, dense_record["duration"])
        bounded_indices = [
            index for index, frame in enumerate(dense_record["frames"])
            if any(start - 0.05 <= frame["timestamp"] <= end + 0.05 for start, end in windows)
        ]
        bounded_scores = dense_vectors[bounded_indices] @ text if bounded_indices else np.array([])
        bounded_ranked = [
            dense_record["frames"][bounded_indices[index]]["timestamp"]
            for index in np.argsort(-bounded_scores)[:5]
        ]
        dense_scores = dense_vectors @ text
        global_ranked = [
            dense_record["frames"][index]["timestamp"]
            for index in np.argsort(-dense_scores)[:5]
        ]
        visibility = {
            row.timestamp: row.visibility for row in event.observations
        }
        evidence = {
            timestamp for timestamp, status in visibility.items()
            if status in {"visible", "partially_cropped"}
        }
        bounded_timestamps = {
            dense_record["frames"][index]["timestamp"] for index in bounded_indices
        }
        rows.append({
            "event_id": event.event_id, "split": event.split,
            "coarse_top_k": [row.timestamp for row in selected_coarse],
            "windows": windows, "secondary_frames": len(bounded_indices),
            "window_contains_visible_evidence": bool(bounded_timestamps & evidence),
            "bounded_clip_top5": bounded_ranked,
            "bounded_clip_hits_visible_evidence": bool(set(bounded_ranked) & evidence),
            "global_2s_top5": global_ranked,
            "global_2s_hits_visible_evidence": bool(set(global_ranked) & evidence),
        })
    summary = {}
    for split in ("all", "calibration", "heldout"):
        selected = rows if split == "all" else [row for row in rows if row["split"] == split]
        summary[split] = {
            "events": len(selected),
            "secondary_window_recall": sum(
                row["window_contains_visible_evidence"] for row in selected
            ) / len(selected),
            "bounded_clip_recall_at_5": sum(
                row["bounded_clip_hits_visible_evidence"] for row in selected
            ) / len(selected),
            "global_2s_clip_recall_at_5": sum(
                row["global_2s_hits_visible_evidence"] for row in selected
            ) / len(selected),
        }
    return {
        "schema_version": "1.0.0",
        "policy": {"radius_seconds": radius, "top_k": top_k, "spacing_seconds": spacing},
        "summary": summary, "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "ml/experiments/small_object_ablation/visibility_v1.json")
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.radius, args.top_k, args.spacing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
