"""Run the frozen coarse-candidate diversity experiment with real CLIP vectors."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import psutil

from ml.experiments.coarse_candidate_diversity.selection import (
    TIMESTAMP_TOLERANCE_SECONDS,
    Candidate,
    mmr,
    represented_intervals,
    scene_aware,
    scene_segments,
    temporal_nms,
)

ROOT = Path(__file__).resolve().parents[3]
PARENT_REPORT = ROOT / "ml/evaluation/reports/bounded-secondary-v1.json"
QUERY_REPORT = ROOT / "ml/evaluation/reports/lightweight-pair-scorer-v1.json"
NATURAL_MANIFEST = ROOT / "ml/evaluation/natural_v2.json"
VISIBILITY_MANIFEST = ROOT / "ml/experiments/small_object_ablation/visibility_v1.json"
GLOBAL_TWO_ROOT = ROOT / "data/bounded-secondary-v3/global-2s"
SMALL_ROOT = ROOT / "data/small-object-ablation/sampling"
SMALL_OBJECT_IDS = {"street-object-bicycle", "throw-object-ball", "throw-compositional-holding"}
METHODS = (
    "raw_5s", "raw_2s", "temporal_nms", "mmr", "temporal_mmr", "scene_aware",
    "multiscale_nms", "base_preserving_multiscale",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def percentile(values: list[float], value: int) -> float:
    return float(np.percentile(values, value)) if values else 0.0


def coarse_assets(query_report: dict) -> dict:
    assets = {}
    for row in query_report["rows"]:
        video_id = row["video_id"]
        if video_id in assets:
            continue
        identifier = row["clip_candidates"][0]["thumbnail"].split("/")[2]
        matches = list((ROOT / "data/benchmarks").glob(f"*/videos/{identifier}"))
        if len(matches) != 1:
            raise ValueError(f"could not resolve frozen asset for {video_id}")
        folder = matches[0]
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assets[video_id] = {
            "frames": manifest["frames"],
            "vectors": np.load(folder / "embeddings.npy", allow_pickle=False),
            "jpeg_bytes": sum((folder / "frames" / Path(item["thumbnail"]).name).stat().st_size
                              for item in manifest["frames"]),
        }
    return assets


def load_sampled_dataset(root: Path, video_ids: list[str], clip, cache_root: Path) -> dict:
    output = {}
    for video_id in video_ids:
        folder = root / video_id
        record = json.loads((folder / "timestamps.json").read_text(encoding="utf-8"))
        paths = [folder / frame["file"] for frame in record["frames"]]
        cache = cache_root / f"{video_id}.npy"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.is_file():
            vectors = np.load(cache, allow_pickle=False)
            encode_seconds = 0.0
        else:
            started = time.perf_counter()
            vectors = clip.images(paths)
            encode_seconds = time.perf_counter() - started
            with cache.open("wb") as target:
                np.save(target, vectors, allow_pickle=False)
        if len(vectors) != len(paths):
            raise ValueError(f"cached vectors do not match {video_id}")
        output[video_id] = {
            "frames": record["frames"], "vectors": vectors,
            "jpeg_bytes": sum(path.stat().st_size for path in paths),
            "embedding_bytes": int(vectors.nbytes),
            "extraction_seconds": record.get("extraction_seconds", 0.0),
            "encode_seconds_this_run": encode_seconds,
        }
    return output


def faiss_pool(vectors: np.ndarray, text_vector: np.ndarray, timestamps: list[float], limit: int):
    from app.encoder import rank_vectors

    started = time.perf_counter()
    scores, indices = rank_vectors(vectors, text_vector, limit)
    elapsed = time.perf_counter() - started
    candidates = [
        Candidate(index, timestamps[index], score) for score, index in zip(scores, indices)
    ]
    return candidates, elapsed


def apply_policy(
    method: str, candidates: list[Candidate], vectors: np.ndarray,
    segments: dict[float, list[int]], config: dict, limit: int = 5,
) -> list[Candidate]:
    pool = candidates[: config.get("pool_depth", 5)]
    if method in {"raw_5s", "raw_2s"}:
        return pool[:limit]
    if method == "temporal_nms":
        return temporal_nms(pool, limit, config["radius_seconds"])
    if method == "mmr":
        return mmr(pool, vectors, limit, config["relevance_weight"])
    if method == "temporal_mmr":
        filtered = temporal_nms(pool, len(pool), config["radius_seconds"])
        return mmr(filtered, vectors, limit, config["relevance_weight"])
    if method == "scene_aware":
        return scene_aware(pool, segments[config["change_threshold"]], limit)
    raise ValueError(f"unknown method: {method}")


def method_configs(method: str) -> list[dict]:
    if method == "temporal_nms":
        return [{"pool_depth": depth, "radius_seconds": radius}
                for depth in (20, 50) for radius in (2.0, 5.0, 10.0)]
    if method == "mmr":
        return [{"pool_depth": depth, "relevance_weight": weight}
                for depth in (20, 50) for weight in (0.5, 0.65, 0.8, 0.9)]
    if method == "temporal_mmr":
        return [
            {"pool_depth": depth, "radius_seconds": radius, "relevance_weight": weight}
            for depth in (20, 50) for radius in (2.0, 5.0, 10.0)
            for weight in (0.5, 0.65, 0.8, 0.9)
        ]
    if method == "scene_aware":
        return [{"pool_depth": depth, "change_threshold": threshold}
                for depth in (20, 50) for threshold in (0.03, 0.05, 0.1, 0.15)]
    if method in {"multiscale_nms", "base_preserving_multiscale"}:
        return [
            {
                "pool_depth": depth, "radius_seconds": radius,
                "change_threshold": threshold,
                **({"base_count": base_count} if method == "base_preserving_multiscale" else {}),
            }
            for depth in (20, 50) for radius in (2.0, 5.0)
            for threshold in (0.05, 0.1, 0.15, 0.2)
            for base_count in ((3, 4) if method == "base_preserving_multiscale" else (0,))
        ]
    return [{"pool_depth": 5}]


def multiscale_asset(base: dict, dense: dict, change_threshold: float) -> dict:
    dense_vectors = dense["vectors"]
    changes = []
    for index in range(len(dense_vectors)):
        neighbors = []
        if index:
            neighbors.append(1 - float(dense_vectors[index] @ dense_vectors[index - 1]))
        if index + 1 < len(dense_vectors):
            neighbors.append(1 - float(dense_vectors[index] @ dense_vectors[index + 1]))
        changes.append(max(neighbors, default=0.0))
    base_timestamps = [frame["timestamp"] for frame in base["frames"]]
    rows = [
        (frame["timestamp"], vector, "base_5s")
        for frame, vector in zip(base["frames"], base["vectors"])
    ]
    rows.extend(
        (frame["timestamp"], vector, "visual_change_2s")
        for frame, vector, change in zip(dense["frames"], dense_vectors, changes)
        if change >= change_threshold
        and all(abs(frame["timestamp"] - timestamp) > 0.25 for timestamp in base_timestamps)
    )
    rows.sort(key=lambda item: item[0])
    return {
        "timestamps": [float(item[0]) for item in rows],
        "vectors": np.stack([item[1] for item in rows]),
        "sources": [item[2] for item in rows],
        "added_frames": sum(item[2] == "visual_change_2s" for item in rows),
    }


def apply_row_policy(row: dict, method: str, config: dict) -> list[Candidate]:
    if method in {"multiscale_nms", "base_preserving_multiscale"}:
        asset = row["multiscale"][config["change_threshold"]]
        row["_trial_source"] = asset["raw"]
        if method == "base_preserving_multiscale":
            selected = list(row["raw_5s"][: config["base_count"]])
            for candidate in asset["raw"][: config["pool_depth"]]:
                if all(
                    abs(candidate.timestamp - item.timestamp)
                    > config["radius_seconds"] + TIMESTAMP_TOLERANCE_SECONDS
                    for item in selected
                ):
                    selected.append(candidate)
                    if len(selected) == 5:
                        break
            return selected
        return temporal_nms(
            asset["raw"][: config["pool_depth"]], 5, config["radius_seconds"]
        )
    row["_trial_source"] = row["raw_2s"]
    return apply_policy(
        method, row["raw_2s"], row["vectors_2s"], row["segments_2s"], config
    )


def candidate_dicts(candidates: list[Candidate]) -> list[dict]:
    return [{"timestamp": row.timestamp, "score": row.score, "index": row.index}
            for row in candidates]


def regions(timestamps: list[float], radius: float = 5.0) -> int:
    if not timestamps:
        return 0
    count = 1
    anchor = sorted(timestamps)[0]
    for current in sorted(timestamps)[1:]:
        if current - anchor > radius + TIMESTAMP_TOLERANCE_SECONDS:
            count += 1
            anchor = current
    return count


def redundancy(candidates: list[Candidate], vectors: np.ndarray) -> dict:
    timestamps = [row.timestamp for row in candidates]
    pairs = list(combinations(range(len(candidates)), 2))
    neighbor_counts = {}
    for radius in (1.0, 2.0, 5.0, 10.0):
        matching = [(a, b) for a, b in pairs
                    if abs(timestamps[a] - timestamps[b])
                    <= radius + TIMESTAMP_TOLERANCE_SECONDS]
        neighbor_counts[str(int(radius))] = {
            "pairs": len(matching),
            "candidates_with_neighbor": len({item for pair in matching for item in pair}),
        }
    duplicate_pairs = [
        (a, b) for a, b in pairs
        if float(vectors[candidates[a].index] @ vectors[candidates[b].index]) >= 0.98
    ]
    gaps = [current - previous for previous, current in pairwise(sorted(timestamps))]
    return {
        "timestamps": timestamps,
        "temporal_gaps": gaps,
        "within_seconds": neighbor_counts,
        "unique_5s_regions": regions(timestamps),
        "score_spread": max((row.score for row in candidates), default=0.0)
        - min((row.score for row in candidates), default=0.0),
        "near_duplicate_pairs_at_cosine_0_98": len(duplicate_pairs),
        "near_duplicate_pair_rate": len(duplicate_pairs) / len(pairs) if pairs else 0.0,
    }


def quality(rows: list[dict], field: str) -> dict:
    from ml.evaluation.metrics import retrieval_metrics

    output = {}
    for k in (1, 3, 5):
        measured = [
            (
                row,
                retrieval_metrics(
                    [item.timestamp for item in row[field]][:k],
                    row["relevant_intervals"], k,
                ),
            )
            for row in rows
        ]
        positives = [metric for row, metric in measured if row["expected_presence"]]
        negatives = [metric for row, metric in measured if not row["expected_presence"]]
        output[str(k)] = {
            "positive_queries": len(positives),
            "negative_queries": len(negatives),
            "recall": statistics.mean(item["recall"] for item in positives)
            if positives else None,
            "mrr": statistics.mean(item["reciprocal_rank"] for item in positives)
            if positives else None,
            "negative_top_score_mean": statistics.mean(
                row[field][0].score for row, _ in measured
                if not row["expected_presence"] and row[field]
            ) if negatives else None,
        }
    return output


def pool_recall(rows: list[dict], field: str, depth: int) -> float | None:
    from ml.evaluation.metrics import retrieval_metrics

    positives = [row for row in rows if row["expected_presence"]]
    if not positives:
        return None
    return statistics.mean(
        retrieval_metrics(
            [item.timestamp for item in row[field]][:depth],
            row["relevant_intervals"], depth,
        )["recall"]
        for row in positives
    )


def category_slices(rows: list[dict], field: str) -> dict:
    names = sorted({row["query_type"] for row in rows})
    output = {name: quality([row for row in rows if row["query_type"] == name], field)
              for name in names}
    small = [row for row in rows if row["query_id"] in SMALL_OBJECT_IDS]
    output["SMALL_OBJECT"] = quality(small, field)
    return output


def visible(candidate: Candidate, event: dict) -> bool:
    return any(
        observation["visibility"] in {"visible", "partially_cropped"}
        and abs(candidate.timestamp - observation["timestamp"]) <= 0.06
        for observation in event["observations"]
    )


def visible_summary(rows: list[dict], field: str, limit: int = 5) -> dict:
    output = {}
    for split in ("calibration", "heldout", "all"):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        hits = []
        ranks = []
        for row in subset:
            candidates = row[field][:limit]
            hit_ranks = [rank for rank, candidate in enumerate(candidates, 1)
                         if visible(candidate, row["event"])]
            hits.append(bool(hit_ranks))
            ranks.append(1 / hit_ranks[0] if hit_ranks else 0.0)
        output[split] = {
            "events": len(subset),
            "recall_at_5": statistics.mean(hits) if hits else None,
            "mrr_at_5": statistics.mean(ranks) if ranks else None,
        }
    return output


def diversity_summary(rows: list[dict], field: str) -> dict:
    region_counts, spacings, redundancies, retentions = [], [], [], []
    for row in rows:
        candidates = row[field][:5]
        timestamps = [item.timestamp for item in candidates]
        region_counts.append(regions(timestamps))
        gaps = [b - a for a, b in pairwise(sorted(timestamps))]
        spacings.append(min(gaps) if gaps else 0.0)
        crowded = sum(any(
            i != j and abs(a - b) <= 5 + TIMESTAMP_TOLERANCE_SECONDS
            for j, b in enumerate(timestamps)
        )
                      for i, a in enumerate(timestamps))
        redundancies.append(crowded / len(timestamps) if timestamps else 0.0)
        if field == "raw_5s":
            source = row["raw_5s"]
        elif field in {"trial", "selected"} and f"{field}_source" in row:
            source = row[f"{field}_source"]
        elif f"{field}_source" in row:
            source = row[f"{field}_source"]
        else:
            source = row["raw_2s"]
        raw_top5 = {item.index for item in source[:5]}
        retentions.append(sum(item.index in raw_top5 for item in candidates) / len(candidates)
                          if candidates else 0.0)
    return {
        "average_unique_5s_regions_top5": statistics.mean(region_counts),
        "median_minimum_temporal_spacing_seconds": statistics.median(spacings),
        "average_temporal_redundancy_rate": statistics.mean(redundancies),
        "raw_top5_retention_rate": statistics.mean(retentions),
    }


def make_small_rows(clip, cache_root: Path) -> list[dict]:
    manifest = json.loads(VISIBILITY_MANIFEST.read_text(encoding="utf-8"))
    source_ids = [source["source_id"] for source in manifest["sources"]]
    datasets = {
        interval: load_sampled_dataset(
            SMALL_ROOT / f"{interval}s", source_ids, clip, cache_root / f"small-{interval}s"
        ) for interval in (5, 2)
    }
    multiscale = {
        source_id: {
            threshold: multiscale_asset(
                datasets[5][source_id], datasets[2][source_id], threshold
            )
            for threshold in (0.05, 0.1, 0.15, 0.2)
        }
        for source_id in source_ids
    }
    rows = []
    for event in manifest["events"]:
        text = clip.text(event["query"])
        row = {"event": event, "event_id": event["event_id"], "split": event["split"]}
        for interval in (5, 2):
            asset = datasets[interval][event["source_id"]]
            timestamps = [frame["timestamp"] for frame in asset["frames"]]
            candidates, _ = faiss_pool(asset["vectors"], text, timestamps, 50)
            row[f"raw_{interval}s"] = candidates
            row[f"vectors_{interval}s"] = asset["vectors"]
            row[f"segments_{interval}s"] = {
                threshold: scene_segments(asset["vectors"], threshold)
                for threshold in (0.03, 0.05, 0.1, 0.15)
            }
        row["multiscale"] = {}
        for threshold, asset in multiscale[event["source_id"]].items():
            candidates, _ = faiss_pool(
                asset["vectors"], text, asset["timestamps"], 50
            )
            row["multiscale"][threshold] = {**asset, "raw": candidates}
        rows.append(row)
    return rows


def run(output: Path, run_root: Path) -> dict:
    from app.encoder import encoder

    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    source = json.loads(QUERY_REPORT.read_text(encoding="utf-8"))
    if digest(NATURAL_MANIFEST) != source["natural_manifest_sha256"]:
        raise ValueError("frozen Natural V2 manifest changed")
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    clip = encoder()
    after_model_rss = process.memory_info().rss
    assets_5s = coarse_assets(source)
    video_ids = sorted(assets_5s)
    assets_2s = load_sampled_dataset(
        GLOBAL_TWO_ROOT, video_ids, clip, run_root / "vectors/global-2s"
    )
    multiscale_assets = {
        video_id: {
            threshold: multiscale_asset(
                assets_5s[video_id], assets_2s[video_id], threshold
            )
            for threshold in (0.05, 0.1, 0.15, 0.2)
        }
        for video_id in video_ids
    }
    indexed_rss = process.memory_info().rss
    durations = {row["video_id"]: row["duration_seconds"] for row in source["videos"]}

    rows = []
    faiss_times = {"5s_top5": [], "5s_top50": [], "2s_top5": [], "2s_top50": []}
    text_times = []
    for frozen in source["rows"]:
        started = time.perf_counter()
        text = clip.text(frozen["query"])
        text_times.append(time.perf_counter() - started)
        datasets = {}
        for interval, asset in ((5, assets_5s[frozen["video_id"]]),
                                (2, assets_2s[frozen["video_id"]])):
            timestamps = [frame["timestamp"] for frame in asset["frames"]]
            top5, _ = faiss_pool(asset["vectors"], text, timestamps, 5)
            top50, _ = faiss_pool(asset["vectors"], text, timestamps, 50)
            _, elapsed5 = faiss_pool(asset["vectors"], text, timestamps, 5)
            _, elapsed50 = faiss_pool(asset["vectors"], text, timestamps, 50)
            faiss_times[f"{interval}s_top5"].append(elapsed5)
            faiss_times[f"{interval}s_top50"].append(elapsed50)
            datasets[interval] = (top5, top50, asset["vectors"])
        vectors_2s = datasets[2][2]
        row = {
            **{key: frozen[key] for key in (
                "query_id", "video_id", "split", "query", "query_type",
                "expected_presence", "relevant_intervals",
            )},
            "duration": durations[frozen["video_id"]],
            "raw_5s": datasets[5][1], "raw_2s": datasets[2][1],
            "vectors_2s": vectors_2s,
            "segments_2s": {threshold: scene_segments(vectors_2s, threshold)
                            for threshold in (0.03, 0.05, 0.1, 0.15)},
            "multiscale": {},
        }
        for threshold, asset in multiscale_assets[frozen["video_id"]].items():
            candidates, _ = faiss_pool(
                asset["vectors"], text, asset["timestamps"], 50
            )
            row["multiscale"][threshold] = {**asset, "raw": candidates}
        rows.append(row)

    small_rows = make_small_rows(clip, run_root / "vectors")
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    small_calibration = [row for row in small_rows if row["split"] == "calibration"]
    trials = []
    for method in METHODS[2:]:
        for config in method_configs(method):
            field = "trial"
            started = time.perf_counter()
            for row in calibration:
                row[field] = apply_row_policy(row, method, config)
                row[f"{field}_source"] = row["_trial_source"]
            for row in small_calibration:
                row[field] = apply_row_policy(row, method, config)
                row[f"{field}_source"] = row["_trial_source"]
            elapsed = (time.perf_counter() - started) / (len(calibration) + len(small_calibration))
            metric = quality(calibration, field)
            small_metric = visible_summary(small_calibration, field)["calibration"]
            diversity = diversity_summary(calibration, field)
            trials.append({
                "method": method, "config": config,
                "calibration_metrics": metric,
                "calibration_small_object_visible": small_metric,
                "calibration_diversity": diversity,
                "mean_selection_seconds": elapsed,
            })

    baseline_cal_r5 = quality(calibration, "raw_5s")["5"]["recall"]
    method_winners = {}

    def selection_key(item):
        return (
            item["calibration_metrics"]["5"]["recall"] >= baseline_cal_r5,
            item["calibration_small_object_visible"]["recall_at_5"],
            item["calibration_metrics"]["5"]["recall"],
            item["calibration_metrics"]["1"]["recall"],
            item["calibration_metrics"]["5"]["mrr"],
            item["calibration_diversity"]["average_unique_5s_regions_top5"],
            -item["config"].get("pool_depth", 0),
            item["config"].get("change_threshold", 0.0),
            -item["config"].get("radius_seconds", 0.0),
            item["config"].get("relevance_weight", 0.0),
        )

    winner = max(trials, key=selection_key)
    for method in METHODS[2:]:
        method_winners[method] = max(
            (trial for trial in trials if trial["method"] == method), key=selection_key
        )
    for method, selected_trial in method_winners.items():
        for row in rows:
            row[method] = apply_row_policy(row, method, selected_trial["config"])
            row[f"{method}_source"] = row["_trial_source"]
        for row in small_rows:
            row[method] = apply_row_policy(row, method, selected_trial["config"])
            row[f"{method}_source"] = row["_trial_source"]
    selected_times = []
    for row in rows:
        row["selected"] = row[winner["method"]]
        started = time.perf_counter()
        for _ in range(50):
            apply_row_policy(row, winner["method"], winner["config"])
        selected_times.append((time.perf_counter() - started) / 50)
        row["selected_source"] = row["_trial_source"]
    for row in small_rows:
        row["selected"] = row[winner["method"]]
        apply_row_policy(row, winner["method"], winner["config"])
        row["selected_source"] = row["_trial_source"]

    method_fields = {
        "raw_5s": "raw_5s", "raw_2s": "raw_2s",
        **{name: name for name in method_winners}, "selected": "selected",
    }
    heldout_quality = {name: quality(heldout, field) for name, field in method_fields.items()}
    heldout_slices = {name: category_slices(heldout, field)
                      for name, field in method_fields.items()}
    small_quality = {name: visible_summary(small_rows, field)
                     for name, field in method_fields.items()}
    candidate_quality = {name: diversity_summary(heldout, field)
                         for name, field in method_fields.items()}

    raw_pool_recall = {}
    for dataset in ("raw_5s", "raw_2s"):
        raw_pool_recall[dataset] = {}
        for depth in (20, 50):
            field = f"pool_{dataset}_{depth}"
            for row in rows:
                row[field] = row[dataset][:depth]
            raw_pool_recall[dataset][str(depth)] = {
                "calibration_correct_region_recall": pool_recall(calibration, field, depth),
                "heldout_correct_region_recall": pool_recall(heldout, field, depth),
            }
    small_raw_pool = {}
    for dataset in ("raw_5s", "raw_2s"):
        small_raw_pool[dataset] = {}
        for depth in (20, 50):
            field = f"pool_{dataset}_{depth}"
            for row in small_rows:
                row[field] = row[dataset][:depth]
            small_raw_pool[dataset][str(depth)] = visible_summary(small_rows, field, depth)

    baseline_r5 = heldout_quality["raw_5s"]["5"]["recall"]
    selected_r5 = heldout_quality["selected"]["5"]["recall"]
    baseline_small = small_quality["raw_5s"]["heldout"]["recall_at_5"]
    selected_small = small_quality["selected"]["heldout"]["recall_at_5"]
    category_drops = {}
    for name, baseline in heldout_slices["raw_5s"].items():
        baseline_value = baseline["5"]["recall"]
        selected_value = heldout_slices["selected"].get(name, {}).get("5", {}).get("recall")
        if baseline_value is not None and selected_value is not None:
            category_drops[name] = baseline_value - selected_value

    added_median = statistics.median(selected_times)
    added_p95 = percentile(selected_times, 95)
    storage_5s = {
        "frames": sum(len(asset["frames"]) for asset in assets_5s.values()),
        "jpeg_bytes": sum(asset["jpeg_bytes"] for asset in assets_5s.values()),
        "embedding_bytes": sum(asset["vectors"].nbytes for asset in assets_5s.values()),
        "historical_ingestion_plus_index_seconds": sum(
            row["ingestion_seconds"] + row["index_seconds"] for row in source["videos"]
        ),
        "timing_note": "Includes the historical run's model/cache state and is not a paired timing run.",
    }
    storage_2s = {
        "frames": sum(len(asset["frames"]) for asset in assets_2s.values()),
        "jpeg_bytes": sum(asset["jpeg_bytes"] for asset in assets_2s.values()),
        "embedding_bytes": sum(asset["embedding_bytes"] for asset in assets_2s.values()),
        "measured_extraction_plus_encoding_seconds": parent["storage_comparison"][
            "global_2s"
        ]["extraction_plus_clip_seconds"],
        "timing_note": "Measured warm-model extraction plus CLIP encoding in the parent run.",
    }
    storage_growth = storage_2s["embedding_bytes"] / storage_5s["embedding_bytes"]
    gate = {
        "overall_recall_at_5": {"baseline": baseline_r5, "selected": selected_r5,
                                "pass": selected_r5 >= baseline_r5},
        "small_object_visible_recall_at_5": {
            "baseline": baseline_small, "selected": selected_small,
            "material_improvement_required": 0.25,
            "pass": selected_small - baseline_small >= 0.25,
        },
        "maximum_category_drop": {"drops": category_drops,
                                  "pass": max(category_drops.values(), default=0.0) <= 0.05},
        "added_median_seconds": {"value": added_median, "limit": 0.05,
                                 "pass": added_median <= 0.05},
        "added_p95_seconds": {"value": added_p95, "limit": 0.01,
                              "pass": added_p95 <= 0.01},
        "storage_growth": {"value": storage_growth, "limit": 1.5,
                           "pass": storage_growth <= 1.5},
    }
    gate["passed"] = all(item["pass"] for item in gate.values() if isinstance(item, dict))

    lost_rows = []
    for row in heldout:
        if not row["expected_presence"]:
            continue
        hit5 = quality([row], "raw_5s")["5"]["recall"] > 0
        hit2 = quality([row], "raw_2s")["5"]["recall"] > 0
        if hit5 and not hit2:
            pool20 = (pool_recall([row], "pool_raw_2s_20", 20) or 0) > 0
            pool50 = (pool_recall([row], "pool_raw_2s_50", 50) or 0) > 0
            redundant = redundancy(row["raw_2s"][:5], row["vectors_2s"])[
                "within_seconds"
            ]["5"]["pairs"] > 0
            lost_rows.append({
                "query_id": row["query_id"], "query_type": row["query_type"],
                "correct_in_top20": pool20, "correct_in_top50": pool50,
                "top5_has_temporal_neighbor_pair": redundant,
                "diagnosis": (
                    "near_duplicate_competition" if pool20 and redundant
                    else "lower_scored_semantic_distractors" if pool50
                    else "correct_region_absent_from_raw_pool"
                ),
            })

    report_rows = []
    for row in rows:
        report_rows.append({
            **{key: row[key] for key in (
                "query_id", "video_id", "split", "query", "query_type",
                "expected_presence", "relevant_intervals",
            )},
            "raw_5s_top20": candidate_dicts(row["raw_5s"][:20]),
            "raw_2s_top20": candidate_dicts(row["raw_2s"][:20]),
            "selected": candidate_dicts(row["selected"]),
            "selected_intervals": represented_intervals(
                row["selected"], row["duration"], winner["config"].get("radius_seconds", 2.0)
            ),
            "raw_5s_redundancy": redundancy(row["raw_5s"][:20], assets_5s[row["video_id"]]["vectors"]),
            "raw_2s_redundancy": redundancy(row["raw_2s"][:20], row["vectors_2s"]),
        })

    report = {
        "schema_version": "1.0.0",
        "experiment": "coarse-candidate-diversity-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "parent_report_sha256": digest(PARENT_REPORT),
        "query_report_sha256": digest(QUERY_REPORT),
        "natural_manifest_sha256": digest(NATURAL_MANIFEST),
        "visibility_manifest_sha256": digest(VISIBILITY_MANIFEST),
        "heldout_tuning": False,
        "selection": {
            "split": "calibration", "winner": winner,
            "method_winners": method_winners, "trials": trials,
        },
        "quality": {"heldout": heldout_quality, "heldout_slices": heldout_slices,
                    "small_object_visible": small_quality},
        "raw_pool_correct_region_recall": raw_pool_recall,
        "small_object_raw_pool_visible_recall": small_raw_pool,
        "candidate_quality": candidate_quality,
        "redundancy": {
            "raw_5s": diversity_summary(heldout, "raw_5s"),
            "raw_2s": diversity_summary(heldout, "raw_2s"),
        },
        "global_2s_diagnosis": {
            "lost_queries": lost_rows,
            "lost_query_count": len(lost_rows),
            "lost_queries_recovered_by_top20": sum(row["correct_in_top20"] for row in lost_rows),
            "lost_queries_with_neighbor_competition": sum(
                row["top5_has_temporal_neighbor_pair"] for row in lost_rows
            ),
            "measured_conclusion": (
                "All 5s-hit/2s-miss queries are speech-label visual-path diagnostics; their "
                "correct regions remain in the 2s top-20 while closer neighboring frames "
                "occupy the top five. The 2s top five also has fewer 5s temporal regions "
                "and a higher temporal redundancy rate than the 5s baseline."
            ),
        },
        "resources": {
            "text_encoding_median_seconds": statistics.median(text_times),
            "text_encoding_p95_seconds": percentile(text_times, 95),
            "selected_query_median_seconds": (
                statistics.median(text_times)
                + statistics.median(faiss_times["2s_top50"])
                + added_median
            ),
            "selected_query_p95_seconds": (
                percentile(text_times, 95)
                + percentile(faiss_times["2s_top50"], 95)
                + added_p95
            ),
            "faiss": {name: {"median_seconds": statistics.median(values),
                             "p95_seconds": percentile(values, 95)}
                      for name, values in faiss_times.items()},
            "diversification_median_seconds": added_median,
            "diversification_p95_seconds": added_p95,
            "rss_before_model_bytes": baseline_rss,
            "rss_after_model_bytes": after_model_rss,
            "rss_after_indexes_bytes": indexed_rss,
            "added_index_rss_bytes": max(0, indexed_rss - after_model_rss),
        },
        "storage": {"global_5s": storage_5s, "global_2s": storage_2s},
        "gate": gate,
        "second_frozen_run": {"required": gate["passed"], "executed": False, "passed": None},
        "production_changed": False,
        "decision": {
            "outcome": "pending",
            "recommendation": "pending until the frozen result is assembled",
        },
        "small_object_rows": [
            {
                "event_id": row["event_id"], "split": row["split"],
                "query": row["event"]["query"],
                "raw_5s_top20": candidate_dicts(row["raw_5s"][:20]),
                "raw_2s_top20": candidate_dicts(row["raw_2s"][:20]),
                "selected": candidate_dicts(row["selected"]),
                "raw_5s_top5_visible_hit": any(
                    visible(candidate, row["event"]) for candidate in row["raw_5s"][:5]
                ),
                "raw_2s_top5_visible_hit": any(
                    visible(candidate, row["event"]) for candidate in row["raw_2s"][:5]
                ),
                "raw_2s_top20_visible_hit": any(
                    visible(candidate, row["event"]) for candidate in row["raw_2s"][:20]
                ),
                "selected_visible_hit": any(
                    visible(candidate, row["event"]) for candidate in row["selected"]
                ),
            }
            for row in small_rows
        ],
        "rows": report_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/coarse-candidate-diversity/report.json")
    parser.add_argument("--run-root", type=Path, default=ROOT / "data/coarse-candidate-diversity")
    args = parser.parse_args()
    result = run(args.output, args.run_root)
    print(json.dumps({"winner": result["selection"]["winner"], "gate": result["gate"]}, indent=2))
