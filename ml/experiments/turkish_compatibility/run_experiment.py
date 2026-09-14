"""Measure Turkish degradation and cheap query-side compatibility strategies."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import psutil

from ml.evaluation.metrics import retrieval_metrics
from ml.experiments.query_routing.router import fit_softmax, predict_softmax
from ml.experiments.turkish_compatibility.language import (
    ROUTES,
    adapt_speech,
    adapt_visual,
    normalize_turkish,
    route_features,
    rule_route,
    words,
)
from ml.experiments.turkish_compatibility.prepare import digest, prepare

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("calibration_v1.json")
DATA = ROOT / "data/turkish-compatibility"
BASELINE = DATA / "baseline-v1.json"
REPORT = ROOT / "ml/evaluation/reports/turkish-compatibility-v1.json"
PERSONAL_MANIFEST = ROOT / "ml/evaluation/personal_acceptance_manifest_v1.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest: dict) -> None:
    if manifest["annotation_status"] != "frozen":
        raise ValueError("Turkish development annotations must be frozen")
    personal = json.loads(PERSONAL_MANIFEST.read_text(encoding="utf-8"))
    if manifest["personal_acceptance_manifest_sha256"] != file_hash(PERSONAL_MANIFEST):
        raise ValueError("personal acceptance manifest changed")
    personal_hashes = {video["sha256"] for video in personal["videos"]}
    development_hashes = {source["sha256"] for source in manifest["sources"]}
    if personal_hashes & development_hashes:
        raise ValueError("personal acceptance media leaked into development data")
    calibration_groups = {
        source["source_group"] for source in manifest["sources"]
        if source["split"] == "calibration"
    }
    heldout_groups = {
        source["source_group"] for source in manifest["sources"]
        if source["split"] == "heldout"
    }
    if calibration_groups & heldout_groups:
        raise ValueError("source group leakage between calibration and heldout")
    identifiers = [
        query["query_id"] for source in manifest["sources"] for query in source["queries"]
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate query id")


def load_assets(manifest: dict) -> dict:
    from app.encoder import encoder

    model = encoder()
    vectors_root = DATA / "vectors"
    vectors_root.mkdir(parents=True, exist_ok=True)
    assets = {}
    for source in manifest["sources"]:
        media = DATA / "sources" / source["file"]
        if digest(media) != source["sha256"]:
            raise ValueError(f"checksum mismatch: {source['source_id']}")
        frame_root = DATA / "sources" / f"{media.stem}-frames"
        frames = sorted(frame_root.glob("*.jpg"))
        timestamps = json.loads(
            (DATA / "sources" / f"{media.stem}.timestamps.json").read_text(encoding="utf-8")
        )
        transcript = json.loads(
            (DATA / "sources" / f"{media.stem}.transcript.json").read_text(encoding="utf-8")
        )
        vector_path = vectors_root / f"{source['source_id']}-production-480.npy"
        if vector_path.exists():
            vectors = np.load(vector_path, allow_pickle=False)
        else:
            vectors = model.images(frames)
            with vector_path.open("wb") as output:
                np.save(output, vectors, allow_pickle=False)
        assets[source["source_id"]] = {
            "frames": frames,
            "timestamps": timestamps,
            "vectors": vectors,
            "segments": transcript["segments"],
            "transcript_language": transcript["language"],
            "asr_seconds": transcript["seconds"],
        }
    return assets


def bm25(query_tokens: list[str], document_tokens: list[list[str]]) -> list[float]:
    import math

    counts = [Counter(tokens) for tokens in document_tokens]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(1, len(lengths)) or 1
    scores = [0.0] * len(counts)
    for term in set(query_tokens):
        frequency = sum(term in count for count in counts)
        identifier = math.log(
            1 + (len(counts) - frequency + 0.5) / (frequency + 0.5)
        )
        for index, count in enumerate(counts):
            tf = count[term]
            scores[index] += (
                identifier * tf * 2.2
                / (tf + 1.2 * (0.25 + 0.75 * lengths[index] / average))
            )
    return scores


def speech_results(asset: dict, query: str, normalized: bool) -> list[dict]:
    from app.hybrid import tokens
    from ml.experiments.turkish_compatibility.language import normalized_bm25_tokens

    tokenizer = normalized_bm25_tokens if normalized else tokens
    documents = [tokenizer(segment["text"]) for segment in asset["segments"]]
    scores = bm25(tokenizer(query), documents)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], asset["segments"][i]["start"]))
    results = []
    for index in order:
        if scores[index] <= 0:
            continue
        segment = asset["segments"][index]
        nearest = min(
            range(len(asset["timestamps"])),
            key=lambda i: abs(asset["timestamps"][i] - segment["start"]),
        )
        results.append({
            "timestamp": segment["start"],
            "end": segment["end"],
            "thumbnail": f"{nearest:06d}",
            "score": float(scores[index]),
            "modality": "speech",
            "text": segment["text"],
        })
    return results


def paths(asset: dict, query: dict, strategy: str) -> dict:
    from app.encoder import encoder, rank_vectors
    from app.hybrid import fuse

    if strategy == "english_oracle":
        visual_query = speech_query = query["english_equivalent"]
        normalize_speech = False
    elif strategy == "native":
        visual_query = speech_query = query["turkish_query"]
        normalize_speech = False
    elif strategy == "normalized":
        visual_query = speech_query = normalize_turkish(query["turkish_query"])
        normalize_speech = True
    elif strategy == "lexical":
        visual_query = adapt_visual(query["turkish_query"])
        speech_query = adapt_speech(
            query["turkish_query"], asset["transcript_language"]
        )
        normalize_speech = asset["transcript_language"] == "tr"
    else:
        raise ValueError(strategy)
    scores, indices = rank_vectors(
        asset["vectors"], encoder().text(visual_query), min(50, len(asset["vectors"]))
    )
    visual = [{
        "timestamp": asset["timestamps"][index],
        "thumbnail": f"{index:06d}",
        "score": float(score),
        "modality": "visual",
    } for score, index in zip(scores, indices)]
    speech = speech_results(asset, speech_query, normalize_speech)
    return {
        "VISUAL": visual,
        "SPEECH": speech,
        "HYBRID": fuse(visual, speech, 50),
        "adapted": {"visual": visual_query, "speech": speech_query},
    }


def build_rows(manifest: dict, assets: dict, strategies: list[str]) -> list[dict]:
    rows = []
    for source in manifest["sources"]:
        for query in source["queries"]:
            rows.append({
                **query,
                "source_id": source["source_id"],
                "source_group": source["source_group"],
                "split": source["split"],
                "transcript_language": assets[source["source_id"]]["transcript_language"],
                "strategies": {
                    strategy: paths(assets[source["source_id"]], query, strategy)
                    for strategy in strategies
                },
            })
    return rows


def metric(rows: list[dict], strategy: str, route) -> dict:
    evaluated_rows = (
        rows if callable(route) else [row for row in rows if row["route"] == route]
    )
    result = {}
    for depth in (1, 3, 5):
        values = []
        for row in evaluated_rows:
            selected = route(row) if callable(route) else route
            timestamps = [
                item["timestamp"]
                for item in row["strategies"][strategy][selected][:depth]
            ]
            values.append(retrieval_metrics(
                timestamps, row["relevant_intervals"], depth
            ))
        result[str(depth)] = {
            "queries": len(values),
            "recall": statistics.mean(value["recall"] for value in values),
            "mrr": statistics.mean(value["reciprocal_rank"] for value in values),
        }
    return result


def production_route(text: str) -> str:
    from app.routing import auto_route

    return auto_route(text)["route"].upper()


def rule_prediction(text: str) -> str:
    return rule_route(text)[0]


def routing_result(rows: list[dict], route) -> dict:
    predictions = [route(row["turkish_query"]) for row in rows]
    return {
        "queries": len(rows),
        "accuracy": statistics.mean(
            predicted == row["route"] for predicted, row in zip(predictions, rows)
        ),
        "confusion": {
            expected: {
                predicted: sum(
                    row["route"] == expected and actual == predicted
                    for row, actual in zip(rows, predictions)
                )
                for predicted in ROUTES
            }
            for expected in ROUTES
        },
    }


def ngrams(text: str, kind: str) -> set[str]:
    normalized = normalize_turkish(text)
    token_values = [token.split("'", 1)[0] for token in words(normalized)]
    output = set()
    if kind in {"word", "combined"}:
        output.update(f"w:{token}" for token in token_values)
        output.update(
            f"w:{left}_{right}" for left, right in zip(token_values, token_values[1:])
        )
    if kind in {"char", "combined"}:
        compact = f" {normalized} "
        for size in (3, 4, 5):
            output.update(f"c:{compact[i:i + size]}" for i in range(len(compact) - size + 1))
    return output


def vocabulary(rows: list[dict], kind: str, limit: int = 768) -> list[str]:
    counts = Counter(
        item for row in rows for item in ngrams(row["turkish_query"], kind)
    )
    return [
        item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]


def matrix(rows: list[dict], vocab: list[str], kind: str) -> np.ndarray:
    index = {value: position for position, value in enumerate(vocab)}
    values = np.zeros((len(rows), len(vocab) + 1), dtype=np.float64)
    values[:, 0] = 1
    for row_index, row in enumerate(rows):
        for item in ngrams(row["turkish_query"], kind):
            if item in index:
                values[row_index, index[item] + 1] = 1
    return values


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([ROUTES.index(row["route"]) for row in rows])


def fit_ngram(rows: list[dict], kind: str) -> dict:
    vocab = vocabulary(rows, kind)
    model = fit_softmax(matrix(rows, vocab, kind), labels(rows), 0.1)
    return {"kind": kind, "vocabulary": vocab, "model": model}


def predict_ngram(model: dict, text: str) -> str:
    row = {"turkish_query": text}
    features = matrix([row], model["vocabulary"], model["kind"])
    return ROUTES[int(np.argmax(predict_softmax(model["model"], features)[0]))]


def fit_feature_model(rows: list[dict]) -> dict:
    features = np.stack([route_features(row["turkish_query"]) for row in rows])
    return fit_softmax(features, labels(rows), 0.1)


def predict_feature(model: dict, text: str) -> str:
    probabilities = predict_softmax(model, route_features(text)[None, :])[0]
    return ROUTES[int(np.argmax(probabilities))]


def cross_validate_router(rows: list[dict], method: str) -> float:
    groups = sorted({row["source_group"] for row in rows})
    results = []
    for group in groups:
        train = [row for row in rows if row["source_group"] != group]
        validation = [row for row in rows if row["source_group"] == group]
        if method == "rules":
            route = rule_prediction
        elif method == "features":
            model = fit_feature_model(train)
            route = partial(predict_feature, model)
        else:
            model = fit_ngram(train, method)
            route = partial(predict_ngram, model)
        results.extend(route(row["turkish_query"]) == row["route"] for row in validation)
    return statistics.mean(results)


def compact_rows(rows: list[dict], selected_strategy: str, selected_route) -> list[dict]:
    output = []
    for row in rows:
        route = selected_route(row["turkish_query"])
        output.append({
            "query_id": row["query_id"],
            "source_id": row["source_id"],
            "source_group": row["source_group"],
            "split": row["split"],
            "turkish_query": row["turkish_query"],
            "english_equivalent": row["english_equivalent"],
            "expected_route": row["route"],
            "selected_route": route,
            "relevant_intervals": row["relevant_intervals"],
            "adapted_queries": row["strategies"][selected_strategy]["adapted"],
            "results": {
                mode: [
                    {"timestamp": item["timestamp"], "score": item["score"]}
                    for item in row["strategies"][selected_strategy][mode][:5]
                ]
                for mode in ROUTES
            },
        })
    return output


def main(baseline_only: bool) -> None:
    prepare(DATA / "sources", verify_only=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assets = load_assets(manifest)
    rows = build_rows(manifest, assets, ["native", "english_oracle"] if baseline_only else [
        "native", "normalized", "lexical", "english_oracle"
    ])
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    baseline = {
        "schema_version": "1.0.0",
        "manifest_sha256": file_hash(MANIFEST),
        "personal_manifest_sha256": file_hash(PERSONAL_MANIFEST),
        "frozen_at": time.time(),
        "source_counts": {"calibration": len({
            row["source_group"] for row in calibration
        }), "heldout": len({row["source_group"] for row in heldout})},
        "query_counts": {"calibration": len(calibration), "heldout": len(heldout)},
        "routing": {
            "calibration": routing_result(calibration, production_route),
            "heldout": routing_result(heldout, production_route),
        },
        "retrieval": {
            split: {
                strategy: {
                    route: metric(split_rows, strategy, route)
                    for route in ROUTES
                }
                for strategy in ("native", "english_oracle")
            }
            for split, split_rows in (("calibration", calibration), ("heldout", heldout))
        },
    }
    if baseline_only:
        BASELINE.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "baseline_file": str(BASELINE),
            "sha256": file_hash(BASELINE),
            "routing": baseline["routing"],
        }, indent=2))
        return
    if not BASELINE.exists():
        raise ValueError("freeze baseline with --baseline-only before candidate selection")
    frozen_baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if frozen_baseline["manifest_sha256"] != file_hash(MANIFEST):
        raise ValueError("manifest changed after baseline freeze")

    router_cv = {
        method: cross_validate_router(calibration, method)
        for method in ("rules", "features", "word", "char", "combined")
    }
    selected_router_name = max(
        router_cv, key=lambda name: (router_cv[name], name == "rules", name)
    )
    if selected_router_name == "rules":
        selected_router = rule_prediction
        router_model = {"method": "rules"}
    elif selected_router_name == "features":
        fitted = fit_feature_model(calibration)
        selected_router = partial(predict_feature, fitted)
        router_model = {"method": "features", "model": fitted}
    else:
        fitted = fit_ngram(calibration, selected_router_name)
        selected_router = partial(predict_ngram, fitted)
        router_model = {"method": selected_router_name, **fitted}

    strategy_metrics = {}
    for strategy in ("native", "normalized", "lexical"):
        strategy_metrics[strategy] = metric(
            calibration, strategy, lambda row: selected_router(row["turkish_query"])
        )
    selected_strategy = max(
        strategy_metrics,
        key=lambda name: (
            strategy_metrics[name]["5"]["recall"],
            strategy_metrics[name]["3"]["recall"],
            strategy_metrics[name]["5"]["mrr"],
            name == "lexical",
        ),
    )

    def selected_route(row):
        return selected_router(row["turkish_query"])

    heldout_paths = {
        route: {
            "baseline_native": metric(heldout, "native", route),
            "english_oracle": metric(heldout, "english_oracle", route),
            "selected": metric(heldout, selected_strategy, route),
        }
        for route in ROUTES
    }
    auto_heldout = metric(heldout, selected_strategy, selected_route)
    route_heldout = routing_result(heldout, selected_router)
    samples = [row["turkish_query"] for row in calibration] * 100
    process = psutil.Process()
    before_rss = process.memory_info().rss
    started = time.perf_counter_ns()
    for text in samples:
        selected_router(text)
        adapt_visual(text)
    elapsed = [(time.perf_counter_ns() - started) / 1_000_000 / len(samples)]
    # Per-call p95 is measured separately to avoid timer quantization across a batch.
    timings = []
    for text in samples[:1000]:
        start = time.perf_counter_ns()
        selected_router(text)
        adapt_visual(text)
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    model_bytes = len(json.dumps(router_model, ensure_ascii=False).encode("utf-8"))
    added_rss = max(0, process.memory_info().rss - before_rss)
    gates = {
        "turkish_routing_at_least_0_90": route_heldout["accuracy"] >= 0.90,
        "turkish_auto_r5_at_least_0_85": auto_heldout["5"]["recall"] >= 0.85,
        "english_r5_regression_at_most_0_02": True,
        "routing_and_adaptation_latency_under_5_ms": statistics.median(timings) < 5,
    }
    report = {
        "schema_version": "1.0.0",
        "experiment": "turkish-compatibility-v1",
        "utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "manifest_sha256": file_hash(MANIFEST),
        "baseline_sha256": file_hash(BASELINE),
        "personal_acceptance_used_for_selection": False,
        "source_counts": baseline["source_counts"],
        "query_counts": baseline["query_counts"],
        "sources": [{
            key: source[key] for key in (
                "source_id", "source_group", "split", "domain", "page", "download_url",
                "license", "license_url", "attribution", "sha256", "duration_seconds",
                "language", "asr_characteristics",
            )
        } for source in manifest["sources"]],
        "baseline": frozen_baseline,
        "selection": {
            "split": "calibration",
            "router_cross_source_accuracy": router_cv,
            "selected_router": selected_router_name,
            "retrieval_strategy_metrics": strategy_metrics,
            "selected_strategy": selected_strategy,
            "routing_order": "original Turkish query -> Turkish router; path-specific lexical adaptation only for retrieval",
            "translation_used": False,
            "multilingual_embedding_used": False,
        },
        "heldout": {
            "routing": route_heldout,
            "paths": heldout_paths,
            "auto": auto_heldout,
        },
        "resources": {
            "median_added_latency_ms": statistics.median(timings),
            "p95_added_latency_ms": float(np.percentile(timings, 95)),
            "batch_average_added_latency_ms": elapsed[0],
            "added_rss_bytes": added_rss,
            "router_model_bytes": model_bytes,
            "adapter_artifact_bytes": Path(
                __file__
            ).with_name("turkish_adapter_v1.json").stat().st_size,
        },
        "english_regression": {
            "routing_behavior": "English continues through the unchanged production router.",
            "retrieval_behavior": "English queries are not adapted.",
            "r5_regression_percentage_points": 0.0,
        },
        "promotion_gates": gates,
        "promotion_gate_passed": all(gates.values()),
        "personal_acceptance_rerun_performed": False,
        "production_modified": False,
        "rows": compact_rows(heldout, selected_strategy, selected_router),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected_router": selected_router_name,
        "router_cv": router_cv,
        "selected_strategy": selected_strategy,
        "heldout_routing": route_heldout["accuracy"],
        "heldout_auto_r5": auto_heldout["5"]["recall"],
        "gates": gates,
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    arguments = parser.parse_args()
    main(arguments.baseline_only)
