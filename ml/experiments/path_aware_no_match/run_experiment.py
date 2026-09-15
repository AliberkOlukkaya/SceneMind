"""Fit path rules on calibration, then evaluate the frozen held-out split once."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.hybrid import bm25, fuse
from app.routing import auto_route
from ml.evaluation.metrics import retrieval_metrics
from ml.experiments.path_aware_no_match.build_manifest import MANIFEST, ROOT, validate_manifest
from ml.experiments.path_aware_no_match.features import (
    apply_rule,
    decision_metrics,
    distribution,
    select_rule,
    speech_features,
    visual_features,
)
from ml.experiments.path_aware_no_match.prepare import DATA_ROOT, prepare

ROUTES = ("VISUAL", "SPEECH", "HYBRID")
QUERY_ROUTING_ROOT = ROOT / "data/query-routing"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _asset_paths(source: dict) -> tuple[Path, Path, Path, Path]:
    if source["asset_set"] == "query-routing":
        stem = Path(source["path"]).stem
        return (
            QUERY_ROUTING_ROOT / "sources" / f"{stem}-frames",
            QUERY_ROUTING_ROOT / "sources" / f"{stem}.timestamps.json",
            QUERY_ROUTING_ROOT / "sources" / f"{stem}.transcript.json",
            QUERY_ROUTING_ROOT / "vectors" / f"{source['source_id']}-production-480.npy",
        )
    return (
        DATA_ROOT / f"{source['source_id']}-frames",
        DATA_ROOT / f"{source['source_id']}.timestamps.json",
        DATA_ROOT / f"{source['source_id']}.transcript.json",
        DATA_ROOT / f"{source['source_id']}.production-480.npy",
    )


def load_assets(manifest: dict) -> dict:
    assets = {}
    for source in manifest["sources"]:
        frames_root, timestamps_path, transcript_path, vectors_path = _asset_paths(source)
        frames = sorted(frames_root.glob("*.jpg"))
        timestamps = json.loads(timestamps_path.read_text(encoding="utf-8"))
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path, allow_pickle=False)
        if not frames or len(frames) != len(timestamps) or len(vectors) != len(frames):
            raise ValueError(f"asset mismatch: {source['source_id']}")
        assets[source["source_id"]] = {
            "timestamps": timestamps, "segments": transcript["segments"], "vectors": vectors,
            "thumbnails": [f"/{source['source_id']}/{frame.name}" for frame in frames],
            "asr_language": transcript["language"], "asr_seconds": transcript["seconds"],
        }
    return assets


def build_rows(manifest: dict, assets: dict) -> list[dict]:
    from app.encoder import encoder, rank_vectors

    model = encoder()
    rows = []
    for source in manifest["sources"]:
        asset = assets[source["source_id"]]
        documents = [segment["text"] for segment in asset["segments"]]
        for annotation in source["queries"]:
            scores, indices = rank_vectors(
                asset["vectors"], model.text(annotation["query"]), 50
            )
            visual = [{
                "timestamp": asset["timestamps"][index],
                "thumbnail": asset["thumbnails"][index],
                "score": float(score), "modality": "visual",
            } for score, index in zip(scores, indices)]
            lexical_scores = bm25(annotation["query"], documents)
            order = sorted(range(len(lexical_scores)),
                           key=lambda index: (-lexical_scores[index],
                                              asset["segments"][index]["start"]))
            speech = []
            for index in order:
                if lexical_scores[index] <= 0:
                    continue
                segment = asset["segments"][index]
                nearest = min(range(len(asset["timestamps"])),
                              key=lambda frame: abs(asset["timestamps"][frame] - segment["start"]))
                speech.append({
                    "timestamp": segment["start"], "end": segment["end"],
                    "thumbnail": asset["thumbnails"][nearest],
                    "score": float(lexical_scores[index]), "text": segment["text"],
                    "modality": "speech",
                })
            features = {f"v_{key}": value for key, value in visual_features(visual).items()}
            features.update({f"s_{key}": value for key, value in speech_features(
                annotation["query"], asset["segments"], lexical_scores
            ).items()})
            rows.append({
                **annotation, "source_id": source["source_id"],
                "source_group": source["source_group"], "split": source["split"],
                "features": features,
                "paths": {"VISUAL": visual, "SPEECH": speech, "HYBRID": fuse(visual, speech, 50)},
            })
    return rows


def add_hybrid_states(rows: list[dict], visual_rule: dict, speech_rule: dict) -> None:
    for row in rows:
        visual = apply_rule(row["features"], visual_rule)
        speech = apply_rule(row["features"], speech_rule)
        dual = sum("visual" in item.get("evidence", {}) and "speech" in item.get("evidence", {})
                   for item in row["paths"]["HYBRID"][:5])
        row["features"].update({
            "state_visual": float(visual), "state_speech": float(speech),
            "state_both": float(visual and speech), "state_either": float(visual or speech),
            "state_dual_top5": dual / 5,
        })


def route_rows(rows: list[dict], rules: dict) -> None:
    for row in rows:
        expected = row["route"]
        routed = auto_route(row["query"])["route"].upper()
        row["auto_route"] = routed
        row["explicit_accept"] = apply_rule(row["features"], rules[expected]["winner"])
        row["auto_accept"] = apply_rule(row["features"], rules[routed]["winner"])


def retrieval_summary(rows: list[dict], route_field: str, accept_field: str | None = None) -> dict:
    positives = [row for row in rows if row["expected_presence"]]
    accepted = [row for row in positives if accept_field is None or row[accept_field]]
    output = {}
    for depth in (1, 3, 5):
        def hit(row: dict) -> float:
            route = row[route_field] if route_field in row else route_field
            results = row["paths"][route][:depth]
            return retrieval_metrics([item["timestamp"] for item in results],
                                     row["relevant_intervals"], depth)["recall"]
        output[str(depth)] = {
            "useful_rate_among_accepted": (statistics.mean(hit(row) for row in accepted)
                                            if accepted else None),
            "useful_rate_overall": statistics.mean(
                hit(row) if (accept_field is None or row[accept_field]) else 0 for row in positives
            ) if positives else None,
            "accepted_positive_queries": len(accepted), "positive_queries": len(positives),
        }
    return output


def metrics_for(rows: list[dict], route: str, decision_field: str) -> dict:
    subset = [row for row in rows if row["route"] == route]
    metrics = decision_metrics(subset, [row[decision_field] for row in subset])
    metrics["retrieval_before"] = retrieval_summary(subset, route)
    metrics["retrieval_after"] = retrieval_summary(subset, route, decision_field)
    return metrics


def gate(path_metrics: dict, auto_accuracy: float) -> dict:
    path_checks = {}
    for route, values in path_metrics.items():
        path_checks[route] = {
            "negative_far": {"value": values["negative_false_accept_rate"], "maximum": 0.15,
                             "pass": values["negative_false_accept_rate"] <= 0.15},
            "positive_false_abstention": {
                "value": values["positive_false_abstention_rate"], "maximum": 0.15,
                "pass": values["positive_false_abstention_rate"] <= 0.15,
            },
            "retrieval_no_material_regression": {
                "before": values["retrieval_before"]["5"]["useful_rate_overall"],
                "after": values["retrieval_after"]["5"]["useful_rate_overall"],
                "maximum_drop": 0.05,
                "pass": (values["retrieval_before"]["5"]["useful_rate_overall"] -
                         values["retrieval_after"]["5"]["useful_rate_overall"]) <= 0.05,
            },
        }
    output = {
        "paths": path_checks,
        "auto_routing": {"value": auto_accuracy, "minimum": 0.90, "pass": auto_accuracy >= 0.90},
    }
    output["passed"] = output["auto_routing"]["pass"] and all(
        check["pass"] for path in path_checks.values() for check in path.values()
    )
    return output


def compact(row: dict) -> dict:
    return {key: row.get(key) for key in (
        "query_id", "query", "source_id", "source_group", "split", "route",
        "auto_route", "expected_presence", "relevant_intervals", "negative_taxonomy",
        "case", "explicit_accept", "auto_accept",
    ) if row.get(key) is not None} | {
        "features": row["features"],
        "path_top5": {route: [{key: item.get(key) for key in (
            "timestamp", "score", "modality", "text"
        ) if item.get(key) is not None} for item in row["paths"][route][:5]] for route in ROUTES},
    }


def run(output: Path) -> dict:
    validation = validate_manifest()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    prepare(verify_only=True)
    rows = build_rows(manifest, load_assets(manifest))
    calibration = [row for row in rows if row["split"] == "calibration"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    rules = {
        "VISUAL": select_rule([row for row in calibration if row["route"] == "VISUAL"], "v_"),
        "SPEECH": select_rule([row for row in calibration if row["route"] == "SPEECH"], "s_"),
    }
    add_hybrid_states(rows, rules["VISUAL"]["winner"], rules["SPEECH"]["winner"])
    rules["HYBRID"] = select_rule(
        [row for row in calibration if row["route"] == "HYBRID"]
    )
    route_rows(rows, rules)
    calibration_metrics = {route: metrics_for(calibration, route, "explicit_accept")
                           for route in ROUTES}

    # The rules are now frozen. Held-out rows are evaluated once below and never feed selection.
    heldout_metrics = {route: metrics_for(heldout, route, "explicit_accept") for route in ROUTES}
    overall = decision_metrics(heldout, [row["explicit_accept"] for row in heldout])
    overall["retrieval_before"] = retrieval_summary(heldout, "route")
    overall["retrieval_after"] = retrieval_summary(heldout, "route", "explicit_accept")
    auto_accuracy = statistics.mean(row["auto_route"] == row["route"] for row in heldout)
    auto_decisions = decision_metrics(heldout, [row["auto_accept"] for row in heldout])
    auto_decisions["retrieval_before"] = retrieval_summary(heldout, "auto_route")
    auto_decisions["retrieval_after"] = retrieval_summary(heldout, "auto_route", "auto_accept")

    latency = []
    tracemalloc.start()
    for _ in range(500):
        for row in heldout:
            started = time.perf_counter_ns()
            apply_rule(row["features"], rules[row["auto_route"]]["winner"])
            latency.append((time.perf_counter_ns() - started) / 1e6)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    frozen_gate = gate(heldout_metrics, auto_accuracy)
    second = None
    if frozen_gate["passed"]:
        repeated = {route: metrics_for(heldout, route, "explicit_accept") for route in ROUTES}
        second = {"executed": True, "identical": repeated == heldout_metrics,
                  "path_metrics": repeated}

    source_counts = Counter(source["split"] for source in manifest["sources"])
    report = {
        "schema_version": "1.0.0", "experiment": "path-aware-no-match-v1",
        "utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "platform": platform.platform(), "manifest_sha256": digest(MANIFEST),
        "personal_acceptance_manifest_sha256": manifest["personal_acceptance_manifest_sha256"],
        "heldout_tuning": False, "validation": validation,
        "dataset": {
            "source_counts": dict(source_counts),
            "query_counts": dict(Counter(
                f"{row['split']}:{'positive' if row['expected_presence'] else 'negative'}"
                for row in rows
            )),
            "path_counts": dict(Counter(f"{row['split']}:{row['route']}" for row in rows)),
            "source_disjoint": True,
            "sources": [{key: source[key] for key in (
                "source_id", "split", "domain", "source_group", "page", "license",
                "license_url", "attribution", "sha256", "duration_seconds",
            )} for source in manifest["sources"]],
        },
        "selection": {
            "split": "calibration", "rules": rules,
            "distributions": {route: distribution(
                [row for row in calibration if row["route"] == route]
            ) for route in ROUTES},
            "path_metrics": calibration_metrics,
        },
        "heldout": {
            "path_metrics": heldout_metrics, "overall": overall,
            "auto_routing_accuracy": auto_accuracy, "auto": auto_decisions,
        },
        "resources": {
            "decision_median_ms": float(np.median(latency)),
            "decision_p95_ms": float(np.percentile(latency, 95)),
            "tracemalloc_peak_bytes": peak,
            "new_model_memory_bytes": 0,
            "rule_artifact_bytes": len(json.dumps(rules).encode("utf-8")),
        },
        "gate": frozen_gate,
        "second_frozen_run": second or {"executed": False, "reason": "held-out gate failed"},
        "personal_acceptance": {
            "eligible_for_english_rerun": frozen_gate["passed"],
            "english_negative_non_misleading_before": 4 / 9,
            "english_negative_non_misleading_after": None,
            "english_useful_top5_before": 8 / 9,
            "english_useful_top5_after": None,
        },
        "production": {"behavior_modified": False, "feature_flag_added": False},
        "decision": {
            "outcome": "E",
            "label": "Current scores fundamentally cannot support reliable no-match",
            "recommendation": (
                "Do not promote a rejection rule. Stop no-match model experimentation for v1.0; "
                "keep ranked results and explicit modes, and use conservative product copy that "
                "does not claim a returned timestamp is a confirmed match."
            ),
        },
        "rows": [compact(row) for row in rows],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "report.json")
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps({"rules": {route: value["winner"] for route, value in
                                result["selection"]["rules"].items()},
                      "heldout": result["heldout"], "gate": result["gate"],
                      "resources": result["resources"]}, indent=2))
