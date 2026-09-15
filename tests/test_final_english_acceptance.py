import hashlib
import json

import pytest

from ml.evaluation.final_english_acceptance import CATEGORIES, summarize, validate


def _manifest(tmp_path):
    media = tmp_path / "real-lecture.mp4"
    media.write_bytes(b"private fixture placeholder")
    categories = sorted(CATEGORIES)
    routes = ("VISUAL", "SPEECH", "HYBRID")
    queries = []
    for index in range(32):
        category = categories[index % len(categories)]
        negative = category == "plausible_negative"
        queries.append({
            "query_id": f"query-{index:02d}", "query": f"Natural query {index}",
            "category": category, "expected_route": routes[index % len(routes)],
            "relevant_intervals": [] if negative else [[index, index + 5]],
            "human_rationale": "Reviewed directly in the source video.",
            "negative": negative,
        })
    return {
        "suite_id": "final-english-long-video-acceptance-v1",
        "annotation_status": "frozen",
        "annotation_frozen_at": "2026-09-15T12:00:00+03:00",
        "videos": [{
            "video_id": "lecture", "local_path": str(media),
            "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            "duration_seconds": 2400, "primary_language": "English",
            "media_kind": "real_continuous", "previously_used_for_tuning": False,
            "usage_rights": "Private local testing permitted.",
            "human_inspection_completed": True, "queries": queries,
        }],
    }


def test_final_manifest_requires_frozen_real_english_long_video(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate(path)["queries"] == 32
    manifest["videos"][0]["media_kind"] = "synthetic_concat"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Synthetic|concatenated|synthetic"):
        validate(path)


def test_empty_draft_template_is_valid_preparation_but_cannot_be_frozen(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text(json.dumps({"annotation_status": "draft", "videos": []}), encoding="utf-8")
    assert validate(path, require_frozen=False)["videos"] == 0
    with pytest.raises(ValueError, match="freeze"):
        validate(path)


def test_final_manifest_rejects_wrong_duration_and_unreviewed_queries(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    manifest["videos"][0]["duration_seconds"] = 1799
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="30 and 60"):
        validate(path)
    manifest = _manifest(tmp_path)
    manifest["videos"][0]["human_inspection_completed"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Human-inspect|human-inspect"):
        validate(path)


def test_final_summary_applies_frozen_gates_and_negative_ux(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = validate(path)["manifest_sha256"]
    rows = []
    for query in manifest["videos"][0]["queries"]:
        rows.append({
            "query_id": query["query_id"],
            "auto_selected_route": query["expected_route"],
            "useful_top_1": not query["negative"],
            "useful_top_3": not query["negative"],
            "useful_top_5": not query["negative"],
            "best_useful_result_rank": None if query["negative"] else 1,
            "timestamp_quality": "direct",
            "user_usefulness": "PASS",
            "failure_categories": [], "failure_reason": "Human-reviewed useful result.",
            "search_latency_ms": 12,
            **({"conservative_ux_understandable": True} if query["negative"] else {}),
        })
    observations = {
        "manifest_sha256": manifest_hash, "run_status": "complete",
        "pipeline": {"upload_seconds": 4, "total_processing_seconds": 200,
                     "peak_worker_ram_bytes": 3_000_000_000, "disk_bytes": 400_000_000,
                     "frame_count": 480, "asr_segment_count": 300,
                     "failures": [], "retries": 0},
        "queries": rows,
    }
    observation_path = tmp_path / "observations.json"
    observation_path.write_text(json.dumps(observations), encoding="utf-8")
    result = summarize(path, observation_path)
    assert result["gate"]["passed"] is True
    assert result["auto_routing_accuracy"] == 1
    assert result["negative_ux_understandable_rate"] == 1
    assert result["negative_misleading_rate"] == 0
    assert result["mrr_at_5"] == 1
    assert result["positive_outcomes"] == {"PASS": 28}
    assert sum(group["queries"] for group in result["category_breakdown"].values()) == 28
