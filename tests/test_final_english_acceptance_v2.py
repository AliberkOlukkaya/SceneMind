import copy
import json
from pathlib import Path

import pytest

from ml.evaluation.final_english_acceptance_v2 import (
    canonical_manifest_sha256,
    summarize_report,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ml/evaluation/final_english_acceptance_v2_manifest.json"
REPORT = ROOT / "ml/evaluation/reports/final-english-acceptance-v2.json"


def test_frozen_v2_manifest_is_self_consistent_and_balanced():
    result = validate_manifest(MANIFEST)
    assert result == {
        "manifest_sha256": "ce633b525af0cc3a347c51017c4019eb173e376246783b8153bd6dc8b4c60b7c",
        "queries": 30,
        "distribution": {"SPEECH": 10, "VISUAL": 8, "MULTIMODAL": 8, "NEGATIVE": 4},
    }


def test_manifest_checksum_detects_post_freeze_changes(tmp_path):
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    document["queries"][0]["text"] += " changed"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert canonical_manifest_sha256(document) != document["manifest_sha256"]
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(path)


def test_report_preserves_hybrid_path_and_pass_level_metrics():
    summary = summarize_report(MANIFEST, REPORT)
    assert summary["queries"] == 30
    assert summary["useful_top_1"] == pytest.approx(20 / 26)
    assert summary["useful_top_3"] == pytest.approx(20 / 26)
    assert summary["useful_top_5"] == pytest.approx(22 / 26)
    assert summary["mrr_at_5"] == pytest.approx(20.45 / 26)
    assert summary["positive_outcomes"] == {"PASS": 22, "PARTIAL": 1, "FAIL": 3}
    assert summary["negative_ux"] == {"ACCEPTABLE": 4}
    assert summary["gates"]["top_3"] is False
    assert summary["gates"]["top_5"] is False


def test_partial_result_cannot_inflate_useful_top_k(tmp_path):
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    partial = next(row for row in report["queries"] if row["outcome"] == "PARTIAL")
    partial["useful_top_5"] = True
    partial["best_useful_rank"] = 5
    changed = tmp_path / "report.json"
    changed.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot inflate"):
        summarize_report(MANIFEST, changed)


def test_primary_auto_or_explicit_route_substitution_is_rejected(tmp_path):
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    changed_document = copy.deepcopy(report)
    changed_document["queries"][0]["selected_route"] = "speech"
    changed = tmp_path / "report.json"
    changed.write_text(json.dumps(changed_document), encoding="utf-8")
    with pytest.raises(ValueError, match="non-Hybrid"):
        summarize_report(MANIFEST, changed)
