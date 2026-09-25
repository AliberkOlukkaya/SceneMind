import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ml" / "evaluation" / "ocr_evidence_v1_manifest.json"


def test_validation_manifest_is_frozen_and_source_disjoint():
    payload = MANIFEST.read_bytes()
    manifest = json.loads(payload)
    expected = MANIFEST.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(payload).hexdigest() == expected
    assert manifest["status"] == "frozen_before_validation"
    assert len(manifest["candidates"]) == 2
    assert manifest["selected_engine"].startswith("rapidocr-")

    development = {row["media_sha256"] for row in manifest["sources"] if row["split"] == "development"}
    validation = {row["media_sha256"] for row in manifest["sources"] if row["split"] == "validation"}
    assert len(development) == len(validation) == 4
    assert development.isdisjoint(validation)


def test_validation_covers_all_categories_and_records_a_sampling_miss():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = {row["source_id"]: row for row in manifest["sources"]}
    validation = [row for row in manifest["targets"] if sources[row["source_id"]]["split"] == "validation"]
    assert {sources[row["source_id"]]["category"] for row in validation} == {
        "presentation",
        "coding",
        "ui",
        "real",
    }
    assert len(validation) == 22
    misses = [row for row in validation if not row["sampled_frame_ids"]]
    assert [row["target_id"] for row in misses] == ["val-real-long"]
    assert misses[0]["diagnostic_1s_frame_ids"]


def test_false_text_audit_frames_are_predeclared():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validation = [row for row in manifest["sources"] if row["split"] == "validation"]
    assert all(len(row["false_text_audit_frame_ids"]) == 2 for row in validation)


def test_false_text_audit_matches_frozen_manifest_and_uses_known_verdicts():
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    audit = json.loads(
        (ROOT / "ml" / "evaluation" / "OCR_EVIDENCE_V1_FALSE_TEXT_AUDIT.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["manifest_sha256"] == manifest_digest
    assert len(audit["rows"]) == 160
    assert {row["verdict"] for row in audit["rows"]} == {
        "false_non_text",
        "supported_by_visible_text",
    }
