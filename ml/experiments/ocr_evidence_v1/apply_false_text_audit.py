"""Validate the frozen manual line audit and attach its rate to the OCR report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "ml" / "evaluation" / "ocr_evidence_v1_manifest.json"
RAW = ROOT / "data" / "ocr-evidence-v1" / "raw-validation-rapidocr.json"
AUDIT = ROOT / "ml" / "evaluation" / "OCR_EVIDENCE_V1_FALSE_TEXT_AUDIT.json"
REPORT = ROOT / "ml" / "evaluation" / "reports" / "ocr-evidence-v1.json"


def main() -> None:
    manifest_bytes = MANIFEST.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    if audit["manifest_sha256"] != manifest_hash or report["manifest_sha256"] != manifest_hash:
        raise SystemExit("audit or report does not match the frozen manifest")

    expected_frames = {
        (source["source_id"], frame_id)
        for source in manifest["sources"]
        if source["split"] == "validation"
        for frame_id in source["false_text_audit_frame_ids"]
    }
    raw_lines = {
        (source["source_id"], frame["frame_id"], index): line["text"]
        for source in raw["sources"]
        for frame in source["frames"]
        if (source["source_id"], frame["frame_id"]) in expected_frames
        for index, line in enumerate(frame["lines"])
    }
    reviewed = {
        (row["source_id"], row["frame_id"], row["line_index"]): row
        for row in audit["rows"]
    }
    if set(raw_lines) != set(reviewed):
        raise SystemExit("manual audit does not cover every line on every predeclared frame")
    if any(raw_lines[key] != row["ocr_text"] for key, row in reviewed.items()):
        raise SystemExit("manual audit line text does not match raw OCR output")

    false_count = sum(row["verdict"] == "false_non_text" for row in reviewed.values())
    report["metrics"]["false_text_rate"] = false_count / len(reviewed)
    report["metrics"]["false_text_audit"] = {
        "reviewed_lines": len(reviewed),
        "false_lines": false_count,
        "predeclared_frames": len(expected_frames),
        "artifact": "ml/evaluation/OCR_EVIDENCE_V1_FALSE_TEXT_AUDIT.json",
    }
    report["metrics"].pop("false_text_rate_note", None)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"]["false_text_audit"], indent=2))


if __name__ == "__main__":
    main()
