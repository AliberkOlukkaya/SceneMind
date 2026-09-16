import json

import pytest

from ml.evaluation.run_hybrid_fusion_holdout_v1 import (
    CAP,
    MANIFEST,
    validate_manifest,
)


def test_frozen_holdout_checksum_composition_and_isolation():
    validated = validate_manifest()
    assert validated["sha256"] == (
        "dcddffb0495cb7c397249ccd13d5818967127b6c62d01173944d0235030b64b5"
    )
    assert validated["counts"] == {
        "SPEECH": 11,
        "VISUAL": 8,
        "MULTIMODAL": 9,
        "NEGATIVE": 6,
    }
    manifest = validated["manifest"]
    assert len(manifest["queries"]) == 34
    assert CAP.variant_id == "cap_1_50"
    assert CAP.overlap_cap == 1.50


def test_frozen_holdout_rejects_manifest_mutation(tmp_path, monkeypatch):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["queries"][0]["text"] += " changed"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "ml.evaluation.run_hybrid_fusion_holdout_v1.CHECKSUM", tmp_path / "expected.sha256"
    )
    (tmp_path / "expected.sha256").write_text(
        "dcddffb0495cb7c397249ccd13d5818967127b6c62d01173944d0235030b64b5  changed.json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum changed"):
        validate_manifest(changed)
