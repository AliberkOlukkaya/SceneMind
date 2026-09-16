import json
from pathlib import Path

import pytest

from ml.evaluation.run_hybrid_fusion_diagnostics import DEFAULT_MANIFEST, validate_manifest


def test_development_manifest_is_source_disjoint_and_balanced():
    result = validate_manifest(DEFAULT_MANIFEST)
    assert result["query_distribution"] == {
        "SPEECH": 11,
        "VISUAL": 8,
        "MULTIMODAL": 9,
        "NEGATIVE": 4,
    }
    assert len(result["manifest"]["queries"]) == 32
    protected = json.loads(
        (DEFAULT_MANIFEST.parents[1] / "final_english_acceptance_v2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        source["sha256"] != protected["video"]["sha256"]
        for source in result["manifest"]["sources"]
    )


def test_development_manifest_rejects_acceptance_query_leakage(tmp_path: Path):
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    protected = json.loads(
        (DEFAULT_MANIFEST.parents[1] / "final_english_acceptance_v2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["queries"][0]["text"] = protected["queries"][0]["text"]
    path = tmp_path / "leaking.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates protected V2"):
        validate_manifest(path)
