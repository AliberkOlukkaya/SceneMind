import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from app.qa import ABSTENTION, GeneratedAnswer, resolve_answer
from app.semantic_qa import EMBEDDING_DIMENSION
from app.video_memory import (
    SectionMemoryIndex,
    build_memory,
    retrieve_from_memory,
    transcript_fingerprint,
)

VIDEO_ID = "00000000-0000-0000-0000-000000000099"


class FakeEncoder:
    dimension = EMBEDDING_DIMENSION

    def encode(self, texts):
        output = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            output[row, 0 if "database" in lowered else 1 if "network" in lowered else 2] = 1
        return output


def segments():
    return [
        SimpleNamespace(id=1, start=0, end=25, text="A database stores durable records."),
        SimpleNamespace(id=2, start=25, end=55, text="Database indexes make lookup efficient."),
        SimpleNamespace(id=3, start=60, end=88, text="A network switch learns device locations."),
        SimpleNamespace(id=4, start=88, end=120, text="Network packets follow learned routes."),
    ]


def test_memory_levels_are_deterministic_traceable_and_semantic():
    first = build_memory(VIDEO_ID, segments(), FakeEncoder())
    second = build_memory(VIDEO_ID, segments(), FakeEncoder())
    assert first == second
    assert len(first.sections) == 2
    assert tuple(fid for section in first.sections for fid in section.fine_ids) == tuple(
        item.unit_id for item in first.fine
    )
    assert set(value for section in first.sections for value in section.segment_ids) == {1, 2, 3, 4}
    assert all(section.summary_fine_ids for section in first.sections)
    assert all(
        section.summary == " ".join(
            item.text for item in first.fine if item.unit_id in section.summary_fine_ids
        )
        for section in first.sections
    )


def test_summary_is_navigation_only_and_evidence_is_original_transcript():
    memory = build_memory(VIDEO_ID, segments(), FakeEncoder())
    index = SectionMemoryIndex.build(memory, FakeEncoder())
    selection = retrieve_from_memory("How does the network route packets?", index, FakeEncoder())
    assert selection.section_ids[0] == memory.sections[1].section_id
    original = {item.unit_id: item for item in memory.fine}
    assert selection.evidence
    assert all(item.evidence_id in original for item in selection.evidence)
    assert all(item.text == original[item.evidence_id].text for item in selection.evidence)
    assert all(not item.evidence_id.startswith("S") for item in selection.evidence)


def test_persistence_round_trip_and_invalidation(tmp_path: Path):
    memory = build_memory(VIDEO_ID, segments(), FakeEncoder())
    index = SectionMemoryIndex.build(memory, FakeEncoder())
    index.save(tmp_path)
    loaded = SectionMemoryIndex.load(
        tmp_path, expected_transcript_fingerprint=memory.transcript_fingerprint
    )
    assert loaded.bundle == memory
    with pytest.raises(ValueError, match="transcript is stale"):
        SectionMemoryIndex.load(tmp_path, expected_transcript_fingerprint="wrong")
    payload = json.loads((tmp_path / "memory.json").read_text())
    payload["configuration_fingerprint"] = "wrong"
    (tmp_path / "memory.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="configuration is stale"):
        SectionMemoryIndex.load(
            tmp_path, expected_transcript_fingerprint=memory.transcript_fingerprint
        )


def test_transcript_fingerprint_changes_with_source_text():
    original = segments()
    changed = segments()
    changed[0].text = "Changed transcript."
    assert transcript_fingerprint(VIDEO_ID, original) != transcript_fingerprint(VIDEO_ID, changed)


def test_evidence_package_bounds_hold():
    memory = build_memory(VIDEO_ID, segments(), FakeEncoder())
    selection = retrieve_from_memory(
        "database", SectionMemoryIndex.build(memory, FakeEncoder()), FakeEncoder(),
        max_units=1, max_characters=100,
    )
    assert len(selection.evidence) == 1
    assert selection.total_characters <= 100


def test_multi_section_navigation_remains_bounded():
    memory = build_memory(VIDEO_ID, segments(), FakeEncoder())
    selection = retrieve_from_memory(
        "How do database records and network routes work?",
        SectionMemoryIndex.build(memory, FakeEncoder()),
        FakeEncoder(),
        section_k=2,
        max_units=2,
        max_characters=160,
    )
    assert len(selection.section_ids) == 2
    assert len(selection.evidence) <= 2
    assert selection.total_characters <= 160
    assert selection.estimated_tokens <= 40


@pytest.mark.parametrize("source_type", ["upload", "url"])
def test_memory_is_compatible_with_upload_and_url_manifests(source_type):
    manifest = {"id": VIDEO_ID, "source_type": source_type}
    memory = build_memory(manifest["id"], segments(), FakeEncoder())
    assert memory.video_id == manifest["id"]
    assert memory.sections


def test_safe_abstention_and_memory_summary_citation_rejection():
    memory = build_memory(VIDEO_ID, segments(), FakeEncoder())
    summary_id = memory.sections[0].section_id
    generated = GeneratedAnswer(
        answerable=True,
        answer="A summary claim.",
        evidence_ids=[summary_id],
        claims=[],
        unsupported_or_missing=[],
    )
    with pytest.raises(HTTPException, match="invalid evidence citation"):
        resolve_answer(generated, [])
    abstention = GeneratedAnswer(
        answerable=False,
        answer="",
        evidence_ids=[],
        claims=[],
        unsupported_or_missing=["not in original transcript evidence"],
    )
    assert resolve_answer(abstention, []) == {
        "answerable": False,
        "answer": ABSTENTION,
        "citations": [],
    }


def test_frozen_manifest_and_report_record_failed_decision():
    root = Path(__file__).parents[1]
    manifest_path = root / "ml/evaluation/hierarchical_video_memory_v1_manifest.json"
    expected = manifest_path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["development"]["questions"]) == 45
    assert len(manifest["validation"]["questions"]) == 45
    assert {q["source_id"] for q in manifest["development"]["questions"]}.isdisjoint(
        {q["source_id"] for q in manifest["validation"]["questions"]}
    )
    assert all(q["relevant_section_ids"] for q in manifest["validation"]["questions"] if q["answerable"])
    report = json.loads(
        (root / "ml/evaluation/reports/hierarchical-video-memory-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["single_frozen_validation_run"] is True
    assert report["decision"]["code"] == "C"
    assert report["decision"]["production_changed"] is False
    assert report["gates"]["passed"] is False
