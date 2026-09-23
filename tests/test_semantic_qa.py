import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.qa import ABSTENTION, GeneratedAnswer, GeneratedClaim, resolve_answer
from app.semantic_qa import (
    EMBEDDING_DIMENSION,
    SemanticIndex,
    build_representations,
    select_hierarchical_evidence,
    semantic_retrieve,
)

VIDEO_ID = "00000000-0000-0000-0000-000000000071"
ROOT = Path(__file__).parents[1]


class FakeEncoder:
    dimension = EMBEDDING_DIMENSION

    def encode(self, texts):
        vectors = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for row, text in enumerate(texts):
            if "cache" in text.lower():
                vectors[row, 0] = 1
            elif "ground" in text.lower():
                vectors[row, 1] = 1
            else:
                vectors[row, 2] = 1
        return vectors


def segments():
    return [
        SimpleNamespace(id=1, start=0.0, end=6.0, text="A short introduction."),
        SimpleNamespace(id=2, start=6.0, end=12.0, text="Semantic search finds meaning."),
        SimpleNamespace(id=3, start=12.0, end=18.0, text="Grounded answers cite evidence."),
        SimpleNamespace(id=4, start=18.0, end=25.0, text="A cache avoids repeated work."),
        SimpleNamespace(id=5, start=25.0, end=31.0, text="Indexes can be reloaded."),
    ]


def test_two_level_representation_is_stable_and_maps_segments():
    fine, context = build_representations(VIDEO_ID, segments())
    assert fine == build_representations(VIDEO_ID, segments())[0]
    assert all(unit.level == "fine" and unit.segment_ids for unit in fine)
    assert all(unit.level == "context" and unit.fine_ids for unit in context)
    assert [fine_id for unit in context for fine_id in unit.fine_ids] == [
        unit.unit_id for unit in fine
    ]
    assert set(segment for unit in fine for segment in unit.segment_ids) == {1, 2, 3, 4, 5}


def test_semantic_query_retrieves_expected_unit():
    fine, _ = build_representations(VIDEO_ID, segments())
    index = SemanticIndex.build(fine, FakeEncoder())
    hits = semantic_retrieve("How does the cache help?", index, FakeEncoder(), 2)
    assert "cache" in hits[0].unit.text.lower()
    assert hits[0].rank == 1


def test_index_persistence_preserves_stable_mapping(tmp_path):
    fine, _ = build_representations(VIDEO_ID, segments())
    original = SemanticIndex.build(fine, FakeEncoder())
    original.save(tmp_path)
    restored = SemanticIndex.load(tmp_path)
    assert restored.units == original.units
    assert restored.index.ntotal == len(fine)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["units"][0]["segment_ids"]


def test_hierarchical_selection_is_bounded_ordered_and_deduplicated():
    fine, context = build_representations(VIDEO_ID, segments())
    encoder = FakeEncoder()
    fine_hits = semantic_retrieve("cache", SemanticIndex.build(fine, encoder), encoder, 5)
    context_hits = semantic_retrieve("cache", SemanticIndex.build(context, encoder), encoder, 2)
    selected = select_hierarchical_evidence(
        "Why does the cache help?",
        fine,
        fine_hits,
        context_hits,
        max_units=3,
        max_characters=150,
    )
    assert len(selected.evidence) <= 3
    assert selected.total_characters <= 150
    assert [item.start_seconds for item in selected.evidence] == sorted(
        item.start_seconds for item in selected.evidence
    )
    assert len({item.evidence_id for item in selected.evidence}) == len(selected.evidence)
    assert all(item.segment_ids for item in selected.evidence)


def test_minimum_sufficient_evidence_can_cover_multiple_intervals():
    fine, context = build_representations(VIDEO_ID, segments())
    encoder = FakeEncoder()
    selection = select_hierarchical_evidence(
        "How do grounded answers and caches help?",
        fine,
        semantic_retrieve("ground cache", SemanticIndex.build(fine, encoder), encoder, 5),
        semantic_retrieve("ground cache", SemanticIndex.build(context, encoder), encoder, 2),
    )
    intervals = [(12, 18), (18, 25)]
    assert all(
        any(item.start_seconds <= end and item.end_seconds >= start for item in selection.evidence)
        for start, end in intervals
    )


def test_candidate_citations_resolve_to_fine_timestamps_and_abstention_is_safe():
    fine, _ = build_representations(VIDEO_ID, segments())
    evidence = select_hierarchical_evidence(
        "cache", fine, [], [], max_units=1
    ).evidence
    generated_answer = GeneratedAnswer(
        answerable=True,
        answer="The cache avoids repeated work.",
        evidence_ids=[evidence[0].evidence_id],
        claims=[
            GeneratedClaim(
                text="The cache avoids repeated work.",
                evidence_ids=[evidence[0].evidence_id],
            )
        ],
        unsupported_or_missing=[],
    )
    resolved = resolve_answer(generated_answer, list(evidence), "How does the cache help?")
    assert resolved["citations"][0]["start_seconds"] == evidence[0].start_seconds
    generated = GeneratedAnswer(
        answerable=False,
        answer="outside knowledge",
        evidence_ids=[],
        claims=[],
        unsupported_or_missing=["missing"],
    )
    assert resolve_answer(generated, []) == {
        "answerable": False,
        "answer": ABSTENTION,
        "citations": [],
    }


@pytest.mark.parametrize("source_type", ["upload", "url"])
def test_representation_is_compatible_with_both_source_types(source_type):
    manifest = {"source_type": source_type, "id": VIDEO_ID}
    fine, context = build_representations(manifest["id"], segments())
    assert fine and context


def test_semantic_hierarchical_manifest_is_frozen_balanced_and_disjoint():
    path = ROOT / "ml/evaluation/semantic_hierarchical_qa_v1_manifest.json"
    expected = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    manifest = json.loads(path.read_text(encoding="utf-8"))
    development = manifest["development"]["questions"]
    validation = manifest["validation"]["questions"]
    assert len(development) == len(validation) == 45
    assert sum(item["answerable"] for item in development) == 33
    assert sum(item["answerable"] for item in validation) == 33
    development_sources = {item["source_id"] for item in development}
    validation_sources = {item["source_id"] for item in validation}
    assert len(development_sources) == len(validation_sources) == 3
    assert development_sources.isdisjoint(validation_sources)
    assert all(item["key_answer_facts"] for item in validation if item["answerable"])
    assert all(
        item["minimum_sufficient_evidence"] for item in validation if item["answerable"]
    )
    frozen = manifest["frozen_configuration"]
    assert frozen["embedding_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert hashlib.sha256((ROOT / "backend/app/semantic_qa.py").read_bytes()).hexdigest() == frozen[
        "semantic_implementation_sha256"
    ]


def test_semantic_hierarchical_report_records_rejected_decision_b():
    report = json.loads(
        (ROOT / "ml/evaluation/reports/semantic-hierarchical-qa-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["single_frozen_validation_run"] is True
    assert report["decision"]["code"] == "B"
    assert report["decision"]["production_changed"] is False
    assert report["gates"]["passed"] is False
    assert report["qa"]["candidate"]["correct_abstention"] == 1
    assert report["qa"]["candidate"]["false_answer_rate"] == 0
    assert report["qa"]["candidate"]["answer_correctness"] < report["qa"]["baseline"][
        "answer_correctness"
    ]
