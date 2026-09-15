"""Build and validate the frozen source-disjoint no-match manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("manifest_v1.json")
PERSONAL_MANIFEST = ROOT / "ml/evaluation/personal_acceptance_manifest_v1.json"
ROUTES = {"VISUAL", "SPEECH", "HYBRID"}
TAXONOMY = set("ABCDEFGHI")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(path: Path = MANIFEST) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["annotation_status"] != "frozen":
        raise ValueError("no-match annotations must be frozen")
    if data["personal_acceptance_manifest_sha256"] != digest(PERSONAL_MANIFEST):
        raise ValueError("personal acceptance manifest changed")
    groups: dict[str, set[str]] = {"calibration": set(), "heldout": set()}
    ids: set[str] = set()
    counts = Counter()
    taxonomies = set()
    for source in data["sources"]:
        split = source["split"]
        if split not in groups:
            raise ValueError(f"invalid split: {split}")
        if source["source_group"] in groups["heldout" if split == "calibration" else "calibration"]:
            raise ValueError("calibration and held-out source groups overlap")
        groups[split].add(source["source_group"])
        if len(source["sha256"]) != 64 or not source["license"] or not source["page"]:
            raise ValueError(f"incomplete provenance: {source['source_id']}")
        for query in source["queries"]:
            if query["query_id"] in ids:
                raise ValueError(f"duplicate query id: {query['query_id']}")
            ids.add(query["query_id"])
            if query["route"] not in ROUTES:
                raise ValueError(f"invalid route: {query['query_id']}")
            if query["expected_presence"] != bool(query["relevant_intervals"]):
                raise ValueError(f"presence/interval mismatch: {query['query_id']}")
            if not query["expected_presence"]:
                taxonomy = query.get("negative_taxonomy")
                if taxonomy not in TAXONOMY:
                    raise ValueError(f"missing negative taxonomy: {query['query_id']}")
                taxonomies.add(taxonomy)
            counts[(split, query["route"], query["expected_presence"])] += 1
    if groups["calibration"] & groups["heldout"]:
        raise ValueError("source leakage")
    expected = {
        ("calibration", route, presence): 12
        for route in ROUTES for presence in (False, True)
    } | {
        ("heldout", "VISUAL", presence): 12 for presence in (False, True)
    } | {
        ("heldout", route, presence): 6
        for route in ("SPEECH", "HYBRID") for presence in (False, True)
    }
    if any(counts[key] != value for key, value in expected.items()):
        raise ValueError(f"unbalanced query counts: {dict(counts)}")
    if taxonomies != TAXONOMY:
        raise ValueError(f"incomplete negative taxonomy: {sorted(taxonomies)}")
    return {
        "manifest_sha256": digest(path),
        "source_groups": {key: sorted(value) for key, value in groups.items()},
        "counts": {f"{s}:{r}:{'positive' if p else 'negative'}": counts[(s, r, p)]
                   for s, r, p in sorted(counts)},
        "negative_taxonomy": sorted(taxonomies),
    }


if __name__ == "__main__":
    print(json.dumps(validate_manifest(), indent=2))
