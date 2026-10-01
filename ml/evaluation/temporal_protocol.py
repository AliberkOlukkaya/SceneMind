"""Fail-closed data/audit/freeze contracts for temporal retrieval research."""

import json
import math
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from ml.evaluation.temporal_retrieval import MODEL, REVISION, sha256

CATEGORIES = {
    "STATIC_OBJECT", "SCENE", "HUMAN_ACTION", "OBJECT_INTERACTION",
    "TEMPORAL_EVENT", "SMALL_OBJECT_DETAIL", "MULTIPLE_SIMILAR_MOMENTS",
}
PROTECTED_MANIFEST = "ml/evaluation/FINAL_DEPLOYMENT_ACCEPTANCE_V1_MANIFEST.json"
PROTECTED_SHA = "d563d6b6292ccf624f9db1cf7de2a3db797efd2b513e1cb4076a50209a96f196"
REQUIRED_CODE = {
    "ml/evaluation/temporal_protocol.py", "ml/evaluation/temporal_retrieval.py",
    "scripts/temporal_benchmark.py", "backend/app/video.py", "backend/app/visual.py",
    "backend/app/encoder.py", "backend/app/config.py", "backend/app/main.py",
    "ml/evaluation/TEMPORAL_VIDEO_RETRIEVAL_V1_PROTOCOL.md",
}
DEPENDENCIES = ("torch", "transformers", "numpy", "opencv-python-headless", "faiss-cpu")


def source_keys(source):
    return {canonical_source(url) for url in
            [source["source_url"], *source.get("source_aliases", [])]}


AUDITS = ("wording", "boundaries", "all_occurrences", "ambiguity", "visual_grounding")


def canonical_source(url):
    parsed = urlsplit(url)
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "web.archive.org" and "/http" in parsed.path:
        original = parsed.path[parsed.path.index("/http") + 1:]
        return canonical_source(original + ("?" + parsed.query if parsed.query else ""))
    if host == "youtu.be":
        return "youtube:" + parsed.path.strip("/")
    if host in {"youtube.com", "m.youtube.com"}:
        identifier = parse_qs(parsed.query).get("v", [None])[0]
        if identifier:
            return "youtube:" + identifier
    # Commons URL spellings and tracking query strings do not create new sources.
    return host + unquote(parsed.path).replace("_", " ").casefold()


def validate_manifest(manifest, excluded_sources=()):
    if manifest.get("split") not in {"development", "validation"}:
        raise ValueError("Unknown split")
    sources, queries = manifest["sources"], manifest["queries"]
    if len(sources) < 5 or len(queries) < 60:
        raise ValueError("At least five sources and sixty queries required")
    ids = [s["source_id"] for s in sources]
    urls = [canonical_source(s["source_url"]) for s in sources]
    hashes = [s["media_sha256"] for s in sources]
    if len(set(ids)) != len(ids) or len(set(urls)) != len(urls) or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate sources")
    aliases = [source_keys(s) for s in sources]
    if any(a.intersection(b) for i, a in enumerate(aliases) for b in aliases[i + 1:]):
        raise ValueError("Duplicate source aliases")
    forbidden_urls = {key for s in excluded_sources for key in source_keys(s)}
    forbidden_hashes = {s["media_sha256"] for s in excluded_sources if s.get("media_sha256")}
    if forbidden_urls.intersection(set().union(*aliases)) or forbidden_hashes.intersection(hashes):
        raise ValueError("Source leakage")
    minimum = 1800 if manifest["split"] == "validation" else 1200
    if max(s["duration_seconds"] for s in sources) < minimum:
        raise ValueError("Required long source missing")
    source_map = {s["source_id"]: s for s in sources}
    for source in sources:
        if (not source.get("license") or not source.get("provenance_review")
                or not source.get("full_timeline_review")
                or not math.isfinite(source["duration_seconds"]) or source["duration_seconds"] <= 0):
            raise ValueError("Source licensing/provenance/timeline review incomplete")
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    counts = Counter(q["category"] for q in queries)
    if set(counts) != CATEGORIES:
        raise ValueError("All seven query categories required")
    if counts["HUMAN_ACTION"] + counts["TEMPORAL_EVENT"] < 15 or counts["OBJECT_INTERACTION"] < 10:
        raise ValueError("Insufficient action/interaction queries")
    per_source = Counter(q["video_id"] for q in queries)
    if set(per_source) != set(ids) or min(per_source.values()) < 8:
        raise ValueError("At least eight queries per source required")
    for query in queries:
        duration = source_map[query["video_id"]]["duration_seconds"]
        if not query["query"].strip() or query["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError("Invalid query")
        intervals = query["intervals"]
        if (not intervals or any(not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b <= duration)
                                 for a, b in intervals)):
            raise ValueError("Invalid intervals")
        audit = query.get("audit", {})
        if (not all(audit.get(key) is True for key in AUDITS)
                or not audit.get("reviewer") or not audit.get("evidence_review")):
            raise ValueError("Annotation audit incomplete")
    return True


def verify_freeze(root, manifest_path, expected_sha):
    root, manifest_path = Path(root), Path(manifest_path)
    if sha256(manifest_path) != expected_sha:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    exclusions = manifest.get("excluded_sources")
    if not exclusions or not manifest.get("exclusion_audit"):
        raise ValueError("Historical source exclusion registry/audit missing")
    validate_manifest(manifest, exclusions)
    if sha256(root / PROTECTED_MANIFEST) != PROTECTED_SHA:
        raise ValueError("Protected V1 manifest changed")
    if manifest.get("candidate") != {"model": MODEL, "revision": REVISION}:
        raise ValueError("Unregistered candidate")
    if not REQUIRED_CODE.issubset(manifest.get("frozen_code_sha256", {})):
        raise ValueError("Required code freeze missing")
    if manifest.get("dependencies") != {name: version(name) for name in DEPENDENCIES}:
        raise ValueError("Frozen dependency versions differ")
    for name, expected in manifest["frozen_code_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f"Frozen code mismatch: {name}")
    for source in manifest["sources"]:
        path = (root / source["media_path"]).resolve()
        if not path.is_relative_to((root / "data/temporal-v1/media").resolve()):
            raise ValueError("Media outside isolated experiment")
        if sha256(path) != source["media_sha256"]:
            raise ValueError(f"Media mismatch: {source['source_id']}")
    return manifest
