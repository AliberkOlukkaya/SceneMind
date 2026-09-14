"""Validate a private, populated personal-video acceptance manifest."""

import argparse
import hashlib
import json
from pathlib import Path

ROUTES = {"VISUAL", "SPEECH", "HYBRID"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def validate(path: Path, require_frozen: bool = True) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if require_frozen and manifest["annotation_status"] != "frozen":
        raise ValueError("freeze manual annotations before retrieval")
    videos = manifest["videos"]
    if not videos:
        raise ValueError("add at least one real personal video")
    query_ids = set()
    for video in videos:
        media = Path(video["local_path"])
        if not media.is_file() or sha256(media) != video["sha256"]:
            raise ValueError(f"missing or changed media: {video['video_id']}")
        for query in video["queries"]:
            if query["query_id"] in query_ids:
                raise ValueError("query IDs must be unique")
            query_ids.add(query["query_id"])
            if query["target_route"] not in ROUTES:
                raise ValueError("target_route must be VISUAL, SPEECH, or HYBRID")
            if not query["relevant_intervals"]:
                raise ValueError("positive personal queries require reviewed intervals")
    return {"videos": len(videos), "queries": len(query_ids),
            "manifest_sha256": sha256(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.manifest, not args.allow_draft), indent=2))
