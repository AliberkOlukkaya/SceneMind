"""BM25 transcript retrieval plus reciprocal-rank fusion over frame neighborhoods."""

import math
import re
from collections import Counter
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Segment, Transcript, engine
from app.video import folder_for, read_manifest
from app.visual import index_status, visual_search

router = APIRouter(prefix="/videos", tags=["hybrid search"])
STOPWORDS = {"a", "an", "the", "is", "are", "where", "when", "does", "find", "show", "in", "of"}


def tokens(text):
    return [token for token in re.findall(r"\w+", text.casefold()) if token not in STOPWORDS]


def bm25(query, documents):
    """Robertson BM25, k1=1.2, b=0.75, positive smoothed IDF."""
    counts = [Counter(tokens(document)) for document in documents]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(1, len(lengths)) or 1
    scores = [0.0] * len(documents)
    for term in set(tokens(query)):
        frequency = sum(term in count for count in counts)
        idf = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
        for i, count in enumerate(counts):
            tf = count[term]
            scores[i] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[i] / average))
    return scores


def speech_results(video_id, query, frames, k):
    with Session(engine()) as session:
        job = session.get(Transcript, video_id)
        if job is None or job.status != "ready":
            return [], False
        segments = session.scalars(
            select(Segment).where(Segment.video_id == video_id).order_by(Segment.start)
        ).all()
        scores = bm25(query, [segment.text for segment in segments])
        ranked = sorted(range(len(segments)), key=lambda i: (-scores[i], segments[i].start))
        results = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            segment = segments[i]
            frame = min(frames, key=lambda frame: abs(frame["timestamp"] - segment.start))
            results.append(
                {
                    "timestamp": segment.start,
                    "end": segment.end,
                    "thumbnail": frame["thumbnail"],
                    "score": scores[i],
                    "text": segment.text,
                    "modality": "speech",
                }
            )
            if len(results) == k:
                break
        return results, True


def fuse(visual, speech, k):
    """One vote per modality per nearest-frame bucket, RRF constant 60."""
    merged = {}
    for modality, results in [("visual", visual), ("speech", speech)]:
        seen = set()
        for rank, result in enumerate(results, 1):
            key = result["thumbnail"]
            if key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(key, {**result, "score": 0.0, "evidence": {}})
            item["score"] += 1 / (60 + rank)
            item["evidence"][modality] = {"rank": rank, "score": result["score"]}
            if modality == "speech":
                item.update(text=result["text"], timestamp=result["timestamp"], end=result["end"])
            item["modality"] = "+".join(item["evidence"])
    return sorted(merged.values(), key=lambda item: (-item["score"], item["timestamp"]))[:k]


@router.get("/{video_id}/search")
def search(
    video_id: str,
    q: str = Query(min_length=1, max_length=500),
    k: int = Query(default=10, ge=1, le=50),
    mode: Literal["visual", "speech", "hybrid"] = "visual",
):
    if not q.strip():
        raise HTTPException(422, "Enter a search query.")
    if mode == "visual":
        return visual_search(video_id, q, k)
    folder = folder_for(video_id)
    record = read_manifest(folder)
    if record["status"] != "ready":
        raise HTTPException(409, "Wait for video processing to finish.")
    speech, has_speech = speech_results(folder.name, q, record["frames"], 50)
    if mode == "speech":
        if not has_speech:
            raise HTTPException(409, "Transcribe this video before searching speech.")
        return {"query": q, "score_type": "bm25", "results": speech[:k]}
    visual = []
    used = ["speech"] if has_speech else []
    if index_status(folder)["status"] == "ready":
        visual = visual_search(video_id, q, 50)["results"]
        used.append("visual")
    if not used:
        raise HTTPException(409, "Build a visual index or transcribe this video first.")
    return {
        "query": q,
        "score_type": "reciprocal_rank_fusion",
        "modalities_used": used,
        "results": fuse(visual, speech, k),
    }
