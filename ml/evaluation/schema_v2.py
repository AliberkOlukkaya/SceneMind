"""Versioned natural-video benchmark schema and split-integrity validation."""

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class QueryType(StrEnum):
    OBJECT = "OBJECT"
    SCENE = "SCENE"
    SPEECH = "SPEECH"
    COMPOSITIONAL = "COMPOSITIONAL"
    ACTION_TEMPORAL = "ACTION_TEMPORAL"
    NEGATIVE = "NEGATIVE"


class Source(BaseModel):
    page_url: HttpUrl
    download_url: HttpUrl
    license: str = Field(min_length=1)
    license_url: HttpUrl
    attribution: str = Field(min_length=1)
    source_sha1: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segment_start: float = Field(ge=0)
    segment_end: float = Field(gt=0)
    preparation: str = Field(min_length=1)

    @model_validator(mode="after")
    def increasing_segment(self):
        if self.segment_end <= self.segment_start:
            raise ValueError("source segment must increase")
        return self


class Query(BaseModel):
    query_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    query_text: str = Field(min_length=3, max_length=500)
    query_type: QueryType
    expected_presence: bool
    relevant_intervals: list[tuple[float, float]]
    modality_requirement: Literal["visual", "speech", "hybrid", "temporal"]
    evaluation_modes: list[Literal["visual", "speech", "hybrid"]] = Field(min_length=1)
    notes: str = Field(min_length=1)
    annotation_version: str = Field(pattern=r"^2\.\d+\.\d+$")

    @model_validator(mode="after")
    def consistent_presence(self):
        if self.expected_presence != bool(self.relevant_intervals):
            raise ValueError("expected_presence must match interval presence")
        if self.query_type == QueryType.NEGATIVE and self.expected_presence:
            raise ValueError("NEGATIVE queries cannot be present")
        if self.query_type != QueryType.NEGATIVE and not self.expected_presence:
            raise ValueError("absent queries must use NEGATIVE taxonomy")
        if len(set(self.evaluation_modes)) != len(self.evaluation_modes):
            raise ValueError("evaluation_modes must be unique")
        for start, end in self.relevant_intervals:
            if start < 0 or end <= start:
                raise ValueError("intervals must be nonnegative and increasing")
        ordered = sorted(self.relevant_intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("intervals must not overlap")
        return self


class Video(BaseModel):
    video_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    path: Path
    domain: Literal["everyday", "street", "education", "sports", "speech"]
    split: Literal["calibration", "heldout"]
    source_group: str = Field(min_length=1)
    source: Source
    queries: list[Query] = Field(min_length=1)


class Benchmark(BaseModel):
    schema_version: Literal["2.0.0"]
    benchmark_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    name: str = Field(min_length=1)
    annotation_status: Literal["frozen"]
    annotation_frozen_at: str
    annotation_policy: str = Field(min_length=1)
    videos: list[Video] = Field(min_length=2)

    @model_validator(mode="after")
    def integrity(self):
        ids = [video.video_id for video in self.videos]
        query_ids = [query.query_id for video in self.videos for query in video.queries]
        if len(ids) != len(set(ids)) or len(query_ids) != len(set(query_ids)):
            raise ValueError("video_id and query_id must be globally unique")
        splits = Counter(video.split for video in self.videos)
        if not splits["calibration"] or not splits["heldout"]:
            raise ValueError("calibration and heldout videos are required")
        calibration_groups = {v.source_group for v in self.videos if v.split == "calibration"}
        heldout_groups = {v.source_group for v in self.videos if v.split == "heldout"}
        if calibration_groups & heldout_groups:
            raise ValueError("source_group leakage across splits")
        calibration_hashes = {v.source.source_sha256 for v in self.videos if v.split == "calibration"}
        heldout_hashes = {v.source.source_sha256 for v in self.videos if v.split == "heldout"}
        if calibration_hashes & heldout_hashes:
            raise ValueError("content checksum leakage across splits")
        for video in self.videos:
            duration = video.source.segment_end - video.source.segment_start
            for query in video.queries:
                if any(end > duration for _, end in query.relevant_intervals):
                    raise ValueError(f"{query.query_id} exceeds prepared segment duration")
        heldout_types = {query.query_type for video in self.videos if video.split == "heldout"
                         for query in video.queries}
        if set(QueryType) - heldout_types:
            raise ValueError("heldout split must cover the complete query taxonomy")
        return self


def load_benchmark(path: Path) -> Benchmark:
    return Benchmark.model_validate_json(path.read_text(encoding="utf-8"))
