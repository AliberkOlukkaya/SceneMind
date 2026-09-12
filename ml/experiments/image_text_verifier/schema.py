"""Frozen calibration-development schema for the verifier experiment."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ml.evaluation.schema_v2 import Query, Source


class CalibrationVideo(BaseModel):
    video_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    path: Path
    source_group: str = Field(min_length=1)
    source: Source
    queries: list[Query] = Field(min_length=1)

    @model_validator(mode="after")
    def visual_only(self):
        if any(query.evaluation_modes != ["visual"] for query in self.queries):
            raise ValueError("verifier calibration queries must be visual-only")
        return self


class CalibrationSet(BaseModel):
    schema_version: Literal["1.0.0"]
    dataset_id: Literal["scenemind-verifier-calibration-v1"]
    annotation_status: Literal["frozen"]
    annotation_frozen_at: str
    parent_benchmark_id: Literal["scenemind-natural-video-v2"]
    parent_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_policy: str = Field(min_length=1)
    videos: list[CalibrationVideo] = Field(min_length=1)

    @model_validator(mode="after")
    def integrity(self):
        video_ids = [video.video_id for video in self.videos]
        query_ids = [query.query_id for video in self.videos for query in video.queries]
        source_groups = [video.source_group for video in self.videos]
        hashes = [video.source.source_sha256 for video in self.videos]
        for name, values in (
            ("video IDs", video_ids),
            ("query IDs", query_ids),
            ("source groups", source_groups),
            ("checksums", hashes),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name}")
        if not any(query.expected_presence for video in self.videos for query in video.queries):
            raise ValueError("positive calibration queries required")
        if not any(not query.expected_presence for video in self.videos for query in video.queries):
            raise ValueError("negative calibration queries required")
        for video in self.videos:
            duration = video.source.segment_end - video.source.segment_start
            if any(end > duration for query in video.queries for _, end in query.relevant_intervals):
                raise ValueError(f"annotation exceeds duration: {video.video_id}")
        return self


def load_calibration(path: Path) -> CalibrationSet:
    return CalibrationSet.model_validate_json(path.read_text(encoding="utf-8"))


def assert_disjoint(calibration: CalibrationSet, heldout_videos) -> None:
    heldout_groups = {video.source_group for video in heldout_videos if video.split == "heldout"}
    heldout_hashes = {
        video.source.source_sha256 for video in heldout_videos if video.split == "heldout"
    }
    if {video.source_group for video in calibration.videos} & heldout_groups:
        raise ValueError("calibration source-group leakage into held-out")
    if {video.source.source_sha256 for video in calibration.videos} & heldout_hashes:
        raise ValueError("calibration checksum leakage into held-out")


class Preprocessing(BaseModel):
    image_size: int = Field(gt=0)
    max_text_tokens: int = Field(gt=0)
    processor: str = Field(min_length=1)


class CalibrationArtifact(BaseModel):
    schema_version: Literal["1.0.0"]
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_license: str = Field(min_length=1)
    preprocessing: Preprocessing
    match_label_index: Literal[0, 1]
    threshold: float = Field(ge=0, le=1)
    threshold_rule: str = Field(min_length=1)
    calibration_dataset: str = Field(min_length=1)
    calibration_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_benchmark: str = Field(min_length=1)
    benchmark_schema_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    promotable: bool

    def assert_compatible(self, model_id: str, revision: str, preprocessing: Preprocessing):
        if (self.model_id, self.model_revision, self.preprocessing) != (
            model_id,
            revision,
            preprocessing,
        ):
            raise ValueError("verifier calibration artifact does not match model configuration")
