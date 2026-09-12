"""Frozen development-set and scorer-artifact schemas."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ml.experiments.image_text_verifier.schema import CalibrationVideo


class DevelopmentSet(BaseModel):
    schema_version: Literal["2.0.0"]
    dataset_id: Literal["scenemind-lightweight-pair-calibration-v2"]
    annotation_status: Literal["frozen"]
    annotation_frozen_at: str = Field(min_length=1)
    parent_benchmark_id: Literal["scenemind-natural-video-v2"]
    parent_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_calibration_id: Literal["scenemind-verifier-calibration-v1"]
    parent_calibration_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_policy: str = Field(min_length=1)
    videos: list[CalibrationVideo] = Field(min_length=1)

    @model_validator(mode="after")
    def integrity(self):
        for label, values in (
            ("video IDs", [video.video_id for video in self.videos]),
            ("source groups", [video.source_group for video in self.videos]),
            ("checksums", [video.source.source_sha256 for video in self.videos]),
            ("query IDs", [query.query_id for video in self.videos for query in video.queries]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label}")
        if not any(query.expected_presence for video in self.videos for query in video.queries):
            raise ValueError("positive development queries required")
        if not any(not query.expected_presence for video in self.videos for query in video.queries):
            raise ValueError("negative development queries required")
        for video in self.videos:
            duration = video.source.segment_end - video.source.segment_start
            if any(end > duration for query in video.queries for _, end in query.relevant_intervals):
                raise ValueError(f"annotation exceeds duration: {video.video_id}")
        return self


def load_development(path: Path) -> DevelopmentSet:
    return DevelopmentSet.model_validate_json(path.read_text(encoding="utf-8"))


def assert_all_disjoint(development: DevelopmentSet, previous, benchmark_videos) -> None:
    heldout = [video for video in benchmark_videos if video.split == "heldout"]
    collections = (development.videos, previous.videos, heldout)
    for index, left in enumerate(collections):
        for right in collections[index + 1 :]:
            if {video.source_group for video in left} & {video.source_group for video in right}:
                raise ValueError("source-group leakage between evidence sets")
            if {video.source.source_sha256 for video in left} & {
                video.source.source_sha256 for video in right
            }:
                raise ValueError("checksum leakage between evidence sets")


class LearnedScorerArtifact(BaseModel):
    schema_version: Literal["1.0.0"]
    scorer_type: Literal["logistic-regression"]
    feature_schema: list[str] = Field(min_length=1)
    feature_mean: list[float]
    feature_scale: list[float]
    weights: list[float]
    intercept: float
    threshold: float = Field(ge=0, le=1)
    training_dataset: str
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_calibration_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_benchmark_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_model_id: str
    candidate_model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime: str
    regularization: float = Field(gt=0)
    random_seed: int
    trainable_parameters: int = Field(gt=0)
    training_seconds: float = Field(ge=0)
    created_at: str
    code_version: str

    @model_validator(mode="after")
    def dimensions(self):
        size = len(self.feature_schema)
        if not size or any(len(values) != size for values in (
            self.feature_mean, self.feature_scale, self.weights
        )):
            raise ValueError("learned scorer feature dimensions differ")
        if any(scale <= 0 for scale in self.feature_scale):
            raise ValueError("feature scales must be positive")
        if self.trainable_parameters != size + 1:
            raise ValueError("trainable parameter count mismatch")
        return self

    def assert_compatible(self, features: list[str], model_id: str, revision: str) -> None:
        if (self.feature_schema, self.candidate_model_id, self.candidate_model_revision) != (
            features, model_id, revision
        ):
            raise ValueError("learned scorer artifact does not match feature/model configuration")


class UFormScorerArtifact(BaseModel):
    schema_version: Literal["1.0.0"]
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_license: str
    runtime: Literal["ONNX Runtime CPUExecutionProvider"]
    image_size: int = Field(gt=0)
    max_text_tokens: int = Field(gt=0)
    embedding_dimensions: int = Field(gt=0)
    model_files: dict[str, int]
    intra_op_threads: int = Field(gt=0)
    inter_op_threads: int = Field(gt=0)
    threshold: float
    threshold_rule: str
    calibration_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_calibration_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_benchmark_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str
    code_version: str
    promotable: bool

    @model_validator(mode="after")
    def files(self):
        if set(self.model_files) != {"image_encoder.onnx", "text_encoder.onnx"}:
            raise ValueError("unexpected UForm ONNX artifact set")
        if any(size <= 0 for size in self.model_files.values()):
            raise ValueError("UForm ONNX artifact sizes must be positive")
        return self

    def assert_compatible(self, model_id: str, revision: str, threads: int) -> None:
        if (self.model_id, self.model_revision, self.intra_op_threads) != (
            model_id, revision, threads
        ):
            raise ValueError("UForm artifact does not match model/runtime configuration")
