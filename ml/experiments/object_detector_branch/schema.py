"""Schemas for detector evidence and the calibration-only artifact."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)

    @model_validator(mode="after")
    def increasing(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding-box coordinates must increase")
        return self


class DetectionEvidence(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    frame_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0)
    detected_class: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    bbox_area_ratio: float = Field(gt=0, le=1)
    center: tuple[float, float]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    coordinate_space: Literal["source_pixels"] = "source_pixels"
    detector_model: str = Field(min_length=1)
    detector_revision: str = Field(min_length=1)

    @model_validator(mode="after")
    def geometry(self):
        if self.bbox.x2 > self.image_width or self.bbox.y2 > self.image_height:
            raise ValueError("bounding box exceeds image")
        if not all(0 <= value <= 1 for value in self.center):
            raise ValueError("center must use normalized coordinates")
        expected = (
            (self.bbox.x2 - self.bbox.x1)
            * (self.bbox.y2 - self.bbox.y1)
            / (self.image_width * self.image_height)
        )
        if abs(expected - self.bbox_area_ratio) > 1e-6:
            raise ValueError("bbox area ratio does not match geometry")
        return self


class QueryObjectMapping(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    normalized_query: str
    query_type: str
    detector_enabled: bool
    required_class_groups: list[list[str]]
    matched_aliases: dict[str, list[str]]
    relationship_required: bool
    unsupported_terms: list[str]

    @model_validator(mode="after")
    def routing_consistency(self):
        if self.detector_enabled != bool(self.required_class_groups):
            raise ValueError("detector routing and required classes disagree")
        if any(not group for group in self.required_class_groups):
            raise ValueError("required class groups cannot be empty")
        return self


class DetectorCalibrationArtifact(BaseModel):
    schema_version: Literal["1.0.0"]
    model_id: Literal["YOLOX-Nano"]
    model_version: Literal["0.1.1rc0"]
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_license: Literal["Apache-2.0"]
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_bytes: int = Field(gt=0)
    runtime: Literal["ONNX Runtime CPUExecutionProvider"]
    input_size: tuple[Literal[416], Literal[416]]
    confidence_floor: float = Field(ge=0, le=1)
    nms_threshold: float = Field(gt=0, le=1)
    presence_threshold: float = Field(ge=0, le=1)
    threshold_rule: str = Field(min_length=1)
    mapping_revision: str = Field(min_length=1)
    parent_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_query_ids: list[str] = Field(min_length=1)
    heldout_query_ids: list[str] = Field(min_length=1)
    created_at: str = Field(min_length=1)
    code_version: str = Field(pattern=r"^[0-9a-f]{40}$")
    promotable: bool

    @model_validator(mode="after")
    def disjoint_splits(self):
        if set(self.calibration_query_ids) & set(self.heldout_query_ids):
            raise ValueError("calibration and held-out query IDs overlap")
        return self

    def assert_compatible(self, model_sha256: str, mapping_revision: str) -> None:
        if (self.model_sha256, self.mapping_revision) != (model_sha256, mapping_revision):
            raise ValueError("detector artifact does not match model or mapping")
