"""Validated human visibility evidence for the small-object ablation."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Visibility = Literal["visible", "not_visible", "partially_cropped", "too_small_to_judge"]
EvidenceClass = Literal[
    "A_visible_in_sampled_frame",
    "B_only_visible_between_sampled_frames",
    "C_partially_cropped",
    "D_too_small_to_judge",
]


class Source(BaseModel):
    source_id: str
    filename: str
    source_group: str
    page_url: str
    download_url: str
    license: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)


class Observation(BaseModel):
    timestamp: float = Field(ge=0)
    visibility: Visibility
    note: str = Field(min_length=1)


class SamplingAssessment(BaseModel):
    interval_seconds: Literal[1, 2, 5]
    sampled_timestamps: list[float]
    evidence_class: EvidenceClass


class Event(BaseModel):
    event_id: str
    source_id: str
    source_group: str
    split: Literal["calibration", "heldout"]
    query: str
    target_class: Literal["bicycle", "sports ball", "bottle"]
    interval: tuple[float, float]
    observations: list[Observation]
    sampling: list[SamplingAssessment]

    @model_validator(mode="after")
    def validate_event(self):
        if self.interval[1] <= self.interval[0]:
            raise ValueError("event interval must increase")
        if len({row.interval_seconds for row in self.sampling}) != 3:
            raise ValueError("each event requires 1s, 2s and 5s assessments")
        observed = {row.timestamp for row in self.observations}
        if any(ts not in observed for row in self.sampling for ts in row.sampled_timestamps):
            raise ValueError("sampled timestamps must have human observations")
        visibility = {row.timestamp: row.visibility for row in self.observations}
        for assessment in self.sampling:
            sampled = [visibility[ts] for ts in assessment.sampled_timestamps]
            expected = (
                "A_visible_in_sampled_frame" if "visible" in sampled else
                "C_partially_cropped" if "partially_cropped" in sampled else
                "D_too_small_to_judge" if "too_small_to_judge" in sampled else
                "B_only_visible_between_sampled_frames"
            )
            if assessment.evidence_class != expected:
                raise ValueError("evidence class disagrees with sampled-frame visibility")
            if expected.startswith("B_") and not any(
                row.visibility in {"visible", "partially_cropped"}
                for row in self.observations
                if row.timestamp not in assessment.sampled_timestamps
            ):
                raise ValueError("between-frame evidence requires a reviewed visible frame")
        return self


class VisibilityManifest(BaseModel):
    schema_version: Literal["1.0.0"]
    annotation_status: Literal["frozen"]
    annotation_protocol: str
    parent_benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: list[Source]
    events: list[Event]

    @model_validator(mode="after")
    def validate_splits(self):
        source_groups = {source.source_id: source.source_group for source in self.sources}
        calibration = {row.source_group for row in self.events if row.split == "calibration"}
        heldout = {row.source_group for row in self.events if row.split == "heldout"}
        if calibration & heldout:
            raise ValueError("calibration and held-out source groups overlap")
        if any(source_groups.get(row.source_id) != row.source_group for row in self.events):
            raise ValueError("event source group disagrees with source record")
        return self
