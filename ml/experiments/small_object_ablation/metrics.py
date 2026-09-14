"""Metrics whose denominators are explicit human visibility labels."""

from collections import Counter

from .schema import Event, VisibilityManifest

ELIGIBLE_VISIBILITY = {"visible", "partially_cropped"}


def sampling_summary(manifest: VisibilityManifest, split: str | None = None) -> dict:
    events = [row for row in manifest.events if split is None or row.split == split]
    result = {}
    for interval in (5, 2, 1):
        assessments = [
            next(item for item in event.sampling if item.interval_seconds == interval)
            for event in events
        ]
        hits = sum(row.evidence_class in {
            "A_visible_in_sampled_frame", "C_partially_cropped"
        } for row in assessments)
        result[str(interval)] = {
            "events": len(events),
            "events_with_visual_evidence": hits,
            "evidence_recall": hits / len(events) if events else None,
            "evidence_classes": dict(Counter(row.evidence_class for row in assessments)),
        }
    return result


def visible_observations(events: list[Event]):
    for event in events:
        for observation in event.observations:
            if observation.visibility == "visible":
                yield event, observation


def conditional_detector_summary(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["visibility"] == "visible"]
    detected = [row for row in eligible if row["detected"]]
    return {
        "verified_visible_frames": len(eligible),
        "detected_frames": len(detected),
        "visibility_conditioned_recall": len(detected) / len(eligible) if eligible else None,
        "mean_confidence_on_hits": (
            sum(row["confidence"] for row in detected) / len(detected) if detected else None
        ),
        "mean_bbox_area_ratio_on_hits": (
            sum(row["bbox_area_ratio"] for row in detected) / len(detected) if detected else None
        ),
    }
