"""Opt-in model-bound score gate. Cosine scores are never probabilities."""

import json
import math

from fastapi import HTTPException

from app.config import settings


def threshold():
    if settings.calibration_path is None:
        return None
    try:
        artifact = json.loads(settings.calibration_path.read_text())
        expected = {
            "visual_model": settings.visual_model,
            "revision": settings.visual_revision,
            "sampling_interval": settings.sampling_interval,
            "mode": "visual",
        }
        if any(artifact.get(key) != value for key, value in expected.items()):
            raise ValueError("Calibration configuration mismatch")
        value = float(artifact["threshold"])
        if not math.isfinite(value) or not -1 <= value <= 1.001:
            raise ValueError("Invalid calibration threshold")
        return value
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            503, "Calibration invalid or incompatible; recalibrate or disable it."
        ) from error
