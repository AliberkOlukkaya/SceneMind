import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ml.experiments.object_detector_branch.detector import YoloXNanoDetector, preprocess
from ml.experiments.object_detector_branch.mapping import MAPPING_REVISION, map_query
from ml.experiments.object_detector_branch.ranking import (
    fit_presence_threshold,
    object_features,
    rerank_candidates,
    summarize,
)
from ml.experiments.object_detector_branch.schema import (
    BoundingBox,
    DetectionEvidence,
    DetectorCalibrationArtifact,
)
from ml.experiments.object_detector_branch.validate import check

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outputs, inputs):
        assert inputs["images"].shape == (1, 3, 416, 416)
        return [self.output]


def evidence(label="bicycle", confidence=0.8, area=0.01):
    side = area**0.5 * 100
    return DetectionEvidence(
        frame_id="frame-1", timestamp=1.0, detected_class=label, confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=side, y2=side), bbox_area_ratio=area,
        center=(side / 200, side / 200), image_width=100, image_height=100,
        detector_model="YOLOX-Nano", detector_revision="test",
    )


def test_preprocessing_letterboxes_without_normalizing_pixels():
    image = np.full((100, 200, 3), 10, dtype=np.uint8)
    tensor, ratio = preprocess(image)
    assert tensor.shape == (1, 3, 416, 416)
    assert tensor.dtype == np.float32
    assert ratio == pytest.approx(2.08)
    assert tensor[0, 0, 0, 0] == 10
    assert tensor[0, 0, 300, 0] == 114


def test_detector_wrapper_decodes_mocked_onnx_output():
    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    raw[0, 0, :4] = [10, 10, np.log(5), np.log(5)]
    raw[0, 0, 4] = 0.9
    raw[0, 0, 6] = 0.8  # COCO class index 1: bicycle
    detector = YoloXNanoDetector(Path("unused.onnx"), session=FakeSession(raw))
    found = detector.detect("frame-1", 2.0, np.zeros((416, 416, 3), dtype=np.uint8))
    assert len(found) == 1
    assert found[0].detected_class == "bicycle"
    assert found[0].confidence == pytest.approx(0.72)
    assert found[0].bbox.model_dump() == pytest.approx(
        {"x1": 60, "y1": 60, "x2": 100, "y2": 100}
    )


def test_detector_reads_unicode_windows_path():
    import cv2

    raw = np.zeros((1, 3549, 85), dtype=np.float32)
    target = ROOT / "data/test-object-detector-unicode/görsel.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))[1]
    encoded.tofile(target)
    detector = YoloXNanoDetector(Path("unused.onnx"), session=FakeSession(raw))
    assert detector.detect_path("frame-1", 0.0, target) == []


def test_alias_mapping_and_query_gate_are_explicit():
    phone = map_query("Find the PHONE!", "OBJECT")
    assert phone.detector_enabled
    assert phone.required_class_groups == [["cell phone"]]
    holding = map_query("Find the man holding a yellow ball.", "COMPOSITIONAL")
    assert holding.required_class_groups == [["sports ball"], ["person"]]
    assert holding.relationship_required
    assert holding.unsupported_terms == ["holding", "yellow"]
    cyclist = map_query("cyclist beside a bus", "COMPOSITIONAL")
    assert cyclist.required_class_groups == [["person"], ["bicycle"], ["bus"]]
    assert not map_query("find moving traffic", "ACTION_TEMPORAL").detector_enabled
    assert not map_query("find a classroom", "NEGATIVE").detector_enabled


def test_object_features_require_every_and_group_and_allow_or_aliases():
    mapping = map_query("person holding a bag", "COMPOSITIONAL")
    detections = [evidence("person", 0.9, 0.2), evidence("handbag", 0.6, 0.02)]
    features = object_features(mapping, detections)
    assert features == pytest.approx({
        "object_confidence": 0.6, "object_area_ratio": 0.02, "matched_groups": 2.0
    })
    assert object_features(mapping, detections[:1])["object_confidence"] == 0


def test_evidence_schema_rejects_inconsistent_geometry():
    payload = evidence().model_dump()
    assert json.loads(evidence().model_dump_json())["coordinate_space"] == "source_pixels"
    with pytest.raises(ValueError, match="area ratio"):
        DetectionEvidence.model_validate({**payload, "bbox_area_ratio": 0.5})


def test_threshold_and_reranking_use_calibration_rows_only():
    mapping = map_query("find a bicycle", "OBJECT")
    rows = [
        {"split": "calibration", "expected_presence": True, "mapping": mapping,
         "detector_candidates": [{"object_confidence": 0.8}]},
        {"split": "calibration", "expected_presence": False, "mapping": mapping,
         "detector_candidates": [{"object_confidence": 0.2}]},
    ]
    assert fit_presence_threshold(rows) > 0.2
    with pytest.raises(ValueError, match="calibration-only"):
        fit_presence_threshold([{**rows[0], "split": "heldout"}, rows[1]])
    row = {
        "mapping": mapping,
        "clip_candidates": [{"timestamp": 1, "score": 0.9}, {"timestamp": 2, "score": 0.8}],
        "detector_candidates": [
            {"timestamp": 1, "score": 0.9, "object_confidence": 0.1, "clip_rank": 1},
            {"timestamp": 2, "score": 0.8, "object_confidence": 0.9, "clip_rank": 2},
        ],
    }
    assert rerank_candidates(row, 1.0)[0]["timestamp"] == 2


def test_artifact_disallows_calibration_heldout_overlap():
    payload = {
        "schema_version": "1.0.0", "model_id": "YOLOX-Nano",
        "model_version": "0.1.1rc0", "model_revision": "a" * 40,
        "model_license": "Apache-2.0", "model_sha256": "b" * 64,
        "model_bytes": 10, "runtime": "ONNX Runtime CPUExecutionProvider",
        "input_size": [416, 416], "confidence_floor": 0.01, "nms_threshold": 0.45,
        "presence_threshold": 0.3, "threshold_rule": "calibration only",
        "mapping_revision": MAPPING_REVISION, "parent_report_sha256": "c" * 64,
        "parent_benchmark_sha256": "d" * 64, "calibration_query_ids": ["cal"],
        "heldout_query_ids": ["held"], "created_at": "2026-09-13T00:00:00Z",
        "code_version": "e" * 40, "promotable": False,
    }
    artifact = DetectorCalibrationArtifact.model_validate(payload)
    artifact.assert_compatible("b" * 64, MAPPING_REVISION)
    with pytest.raises(ValueError, match="overlap"):
        DetectorCalibrationArtifact.model_validate({**payload, "heldout_query_ids": ["cal"]})


def test_metric_summary_counts_abstention_and_interval_recall():
    rows = [
        {"expected_presence": True, "relevant_intervals": [[1, 2]],
         "results": [{"timestamp": 1.5}]},
        {"expected_presence": True, "relevant_intervals": [[1, 2]], "results": []},
        {"expected_presence": False, "relevant_intervals": [],
         "results": [{"timestamp": 4.0}]},
    ]
    result = summarize(rows, "results")["5"]
    assert result["recall"] == 0.5
    assert result["positive_false_abstention_rate"] == 0.5
    assert result["negative_false_accept_rate"] == 1.0


def test_failure_inventory_hashes_still_match_parents():
    inventory = json.loads((
        ROOT / "ml/experiments/object_detector_branch/failure_inventory_v1.json"
    ).read_text(encoding="utf-8"))
    import hashlib

    for key in ("parent_benchmark", "parent_report"):
        parent = ROOT / inventory[key]["path"]
        assert hashlib.sha256(parent.read_bytes()).hexdigest() == inventory[key]["sha256"]


def test_committed_report_artifact_and_evidence_validate():
    assert check(
        ROOT / "ml/evaluation/reports/object-detector-v1.json",
        ROOT / "ml/experiments/object_detector_branch/calibration_v1.json",
    ) == []
