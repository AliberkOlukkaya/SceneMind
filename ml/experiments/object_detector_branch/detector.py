"""Minimal YOLOX-Nano ONNX Runtime CPU wrapper."""

from pathlib import Path

import cv2
import numpy as np

from ml.experiments.object_detector_branch.schema import BoundingBox, DetectionEvidence

MODEL_ID = "YOLOX-Nano"
MODEL_VERSION = "0.1.1rc0"
MODEL_REVISION = "e1052df71842031413f6030723c3607b839c80ce"
INPUT_SIZE = (416, 416)
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


def preprocess(
    image: np.ndarray, input_size: tuple[int, int] = INPUT_SIZE
) -> tuple[np.ndarray, float]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detector input must be a BGR image")
    height, width = image.shape[:2]
    ratio = min(input_size[0] / height, input_size[1] / width)
    resized = cv2.resize(
        image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8)
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None]
    return tensor, ratio


def _decode(raw: np.ndarray, input_size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    expected_rows = sum((input_size[0] // stride) * (input_size[1] // stride) for stride in (8, 16, 32))
    if raw.shape != (1, expected_rows, 85):
        raise ValueError(f"unexpected YOLOX output shape: {raw.shape}")
    decoded = raw[0].copy()
    grids = []
    strides = []
    for stride in (8, 16, 32):
        height, width = input_size[0] // stride, input_size[1] // stride
        yv, xv = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        strides.append(np.full((height * width, 1), stride))
    grid = np.concatenate(grids)
    stride_values = np.concatenate(strides)
    decoded[:, :2] = (decoded[:, :2] + grid) * stride_values
    decoded[:, 2:4] = np.exp(np.clip(decoded[:, 2:4], -20, 20)) * stride_values
    return decoded


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
    box_area = np.prod(box[2:] - box[:2])
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    return intersection / np.maximum(box_area + areas - intersection, 1e-12)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[_iou(boxes[current], boxes[remaining]) <= threshold]
    return keep


class YoloXNanoDetector:
    def __init__(
        self, model_path: Path, *, threads: int = 4, confidence_floor: float = 0.01,
        nms_threshold: float = 0.45, input_size: tuple[int, int] = INPUT_SIZE, session=None
    ):
        self.model_path = Path(model_path)
        self.confidence_floor = confidence_floor
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        if not 0 <= confidence_floor <= 1 or not 0 < nms_threshold <= 1:
            raise ValueError("invalid detector thresholds")
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        self.session = session
        self.input_name = session.get_inputs()[0].name

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def detect_path(self, frame_id: str, timestamp: float, path: Path) -> list[DetectionEvidence]:
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError as error:
            raise ValueError(f"cannot read detector frame: {path}") from error
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read detector frame: {path}")
        return self.detect(frame_id, timestamp, image)

    def detect(
        self, frame_id: str, timestamp: float, image: np.ndarray
    ) -> list[DetectionEvidence]:
        tensor, ratio = preprocess(image, self.input_size)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        return self.postprocess(frame_id, timestamp, image.shape[:2], raw, ratio)

    def postprocess(
        self, frame_id: str, timestamp: float, image_shape: tuple[int, int],
        raw: np.ndarray, ratio: float
    ) -> list[DetectionEvidence]:
        predictions = _decode(raw, self.input_size)
        height, width = image_shape
        boxes = np.empty_like(predictions[:, :4])
        boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2
        boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2
        boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2
        boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2
        boxes /= ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        class_ids = predictions[:, 5:].argmax(axis=1)
        scores = predictions[:, 4] * predictions[np.arange(len(predictions)), class_ids + 5]
        valid = (
            (scores >= self.confidence_floor)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
        selected = []
        for class_id in np.unique(class_ids):
            indices = np.flatnonzero(class_ids == class_id)
            selected.extend(indices[index] for index in _nms(
                boxes[indices], scores[indices], self.nms_threshold
            ))
        evidence = []
        for index in sorted(selected, key=lambda item: -scores[item]):
            x1, y1, x2, y2 = (float(value) for value in boxes[index])
            area_ratio = (x2 - x1) * (y2 - y1) / (width * height)
            evidence.append(DetectionEvidence(
                frame_id=frame_id,
                timestamp=timestamp,
                detected_class=COCO_CLASSES[int(class_ids[index])],
                confidence=float(scores[index]),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                bbox_area_ratio=area_ratio,
                center=((x1 + x2) / (2 * width), (y1 + y2) / (2 * height)),
                image_width=width,
                image_height=height,
                detector_model=MODEL_ID,
                detector_revision=f"{MODEL_VERSION}:{MODEL_REVISION}",
            ))
        return evidence
