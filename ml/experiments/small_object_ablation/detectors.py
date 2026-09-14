"""Detector adapters used only by the frozen ablation."""

from pathlib import Path

import cv2
import numpy as np

from ml.experiments.object_detector_branch.detector import YoloXNanoDetector


class YoloXAdapter:
    model_id = "YOLOX-Nano"

    def __init__(self, model_path: Path, size: int, threads: int = 4):
        self.size = size
        self.detector = YoloXNanoDetector(
            model_path, threads=threads, confidence_floor=0.01, input_size=(size, size)
        )

    def detect(self, path: Path, target_class: str) -> list[dict]:
        rows = self.detector.detect_path(path.stem, 0, path)
        return [row.model_dump() for row in rows if row.detected_class == target_class]


class RTDetrR18Adapter:
    model_id = "RT-DETR-R18"

    def __init__(self, cache_dir: Path, revision: str, threads: int = 4):
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        torch.set_num_threads(threads)
        options = {"cache_dir": cache_dir, "revision": revision, "local_files_only": True}
        self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd", **options)
        self.model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r18vd", **options
        ).eval()

    def detect(self, path: Path, target_class: str) -> list[dict]:
        import torch
        from PIL import Image

        encoded = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read detector frame: {path}")
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=[image.size[::-1]], threshold=0.01
        )[0]
        rows = []
        width, height = image.size
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            if self.model.config.id2label[int(label)] != target_class:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            rows.append({
                "detected_class": target_class,
                "confidence": float(score),
                "bbox_area_ratio": (x2 - x1) * (y2 - y1) / (width * height),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "image_width": width,
                "image_height": height,
            })
        return rows
