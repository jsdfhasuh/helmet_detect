"""OpenCV-DNN inference for Ultralytics YOLO ONNX exports."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .preprocess import LetterboxTransform, letterbox
from .types import Detection, Rect

CLASS_NAMES = {0: "With Helmet", 1: "Without Helmet"}


class OpenCVDNNDetector:
    """Load one fixed-size ONNX model and decode its detections."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "OpenCV is required. Install with: pip install -e '.[runtime]'"
            ) from exc

        if not config.path.is_file():
            raise FileNotFoundError(
                f"Model not found: {config.path}. Run: python scripts/prepare_models.py"
            )
        self.config = config
        self._cv2 = cv2
        self._net = cv2.dnn.readNetFromONNX(str(config.path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @property
    def name(self) -> str:
        return self.config.name

    def infer(
        self,
        image: np.ndarray,
        *,
        origin: tuple[int, int],
        candidate_threshold: float,
        nms_threshold: float,
    ) -> list[Detection]:
        padded, transform = letterbox(image, self.config.input_size)
        blob = self._cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=(self.config.input_size, self.config.input_size),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        raw = self._net.forward()
        rows = normalise_output(raw)
        return self._decode(
            rows,
            transform,
            origin=origin,
            candidate_threshold=candidate_threshold,
            nms_threshold=nms_threshold,
        )

    def _decode(
        self,
        rows: np.ndarray,
        transform: LetterboxTransform,
        *,
        origin: tuple[int, int],
        candidate_threshold: float,
        nms_threshold: float,
    ) -> list[Detection]:
        if rows.size == 0:
            return []
        if rows.shape[1] < 6:
            raise ValueError(f"Unexpected model output shape: {rows.shape}")

        class_scores = rows[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(rows)), class_ids]
        keep = confidences >= candidate_threshold
        rows = rows[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        origin_x, origin_y = origin
        provisional: list[Detection] = []
        for row, class_id_raw, confidence_raw in zip(rows, class_ids, confidences, strict=True):
            class_id = int(class_id_raw)
            confidence = float(confidence_raw)
            x1, y1, x2, y2 = transform.inverse_xyxy(*map(float, row[:4]))
            if x2 <= x1 or y2 <= y1:
                continue
            local_box = Rect(x1, y1, x2, y2).clamp(
                transform.source_width,
                transform.source_height,
            )
            if local_box is None:
                continue
            box = local_box.translate(origin_x, origin_y)
            provisional.append(
                Detection(
                    model_name=self.config.name,
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    box=box,
                )
            )

        return class_aware_nms(provisional, candidate_threshold, nms_threshold, self._cv2)


def normalise_output(raw: np.ndarray) -> np.ndarray:
    """Return an ``N x (4 + classes)`` view for common YOLO export layouts."""

    array = np.asarray(raw)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported ONNX output shape: {np.asarray(raw).shape}")

    # Ultralytics usually exports (channels, anchors). Other runtimes may expose
    # (anchors, channels). The channel dimension is the small one.
    if array.shape[0] <= 256 and array.shape[1] > array.shape[0]:
        array = array.T
    return np.ascontiguousarray(array, dtype=np.float32)


def class_aware_nms(
    detections: Iterable[Detection],
    score_threshold: float,
    nms_threshold: float,
    cv2_module: object,
) -> list[Detection]:
    """Apply NMS independently for each class."""

    grouped: dict[int, list[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.class_id, []).append(detection)

    result: list[Detection] = []
    for class_detections in grouped.values():
        boxes = [
            [item.box.x1, item.box.y1, item.box.width, item.box.height]
            for item in class_detections
        ]
        scores = [item.confidence for item in class_detections]
        indices = cv2_module.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold,
            nms_threshold,
        )
        if len(indices) == 0:
            continue
        for index in np.asarray(indices).reshape(-1):
            result.append(class_detections[int(index)])

    return sorted(result, key=lambda item: item.confidence, reverse=True)
