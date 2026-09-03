"""Lazy Ultralytics adapters used by the dynamic V2 pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from .dynamic_config import HelmetModelConfig, SceneModelConfig
from .dynamic_types import SceneObject
from .types import Detection, Rect


class SceneDetector(Protocol):
    def detect(self, frame: np.ndarray, *, persist: bool) -> list[SceneObject]: ...


class HelmetDetector(Protocol):
    def detect_batch(self, crops: Sequence[np.ndarray]) -> list[list[Detection]]: ...


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


class UltralyticsSceneDetector:
    """Run official COCO YOLO detection and ByteTrack on the full frame."""

    def __init__(self, config: SceneModelConfig) -> None:
        if not config.path.is_file():
            raise FileNotFoundError(
                f"Scene model not found: {config.path}. "
                "Run: python scripts/prepare_dynamic_models.py"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "Dynamic V2 requires Ultralytics. Install with: pip install -e '.[dynamic]'"
            ) from exc
        self.config = config
        self._model = YOLO(str(config.path))

    def detect(self, frame: np.ndarray, *, persist: bool) -> list[SceneObject]:
        arguments: dict[str, object] = {
            "source": frame,
            "imgsz": self.config.image_size,
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "classes": list(self.config.inference_class_ids),
            "verbose": False,
            "max_det": 200,
        }
        if self.config.device is not None:
            arguments["device"] = self.config.device
        if persist:
            arguments.update(
                {
                    "persist": True,
                    "tracker": self.config.tracker,
                }
            )
            results = self._model.track(**arguments)
        else:
            results = self._model.predict(**arguments)
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        coordinates = _numpy(boxes.xyxy)
        confidences = _numpy(boxes.conf).reshape(-1)
        classes = _numpy(boxes.cls).reshape(-1)
        track_values: np.ndarray | None = None
        if getattr(boxes, "id", None) is not None:
            track_values = _numpy(boxes.id).reshape(-1)

        frame_height, frame_width = frame.shape[:2]
        detections: list[SceneObject] = []
        for index, (xyxy, confidence, class_raw) in enumerate(
            zip(coordinates, confidences, classes, strict=True)
        ):
            class_id = int(class_raw)
            x1, y1, x2, y2 = (int(round(float(item))) for item in xyxy)
            raw_box = Rect(x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))
            box = raw_box.clamp(frame_width, frame_height)
            if box is None:
                continue
            track_id = None
            if track_values is not None and index < len(track_values):
                track_id = int(round(float(track_values[index])))
            detections.append(
                SceneObject(
                    class_id=class_id,
                    class_name=_name(result.names, class_id),
                    confidence=float(confidence),
                    box=box,
                    track_id=track_id,
                )
            )
        return detections


class UltralyticsHelmetDetector:
    """Batch head-state inference over person-centred context crops."""

    def __init__(self, config: HelmetModelConfig) -> None:
        if not config.path.is_file():
            raise FileNotFoundError(
                f"Helmet model not found: {config.path}. "
                "Run: python scripts/prepare_dynamic_models.py"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "Dynamic V2 requires Ultralytics. Install with: pip install -e '.[dynamic]'"
            ) from exc
        self.config = config
        self._model = YOLO(str(config.path))

    def detect_batch(self, crops: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not crops:
            return []
        arguments: dict[str, object] = {
            "source": list(crops),
            "imgsz": self.config.image_size,
            "conf": self.config.candidate_threshold,
            "iou": self.config.iou,
            "verbose": False,
            "max_det": 100,
        }
        if self.config.device is not None:
            arguments["device"] = self.config.device
        results = self._model.predict(**arguments)
        output: list[list[Detection]] = []
        for crop, result in zip(crops, results, strict=True):
            crop_height, crop_width = crop.shape[:2]
            boxes = result.boxes
            detections: list[Detection] = []
            if boxes is not None and len(boxes) > 0:
                coordinates = _numpy(boxes.xyxy)
                confidences = _numpy(boxes.conf).reshape(-1)
                classes = _numpy(boxes.cls).reshape(-1)
                for xyxy, confidence, class_raw in zip(
                    coordinates,
                    confidences,
                    classes,
                    strict=True,
                ):
                    class_id = int(class_raw)
                    x1, y1, x2, y2 = (int(round(float(item))) for item in xyxy)
                    raw_box = Rect(x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))
                    box = raw_box.clamp(crop_width, crop_height)
                    if box is None:
                        continue
                    detections.append(
                        Detection(
                            model_name="helmet_yolov8n",
                            class_id=class_id,
                            class_name=_name(result.names, class_id),
                            confidence=float(confidence),
                            box=box,
                        )
                    )
            output.append(detections)
        return output
