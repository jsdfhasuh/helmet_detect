"""Frame-level helmet detection and model fusion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig
from .model import OpenCVDNNDetector
from .types import Detection


@dataclass(frozen=True, slots=True)
class FrameResult:
    alarm: bool
    best_no_helmet: dict[str, float]
    best_helmet: dict[str, float]
    detections: tuple[Detection, ...]

    @property
    def fused_no_helmet_score(self) -> float:
        return max(self.best_no_helmet.values(), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "alarm": self.alarm,
            "fused_no_helmet_score": round(self.fused_no_helmet_score, 6),
            "best_no_helmet": {
                key: round(value, 6) for key, value in self.best_no_helmet.items()
            },
            "best_helmet": {
                key: round(value, 6) for key, value in self.best_helmet.items()
            },
            "detections": [item.to_dict() for item in self.detections],
        }


class HelmetDetectionPipeline:
    """Apply all configured models to a fixed camera ROI and fuse their alarms."""

    def __init__(
        self,
        config: DetectorConfig,
        *,
        enabled_models: Iterable[str] | None = None,
    ) -> None:
        enabled = set(enabled_models or ())
        model_configs = [
            item for item in config.models if not enabled or item.name in enabled
        ]
        if not model_configs:
            raise ValueError("No enabled models remain after filtering")
        self.config = config
        self.models = tuple(OpenCVDNNDetector(item) for item in model_configs)
        self._model_config = {item.name: item for item in model_configs}

    def detect_frame(self, frame: np.ndarray) -> FrameResult:
        frame_height, frame_width = frame.shape[:2]
        roi = self.config.roi.clamp(frame_width, frame_height)
        gate = self.config.gate.clamp(frame_width, frame_height)
        if roi is None:
            raise ValueError(
                "Configured ROI "
                f"{self.config.roi} does not overlap frame {frame_width}x{frame_height}"
            )
        if gate is None:
            raise ValueError(
                "Configured gate "
                f"{self.config.gate} does not overlap frame {frame_width}x{frame_height}"
            )

        crop = frame[roi.y1 : roi.y2, roi.x1 : roi.x2]
        all_detections: list[Detection] = []
        best_no_helmet: dict[str, float] = {}
        best_helmet: dict[str, float] = {}
        alarm = False

        for model in self.models:
            detections = model.infer(
                crop,
                origin=(roi.x1, roi.y1),
                candidate_threshold=self.config.candidate_threshold,
                nms_threshold=self.config.nms_threshold,
            )
            gated = [item for item in detections if gate.contains_centre(item.box)]
            all_detections.extend(gated)

            model_config = self._model_config[model.name]
            no_helmet = max(
                (
                    item.confidence
                    for item in gated
                    if item.class_id == model_config.no_helmet_class_id
                ),
                default=0.0,
            )
            helmet = max(
                (
                    item.confidence
                    for item in gated
                    if item.class_id == model_config.helmet_class_id
                ),
                default=0.0,
            )
            best_no_helmet[model.name] = no_helmet
            best_helmet[model.name] = helmet
            if no_helmet >= model_config.alarm_threshold:
                alarm = True

        return FrameResult(
            alarm=alarm,
            best_no_helmet=best_no_helmet,
            best_helmet=best_helmet,
            detections=tuple(
                sorted(all_detections, key=lambda item: item.confidence, reverse=True)
            ),
        )
