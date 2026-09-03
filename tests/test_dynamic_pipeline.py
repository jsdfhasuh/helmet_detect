from __future__ import annotations

from pathlib import Path

import numpy as np

from helmet_detect.dynamic_config import (
    DynamicDetectorConfig,
    HelmetModelConfig,
    SceneModelConfig,
    TrackTemporalConfig,
)
from helmet_detect.dynamic_pipeline import DynamicHelmetDetectionPipeline
from helmet_detect.dynamic_types import HelmetState, SceneObject
from helmet_detect.types import Detection, Rect


class FakeSceneDetector:
    def __init__(self, box: Rect) -> None:
        self.box = box

    def detect(self, frame: np.ndarray, *, persist: bool) -> list[SceneObject]:
        del frame, persist
        return [SceneObject(0, "person", 0.9, self.box, 12)]


class FakeHelmetDetector:
    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]:
        output: list[list[Detection]] = []
        for crop in crops:
            height, width = crop.shape[:2]
            cx, cy = width // 2, height // 2
            output.append(
                [
                    Detection(
                        model_name="helmet_yolov8n",
                        class_id=1,
                        class_name="Without Helmet",
                        confidence=0.8,
                        box=Rect(cx - 8, cy - 8, cx + 8, cy + 8),
                    )
                ]
            )
        return output


def _config() -> DynamicDetectorConfig:
    return DynamicDetectorConfig(
        scene_model=SceneModelConfig(
            path=Path("scene.pt"),
            minimum_person_height=20,
        ),
        helmet_model=HelmetModelConfig(
            path=Path("helmet.pt"),
            context_height_multipliers=(2.0,),
            context_frame_scales=(),
            maximum_contexts_per_person=1,
            minimum_context_size=100,
            no_helmet_threshold=0.2,
        ),
        temporal=TrackTemporalConfig(
            window=3,
            minimum_no_helmet_hits=2,
            minimum_valid_observations=2,
        ),
    )


def test_dynamic_pipeline_detects_person_at_arbitrary_position() -> None:
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    pipeline = DynamicHelmetDetectionPipeline(
        _config(),
        scene_detector=FakeSceneDetector(Rect(600, 120, 680, 300)),
        helmet_detector=FakeHelmetDetector(),
    )

    result = pipeline.detect_frame(
        frame,
        persist_tracks=False,
        apply_temporal=False,
    )

    assert result.alarm
    assert len(result.persons) == 1
    assert result.persons[0].track_id == 12
    assert result.persons[0].state is HelmetState.NO_HELMET
    assert result.persons[0].person.box.x1 == 600


def test_dynamic_pipeline_temporal_vote_is_per_track() -> None:
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    pipeline = DynamicHelmetDetectionPipeline(
        _config(),
        scene_detector=FakeSceneDetector(Rect(200, 120, 280, 300)),
        helmet_detector=FakeHelmetDetector(),
    )

    first = pipeline.detect_frame(frame, timestamp_seconds=0.0)
    second = pipeline.detect_frame(frame, timestamp_seconds=0.2)

    assert not first.alarm
    assert second.alarm
    assert second.event_count == 1
