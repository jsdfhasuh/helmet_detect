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
            high_confidence_minimum_hits=2,
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


class FakeSceneSequence:
    def __init__(self, frames: list[list[SceneObject]]) -> None:
        self.frames = frames
        self.index = 0

    def detect(self, frame: np.ndarray, *, persist: bool) -> list[SceneObject]:
        del frame, persist
        result = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return result


class CountingHelmetDetector(FakeHelmetDetector):
    def __init__(self) -> None:
        self.calls = 0

    def detect_batch(self, crops: list[np.ndarray]) -> list[list[Detection]]:
        self.calls += len(crops)
        return super().detect_batch(crops)


def test_rider_only_mode_filters_unmatched_pedestrian() -> None:
    from helmet_detect.dynamic_config import AssociationConfig

    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    detector = CountingHelmetDetector()
    config = _config()
    config = DynamicDetectorConfig(
        scene_model=config.scene_model,
        helmet_model=config.helmet_model,
        association=AssociationConfig(mode="riders_only"),
        temporal=config.temporal,
    )
    pipeline = DynamicHelmetDetectionPipeline(
        config,
        scene_detector=FakeSceneDetector(Rect(200, 120, 280, 300)),
        helmet_detector=detector,
    )

    result = pipeline.detect_frame(
        frame,
        persist_tracks=False,
        apply_temporal=False,
    )

    assert len(result.persons) == 1
    assert not result.persons[0].rider_eligible
    assert result.persons[0].state is HelmetState.UNKNOWN
    assert not result.alarm
    assert detector.calls == 0


def test_cross_id_switch_keeps_one_temporal_identity_and_event() -> None:
    from helmet_detect.dynamic_config import AssociationConfig, IdentityConfig

    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    frame[100:320, 180:330] = (20, 80, 220)
    person_first = SceneObject(0, "person", 0.9, Rect(200, 120, 280, 300), 12)
    person_second = SceneObject(0, "person", 0.9, Rect(208, 122, 288, 302), 99)
    vehicle_first = SceneObject(3, "motorcycle", 0.8, Rect(170, 230, 330, 350), None)
    vehicle_second = SceneObject(3, "motorcycle", 0.8, Rect(178, 232, 338, 352), None)
    base = _config()
    config = DynamicDetectorConfig(
        scene_model=base.scene_model,
        helmet_model=base.helmet_model,
        association=AssociationConfig(mode="riders_only"),
        identity=IdentityConfig(maximum_gap_seconds=2.0),
        temporal=TrackTemporalConfig(
            window=3,
            minimum_no_helmet_hits=2,
            minimum_valid_observations=2,
            high_confidence_minimum_hits=2,
        ),
    )
    pipeline = DynamicHelmetDetectionPipeline(
        config,
        scene_detector=FakeSceneSequence(
            [
                [person_first, vehicle_first],
                [person_second, vehicle_second],
                [person_second, vehicle_second],
            ]
        ),
        helmet_detector=FakeHelmetDetector(),
    )

    first = pipeline.detect_frame(frame, timestamp_seconds=0.0)
    second = pipeline.detect_frame(frame, timestamp_seconds=0.2)
    third = pipeline.detect_frame(frame, timestamp_seconds=0.4)

    assert first.persons[0].track_id == 12
    assert second.persons[0].track_id == 12
    assert second.persons[0].source_track_id == 99
    assert second.persons[0].identity_stitched
    assert second.event_count == 1
    assert third.event_count == 0
