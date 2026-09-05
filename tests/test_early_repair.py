from dataclasses import replace
from pathlib import Path

import numpy as np

from helmet_detect.dynamic_config import (
    AssociationConfig,
    DynamicDetectorConfig,
    EarlyCheckConfig,
    HelmetModelConfig,
    SceneModelConfig,
    TrackTemporalConfig,
)
from helmet_detect.dynamic_pipeline import DynamicHelmetDetectionPipeline, _helmet_state
from helmet_detect.dynamic_types import HelmetState, SceneObject
from helmet_detect.early_check import EarlyCheckScheduler
from helmet_detect.fallback_tracker import GreedyPersonTracker
from helmet_detect.rider_state import RiderEvidenceTracker
from helmet_detect.track_state import PerTrackTemporalAlarm
from helmet_detect.types import Detection, Rect
from helmet_detect.ultralytics_backend import track_ids_from_rows


def test_tracker_mapping_keeps_unmatched_indices_available():
    tracks = np.array([[0, 0, 20, 20, 7, .9, 0, 1]], dtype=np.float32)
    ids = track_ids_from_rows(tracks, np.array([0, 0, 3]))
    assert ids == {1: 7}
    assert [ids.get(i) for i in range(3)] == [None, 7, None]


def test_track_mapping_rejects_wrong_class_and_invalid_index():
    tracks = np.array([[0, 0, 20, 20, 7, .9, 3, 0],
                       [0, 0, 20, 20, 8, .9, 0, 100]], dtype=np.float32)
    assert track_ids_from_rows(tracks, np.array([0])) == {}


def test_fallback_does_not_steal_an_id_reserved_later_in_frame():
    tracker = GreedyPersonTracker()
    a = SceneObject(0, 'person', .9, Rect(100, 100, 180, 300), 4)
    tracker.assign([a], 0)
    untracked = replace(a, track_id=None, box=Rect(105, 100, 185, 300))
    result = tracker.assign([untracked, a], .2)
    assert result[1].track_id == 4
    assert result[0].track_id != 4


def test_vehicle_minimum_hits_cannot_be_bypassed_by_current_match():
    tracker = RiderEvidenceTracker(
        AssociationConfig(mode='riders_only', minimum_vehicle_hits=3),
        maximum_age_seconds=4,
    )
    assert not tracker.update(1, 0, has_vehicle_match=True).eligible
    assert not tracker.update(1, .2, has_vehicle_match=True).eligible
    assert tracker.update(1, .4, has_vehicle_match=True).eligible


def test_early_probe_budget_is_fair_and_expires():
    scheduler = EarlyCheckScheduler(EarlyCheckConfig(enabled=True, maximum_pending_people=1))
    assert scheduler.select([(1, False), (2, False)], 0) == {1}
    assert scheduler.select([(1, False), (2, False)], .2) == {2}
    assert scheduler.select([(1, False), (2, False)], 2.1) == set()
    assert scheduler.select([(1, True), (2, False)], 2.2) == {1}


def test_pre_rider_votes_do_not_emit_or_consume_event():
    voter = PerTrackTemporalAlarm(TrackTemporalConfig(high_confidence_minimum_hits=2))
    for now in [0, .2]:
        v = voter.update(1, now, HelmetState.NO_HELMET, no_helmet_score=.9,
                         high_confidence_threshold=.5, allow_event=False)
        assert not v.active and not v.event_triggered
    v = voter.update(1, .4, HelmetState.UNKNOWN, allow_event=True,
                     high_confidence_threshold=.5)
    assert v.event_triggered
    assert not voter.update(1, .6, HelmetState.NO_HELMET, no_helmet_score=.9,
                            high_confidence_threshold=.5).event_triggered


def test_expired_pre_rider_observation_cannot_trigger():
    voter = PerTrackTemporalAlarm(TrackTemporalConfig(observation_ttl_seconds=.5))
    voter.update(1, 0, HelmetState.NO_HELMET, no_helmet_score=.9,
                 high_confidence_threshold=.5, allow_event=False)
    v = voter.update(1, 1, HelmetState.UNKNOWN, high_confidence_threshold=.5)
    assert v.hit_count == 0 and not v.event_triggered


def test_current_helmet_blocks_cached_no_helmet():
    voter = PerTrackTemporalAlarm(TrackTemporalConfig())
    voter.update(1, 0, HelmetState.NO_HELMET, no_helmet_score=.9,
                 high_confidence_threshold=.5, allow_event=False)
    v = voter.update(1, .2, HelmetState.HELMET, high_confidence_threshold=.5)
    assert not v.event_triggered


def config():
    return DynamicDetectorConfig(
        scene_model=SceneModelConfig(path=Path('scene.pt')),
        helmet_model=HelmetModelConfig(
            path=Path('head.pt'), context_height_multipliers=(2,), context_frame_scales=(),
            maximum_contexts_per_person=1,
        ),
        early_check=EarlyCheckConfig(enabled=True),
        association=AssociationConfig(mode='riders_only'),
    )


class Scene:
    def __init__(self):
        self.with_vehicle = False
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def detect(self, frame, *, persist):
        result = [SceneObject(0, 'person', .9, Rect(300, 200, 380, 380), 1)]
        if self.with_vehicle:
            result.append(SceneObject(3, 'motorcycle', .9, Rect(270, 280, 420, 440), 2))
        return result


class Head:
    def detect_batch(self, crops):
        return [[Detection('test', 1, 'Without Helmet', .9,
                           Rect(c.shape[1]//2-10, c.shape[0]//2-10,
                                c.shape[1]//2+10, c.shape[0]//2+10))] for c in crops]


def test_pipeline_checks_head_before_vehicle_but_only_alerts_after_confirmation():
    scene = Scene()
    pipeline = DynamicHelmetDetectionPipeline(config(), scene_detector=scene,
                                               helmet_detector=Head())
    image = np.zeros((600, 800, 3), np.uint8)
    first = pipeline.detect_frame(image, timestamp_seconds=0)
    person = first.persons[0]
    assert person.helmet_evaluated and person.head_check_reason == 'early_probe'
    assert person.state == HelmetState.NO_HELMET
    assert not person.rider_eligible and not first.alarm and not first.event_count
    scene.with_vehicle = True
    second = pipeline.detect_frame(image, timestamp_seconds=.2)
    assert second.event_count == 1
    assert second.diagnostics['helmet_crops'] == 1
    pipeline.reset()
    assert scene.reset_count == 1


def test_alarm_zone_does_not_consume_event_before_entry():
    scene = Scene()
    scene.with_vehicle = True
    cfg = replace(config(), alarm_zone=((0, 0), (.1, 0), (.1, .1), (0, .1)))
    pipeline = DynamicHelmetDetectionPipeline(cfg, scene_detector=scene, helmet_detector=Head())
    image = np.zeros((600, 800, 3), np.uint8)
    first = pipeline.detect_frame(image, timestamp_seconds=0)
    assert not first.event_count
    pipeline.config = replace(cfg, alarm_zone=None)
    assert pipeline.detect_frame(image, timestamp_seconds=.2).event_count == 1


def test_conflicting_head_states_are_unknown_not_no_helmet():
    cfg = config()
    assert _helmet_state(.51, .55, config=cfg) == HelmetState.UNKNOWN
    assert _helmet_state(.45, .75, config=cfg) == HelmetState.HELMET
    assert _helmet_state(.85, .1, config=cfg) == HelmetState.NO_HELMET


def test_no_video_records_measured_time_without_continuous_mp4(tmp_path):
    import json

    import pytest

    cv2 = pytest.importorskip('cv2')
    from helmet_detect.dynamic_types import DynamicFrameResult
    from helmet_detect.dynamic_video import process_dynamic_video

    source = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 5, (64, 64))
    if not writer.isOpened():
        pytest.skip('mp4v encoder unavailable')
    for _ in range(4):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()

    class EmptyPipeline:
        def reset(self):
            pass

        def detect_frame(self, frame, *, timestamp_seconds, **kwargs):
            return DynamicFrameResult(timestamp_seconds, (), (), {'helmet_crops': 0})

    output = tmp_path / 'output.mp4'
    records = tmp_path / 'frames.jsonl'
    result = process_dynamic_video(EmptyPipeline(), source, output, records, save_video=False)
    assert result.processed_frames == 4
    assert not output.exists()
    assert not result.annotated_video_written
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    assert len(rows) == 4 and rows[0]['processing_ms'] >= 0
    assert result.to_dict()['measured_processing_fps'] > 0
