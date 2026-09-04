from __future__ import annotations

from helmet_detect.dynamic_config import TrackTemporalConfig
from helmet_detect.dynamic_types import HelmetState
from helmet_detect.track_state import PerTrackTemporalAlarm


def test_per_track_voting_triggers_once_and_keeps_tracks_independent() -> None:
    alarm = PerTrackTemporalAlarm(
        TrackTemporalConfig(
            window=4,
            minimum_no_helmet_hits=2,
            minimum_valid_observations=2,
            cooldown_seconds=10,
            maximum_track_age_seconds=3,
        )
    )

    first = alarm.update(7, 0.0, HelmetState.NO_HELMET)
    second = alarm.update(7, 0.2, HelmetState.NO_HELMET)
    repeated = alarm.update(7, 0.4, HelmetState.NO_HELMET)
    other = alarm.update(8, 0.4, HelmetState.HELMET)

    assert not first.event_triggered
    assert second.event_triggered
    assert not repeated.event_triggered
    assert not other.active


def test_unknown_does_not_count_as_no_helmet() -> None:
    alarm = PerTrackTemporalAlarm(
        TrackTemporalConfig(
            window=3,
            minimum_no_helmet_hits=2,
            minimum_valid_observations=2,
        )
    )

    alarm.update(1, 0.0, HelmetState.NO_HELMET)
    result = alarm.update(1, 0.2, HelmetState.UNKNOWN)

    assert result.hit_count == 1
    assert result.valid_observations == 1
    assert not result.active


def test_high_confidence_path_triggers_with_fewer_hits() -> None:
    alarm = PerTrackTemporalAlarm(
        TrackTemporalConfig(
            window=5,
            minimum_no_helmet_hits=3,
            minimum_valid_observations=3,
            high_confidence_minimum_hits=2,
        )
    )

    first = alarm.update(
        7,
        0.0,
        HelmetState.NO_HELMET,
        no_helmet_score=0.80,
        high_confidence_threshold=0.65,
    )
    second = alarm.update(
        7,
        0.2,
        HelmetState.NO_HELMET,
        no_helmet_score=0.75,
        high_confidence_threshold=0.65,
    )

    assert not first.event_triggered
    assert second.event_triggered
    assert second.trigger_mode == "high_confidence"
    assert second.high_confidence_hit_count == 2


def test_one_event_per_session_suppresses_repeat_until_helmet_rearm() -> None:
    alarm = PerTrackTemporalAlarm(
        TrackTemporalConfig(
            window=4,
            minimum_no_helmet_hits=2,
            minimum_valid_observations=2,
            high_confidence_minimum_hits=2,
            helmet_rearm_hits=2,
            one_event_per_track_session=True,
        )
    )

    alarm.update(3, 0.0, HelmetState.NO_HELMET)
    first_event = alarm.update(3, 0.2, HelmetState.NO_HELMET)
    alarm.update(3, 0.4, HelmetState.UNKNOWN)
    alarm.update(3, 0.6, HelmetState.UNKNOWN)
    alarm.update(3, 0.8, HelmetState.UNKNOWN)
    alarm.update(3, 1.0, HelmetState.UNKNOWN)
    alarm.update(3, 1.2, HelmetState.NO_HELMET)
    suppressed = alarm.update(3, 1.4, HelmetState.NO_HELMET)

    assert first_event.event_triggered
    assert not suppressed.event_triggered
    assert suppressed.event_suppressed

    alarm.update(3, 1.6, HelmetState.HELMET)
    alarm.update(3, 1.8, HelmetState.HELMET)
    alarm.update(3, 2.0, HelmetState.NO_HELMET)
    rearmed = alarm.update(3, 2.2, HelmetState.NO_HELMET)

    assert rearmed.event_triggered
