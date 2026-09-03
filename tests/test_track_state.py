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
