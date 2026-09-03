"""Per-track temporal voting and event de-duplication."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .dynamic_config import TrackTemporalConfig
from .dynamic_types import HelmetState, TrackVote


@dataclass(slots=True)
class _History:
    states: deque[HelmetState]
    active: bool = False
    last_event_time: float = float("-inf")
    last_seen_time: float = float("-inf")


class PerTrackTemporalAlarm:
    """Keep independent helmet state histories for every tracked person."""

    def __init__(self, config: TrackTemporalConfig) -> None:
        self.config = config
        self._histories: dict[int, _History] = {}

    def update(
        self,
        track_id: int,
        timestamp_seconds: float,
        state: HelmetState,
    ) -> TrackVote:
        history = self._histories.get(track_id)
        if history is None:
            history = _History(states=deque(maxlen=self.config.window))
            self._histories[track_id] = history
        history.states.append(state)
        history.last_seen_time = timestamp_seconds

        valid = [item for item in history.states if item is not HelmetState.UNKNOWN]
        hit_count = sum(item is HelmetState.NO_HELMET for item in history.states)
        qualifies = (
            len(valid) >= self.config.minimum_valid_observations
            and hit_count >= self.config.minimum_no_helmet_hits
        )
        event_triggered = False
        if qualifies and not history.active:
            if timestamp_seconds - history.last_event_time >= self.config.cooldown_seconds:
                event_triggered = True
                history.last_event_time = timestamp_seconds
            history.active = True
        elif valid and hit_count == 0:
            history.active = False

        self.prune(timestamp_seconds)
        return TrackVote(
            hit_count=hit_count,
            valid_observations=len(valid),
            window_size=len(history.states),
            active=qualifies,
            event_triggered=event_triggered,
        )

    def prune(self, timestamp_seconds: float) -> None:
        stale = [
            track_id
            for track_id, history in self._histories.items()
            if timestamp_seconds - history.last_seen_time
            > self.config.maximum_track_age_seconds
        ]
        for track_id in stale:
            del self._histories[track_id]

    def reset(self) -> None:
        self._histories.clear()
