"""Temporal voting that turns intermittent frame detections into one event."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import TemporalConfig


@dataclass(frozen=True, slots=True)
class VoteResult:
    hit_count: int
    window_size: int
    active: bool
    event_triggered: bool


class TemporalAlarm:
    def __init__(self, config: TemporalConfig) -> None:
        self.config = config
        self._votes: deque[bool] = deque(maxlen=config.window)
        self._active = False
        self._last_event_time = float("-inf")

    def update(self, timestamp_seconds: float, frame_alarm: bool) -> VoteResult:
        self._votes.append(bool(frame_alarm))
        hit_count = sum(self._votes)
        qualifies = hit_count >= self.config.min_hits
        event_triggered = False

        if qualifies and not self._active:
            if timestamp_seconds - self._last_event_time >= self.config.cooldown_seconds:
                event_triggered = True
                self._last_event_time = timestamp_seconds
            self._active = True
        elif hit_count == 0:
            self._active = False

        return VoteResult(
            hit_count=hit_count,
            window_size=len(self._votes),
            active=qualifies,
            event_triggered=event_triggered,
        )
