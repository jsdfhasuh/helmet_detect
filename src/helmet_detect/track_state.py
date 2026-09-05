"""Per-canonical-track dual-threshold voting and event de-duplication."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .dynamic_config import TrackTemporalConfig
from .dynamic_types import HelmetState, TrackVote


@dataclass(frozen=True, slots=True)
class _Observation:
    state: HelmetState
    no_helmet_score: float
    timestamp_seconds: float


@dataclass(slots=True)
class _History:
    observations: deque[_Observation]
    active: bool = False
    event_sent: bool = False
    last_event_time: float = float("-inf")
    last_seen_time: float = float("-inf")
    helmet_streak: int = 0


class PerTrackTemporalAlarm:
    """Keep independent helmet histories for each stitched canonical person.

    Two paths may qualify a track:

    * normal confidence: the regular no-helmet hit and valid-observation counts;
    * high confidence: fewer observations whose score exceeds the configured high
      threshold.

    Once an event is sent, the same canonical track cannot emit another event during
    the same session unless it is explicitly re-armed by consecutive HELMET states.
    """

    def __init__(self, config: TrackTemporalConfig) -> None:
        self.config = config
        self._histories: dict[int, _History] = {}

    def update(
        self,
        track_id: int,
        timestamp_seconds: float,
        state: HelmetState,
        *,
        no_helmet_score: float = 0.0,
        high_confidence_threshold: float = 1.0,
        allow_event: bool = True,
    ) -> TrackVote:
        history = self._histories.get(track_id)
        if history is None:
            history = _History(observations=deque(maxlen=self.config.window))
            self._histories[track_id] = history

        while history.observations and (
            timestamp_seconds - history.observations[0].timestamp_seconds
            > self.config.observation_ttl_seconds
        ):
            history.observations.popleft()

        if state is HelmetState.HELMET:
            history.helmet_streak += 1
        else:
            history.helmet_streak = 0

        # A clear helmet sequence means the same person may later remove the helmet;
        # start a fresh voting session at that point.
        if (
            history.event_sent
            and history.helmet_streak >= self.config.helmet_rearm_hits
        ):
            history.observations.clear()
            history.active = False
            history.event_sent = False
            history.last_event_time = float("-inf")
            history.helmet_streak = 0

        observation = _Observation(
            state=state,
            no_helmet_score=float(no_helmet_score),
            timestamp_seconds=timestamp_seconds,
        )
        history.observations.append(observation)
        history.last_seen_time = timestamp_seconds

        valid = [
            item
            for item in history.observations
            if item.state is not HelmetState.UNKNOWN
        ]
        hit_count = sum(
            item.state is HelmetState.NO_HELMET for item in history.observations
        )
        high_hit_count = sum(
            item.state is HelmetState.NO_HELMET
            and item.no_helmet_score >= high_confidence_threshold
            for item in history.observations
        )
        normal_qualifies = (
            len(valid) >= self.config.minimum_valid_observations
            and hit_count >= self.config.minimum_no_helmet_hits
        )
        high_qualifies = (
            high_hit_count >= self.config.high_confidence_minimum_hits
        )
        qualifies = normal_qualifies or high_qualifies
        trigger_mode = (
            "high_confidence"
            if high_qualifies
            else ("normal" if normal_qualifies else None)
        )

        event_triggered = False
        event_suppressed = False
        # HELMET contradicts a cached NO_HELMET hit: never fire a stale alert.
        actionable = qualifies and allow_event and state is not HelmetState.HELMET
        entering_alarm = actionable and not history.active
        if entering_alarm:
            blocked_by_session = (
                self.config.one_event_per_track_session and history.event_sent
            )
            blocked_by_cooldown = (
                timestamp_seconds - history.last_event_time
                < self.config.cooldown_seconds
            )
            if blocked_by_session or blocked_by_cooldown:
                event_suppressed = True
            else:
                event_triggered = True
                history.event_sent = True
                history.last_event_time = timestamp_seconds

        history.active = actionable
        self.prune(timestamp_seconds)
        return TrackVote(
            hit_count=hit_count,
            high_confidence_hit_count=high_hit_count,
            valid_observations=len(valid),
            window_size=len(history.observations),
            active=actionable,
            event_triggered=event_triggered,
            event_suppressed=event_suppressed,
            trigger_mode=trigger_mode,
        )

    def inactive_vote(self) -> TrackVote:
        """Return a non-alarming vote for a person filtered by rider mode."""

        return TrackVote(
            hit_count=0,
            high_confidence_hit_count=0,
            valid_observations=0,
            window_size=0,
            active=False,
            event_triggered=False,
            event_suppressed=False,
            trigger_mode=None,
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
