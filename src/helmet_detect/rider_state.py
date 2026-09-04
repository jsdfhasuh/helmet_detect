"""Temporal rider evidence derived from person-to-two-wheel-vehicle association."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .dynamic_config import AssociationConfig


@dataclass(frozen=True, slots=True)
class RiderEvidence:
    current_vehicle_match: bool
    vehicle_hit_count: int
    window_size: int
    confirmed: bool
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "current_vehicle_match": self.current_vehicle_match,
            "vehicle_hit_count": self.vehicle_hit_count,
            "window_size": self.window_size,
            "confirmed": self.confirmed,
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass(slots=True)
class _RiderHistory:
    matches: deque[bool]
    confirmed: bool = False
    last_vehicle_time: float = float("-inf")
    last_seen_time: float = float("-inf")


class RiderEvidenceTracker:
    """Keep intermittent vehicle matches alive for the same canonical person."""

    def __init__(self, config: AssociationConfig, *, maximum_age_seconds: float) -> None:
        self.config = config
        self.maximum_age_seconds = maximum_age_seconds
        self._histories: dict[int, _RiderHistory] = {}

    def update(
        self,
        canonical_track_id: int,
        timestamp_seconds: float,
        *,
        has_vehicle_match: bool,
    ) -> RiderEvidence:
        history = self._histories.get(canonical_track_id)
        if history is None:
            history = _RiderHistory(
                matches=deque(maxlen=self.config.vehicle_evidence_window)
            )
            self._histories[canonical_track_id] = history
        history.matches.append(bool(has_vehicle_match))
        history.last_seen_time = timestamp_seconds
        if has_vehicle_match:
            history.last_vehicle_time = timestamp_seconds

        hit_count = sum(history.matches)
        if hit_count >= self.config.minimum_vehicle_hits:
            history.confirmed = True
        recent_vehicle = (
            timestamp_seconds - history.last_vehicle_time
            <= self.config.vehicle_grace_seconds
        )

        if not self.config.riders_only:
            eligible = True
            reason = "all_persons_mode"
        elif has_vehicle_match:
            eligible = True
            reason = "current_vehicle_match"
        elif history.confirmed and recent_vehicle:
            eligible = True
            reason = "recent_vehicle_evidence"
        else:
            eligible = False
            reason = "no_rider_evidence"

        self.prune(timestamp_seconds)
        return RiderEvidence(
            current_vehicle_match=has_vehicle_match,
            vehicle_hit_count=hit_count,
            window_size=len(history.matches),
            confirmed=history.confirmed,
            eligible=eligible,
            reason=reason,
        )

    def prune(self, timestamp_seconds: float) -> None:
        stale = [
            track_id
            for track_id, history in self._histories.items()
            if timestamp_seconds - history.last_seen_time > self.maximum_age_seconds
        ]
        for track_id in stale:
            del self._histories[track_id]

    def reset(self) -> None:
        self._histories.clear()
