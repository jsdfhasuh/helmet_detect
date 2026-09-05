"""Small tracking fallback for detections that do not receive a ByteTrack ID."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .dynamic_types import SceneObject
from .geometry import iou
from .types import Rect


@dataclass(slots=True)
class _Track:
    box: Rect
    last_seen: float


class GreedyPersonTracker:
    """Assign stable IDs by IoU/centre distance when the primary tracker has no ID."""

    def __init__(self, *, maximum_age_seconds: float = 3.0) -> None:
        self.maximum_age_seconds = maximum_age_seconds
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1_000_000

    def assign(
        self,
        people: list[SceneObject],
        timestamp_seconds: float,
    ) -> list[SceneObject]:
        self._prune(timestamp_seconds)
        reserved = {p.track_id for p in people if p.track_id is not None}
        available = set(self._tracks) - reserved
        output: list[SceneObject] = []
        for person in people:
            if person.track_id is not None:
                track_id = person.track_id
            else:
                track_id = self._match(person.box, available)
                if track_id is None:
                    track_id = self._next_id
                    self._next_id += 1
            available.discard(track_id)
            self._tracks[track_id] = _Track(person.box, timestamp_seconds)
            output.append(replace(person, track_id=track_id))
        return output

    def _match(self, box: Rect, available: set[int]) -> int | None:
        best: tuple[float, int] | None = None
        for track_id in available:
            previous = self._tracks[track_id].box
            overlap = iou(box, previous)
            distance = math.dist(box.centre, previous.centre)
            normalised_distance = distance / max(box.height, previous.height, 1)
            if overlap < 0.10 and normalised_distance > 0.65:
                continue
            score = overlap - 0.25 * normalised_distance
            if best is None or score > best[0]:
                best = (score, track_id)
        return best[1] if best is not None else None

    def _prune(self, timestamp_seconds: float) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp_seconds - track.last_seen > self.maximum_age_seconds
        ]
        for track_id in stale:
            del self._tracks[track_id]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1_000_000
