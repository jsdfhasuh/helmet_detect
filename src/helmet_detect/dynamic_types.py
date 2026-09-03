"""Data types for the dynamic per-person detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import Detection, Rect


class HelmetState(str, Enum):
    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SceneObject:
    class_id: int
    class_name: str
    confidence: float
    box: Rect
    track_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "box": self.box.to_list(),
        }


@dataclass(frozen=True, slots=True)
class ContextCrop:
    person_index: int
    box: Rect
    scale_source: str


@dataclass(frozen=True, slots=True)
class TrackVote:
    hit_count: int
    valid_observations: int
    window_size: int
    active: bool
    event_triggered: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "hit_count": self.hit_count,
            "valid_observations": self.valid_observations,
            "window_size": self.window_size,
            "active": self.active,
            "event_triggered": self.event_triggered,
        }


@dataclass(frozen=True, slots=True)
class PersonObservation:
    track_id: int
    person: SceneObject
    matched_vehicle: SceneObject | None
    state: HelmetState
    no_helmet_score: float
    helmet_score: float
    head_detections: tuple[Detection, ...]
    context_boxes: tuple[Rect, ...]
    vote: TrackVote
    alarm_zone_eligible: bool

    @property
    def is_likely_rider(self) -> bool:
        return self.matched_vehicle is not None

    @property
    def alarm(self) -> bool:
        return self.vote.active and self.alarm_zone_eligible

    @property
    def event_triggered(self) -> bool:
        return self.vote.event_triggered and self.alarm_zone_eligible

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "person": self.person.to_dict(),
            "matched_vehicle": (
                self.matched_vehicle.to_dict() if self.matched_vehicle is not None else None
            ),
            "is_likely_rider": self.is_likely_rider,
            "state": self.state.value,
            "no_helmet_score": round(self.no_helmet_score, 6),
            "helmet_score": round(self.helmet_score, 6),
            "head_detections": [item.to_dict() for item in self.head_detections],
            "context_boxes": [item.to_list() for item in self.context_boxes],
            "vote": self.vote.to_dict(),
            "alarm_zone_eligible": self.alarm_zone_eligible,
            "alarm": self.alarm,
            "event_triggered": self.event_triggered,
        }


@dataclass(frozen=True, slots=True)
class DynamicFrameResult:
    timestamp_seconds: float
    scene_objects: tuple[SceneObject, ...]
    persons: tuple[PersonObservation, ...]

    @property
    def alarm(self) -> bool:
        return any(item.alarm for item in self.persons)

    @property
    def event_count(self) -> int:
        return sum(item.event_triggered for item in self.persons)

    @property
    def maximum_no_helmet_score(self) -> float:
        return max((item.no_helmet_score for item in self.persons), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "alarm": self.alarm,
            "event_count": self.event_count,
            "maximum_no_helmet_score": round(self.maximum_no_helmet_score, 6),
            "scene_objects": [item.to_dict() for item in self.scene_objects],
            "persons": [item.to_dict() for item in self.persons],
        }
