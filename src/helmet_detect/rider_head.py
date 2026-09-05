"""Experimental one-pass rider + worn-head inference; no fixed camera coordinates.

Requires weights whose exact classes are rider/helmet_head/no_helmet_head.
The public COCO model and the old two-class helmet weights are NOT compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .dynamic_config import IdentityConfig, TrackTemporalConfig
from .dynamic_types import HelmetState, SceneObject
from .fallback_tracker import GreedyPersonTracker
from .geometry import iou
from .identity import CrossIdIdentityResolver
from .track_state import PerTrackTemporalAlarm
from .types import Rect

EXPECTED_NAMES = {0: "rider", 1: "helmet_head", 2: "no_helmet_head"}


@dataclass(frozen=True, slots=True)
class RiderHeadConfig:
    rider_threshold: float = 0.35
    head_threshold: float = 0.35
    conflict_margin: float = 0.15
    minimum_head_pixels: int = 10
    minimum_hits: int = 2
    evidence_seconds: float = 0.8

    def __post_init__(self) -> None:
        for value in (self.rider_threshold, self.head_threshold, self.conflict_margin):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Thresholds must be finite and in [0,1]")
        if not 2 <= self.minimum_hits <= 8 or self.minimum_head_pixels < 1:
            raise ValueError("Use at least two distinct observations and positive head size")
        if not isfinite(self.evidence_seconds) or self.evidence_seconds <= 0:
            raise ValueError("Evidence lifetime must be finite and positive")


def associate_heads(riders: list[SceneObject], heads: list[SceneObject]) -> dict[int, list]:
    """Assign a head to one plausible rider; overlapping ownership remains unknown."""
    output: dict[int, list] = {i: [] for i in range(len(riders))}
    for head in heads:
        hx, hy = head.box.centre
        owners = []
        for i, rider in enumerate(riders):
            b = rider.box
            if not (b.x1 - 0.12 * b.width <= hx <= b.x2 + 0.12 * b.width):
                continue
            if not (b.y1 - 0.12 * b.height <= hy <= b.y1 + 0.45 * b.height):
                continue
            if head.box.height > 0.55 * b.height or head.box.width > 0.9 * b.width:
                continue
            score = abs(hx - b.centre[0]) / b.width + abs(hy - b.y1 - 0.13 * b.height) / b.height
            owners.append((score, i))
        owners.sort()
        if owners and (len(owners) == 1 or owners[1][0] - owners[0][0] > 0.15):
            output[owners[0][1]].append(head)
    return output


class RiderHeadLogic:
    """Timestamped per-rider events. Cached/duplicate frames never add a vote."""

    def __init__(self, config: RiderHeadConfig | None = None) -> None:
        config = config or RiderHeadConfig()
        self.config = config
        self.tracker = GreedyPersonTracker(maximum_age_seconds=1.5)
        self.identity = CrossIdIdentityResolver(IdentityConfig(maximum_gap_seconds=1.5))
        self.vote = PerTrackTemporalAlarm(
            TrackTemporalConfig(
                window=8,
                minimum_no_helmet_hits=config.minimum_hits,
                minimum_valid_observations=config.minimum_hits,
                high_confidence_minimum_hits=8,
                observation_ttl_seconds=config.evidence_seconds,
                one_event_per_track_session=True,
            )
        )
        self._last_timestamp = float("-inf")
        self._event_sessions: dict[int, float] = {}

    def reset(self) -> None:
        self.tracker.reset()
        self.identity.reset()
        self.vote.reset()
        self._event_sessions.clear()
        self._last_timestamp = float("-inf")

    def update(self, frame: np.ndarray, rows: Any, timestamp: float) -> dict:
        if not isfinite(timestamp) or timestamp <= self._last_timestamp:
            raise ValueError("Each processed frame needs a fresh increasing timestamp")
        self._last_timestamp = timestamp
        self._event_sessions = {
            key: seen for key, seen in self._event_sessions.items() if timestamp - seen <= 1.5
        }
        height, width = frame.shape[:2]
        objects = []
        for row in rows:
            if len(row) != 6 or not all(isfinite(float(v)) for v in row):
                continue
            x1, y1, x2, y2, score, cls = row
            cls = int(cls)
            if cls not in EXPECTED_NAMES or x2 <= x1 or y2 <= y1:
                continue
            box = Rect(int(x1), int(y1), int(x2), int(y2)).clamp(width, height)
            threshold = self.config.rider_threshold if cls == 0 else self.config.head_threshold
            if box is not None and score >= threshold:
                if cls != 0 and min(box.width, box.height) < self.config.minimum_head_pixels:
                    continue
                objects.append(SceneObject(cls, EXPECTED_NAMES[cls], float(score), box))
        riders = [o for o in objects if o.class_id == 0]
        heads = [o for o in objects if o.class_id != 0]
        matched = associate_heads(riders, heads)
        tracked = self.tracker.assign(riders, timestamp)
        identities = self.identity.resolve_frame(tracked, frame, timestamp)
        observations = []
        for index, person in enumerate(identities):
            choices = sorted(matched[index], key=lambda h: h.confidence, reverse=True)
            state = HelmetState.UNKNOWN
            chosen = None
            if choices:
                chosen = choices[0]
                # Reject multiple spatially distinct heads as well as competing classes.
                distinct = any(iou(chosen.box, h.box) < 0.25 for h in choices[1:])
                conflicting = any(
                    h.class_id != chosen.class_id
                    and chosen.confidence - h.confidence < self.config.conflict_margin
                    for h in choices[1:]
                )
                if not distinct and not conflicting:
                    state = HelmetState.HELMET if chosen.class_id == 1 else HelmetState.NO_HELMET
            score = chosen.confidence if chosen is not None else 0.0
            vote = self.vote.update(
                person.canonical_track_id,
                timestamp,
                state,
                no_helmet_score=score if state is HelmetState.NO_HELMET else 0,
                high_confidence_threshold=1.0,
                allow_event=(
                    state is HelmetState.NO_HELMET
                    and person.canonical_track_id not in self._event_sessions
                ),
            )
            if vote.event_triggered or person.canonical_track_id in self._event_sessions:
                self._event_sessions[person.canonical_track_id] = timestamp
            observations.append(
                {
                    "track_id": person.canonical_track_id,
                    "source_track_id": person.source_track_id,
                    "state": state.value,
                    "confidence": score,
                    "rider_box": riders[index].box.to_list(),
                    "head_box": None if chosen is None else chosen.box.to_list(),
                    "event": vote.event_triggered,
                    "vote": vote.to_dict(),
                }
            )
        self.identity.prune(timestamp)
        self.vote.prune(timestamp)
        return {
            "timestamp_seconds": timestamp,
            "observations": observations,
            "events": sum(o["event"] for o in observations),
        }


class RiderHeadDetector:
    def __init__(
        self,
        weights: str,
        *,
        image_size: int = 1280,
        device: str = "cpu",
        config: RiderHeadConfig | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        names = {int(k): v for k, v in self.model.names.items()}
        if names != EXPECTED_NAMES:
            raise ValueError(f"Wrong weight class contract: {names}; expected {EXPECTED_NAMES}")
        self.image_size = image_size
        self.device = device
        self.logic = RiderHeadLogic(config)

    def detect(self, frame: np.ndarray, timestamp: float) -> dict:
        import time

        start = time.perf_counter()
        result = self.model.predict(
            frame, imgsz=self.image_size, device=self.device, conf=0.1, verbose=False, max_det=100
        )[0]
        rows = result.boxes.data.cpu().numpy()
        output = self.logic.update(frame, rows, timestamp)
        output["processing_ms"] = (time.perf_counter() - start) * 1000
        output["device"] = str(self.model.device)
        return output
