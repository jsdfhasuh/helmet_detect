"""Cross-ID stitching for short ByteTrack/fallback tracker ID switches."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .dynamic_config import IdentityConfig
from .dynamic_types import SceneObject
from .types import Rect


@dataclass(frozen=True, slots=True)
class ResolvedPerson:
    """One scene person with a stable canonical ID."""

    person: SceneObject
    source_track_id: int
    canonical_track_id: int
    stitched: bool
    appearance_similarity: float | None = None


@dataclass(slots=True)
class _Identity:
    canonical_id: int
    raw_track_ids: set[int] = field(default_factory=set)
    box: Rect | None = None
    last_seen_time: float = float("-inf")
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    appearance: np.ndarray | None = None


class CrossIdIdentityResolver:
    """Map unstable raw tracker IDs onto short-lived canonical identities.

    ByteTrack can change an ID when a small, partially occluded person disappears for a
    few frames.  The resolver performs a one-to-one match against recently seen
    identities using predicted centre position, body size and a compact colour
    signature.  It is intentionally conservative: a poor match creates a new identity
    rather than merging two nearby people.
    """

    def __init__(self, config: IdentityConfig) -> None:
        self.config = config
        self._identities: dict[int, _Identity] = {}
        self._raw_to_canonical: dict[int, int] = {}
        self._used_canonical_ids: set[int] = set()
        self._next_synthetic_id = 10_000_000

    def resolve_frame(
        self,
        people: list[SceneObject],
        frame: np.ndarray,
        timestamp_seconds: float,
    ) -> list[ResolvedPerson]:
        self.prune(timestamp_seconds)
        if not people:
            return []

        signatures = [_appearance_signature(frame, person.box) for person in people]
        resolved: list[ResolvedPerson | None] = [None] * len(people)
        assigned_canonical: set[int] = set()

        # Preserve a known raw-ID mapping first.  The one-to-one guard prevents a
        # duplicate tracker box from claiming the same canonical identity twice.
        for index, person in enumerate(people):
            raw_id = _required_track_id(person)
            canonical_id = self._raw_to_canonical.get(raw_id)
            identity = self._identities.get(canonical_id) if canonical_id is not None else None
            if identity is None or canonical_id in assigned_canonical:
                continue
            similarity = _cosine_similarity(identity.appearance, signatures[index])
            resolved[index] = ResolvedPerson(
                person=person,
                source_track_id=raw_id,
                canonical_track_id=canonical_id,
                stitched=len(identity.raw_track_ids) > 1,
                appearance_similarity=similarity,
            )
            assigned_canonical.add(canonical_id)

        # Build all admissible new-ID -> recent-identity candidates, then greedily
        # take the best one-to-one assignments globally.
        candidates: list[tuple[float, int, int, float | None]] = []
        if self.config.enabled:
            for index, person in enumerate(people):
                if resolved[index] is not None:
                    continue
                raw_id = _required_track_id(person)
                signature = signatures[index]
                for canonical_id, identity in self._identities.items():
                    if canonical_id in assigned_canonical or identity.box is None:
                        continue
                    if raw_id in identity.raw_track_ids:
                        continue
                    scored = self._match_score(
                        person.box,
                        signature,
                        identity,
                        timestamp_seconds,
                    )
                    if scored is None:
                        continue
                    score, similarity = scored
                    candidates.append((score, index, canonical_id, similarity))

            claimed_people: set[int] = set()
            for _, index, canonical_id, similarity in sorted(candidates):
                if index in claimed_people or canonical_id in assigned_canonical:
                    continue
                person = people[index]
                raw_id = _required_track_id(person)
                resolved[index] = ResolvedPerson(
                    person=person,
                    source_track_id=raw_id,
                    canonical_track_id=canonical_id,
                    stitched=True,
                    appearance_similarity=similarity,
                )
                claimed_people.add(index)
                assigned_canonical.add(canonical_id)

        # Anything not stitched starts a new canonical identity.  Reuse the raw ID as
        # the first canonical ID where possible so logs remain easy to interpret.
        for index, person in enumerate(people):
            if resolved[index] is not None:
                continue
            raw_id = _required_track_id(person)
            canonical_id = self._new_canonical_id(raw_id)
            resolved[index] = ResolvedPerson(
                person=person,
                source_track_id=raw_id,
                canonical_track_id=canonical_id,
                stitched=False,
                appearance_similarity=None,
            )
            assigned_canonical.add(canonical_id)

        output = [item for item in resolved if item is not None]
        for item, signature in zip(output, signatures, strict=True):
            self._update_identity(item, signature, timestamp_seconds)
        return output

    def _match_score(
        self,
        box: Rect,
        appearance: np.ndarray | None,
        identity: _Identity,
        timestamp_seconds: float,
    ) -> tuple[float, float | None] | None:
        previous = identity.box
        if previous is None:
            return None
        gap = timestamp_seconds - identity.last_seen_time
        if gap < 0 or gap > self.config.maximum_gap_seconds:
            return None

        prediction_seconds = min(gap, self.config.maximum_prediction_seconds)
        predicted_x = previous.centre[0] + identity.velocity_x * prediction_seconds
        predicted_y = previous.centre[1] + identity.velocity_y * prediction_seconds
        centre_distance = math.dist(box.centre, (predicted_x, predicted_y))
        distance_ratio = centre_distance / max(box.height, previous.height, 1)
        allowed_distance = self.config.maximum_center_distance_ratio * (
            1.0 + 0.10 * min(gap, 2.0)
        )
        if distance_ratio > allowed_distance:
            return None

        height_ratio = max(box.height, previous.height) / max(
            min(box.height, previous.height),
            1,
        )
        if height_ratio > self.config.maximum_height_ratio:
            return None

        similarity = _cosine_similarity(identity.appearance, appearance)
        if (
            similarity is not None
            and similarity < self.config.minimum_appearance_similarity
        ):
            return None

        appearance_penalty = 0.5 if similarity is None else 1.0 - similarity
        size_penalty = abs(math.log(max(height_ratio, 1.0)))
        gap_penalty = gap / max(self.config.maximum_gap_seconds, 1e-6)
        score = (
            distance_ratio
            + 0.35 * size_penalty
            + self.config.appearance_weight * appearance_penalty
            + 0.10 * gap_penalty
        )
        return score, similarity

    def _new_canonical_id(self, raw_id: int) -> int:
        if raw_id not in self._used_canonical_ids:
            canonical_id = raw_id
        else:
            while self._next_synthetic_id in self._used_canonical_ids:
                self._next_synthetic_id += 1
            canonical_id = self._next_synthetic_id
            self._next_synthetic_id += 1
        self._used_canonical_ids.add(canonical_id)
        self._identities[canonical_id] = _Identity(canonical_id=canonical_id)
        return canonical_id

    def _update_identity(
        self,
        resolved: ResolvedPerson,
        signature: np.ndarray | None,
        timestamp_seconds: float,
    ) -> None:
        identity = self._identities.setdefault(
            resolved.canonical_track_id,
            _Identity(canonical_id=resolved.canonical_track_id),
        )
        previous_box = identity.box
        previous_time = identity.last_seen_time
        if previous_box is not None and timestamp_seconds > previous_time:
            dt = timestamp_seconds - previous_time
            measured_x = (resolved.person.box.centre[0] - previous_box.centre[0]) / dt
            measured_y = (resolved.person.box.centre[1] - previous_box.centre[1]) / dt
            smoothing = self.config.velocity_smoothing
            identity.velocity_x = smoothing * identity.velocity_x + (1.0 - smoothing) * measured_x
            identity.velocity_y = smoothing * identity.velocity_y + (1.0 - smoothing) * measured_y

        identity.box = resolved.person.box
        identity.last_seen_time = timestamp_seconds
        identity.raw_track_ids.add(resolved.source_track_id)
        self._raw_to_canonical[resolved.source_track_id] = resolved.canonical_track_id
        if signature is not None:
            if identity.appearance is None:
                identity.appearance = signature
            else:
                blended = 0.75 * identity.appearance + 0.25 * signature
                norm = float(np.linalg.norm(blended))
                identity.appearance = blended / norm if norm > 0 else blended

    def prune(self, timestamp_seconds: float) -> None:
        stale = [
            canonical_id
            for canonical_id, identity in self._identities.items()
            if timestamp_seconds - identity.last_seen_time
            > self.config.maximum_gap_seconds
        ]
        for canonical_id in stale:
            identity = self._identities.pop(canonical_id)
            for raw_id in identity.raw_track_ids:
                if self._raw_to_canonical.get(raw_id) == canonical_id:
                    del self._raw_to_canonical[raw_id]

    def reset(self) -> None:
        self._identities.clear()
        self._raw_to_canonical.clear()
        self._used_canonical_ids.clear()
        self._next_synthetic_id = 10_000_000


def _required_track_id(person: SceneObject) -> int:
    if person.track_id is None:  # pragma: no cover - fallback tracker assigns one
        raise RuntimeError("Person has no raw track ID")
    return int(person.track_id)


def _appearance_signature(frame: np.ndarray, box: Rect) -> np.ndarray | None:
    """Return a small normalised colour descriptor from the person's torso region."""

    x_margin = max(0, int(round(box.width * 0.10)))
    x1 = min(box.x2 - 1, box.x1 + x_margin)
    x2 = max(x1 + 1, box.x2 - x_margin)
    y1 = min(box.y2 - 1, box.y1 + int(round(box.height * 0.15)))
    y2 = max(y1 + 1, box.y1 + int(round(box.height * 0.88)))
    crop = frame[y1:y2, x1:x2]
    if crop.size < 12:
        return None

    features: list[np.ndarray] = []
    for channel in range(min(3, crop.shape[2])):
        histogram, _ = np.histogram(
            crop[..., channel],
            bins=8,
            range=(0, 256),
        )
        features.append(histogram.astype(np.float32))
    vector = np.concatenate(features)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def _cosine_similarity(
    first: np.ndarray | None,
    second: np.ndarray | None,
) -> float | None:
    if first is None or second is None:
        return None
    return float(np.clip(np.dot(first, second), 0.0, 1.0))
