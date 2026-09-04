"""Dynamic full-frame person tracking followed by per-person helmet inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .dynamic_config import DynamicDetectorConfig
from .dynamic_types import (
    DynamicFrameResult,
    HelmetState,
    PersonObservation,
    SceneObject,
    TrackVote,
)
from .fallback_tracker import GreedyPersonTracker
from .geometry import (
    build_context_crops,
    build_head_zone,
    deduplicate_people,
    iou,
    match_vehicle,
    point_in_normalised_polygon,
)
from .identity import CrossIdIdentityResolver, ResolvedPerson
from .rider_state import RiderEvidence, RiderEvidenceTracker
from .track_state import PerTrackTemporalAlarm
from .types import Detection
from .ultralytics_backend import (
    HelmetDetector,
    SceneDetector,
    UltralyticsHelmetDetector,
    UltralyticsSceneDetector,
)


class DynamicHelmetDetectionPipeline:
    """Detect people globally, stitch IDs, select riders, then judge helmet state."""

    def __init__(
        self,
        config: DynamicDetectorConfig,
        *,
        scene_detector: SceneDetector | None = None,
        helmet_detector: HelmetDetector | None = None,
    ) -> None:
        self.config = config
        self.scene_detector = scene_detector or UltralyticsSceneDetector(config.scene_model)
        self.helmet_detector = helmet_detector or UltralyticsHelmetDetector(config.helmet_model)
        self.temporal = PerTrackTemporalAlarm(config.temporal)
        self.fallback_tracker = GreedyPersonTracker(
            maximum_age_seconds=config.temporal.maximum_track_age_seconds
        )
        self.identity_resolver = CrossIdIdentityResolver(config.identity)
        rider_age = max(
            config.temporal.maximum_track_age_seconds,
            config.identity.maximum_gap_seconds,
            config.association.vehicle_grace_seconds,
        )
        self.rider_tracker = RiderEvidenceTracker(
            config.association,
            maximum_age_seconds=rider_age,
        )

    def detect_frame(
        self,
        frame: np.ndarray,
        *,
        timestamp_seconds: float = 0.0,
        persist_tracks: bool = True,
        apply_temporal: bool = True,
    ) -> DynamicFrameResult:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape H x W x 3")
        frame_height, frame_width = frame.shape[:2]
        scene_objects = self.scene_detector.detect(frame, persist=persist_tracks)

        people = [
            item
            for item in scene_objects
            if item.class_id == self.config.scene_model.person_class_id
            and item.confidence >= self.config.scene_model.minimum_person_confidence
            and item.box.height >= self.config.scene_model.minimum_person_height
        ]
        people = deduplicate_people(
            people,
            maximum_people=self.config.scene_model.maximum_persons,
        )
        people = self.fallback_tracker.assign(people, timestamp_seconds)
        resolved_people = self.identity_resolver.resolve_frame(
            people,
            frame,
            timestamp_seconds,
        )
        vehicles = [
            item
            for item in scene_objects
            if item.class_id in self.config.scene_model.vehicle_class_ids
        ]

        candidates: list[tuple[ResolvedPerson, SceneObject | None, RiderEvidence]] = []
        for resolved in resolved_people:
            vehicle = match_vehicle(
                resolved.person,
                vehicles,
                config=self.config.association,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            rider_evidence = self.rider_tracker.update(
                resolved.canonical_track_id,
                timestamp_seconds,
                has_vehicle_match=vehicle is not None,
            )
            candidates.append((resolved, vehicle, rider_evidence))

        crop_metadata = []
        crop_images: list[np.ndarray] = []
        context_boxes_by_person: list[list] = [[] for _ in candidates]
        for person_index, (resolved, _, rider_evidence) in enumerate(candidates):
            if not rider_evidence.eligible:
                continue
            contexts = build_context_crops(
                person_index,
                resolved.person,
                config=self.config.helmet_model,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            for context in contexts:
                crop = frame[
                    context.box.y1 : context.box.y2,
                    context.box.x1 : context.box.x2,
                ]
                if crop.size == 0:
                    continue
                crop_metadata.append(context)
                crop_images.append(crop)
                context_boxes_by_person[person_index].append(context.box)

        crop_results = self.helmet_detector.detect_batch(crop_images)
        if len(crop_results) != len(crop_metadata):
            raise RuntimeError(
                "Helmet backend returned a different number of result batches "
                f"({len(crop_results)}) than input crops ({len(crop_metadata)})"
            )

        detections_by_person: list[list[Detection]] = [[] for _ in candidates]
        for context, detections in zip(crop_metadata, crop_results, strict=True):
            person = candidates[context.person_index][0].person
            head_zone = build_head_zone(
                person,
                config=self.config.helmet_model,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            for detection in detections:
                if detection.class_id not in {
                    self.config.helmet_model.helmet_class_id,
                    self.config.helmet_model.no_helmet_class_id,
                }:
                    continue
                global_detection = replace(
                    detection,
                    box=detection.box.translate(context.box.x1, context.box.y1),
                )
                if head_zone.contains_centre(global_detection.box):
                    detections_by_person[context.person_index].append(global_detection)

        observations: list[PersonObservation] = []
        for person_index, (resolved, vehicle, rider_evidence) in enumerate(candidates):
            head_detections = _merge_duplicate_head_detections(
                detections_by_person[person_index]
            )
            best_no_helmet = max(
                (
                    item.confidence
                    for item in head_detections
                    if item.class_id == self.config.helmet_model.no_helmet_class_id
                ),
                default=0.0,
            )
            best_helmet = max(
                (
                    item.confidence
                    for item in head_detections
                    if item.class_id == self.config.helmet_model.helmet_class_id
                ),
                default=0.0,
            )
            state = (
                _helmet_state(
                    best_no_helmet,
                    best_helmet,
                    config=self.config,
                )
                if rider_evidence.eligible
                else HelmetState.UNKNOWN
            )
            canonical_id = resolved.canonical_track_id

            if apply_temporal and rider_evidence.eligible:
                vote = self.temporal.update(
                    canonical_id,
                    timestamp_seconds,
                    state,
                    no_helmet_score=best_no_helmet,
                    high_confidence_threshold=(
                        self.config.helmet_model.high_confidence_no_helmet_threshold
                    ),
                )
            elif apply_temporal:
                vote = self.temporal.inactive_vote()
            else:
                is_alarm = (
                    rider_evidence.eligible
                    and state is HelmetState.NO_HELMET
                )
                vote = TrackVote(
                    hit_count=int(is_alarm),
                    high_confidence_hit_count=int(
                        is_alarm
                        and best_no_helmet
                        >= self.config.helmet_model.high_confidence_no_helmet_threshold
                    ),
                    valid_observations=int(
                        rider_evidence.eligible
                        and state is not HelmetState.UNKNOWN
                    ),
                    window_size=1,
                    active=is_alarm,
                    event_triggered=is_alarm,
                    event_suppressed=False,
                    trigger_mode="single_frame" if is_alarm else None,
                )

            foot_x = resolved.person.box.centre[0]
            foot_y = resolved.person.box.y2
            zone_eligible = point_in_normalised_polygon(
                foot_x,
                foot_y,
                self.config.alarm_zone,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            observations.append(
                PersonObservation(
                    track_id=canonical_id,
                    source_track_id=resolved.source_track_id,
                    identity_stitched=resolved.stitched,
                    appearance_similarity=resolved.appearance_similarity,
                    person=resolved.person,
                    matched_vehicle=vehicle,
                    rider_evidence=rider_evidence,
                    state=state,
                    no_helmet_score=best_no_helmet,
                    helmet_score=best_helmet,
                    head_detections=tuple(head_detections),
                    context_boxes=tuple(context_boxes_by_person[person_index]),
                    vote=vote,
                    alarm_zone_eligible=zone_eligible,
                )
            )

        self.temporal.prune(timestamp_seconds)
        self.rider_tracker.prune(timestamp_seconds)
        self.identity_resolver.prune(timestamp_seconds)
        return DynamicFrameResult(
            timestamp_seconds=timestamp_seconds,
            scene_objects=tuple(scene_objects),
            persons=tuple(observations),
        )

    def reset(self) -> None:
        self.temporal.reset()
        self.fallback_tracker.reset()
        self.identity_resolver.reset()
        self.rider_tracker.reset()


def _helmet_state(
    best_no_helmet: float,
    best_helmet: float,
    *,
    config: DynamicDetectorConfig,
) -> HelmetState:
    helmet = config.helmet_model
    if (
        best_no_helmet >= helmet.no_helmet_threshold
        and best_no_helmet >= best_helmet * helmet.no_helmet_to_helmet_ratio
    ):
        return HelmetState.NO_HELMET
    if best_helmet >= helmet.helmet_threshold and best_helmet > best_no_helmet:
        return HelmetState.HELMET
    return HelmetState.UNKNOWN


def _merge_duplicate_head_detections(
    detections: Sequence[Detection],
) -> list[Detection]:
    """Merge the same head found in several dynamic context scales."""

    output: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            item.class_id == candidate.class_id and iou(item.box, candidate.box) >= 0.45
            for item in output
        )
        if not duplicate:
            output.append(candidate)
    return output
