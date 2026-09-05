"""Dynamic full-frame person tracking followed by per-person helmet inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter

import numpy as np

from .dynamic_config import DynamicDetectorConfig
from .dynamic_types import (
    DynamicFrameResult,
    HelmetState,
    PersonObservation,
    SceneObject,
    TrackVote,
)
from .early_check import EarlyCheckScheduler
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
        self.early_scheduler = EarlyCheckScheduler(config.early_check, maximum_age=rider_age)
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
        started = perf_counter()
        frame_height, frame_width = frame.shape[:2]
        scene_objects = self.scene_detector.detect(frame, persist=persist_tracks)

        scene_done = perf_counter()
        people = [
            item
            for item in scene_objects
            if item.class_id == self.config.scene_model.person_class_id
            and item.confidence >= self.config.scene_model.minimum_person_confidence
            and item.box.height >= self.config.scene_model.minimum_person_height
        ]
        raw_people_count = len(people)
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

        selected_ids = self.early_scheduler.select(
            [(person.canonical_track_id, evidence.eligible)
             for person, _, evidence in candidates], timestamp_seconds
        )
        crop_metadata = []
        crop_images: list[np.ndarray] = []
        context_boxes_by_person: list[list] = [[] for _ in candidates]
        for person_index, (resolved, _, _rider_evidence) in enumerate(candidates):
            if resolved.canonical_track_id not in selected_ids:
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

        head_started = perf_counter()
        crop_results = self.helmet_detector.detect_batch(crop_images)
        head_done = perf_counter()
        if len(crop_results) != len(crop_metadata):
            raise RuntimeError(
                "Helmet backend returned a different number of result batches "
                f"({len(crop_results)}) than input crops ({len(crop_metadata)})"
            )

        # Merge crops globally, then give each physical head at most one owner.
        mapped: list[Detection] = []
        for context, detections in zip(crop_metadata, crop_results, strict=True):
            for detection in detections:
                if detection.class_id in {
                    self.config.helmet_model.helmet_class_id,
                    self.config.helmet_model.no_helmet_class_id,
                }:
                    mapped.append(replace(
                        detection,
                        box=detection.box.translate(context.box.x1, context.box.y1),
                    ))
        detections_by_person: list[list[Detection]] = [[] for _ in candidates]
        for detection in _merge_duplicate_head_detections(mapped):
            possible = []
            for index, (resolved, _, _) in enumerate(candidates):
                if resolved.canonical_track_id not in selected_ids:
                    continue
                person = resolved.person
                zone = build_head_zone(
                    person, config=self.config.helmet_model,
                    frame_width=frame_width, frame_height=frame_height,
                )
                if zone.contains_centre(detection.box):
                    x, y = detection.box.centre
                    px, _ = person.box.centre
                    py = person.box.y1 + person.box.height * 0.12
                    distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                    possible.append((distance / max(person.box.height, 1), index))
            possible.sort()
            # Overlapping people with equally plausible ownership are UNKNOWN.
            if possible and (len(possible) == 1 or possible[1][0] - possible[0][0] > 0.05):
                detections_by_person[possible[0][1]].append(detection)

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
                if resolved.canonical_track_id in selected_ids
                else HelmetState.UNKNOWN
            )
            canonical_id = resolved.canonical_track_id

            foot_x = resolved.person.box.centre[0]
            foot_y = resolved.person.box.y2
            zone_eligible = point_in_normalised_polygon(
                foot_x, foot_y, self.config.alarm_zone,
                frame_width=frame_width, frame_height=frame_height,
            )
            allowed = rider_evidence.eligible and zone_eligible
            if apply_temporal:
                vote = self.temporal.update(
                    canonical_id, timestamp_seconds, state,
                    no_helmet_score=best_no_helmet,
                    high_confidence_threshold=(
                        self.config.helmet_model.high_confidence_no_helmet_threshold
                    ),
                    allow_event=allowed,
                )
            else:
                is_alarm = allowed and state is HelmetState.NO_HELMET
                vote = TrackVote(
                    hit_count=int(state is HelmetState.NO_HELMET),
                    high_confidence_hit_count=int(
                        state is HelmetState.NO_HELMET and best_no_helmet
                        >= self.config.helmet_model.high_confidence_no_helmet_threshold
                    ),
                    valid_observations=int(state is not HelmetState.UNKNOWN),
                    window_size=1, active=is_alarm, event_triggered=is_alarm,
                    event_suppressed=False,
                    trigger_mode="single_frame" if is_alarm else None,
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
                    helmet_evaluated=canonical_id in selected_ids,
                    head_check_reason=(
                        "rider" if rider_evidence.eligible else
                        "early_probe" if canonical_id in selected_ids else "not_scheduled"
                    ),
                )
            )

        self.temporal.prune(timestamp_seconds)
        self.rider_tracker.prune(timestamp_seconds)
        self.identity_resolver.prune(timestamp_seconds)
        return DynamicFrameResult(
            timestamp_seconds=timestamp_seconds,
            scene_objects=tuple(scene_objects),
            persons=tuple(observations),
            diagnostics={
                "scene_ms": round((scene_done - started) * 1000, 3),
                "helmet_ms": round((head_done - head_started) * 1000, 3),
                "pipeline_ms": round((perf_counter() - started) * 1000, 3),
                "helmet_crops": len(crop_images),
                "raw_people": raw_people_count,
                "selected_people": len(candidates),
                "scene_device": str(getattr(self.scene_detector, "device", "unknown")),
                "helmet_device": str(getattr(self.helmet_detector, "device", "unknown")),
            },
        )

    def reset(self) -> None:
        reset_scene = getattr(self.scene_detector, "reset", None)
        if callable(reset_scene):
            reset_scene()
        self.early_scheduler.reset()
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
        config.early_check.enabled
        and best_helmet >= helmet.helmet_threshold
        and best_no_helmet >= helmet.no_helmet_threshold
        and abs(best_helmet - best_no_helmet) < config.early_check.conflict_margin
    ):
        return HelmetState.UNKNOWN
    if (
        best_no_helmet >= helmet.no_helmet_threshold
        and best_no_helmet >= best_helmet * helmet.no_helmet_to_helmet_ratio
        and (not config.early_check.enabled or best_no_helmet > best_helmet)
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
