"""Rendering for dynamic per-person helmet results."""

from __future__ import annotations

import numpy as np

from .dynamic_config import DynamicDetectorConfig
from .dynamic_types import DynamicFrameResult, HelmetState


def annotate_dynamic_frame(
    frame: np.ndarray,
    result: DynamicFrameResult,
    *,
    config: DynamicDetectorConfig,
    show_contexts: bool = False,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("OpenCV is required for rendering") from exc

    output = frame.copy()
    if config.draw_scene_objects:
        for scene_object in result.scene_objects:
            if scene_object.class_id == config.scene_model.person_class_id:
                continue
            if scene_object.class_id not in config.scene_model.vehicle_class_ids:
                continue
            box = scene_object.box
            cv2.rectangle(output, (box.x1, box.y1), (box.x2, box.y2), (180, 140, 40), 1)

    for observation in result.persons:
        if show_contexts:
            for context in observation.context_boxes:
                cv2.rectangle(
                    output,
                    (context.x1, context.y1),
                    (context.x2, context.y2),
                    (255, 0, 255),
                    1,
                )

        if not observation.rider_eligible:
            colour = (130, 130, 130)
            status = "RIDER-PENDING" if observation.helmet_evaluated else "NOT-EVALUATED"
        elif observation.state is HelmetState.NO_HELMET:
            colour = (0, 0, 255)
            status = "NO-HELMET"
        elif observation.state is HelmetState.HELMET:
            colour = (0, 190, 0)
            status = "HELMET"
        else:
            colour = (160, 160, 160)
            status = "UNKNOWN"
        if observation.event_triggered:
            colour = (0, 0, 255)
            suffix = observation.vote.trigger_mode or "event"
            status += f" EVENT:{suffix}"
        elif observation.vote.event_suppressed:
            status += " DEDUP"

        person_box = observation.person.box
        cv2.rectangle(
            output,
            (person_box.x1, person_box.y1),
            (person_box.x2, person_box.y2),
            colour,
            2,
        )
        if observation.matched_vehicle is not None:
            vehicle_box = observation.matched_vehicle.box
            cv2.line(
                output,
                (int(person_box.centre[0]), person_box.y2),
                (int(vehicle_box.centre[0]), int(vehicle_box.centre[1])),
                (180, 140, 40),
                1,
            )

        for detection in observation.head_detections:
            threshold = (config.helmet_model.no_helmet_threshold
                         if detection.class_id == config.helmet_model.no_helmet_class_id
                         else config.helmet_model.helmet_threshold)
            if detection.confidence < threshold:
                continue
            if observation.state is HelmetState.UNKNOWN or not observation.rider_eligible:
                head_colour = (160, 160, 160)
            elif detection.class_id == config.helmet_model.no_helmet_class_id:
                head_colour = (0, 0, 255)
            else:
                head_colour = (0, 210, 0)
            box = detection.box
            cv2.rectangle(output, (box.x1, box.y1), (box.x2, box.y2), head_colour, 2)

        raw_suffix = (
            f"/raw{observation.source_track_id}"
            if observation.source_track_id != observation.track_id
            else ""
        )
        label = (
            f"ID {observation.track_id}{raw_suffix} {status} "
            f"N={observation.no_helmet_score:.2f} H={observation.helmet_score:.2f} "
            f"vote={observation.vote.hit_count}/{config.temporal.minimum_no_helmet_hits} "
            f"hi={observation.vote.high_confidence_hit_count}/"
            f"{config.temporal.high_confidence_minimum_hits} "
            f"veh={observation.rider_evidence.vehicle_hit_count}"
        )
        _draw_label(output, label, person_box.x1, person_box.y1, colour)

    if config.alarm_zone is not None:
        points = np.asarray(
            [
                [
                    int(round(x * output.shape[1])),
                    int(round(y * output.shape[0])),
                ]
                for x, y in config.alarm_zone
            ],
            dtype=np.int32,
        )
        cv2.polylines(output, [points], True, (255, 255, 0), 2)

    cv2.rectangle(output, (0, 0), (output.shape[1], 58), (0, 0, 0), -1)
    status = (
        "NO-HELMET EVENT"
        if result.event_count
        else ("NO-HELMET" if result.alarm else "MONITORING")
    )
    status_colour = (0, 0, 255) if result.alarm or result.event_count else (240, 240, 240)
    cv2.putText(
        output,
        status,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_colour,
        2,
        cv2.LINE_AA,
    )
    detail = (
        f"t={result.timestamp_seconds:.2f}s  people={len(result.persons)}  "
        f"riders={result.rider_count}  events={result.event_count}  "
        f"max_no_helmet={result.maximum_no_helmet_score:.3f}"
    )
    cv2.putText(
        output,
        detail,
        (12, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return output


def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
) -> None:
    import cv2

    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    left = max(0, min(image.shape[1] - size[0] - 6, x))
    bottom = max(64 + size[1], y - 5)
    cv2.rectangle(
        image,
        (left, bottom - size[1] - 5),
        (left + size[0] + 6, bottom + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (left + 3, bottom - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        colour,
        1,
        cv2.LINE_AA,
    )
