"""Drawing helpers."""

from __future__ import annotations

import numpy as np

from .pipeline import FrameResult
from .types import Rect


def _short_model_name(name: str) -> str:
    aliases = {
        "helmet_yolov8n": "V8N",
        "rider_yolo11m": "V11M",
    }
    return aliases.get(name, name[:8])


def annotate_frame(
    frame: np.ndarray,
    result: FrameResult,
    *,
    roi: Rect,
    gate: Rect,
    timestamp_seconds: float | None = None,
    temporal_hits: tuple[int, int] | None = None,
    event_triggered: bool = False,
    draw_threshold: float = 0.05,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("OpenCV is required for rendering") from exc

    output = frame.copy()
    cv2.rectangle(output, (roi.x1, roi.y1), (roi.x2 - 1, roi.y2 - 1), (255, 0, 255), 2)
    cv2.rectangle(output, (gate.x1, gate.y1), (gate.x2, gate.y2), (255, 255, 0), 2)

    visible = [
        item
        for item in result.detections
        if (item.class_id == 1 and item.confidence >= draw_threshold)
        or (item.class_id == 0 and item.confidence >= 0.50)
    ][:8]
    for detection in visible:
        is_no_helmet = detection.class_id == 1
        colour = (0, 0, 255) if is_no_helmet else (0, 200, 0)
        box = detection.box
        cv2.rectangle(output, (box.x1, box.y1), (box.x2, box.y2), colour, 2)
        state = "NO" if is_no_helmet else "HELMET"
        label = f"{_short_model_name(detection.model_name)} {state} {detection.confidence:.3f}"
        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        label_x = max(0, min(output.shape[1] - text_size[0] - 4, box.x1))
        if is_no_helmet:
            label_y = max(76, box.y1 - 7)
        else:
            label_y = min(output.shape[0] - baseline - 4, box.y2 + text_size[1] + 7)
        cv2.rectangle(
            output,
            (label_x, label_y - text_size[1] - 4),
            (label_x + text_size[0] + 4, label_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output,
            label,
            (label_x + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            1,
            cv2.LINE_AA,
        )

    banner_height = 68
    cv2.rectangle(output, (0, 0), (output.shape[1], banner_height), (0, 0, 0), -1)
    status = "NO-HELMET" if result.alarm else "CLEAR"
    if event_triggered:
        status += " EVENT"
    status_colour = (0, 0, 255) if result.alarm else (255, 255, 255)
    cv2.putText(
        output,
        status,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_colour,
        2,
        cv2.LINE_AA,
    )

    details: list[str] = []
    if timestamp_seconds is not None:
        details.append(f"t={timestamp_seconds:.2f}s")
    details.extend(
        f"{_short_model_name(name)}={score:.3f}"
        for name, score in result.best_no_helmet.items()
    )
    if temporal_hits is not None:
        details.append(f"vote={temporal_hits[0]}/{temporal_hits[1]}")
    cv2.putText(
        output,
        "  ".join(details),
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return output
