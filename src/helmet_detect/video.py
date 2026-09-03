"""Video processing with sampled inference and temporal voting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .pipeline import HelmetDetectionPipeline
from .render import annotate_frame
from .temporal import TemporalAlarm


@dataclass(frozen=True, slots=True)
class VideoSummary:
    source_fps: float
    output_fps: float
    processed_frames: int
    alarm_frames: int
    events: int
    maximum_no_helmet_score: float
    output_path: str
    records_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_fps": round(self.source_fps, 6),
            "output_fps": round(self.output_fps, 6),
            "processed_frames": self.processed_frames,
            "alarm_frames": self.alarm_frames,
            "events": self.events,
            "maximum_no_helmet_score": round(self.maximum_no_helmet_score, 6),
            "output_path": self.output_path,
            "records_path": self.records_path,
        }


def process_video(
    pipeline: HelmetDetectionPipeline,
    source: str | Path,
    output: str | Path,
    records: str | Path,
    *,
    sample_fps: float = 5.0,
) -> VideoSummary:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("OpenCV is required. Install with: pip install -e '.[runtime]'") from exc

    source_path = str(source)
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid dimensions: {frame_width}x{frame_height}")

    if sample_fps <= 0:
        stride = 1
    else:
        stride = max(1, int(round(source_fps / sample_fps)))
    output_fps = max(1.0, source_fps / stride)

    output_path = Path(output)
    records_path = Path(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    temporal = TemporalAlarm(pipeline.config.temporal)
    frame_index = 0
    processed_frames = 0
    alarm_frames = 0
    events = 0
    maximum_score = 0.0

    with records_path.open("w", encoding="utf-8") as records_file:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            timestamp = frame_index / source_fps
            frame_result = pipeline.detect_frame(frame)
            vote = temporal.update(timestamp, frame_result.alarm)
            maximum_score = max(maximum_score, frame_result.fused_no_helmet_score)
            alarm_frames += int(frame_result.alarm)
            events += int(vote.event_triggered)

            record = frame_result.to_dict()
            record.update(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp, 6),
                    "temporal_hit_count": vote.hit_count,
                    "temporal_window_size": vote.window_size,
                    "temporal_active": vote.active,
                    "event_triggered": vote.event_triggered,
                }
            )
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            annotated = annotate_frame(
                frame,
                frame_result,
                roi=pipeline.config.roi,
                gate=pipeline.config.gate,
                timestamp_seconds=timestamp,
                temporal_hits=(vote.hit_count, pipeline.config.temporal.min_hits),
                event_triggered=vote.event_triggered,
            )
            writer.write(annotated)
            processed_frames += 1
            frame_index += 1

    capture.release()
    writer.release()
    return VideoSummary(
        source_fps=source_fps,
        output_fps=output_fps,
        processed_frames=processed_frames,
        alarm_frames=alarm_frames,
        events=events,
        maximum_no_helmet_score=maximum_score,
        output_path=str(output_path),
        records_path=str(records_path),
    )
