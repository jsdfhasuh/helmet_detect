"""Video processing for the dynamic V2 pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .dynamic_pipeline import DynamicHelmetDetectionPipeline
from .dynamic_render import annotate_dynamic_frame
from .dynamic_types import HelmetState


@dataclass(frozen=True, slots=True)
class DynamicVideoSummary:
    source_fps: float
    output_fps: float
    processed_frames: int
    frames_with_people: int
    frames_with_riders: int
    no_helmet_observation_frames: int
    alarm_frames: int
    events: int
    high_confidence_events: int
    unique_tracks: int
    raw_unique_tracks: int
    maximum_no_helmet_score: float
    output_path: str
    records_path: str
    events_directory: str

    def to_dict(self) -> dict[str, object]:
        return {
            "pipeline": "dynamic_v2",
            "source_fps": round(self.source_fps, 6),
            "output_fps": round(self.output_fps, 6),
            "processed_frames": self.processed_frames,
            "frames_with_people": self.frames_with_people,
            "frames_with_riders": self.frames_with_riders,
            "no_helmet_observation_frames": self.no_helmet_observation_frames,
            "alarm_frames": self.alarm_frames,
            "events": self.events,
            "high_confidence_events": self.high_confidence_events,
            "unique_tracks": self.unique_tracks,
            "raw_unique_tracks": self.raw_unique_tracks,
            "maximum_no_helmet_score": round(self.maximum_no_helmet_score, 6),
            "output_path": self.output_path,
            "records_path": self.records_path,
            "events_directory": self.events_directory,
        }


def process_dynamic_video(
    pipeline: DynamicHelmetDetectionPipeline,
    source: str | Path,
    output: str | Path,
    records: str | Path,
    *,
    sample_fps: float = 5.0,
    show_contexts: bool = False,
) -> DynamicVideoSummary:
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

    stride = 1 if sample_fps <= 0 else max(1, int(round(source_fps / sample_fps)))
    output_fps = max(1.0, source_fps / stride)
    output_path = Path(output)
    records_path = Path(records)
    events_directory = output_path.parent / f"{output_path.stem}_events"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    events_directory.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    pipeline.reset()
    frame_index = 0
    processed_frames = 0
    frames_with_people = 0
    frames_with_riders = 0
    no_helmet_observation_frames = 0
    alarm_frames = 0
    event_count = 0
    high_confidence_events = 0
    maximum_score = 0.0
    canonical_tracks: set[int] = set()
    raw_tracks: set[int] = set()

    with records_path.open("w", encoding="utf-8") as records_file:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            timestamp = frame_index / source_fps
            result = pipeline.detect_frame(
                frame,
                timestamp_seconds=timestamp,
                persist_tracks=True,
                apply_temporal=True,
            )
            for person in result.persons:
                canonical_tracks.add(person.track_id)
                raw_tracks.add(person.source_track_id)
            frames_with_people += int(bool(result.persons))
            frames_with_riders += int(any(person.rider_eligible for person in result.persons))
            no_helmet_observation_frames += int(
                any(
                    person.rider_eligible and person.state is HelmetState.NO_HELMET
                    for person in result.persons
                )
            )
            alarm_frames += int(result.alarm)
            maximum_score = max(maximum_score, result.maximum_no_helmet_score)

            record = result.to_dict()
            record["frame_index"] = frame_index
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            annotated = annotate_dynamic_frame(
                frame,
                result,
                config=pipeline.config,
                show_contexts=show_contexts,
            )
            writer.write(annotated)
            for person in result.persons:
                if person.event_triggered:
                    event_count += 1
                    if person.vote.trigger_mode == "high_confidence":
                        high_confidence_events += 1
                    event_name = (
                        f"event_{event_count:04d}_id_{person.track_id}_"
                        f"raw_{person.source_track_id}_"
                        f"{person.vote.trigger_mode or 'event'}_t_{timestamp:.2f}.jpg"
                    )
                    event_path = events_directory / event_name
                    if not cv2.imwrite(str(event_path), annotated):
                        raise RuntimeError(f"Cannot write event snapshot: {event_path}")

            processed_frames += 1
            frame_index += 1

    capture.release()
    writer.release()
    return DynamicVideoSummary(
        source_fps=source_fps,
        output_fps=output_fps,
        processed_frames=processed_frames,
        frames_with_people=frames_with_people,
        frames_with_riders=frames_with_riders,
        no_helmet_observation_frames=no_helmet_observation_frames,
        alarm_frames=alarm_frames,
        events=event_count,
        high_confidence_events=high_confidence_events,
        unique_tracks=len(canonical_tracks),
        raw_unique_tracks=len(raw_tracks),
        maximum_no_helmet_score=maximum_score,
        output_path=str(output_path),
        records_path=str(records_path),
        events_directory=str(events_directory),
    )
