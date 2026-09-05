"""Opt-in pilot runner: ordered file evaluation OR bounded latest-frame streaming.

The pilot weights are private, camera-specific experiments. Do not connect this
runner to a safety interlock. It writes candidate events for human review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .latest_frame import LatestFrameSlot
from .rider_head import RiderHeadDetector


def draw(frame: np.ndarray, result: dict) -> np.ndarray:
    import cv2

    image = frame.copy()
    for obs in result["observations"]:
        colour = {"helmet": (0, 180, 0), "no_helmet": (0, 0, 240)}.get(
            obs["state"], (150, 150, 150)
        )
        x1, y1, x2, y2 = obs["rider_box"]
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        if obs["head_box"]:
            a, b, c, d = obs["head_box"]
            cv2.rectangle(image, (a, b), (c, d), colour, 2)
        text = f"ID={obs['track_id']} {obs['state']} {obs['confidence']:.2f}"
        if obs["event"]:
            text += " CANDIDATE-EVENT"
        cv2.putText(
            image, text, (max(0, x1 - 80), max(65, y1 - 12)), 0, 0.48, colour, 1, cv2.LINE_AA
        )
    cv2.rectangle(image, (0, 0), (image.shape[1], 48), (0, 0, 0), -1)
    text = (
        f"PILOT / NOT ACCEPTED  t={result['timestamp_seconds']:.2f}  "
        f"{result['processing_ms']:.0f}ms"
    )
    cv2.putText(image, text, (12, 30), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


class EventWriter:
    """Bounded image worker; never silently drop event evidence on overflow."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.jobs: queue.Queue = queue.Queue(maxsize=8)
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        import cv2

        while True:
            item = self.jobs.get()
            try:
                if item is None:
                    return
                index, frame, result = item
                ok = cv2.imwrite(str(self.folder / f"event_{index:04d}.jpg"), draw(frame, result))
                if not ok:
                    self.errors.append("Event image write failed")
            except Exception as exc:
                self.errors.append(type(exc).__name__)
            finally:
                self.jobs.task_done()

    def submit(self, index: int, frame: np.ndarray, result: dict) -> None:
        if self.errors:
            raise RuntimeError(f"Event writer failed: {self.errors[0]}")
        try:
            self.jobs.put_nowait((index, frame.copy(), result))
        except queue.Full as exc:
            raise RuntimeError(
                "Event evidence queue full; stopping instead of silently losing events"
            ) from exc

    def close(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=20)
        if self.thread.is_alive() or self.errors:
            raise RuntimeError("Event evidence writer did not finish cleanly")


def file_frames(source: str, sample_fps: float):
    import cv2

    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open local video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Invalid source FPS; normalize the file before evaluation")
        stride = max(1, round(fps / sample_fps))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                yield frame, index / fps, time.monotonic(), fps / stride
            index += 1
    finally:
        cap.release()


def live_frames(source: str, stop: threading.Event, slot: LatestFrameSlot) -> None:
    import cv2

    cap = None
    try:
        # Open/read timeouts are supported by FFmpeg network capture, not all backends.
        cap = cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000],
        )
        if not cap.isOpened():
            raise RuntimeError("Cannot open stream using timeout-capable FFmpeg backend")
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Stream stopped; reconnect explicitly after checking camera")
            slot.publish(frame)
    except Exception as exc:
        slot.close(type(exc).__name__)  # Do not log a URL containing credentials.
    finally:
        if cap is not None:
            cap.release()
        slot.close(slot.error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Local video, processed in order")
    source.add_argument("--stream-env", help="Environment variable containing RTSP URL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--sample-fps", type=float, default=5)
    parser.add_argument("--duration", type=float, default=60, help="Maximum live test duration")
    parser.add_argument("--save-video", action="store_true", help="File evaluation only")
    parser.add_argument("--accept-experimental", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_experimental:
        parser.error(
            "This small-data pilot is not production accepted; --accept-experimental is required"
        )
    if args.sample_fps <= 0 or not math.isfinite(args.sample_fps):
        parser.error("sample-fps must be finite and positive")
    if (
        args.imgsz < 32
        or args.cpu_threads < 1
        or args.duration <= 0
        or not math.isfinite(args.duration)
    ):
        parser.error("Invalid image size, CPU thread count or duration")
    if args.stream_env and args.save_video:
        parser.error("Continuous synchronous recording is disabled in live mode")
    if not Path(args.weights).is_file():
        parser.error("A local verified pilot weight file is required")

    if args.source and not Path(args.source).is_file():
        parser.error("--source must be an existing local video file")

    import cv2
    import torch

    torch.set_num_threads(args.cpu_threads)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frames.jsonl").exists():
        parser.error("Output already contains a run; choose a new directory")
    detector = RiderHeadDetector(args.weights, image_size=args.imgsz, device=args.device)
    writer = None
    events = EventWriter(out / "events")
    stop = threading.Event()
    slot = LatestFrameSlot()
    producer = None
    iterator = None
    start = time.monotonic()
    count = event_count = 0
    timings: list[float] = []
    ages: list[float] = []
    failure = None
    try:
        if args.source:
            iterator = file_frames(args.source, args.sample_fps)
        else:
            value = os.environ.get(args.stream_env, "")
            if not value:
                raise RuntimeError("Stream environment variable is empty")
            # Warm up before capture so initialization does not age the first frame.
            detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8), 0.0)
            detector.logic.reset()
            start = time.monotonic()
            producer = threading.Thread(target=live_frames, args=(value, stop, slot), daemon=True)
            producer.start()

            def latest_iterator():
                while time.monotonic() - start < args.duration:
                    packet = slot.take(timeout=0.25)
                    if packet is None:
                        if slot.closed:
                            if slot.error:
                                raise RuntimeError(f"Stream worker failed: {slot.error}")
                            break
                        continue
                    yield packet.image, packet.received_at - start, packet.received_at, 0.0

            iterator = latest_iterator()
        with (out / "frames.jsonl").open("w", encoding="utf-8") as log:
            for frame, timestamp, received_at, output_fps in iterator:
                result = detector.detect(frame, timestamp)
                result["post_decode_to_result_ms"] = (time.monotonic() - received_at) * 1000
                result["mode"] = "live" if args.stream_env else "file"
                timings.append(result["processing_ms"])
                ages.append(result["post_decode_to_result_ms"])
                # Deliver the small event record before image encoding.
                if result["events"]:
                    event_count += result["events"]
                    print(json.dumps({"candidate_event": result}, ensure_ascii=False), flush=True)
                    events.submit(event_count, frame, result)
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(out / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            (frame.shape[1], frame.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create video writer")
                    writer.write(draw(frame, result))
                count += 1
    except BaseException as exc:
        failure = type(exc).__name__
        raise
    finally:
        if iterator is not None:
            iterator.close()
        stop.set()
        if producer is not None:
            producer.join(timeout=7)
            if producer.is_alive():
                failure = failure or "CaptureThreadStopTimeout"
        if writer is not None:
            writer.release()
        try:
            events.close()
        except RuntimeError:
            failure = failure or "EventWriterFailure"
            raise
        finally:
            elapsed = time.monotonic() - start
            summary = {
                "acceptance": "EXPERIMENTAL_NOT_PRODUCTION_ACCEPTED",
                "processed_frames": count,
                "events": event_count,
                "elapsed_seconds": elapsed,
                "processing_fps_wall": count / max(elapsed, 0.001),
                "mean_model_pipeline_ms": float(np.mean(timings)) if timings else None,
                "p95_model_pipeline_ms": float(np.percentile(timings, 95)) if timings else None,
                "p95_post_decode_to_result_ms": float(np.percentile(ages, 95)) if ages else None,
                "dropped_unread_frames": slot.dropped,
                "error": failure,
                "frame_age_note": "Post-decode only; excludes camera/network/decoder buffering",
            }
            (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 1 if failure or not count else 0


if __name__ == "__main__":
    raise SystemExit(main())
