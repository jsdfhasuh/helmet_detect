"""Build a compact video from ordered real camera frames for regression tests."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def build_video_from_images(
    image_paths: Iterable[str | Path],
    output: str | Path,
    *,
    fps: float = 6.0,
    repeat_each: int = 1,
    codec: str | None = None,
) -> Path:
    """Create a video from ordered image paths and return the output path.

    AVI outputs default to lossless FFV1 so real-model regression is not changed
    by a second lossy encode. Other extensions default to ``mp4v``.
    """

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("OpenCV is required to build the sample video") from exc

    paths = [Path(item) for item in image_paths]
    if not paths:
        raise ValueError("At least one image is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if repeat_each <= 0:
        raise ValueError("repeat_each must be positive")

    first = cv2.imread(str(paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read sample image: {paths[0]}")
    height, width = first.shape[:2]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_codec = codec or ("FFV1" if output_path.suffix.lower() == ".avi" else "mp4v")
    if len(selected_codec) != 4:
        raise ValueError("codec must be a four-character code")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*selected_codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create sample video: {output_path}")

    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Cannot read sample image: {path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"All sample frames must be {width}x{height}; "
                    f"{path} is {frame.shape[1]}x{frame.shape[0]}"
                )
            for _ in range(repeat_each):
                writer.write(frame)
    finally:
        writer.release()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Sample video was not created: {output_path}")
    return output_path
