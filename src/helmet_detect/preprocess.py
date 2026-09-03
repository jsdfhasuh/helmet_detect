"""Image preprocessing and coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    scale: float
    pad_x: int
    pad_y: int
    input_size: int
    source_width: int
    source_height: int

    def inverse_xyxy(
        self,
        cx: float,
        cy: float,
        width: float,
        height: float,
    ) -> tuple[int, int, int, int]:
        x1 = int(round((cx - width / 2.0 - self.pad_x) / self.scale))
        y1 = int(round((cy - height / 2.0 - self.pad_y) / self.scale))
        x2 = int(round((cx + width / 2.0 - self.pad_x) / self.scale))
        y2 = int(round((cy + height / 2.0 - self.pad_y) / self.scale))
        return x1, y1, x2, y2


def letterbox(image: np.ndarray, input_size: int = 960) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize while preserving aspect ratio, then pad to a square."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("OpenCV is required. Install with: pip install -e '.[runtime]'") from exc

    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Input image is empty")

    scale = min(input_size / source_width, input_size / source_height)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    delta_width = input_size - resized_width
    delta_height = input_size - resized_height
    left = int(round(delta_width / 2.0 - 0.1))
    right = int(round(delta_width / 2.0 + 0.1))
    top = int(round(delta_height / 2.0 - 0.1))
    bottom = int(round(delta_height / 2.0 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    transform = LetterboxTransform(
        scale=scale,
        pad_x=left,
        pad_y=top,
        input_size=input_size,
        source_width=source_width,
        source_height=source_height,
    )
    return padded, transform
