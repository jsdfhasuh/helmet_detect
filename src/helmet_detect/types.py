"""Small immutable data types used by the detector."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle using half-open image coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Invalid rectangle: {self}")

    @classmethod
    def from_iterable(cls, values: Iterable[int]) -> Rect:
        items = tuple(int(value) for value in values)
        if len(items) != 4:
            raise ValueError("Rectangle must contain exactly four integers: x1,y1,x2,y2")
        return cls(*items)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2

    def contains_centre(self, other: Rect) -> bool:
        return self.contains_point(*other.centre)

    def translate(self, dx: int, dy: int) -> Rect:
        return Rect(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def clamp(self, width: int, height: int) -> Rect | None:
        x1 = max(0, min(width, self.x1))
        y1 = max(0, min(height, self.y1))
        x2 = max(0, min(width, self.x2))
        y2 = max(0, min(height, self.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return Rect(x1, y1, x2, y2)

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True, slots=True)
class Detection:
    model_name: str
    class_id: int
    class_name: str
    confidence: float
    box: Rect

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "box": self.box.to_list(),
        }
