"""JSON configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import Rect


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    path: Path
    alarm_threshold: float
    no_helmet_class_id: int = 1
    helmet_class_id: int = 0
    input_size: int = 960

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path) -> ModelConfig:
        raw_path = Path(str(data["path"]))
        path = raw_path if raw_path.is_absolute() else (base_dir / raw_path).resolve()
        threshold = float(data["alarm_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"alarm_threshold must be in [0, 1], got {threshold}")
        return cls(
            name=str(data["name"]),
            path=path,
            alarm_threshold=threshold,
            no_helmet_class_id=int(data.get("no_helmet_class_id", 1)),
            helmet_class_id=int(data.get("helmet_class_id", 0)),
            input_size=int(data.get("input_size", 960)),
        )


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    window: int = 5
    min_hits: int = 2
    cooldown_seconds: float = 5.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TemporalConfig:
        data = data or {}
        result = cls(
            window=int(data.get("window", 5)),
            min_hits=int(data.get("min_hits", 2)),
            cooldown_seconds=float(data.get("cooldown_seconds", 5.0)),
        )
        if result.window <= 0:
            raise ValueError("temporal.window must be positive")
        if not 1 <= result.min_hits <= result.window:
            raise ValueError("temporal.min_hits must be between 1 and temporal.window")
        if result.cooldown_seconds < 0:
            raise ValueError("temporal.cooldown_seconds cannot be negative")
        return result


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    roi: Rect
    gate: Rect
    models: tuple[ModelConfig, ...]
    candidate_threshold: float = 0.02
    nms_threshold: float = 0.45
    temporal: TemporalConfig = TemporalConfig()

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path) -> DetectorConfig:
        models = tuple(ModelConfig.from_dict(item, base_dir) for item in data["models"])
        if not models:
            raise ValueError("At least one model must be configured")
        candidate_threshold = float(data.get("candidate_threshold", 0.02))
        nms_threshold = float(data.get("nms_threshold", 0.45))
        if not 0.0 <= candidate_threshold <= 1.0:
            raise ValueError("candidate_threshold must be in [0, 1]")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be in [0, 1]")
        return cls(
            roi=Rect.from_iterable(data["roi"]),
            gate=Rect.from_iterable(data["gate"]),
            models=models,
            candidate_threshold=candidate_threshold,
            nms_threshold=nms_threshold,
            temporal=TemporalConfig.from_dict(data.get("temporal")),
        )


def load_config(path: str | Path) -> DetectorConfig:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return DetectorConfig.from_dict(data, config_path.parent)
