"""Configuration for the dynamic, per-person helmet detection pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _resolve_path(raw: object, base_dir: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else (base_dir / path).resolve()


def _probability(value: object, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {number}")
    return number


def _positive(value: object, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _device(value: object | None) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    return text


@dataclass(frozen=True, slots=True)
class SceneModelConfig:
    """Full-frame person and vehicle detector settings."""

    path: Path
    image_size: int = 1280
    confidence: float = 0.05
    iou: float = 0.45
    device: str | int | None = None
    tracker: str = "bytetrack.yaml"
    person_class_id: int = 0
    vehicle_class_ids: tuple[int, ...] = (1, 3)
    minimum_person_confidence: float = 0.08
    minimum_person_height: int = 55
    maximum_persons: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path) -> SceneModelConfig:
        image_size = int(data.get("image_size", data.get("imgsz", 1280)))
        minimum_person_height = int(data.get("minimum_person_height", 55))
        maximum_persons = int(data.get("maximum_persons", 8))
        if image_size <= 0:
            raise ValueError("scene_model.image_size must be positive")
        if minimum_person_height <= 0:
            raise ValueError("scene_model.minimum_person_height must be positive")
        if maximum_persons <= 0:
            raise ValueError("scene_model.maximum_persons must be positive")
        vehicle_ids = tuple(int(item) for item in data.get("vehicle_class_ids", [1, 3]))
        return cls(
            path=_resolve_path(data["path"], base_dir),
            image_size=image_size,
            confidence=_probability(data.get("confidence", 0.05), "scene_model.confidence"),
            iou=_probability(data.get("iou", 0.45), "scene_model.iou"),
            device=_device(data.get("device")),
            tracker=str(data.get("tracker", "bytetrack.yaml")),
            person_class_id=int(data.get("person_class_id", 0)),
            vehicle_class_ids=vehicle_ids,
            minimum_person_confidence=_probability(
                data.get("minimum_person_confidence", 0.08),
                "scene_model.minimum_person_confidence",
            ),
            minimum_person_height=minimum_person_height,
            maximum_persons=maximum_persons,
        )

    @property
    def inference_class_ids(self) -> tuple[int, ...]:
        return (self.person_class_id, *self.vehicle_class_ids)


@dataclass(frozen=True, slots=True)
class HelmetModelConfig:
    """Per-person head-state detector and dynamic context settings."""

    path: Path
    image_size: int = 960
    candidate_threshold: float = 0.005
    iou: float = 0.45
    device: str | int | None = None
    helmet_class_id: int = 0
    no_helmet_class_id: int = 1
    no_helmet_threshold: float = 0.20
    helmet_threshold: float = 0.35
    no_helmet_to_helmet_ratio: float = 0.80
    context_height_multipliers: tuple[float, ...] = (3.1, 3.7)
    context_frame_scales: tuple[float, ...] = (0.45, 0.52)
    maximum_contexts_per_person: int = 4
    context_center_y_fraction: float = 0.10
    context_deduplication_ratio: float = 0.08
    minimum_context_size: int = 160
    maximum_context_frame_ratio: float = 0.80
    head_zone_expand_x: float = 0.60
    head_zone_above: float = 0.40
    head_zone_below: float = 0.55

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path) -> HelmetModelConfig:
        image_size = int(data.get("image_size", data.get("imgsz", 960)))
        maximum_contexts = int(data.get("maximum_contexts_per_person", 4))
        minimum_context_size = int(data.get("minimum_context_size", 160))
        if image_size <= 0:
            raise ValueError("helmet_model.image_size must be positive")
        if maximum_contexts <= 0:
            raise ValueError("helmet_model.maximum_contexts_per_person must be positive")
        if minimum_context_size <= 0:
            raise ValueError("helmet_model.minimum_context_size must be positive")
        context_multipliers = tuple(
            _positive(item, "helmet_model.context_height_multipliers")
            for item in data.get("context_height_multipliers", [3.1, 3.7])
        )
        frame_scales = tuple(
            _positive(item, "helmet_model.context_frame_scales")
            for item in data.get("context_frame_scales", [0.45, 0.52])
        )
        return cls(
            path=_resolve_path(data["path"], base_dir),
            image_size=image_size,
            candidate_threshold=_probability(
                data.get("candidate_threshold", 0.005),
                "helmet_model.candidate_threshold",
            ),
            iou=_probability(data.get("iou", 0.45), "helmet_model.iou"),
            device=_device(data.get("device")),
            helmet_class_id=int(data.get("helmet_class_id", 0)),
            no_helmet_class_id=int(data.get("no_helmet_class_id", 1)),
            no_helmet_threshold=_probability(
                data.get("no_helmet_threshold", 0.20),
                "helmet_model.no_helmet_threshold",
            ),
            helmet_threshold=_probability(
                data.get("helmet_threshold", 0.35),
                "helmet_model.helmet_threshold",
            ),
            no_helmet_to_helmet_ratio=_positive(
                data.get("no_helmet_to_helmet_ratio", 0.80),
                "helmet_model.no_helmet_to_helmet_ratio",
            ),
            context_height_multipliers=context_multipliers,
            context_frame_scales=frame_scales,
            maximum_contexts_per_person=maximum_contexts,
            context_center_y_fraction=float(data.get("context_center_y_fraction", 0.10)),
            context_deduplication_ratio=_probability(
                data.get("context_deduplication_ratio", 0.08),
                "helmet_model.context_deduplication_ratio",
            ),
            minimum_context_size=minimum_context_size,
            maximum_context_frame_ratio=_probability(
                data.get("maximum_context_frame_ratio", 0.80),
                "helmet_model.maximum_context_frame_ratio",
            ),
            head_zone_expand_x=_positive(
                data.get("head_zone_expand_x", 0.60),
                "helmet_model.head_zone_expand_x",
            ),
            head_zone_above=_positive(
                data.get("head_zone_above", 0.40),
                "helmet_model.head_zone_above",
            ),
            head_zone_below=_positive(
                data.get("head_zone_below", 0.55),
                "helmet_model.head_zone_below",
            ),
        )


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    """Rules for marking a tracked person as a likely rider."""

    require_vehicle: bool = False
    vehicle_box_expansion: float = 0.25
    maximum_foot_distance_ratio: float = 1.60

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AssociationConfig:
        data = data or {}
        return cls(
            require_vehicle=bool(data.get("require_vehicle", False)),
            vehicle_box_expansion=float(data.get("vehicle_box_expansion", 0.25)),
            maximum_foot_distance_ratio=_positive(
                data.get("maximum_foot_distance_ratio", 1.60),
                "association.maximum_foot_distance_ratio",
            ),
        )


@dataclass(frozen=True, slots=True)
class TrackTemporalConfig:
    """Per-track temporal voting and event de-duplication."""

    window: int = 8
    minimum_no_helmet_hits: int = 3
    minimum_valid_observations: int = 3
    cooldown_seconds: float = 8.0
    maximum_track_age_seconds: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrackTemporalConfig:
        data = data or {}
        result = cls(
            window=int(data.get("window", 8)),
            minimum_no_helmet_hits=int(data.get("minimum_no_helmet_hits", 3)),
            minimum_valid_observations=int(data.get("minimum_valid_observations", 3)),
            cooldown_seconds=float(data.get("cooldown_seconds", 8.0)),
            maximum_track_age_seconds=float(data.get("maximum_track_age_seconds", 3.0)),
        )
        if result.window <= 0:
            raise ValueError("temporal.window must be positive")
        if not 1 <= result.minimum_no_helmet_hits <= result.window:
            raise ValueError(
                "temporal.minimum_no_helmet_hits must be between 1 and temporal.window"
            )
        if not 1 <= result.minimum_valid_observations <= result.window:
            raise ValueError(
                "temporal.minimum_valid_observations must be between 1 and temporal.window"
            )
        if result.cooldown_seconds < 0:
            raise ValueError("temporal.cooldown_seconds cannot be negative")
        if result.maximum_track_age_seconds <= 0:
            raise ValueError("temporal.maximum_track_age_seconds must be positive")
        return result


@dataclass(frozen=True, slots=True)
class DynamicDetectorConfig:
    """Top-level dynamic V2 configuration."""

    scene_model: SceneModelConfig
    helmet_model: HelmetModelConfig
    association: AssociationConfig = AssociationConfig()
    temporal: TrackTemporalConfig = TrackTemporalConfig()
    alarm_zone: tuple[tuple[float, float], ...] | None = None
    draw_scene_objects: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path) -> DynamicDetectorConfig:
        pipeline = str(data.get("pipeline", "dynamic_v2"))
        if pipeline not in {"dynamic", "dynamic_v2"}:
            raise ValueError(f"Expected pipeline='dynamic_v2', got {pipeline!r}")
        raw_zone = data.get("alarm_zone")
        zone: tuple[tuple[float, float], ...] | None = None
        if raw_zone is not None:
            zone = tuple((float(point[0]), float(point[1])) for point in raw_zone)
            if len(zone) < 3:
                raise ValueError("alarm_zone must contain at least three normalized points")
            for x, y in zone:
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError("alarm_zone coordinates must be normalized to [0, 1]")
        return cls(
            scene_model=SceneModelConfig.from_dict(data["scene_model"], base_dir),
            helmet_model=HelmetModelConfig.from_dict(data["helmet_model"], base_dir),
            association=AssociationConfig.from_dict(data.get("association")),
            temporal=TrackTemporalConfig.from_dict(data.get("temporal")),
            alarm_zone=zone,
            draw_scene_objects=bool(data.get("draw_scene_objects", True)),
        )


def load_dynamic_config(path: str | Path) -> DynamicDetectorConfig:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return DynamicDetectorConfig.from_dict(data, config_path.parent)


def config_pipeline(path: str | Path) -> str:
    """Return the configured pipeline kind without fully parsing the file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(data.get("pipeline", "legacy"))
