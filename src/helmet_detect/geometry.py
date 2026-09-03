"""Geometry helpers for dynamic person, vehicle and head association."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .dynamic_config import AssociationConfig, HelmetModelConfig
from .dynamic_types import ContextCrop, SceneObject
from .types import Rect


def intersection_area(first: Rect, second: Rect) -> int:
    width = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def area(rect: Rect) -> int:
    return rect.width * rect.height


def iou(first: Rect, second: Rect) -> float:
    intersection = intersection_area(first, second)
    if intersection == 0:
        return 0.0
    return intersection / float(area(first) + area(second) - intersection)


def overlap_over_smaller(first: Rect, second: Rect) -> float:
    intersection = intersection_area(first, second)
    if intersection == 0:
        return 0.0
    return intersection / float(min(area(first), area(second)))


def deduplicate_people(
    people: Iterable[SceneObject],
    *,
    maximum_people: int,
) -> list[SceneObject]:
    """Suppress nested/duplicate person boxes while preferring a complete body box."""

    candidates = sorted(people, key=lambda item: item.confidence, reverse=True)
    clusters: list[list[SceneObject]] = []
    for candidate in candidates:
        assigned = False
        for cluster in clusters:
            representative = cluster[0]
            first = representative.box
            second = candidate.box
            centre_distance = math.dist(first.centre, second.centre)
            scale = max(first.height, second.height, 1)
            top_distance = abs(first.y1 - second.y1) / scale
            same_person = (
                overlap_over_smaller(first, second) >= 0.65
                or iou(first, second) >= 0.30
                or (centre_distance <= 0.45 * scale and top_distance <= 0.20)
            )
            if same_person:
                cluster.append(candidate)
                assigned = True
                break
        if not assigned:
            clusters.append([candidate])

    selected: list[SceneObject] = []
    for cluster in clusters:
        maximum_confidence = max(item.confidence for item in cluster)
        credible = [
            item for item in cluster if item.confidence >= maximum_confidence * 0.45
        ]
        # Partial upper-body boxes often have a slightly higher confidence than the
        # complete rider. Prefer the largest credible box for stable context crops.
        chosen = max(credible, key=lambda item: (area(item.box), item.confidence))
        selected.append(chosen)

    return sorted(
        selected,
        key=lambda item: (item.confidence, area(item.box)),
        reverse=True,
    )[:maximum_people]


def _expanded(rect: Rect, ratio: float, frame_width: int, frame_height: int) -> Rect:
    dx = int(round(rect.width * ratio))
    dy = int(round(rect.height * ratio))
    expanded = Rect(rect.x1 - dx, rect.y1 - dy, rect.x2 + dx, rect.y2 + dy)
    clamped = expanded.clamp(frame_width, frame_height)
    if clamped is None:  # pragma: no cover - the original rectangle is valid
        return rect
    return clamped


def _point_to_rect_distance(x: float, y: float, rect: Rect) -> float:
    dx = max(rect.x1 - x, 0.0, x - rect.x2)
    dy = max(rect.y1 - y, 0.0, y - rect.y2)
    return math.hypot(dx, dy)


def match_vehicle(
    person: SceneObject,
    vehicles: Iterable[SceneObject],
    *,
    config: AssociationConfig,
    frame_width: int,
    frame_height: int,
) -> SceneObject | None:
    """Match a vehicle near the person's lower body; return ``None`` when uncertain."""

    foot_x = person.box.centre[0]
    foot_y = person.box.y2 - 0.05 * person.box.height
    best: tuple[float, SceneObject] | None = None
    for vehicle in vehicles:
        expanded = _expanded(
            vehicle.box,
            config.vehicle_box_expansion,
            frame_width,
            frame_height,
        )
        distance = _point_to_rect_distance(foot_x, foot_y, expanded)
        normalised = distance / max(person.box.height, 1)
        intersects = intersection_area(person.box, expanded) > 0
        if not intersects and normalised > config.maximum_foot_distance_ratio:
            continue
        ranking = normalised - 0.20 * vehicle.confidence
        if best is None or ranking < best[0]:
            best = (ranking, vehicle)
    return best[1] if best is not None else None


def _square_around(
    centre_x: float,
    centre_y: float,
    side: int,
    frame_width: int,
    frame_height: int,
) -> Rect:
    side = max(1, min(side, frame_width, frame_height))
    x1 = int(round(centre_x - side / 2.0))
    y1 = int(round(centre_y - side / 2.0))
    x1 = max(0, min(frame_width - side, x1))
    y1 = max(0, min(frame_height - side, y1))
    return Rect(x1, y1, x1 + side, y1 + side)


def build_context_crops(
    person_index: int,
    person: SceneObject,
    *,
    config: HelmetModelConfig,
    frame_width: int,
    frame_height: int,
) -> list[ContextCrop]:
    """Create multi-scale, position-adaptive square crops centred near the head."""

    short_side = min(frame_width, frame_height)
    long_side = max(frame_width, frame_height)
    raw_sizes: list[tuple[float, str]] = [
        (person.box.height * multiplier, f"person_x{multiplier:g}")
        for multiplier in config.context_height_multipliers
    ]
    raw_sizes.extend(
        (short_side * scale, f"frame_x{scale:g}")
        for scale in config.context_frame_scales
    )

    maximum_side = max(
        config.minimum_context_size,
        int(round(long_side * config.maximum_context_frame_ratio)),
    )
    chosen_sizes: list[tuple[int, str]] = []
    for raw_size, source in raw_sizes:
        side = int(
            round(
                max(
                    config.minimum_context_size,
                    min(maximum_side, raw_size),
                )
            )
        )
        if any(
            abs(side - previous) <= previous * config.context_deduplication_ratio
            for previous, _ in chosen_sizes
        ):
            continue
        chosen_sizes.append((side, source))
        if len(chosen_sizes) >= config.maximum_contexts_per_person:
            break

    centre_x = person.box.centre[0]
    centre_y = person.box.y1 + person.box.height * config.context_center_y_fraction
    return [
        ContextCrop(
            person_index=person_index,
            box=_square_around(
                centre_x,
                centre_y,
                side,
                frame_width,
                frame_height,
            ),
            scale_source=source,
        )
        for side, source in chosen_sizes
    ]


def build_head_zone(
    person: SceneObject,
    *,
    config: HelmetModelConfig,
    frame_width: int,
    frame_height: int,
) -> Rect:
    x_margin = int(round(person.box.width * config.head_zone_expand_x))
    y_above = int(round(person.box.height * config.head_zone_above))
    y_below = int(round(person.box.height * config.head_zone_below))
    raw = Rect(
        person.box.x1 - x_margin,
        person.box.y1 - y_above,
        person.box.x2 + x_margin,
        person.box.y1 + y_below,
    )
    clamped = raw.clamp(frame_width, frame_height)
    if clamped is None:  # pragma: no cover - person box is valid
        return person.box
    return clamped


def point_in_normalised_polygon(
    x: float,
    y: float,
    polygon: tuple[tuple[float, float], ...] | None,
    *,
    frame_width: int,
    frame_height: int,
) -> bool:
    """Return whether a pixel point is inside an optional normalised polygon."""

    if polygon is None:
        return True
    nx = x / max(frame_width, 1)
    ny = y / max(frame_height, 1)
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        crosses = (current_y > ny) != (previous_y > ny)
        if crosses:
            denominator = previous_y - current_y
            intersection_x = (
                (previous_x - current_x) * (ny - current_y) / denominator + current_x
            )
            if nx < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside
