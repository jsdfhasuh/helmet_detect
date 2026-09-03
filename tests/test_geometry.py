from __future__ import annotations

from pathlib import Path

from helmet_detect.dynamic_config import HelmetModelConfig
from helmet_detect.dynamic_types import SceneObject
from helmet_detect.geometry import build_context_crops, deduplicate_people
from helmet_detect.types import Rect


def _person(confidence: float, box: Rect, track_id: int) -> SceneObject:
    return SceneObject(0, "person", confidence, box, track_id)


def test_deduplicate_people_prefers_complete_credible_body_box() -> None:
    upper = _person(0.45, Rect(110, 100, 170, 180), 1)
    full = _person(0.25, Rect(100, 100, 180, 270), 2)

    result = deduplicate_people([upper, full], maximum_people=5)

    assert result == [full]


def test_context_crops_follow_person_position_without_fixed_roi() -> None:
    config = HelmetModelConfig(
        path=Path("helmet.pt"),
        context_height_multipliers=(3.0,),
        context_frame_scales=(),
        maximum_contexts_per_person=1,
        minimum_context_size=100,
    )
    left = _person(0.9, Rect(80, 200, 140, 360), 1)
    right = _person(0.9, Rect(680, 200, 740, 360), 2)

    left_crop = build_context_crops(
        0,
        left,
        config=config,
        frame_width=800,
        frame_height=600,
    )[0]
    right_crop = build_context_crops(
        0,
        right,
        config=config,
        frame_width=800,
        frame_height=600,
    )[0]

    assert left_crop.box.centre[0] < right_crop.box.centre[0]
    assert left_crop.box.width == right_crop.box.width
    assert left_crop.box.x1 != right_crop.box.x1
