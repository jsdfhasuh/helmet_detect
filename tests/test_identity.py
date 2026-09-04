from __future__ import annotations

import numpy as np

from helmet_detect.dynamic_config import IdentityConfig
from helmet_detect.dynamic_types import SceneObject
from helmet_detect.identity import CrossIdIdentityResolver
from helmet_detect.types import Rect


def _person(track_id: int, box: Rect) -> SceneObject:
    return SceneObject(0, "person", 0.9, box, track_id)


def test_cross_id_resolver_stitches_nearby_same_appearance() -> None:
    frame = np.zeros((320, 640, 3), dtype=np.uint8)
    frame[80:260, 200:280] = (20, 80, 210)
    resolver = CrossIdIdentityResolver(IdentityConfig())

    first = resolver.resolve_frame(
        [_person(10, Rect(200, 80, 280, 260))],
        frame,
        0.0,
    )[0]
    second = resolver.resolve_frame(
        [_person(99, Rect(210, 82, 290, 262))],
        frame,
        0.2,
    )[0]

    assert first.canonical_track_id == 10
    assert second.canonical_track_id == 10
    assert second.source_track_id == 99
    assert second.stitched


def test_cross_id_resolver_keeps_far_people_separate() -> None:
    frame = np.full((320, 640, 3), 120, dtype=np.uint8)
    resolver = CrossIdIdentityResolver(IdentityConfig(maximum_center_distance_ratio=0.8))

    first = resolver.resolve_frame(
        [_person(10, Rect(50, 80, 130, 260))],
        frame,
        0.0,
    )[0]
    second = resolver.resolve_frame(
        [_person(99, Rect(470, 80, 550, 260))],
        frame,
        0.2,
    )[0]

    assert first.canonical_track_id == 10
    assert second.canonical_track_id == 99
    assert not second.stitched
