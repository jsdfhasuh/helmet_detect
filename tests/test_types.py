import pytest

from helmet_detect.types import Rect


def test_rect_geometry_and_translation() -> None:
    rect = Rect(10, 20, 30, 50)
    assert rect.width == 20
    assert rect.height == 30
    assert rect.centre == (20.0, 35.0)
    assert rect.translate(5, -5) == Rect(15, 15, 35, 45)


def test_rect_clamp() -> None:
    assert Rect(-5, -10, 30, 40).clamp(20, 25) == Rect(0, 0, 20, 25)
    assert Rect(30, 30, 40, 40).clamp(20, 20) is None


def test_invalid_rect_rejected() -> None:
    with pytest.raises(ValueError):
        Rect(10, 10, 10, 20)
