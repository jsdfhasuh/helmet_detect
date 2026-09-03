from __future__ import annotations

import pytest

from helmet_detect.sample import build_video_from_images


def test_build_video_rejects_empty_input(tmp_path) -> None:
    with pytest.raises(ValueError, match="At least one image"):
        build_video_from_images([], tmp_path / "empty.mp4")


def test_build_video_rejects_invalid_settings(tmp_path) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        build_video_from_images([tmp_path / "missing.jpg"], tmp_path / "x.mp4", fps=0)
    with pytest.raises(ValueError, match="repeat_each must be positive"):
        build_video_from_images(
            [tmp_path / "missing.jpg"],
            tmp_path / "x.mp4",
            repeat_each=0,
        )
