import numpy as np

from helmet_detect.model import normalise_output
from helmet_detect.preprocess import letterbox


def test_letterbox_preserves_shape_and_transform() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, transform = letterbox(image, 320)
    assert padded.shape == (320, 320, 3)
    assert transform.scale == 1.6
    assert transform.pad_x == 0
    assert transform.pad_y == 80
    assert transform.inverse_xyxy(160, 160, 160, 80) == (50, 25, 150, 75)


def test_normalise_output_transposes_channel_first_layout() -> None:
    raw = np.zeros((1, 6, 100), dtype=np.float32)
    assert normalise_output(raw).shape == (100, 6)


def test_normalise_output_keeps_anchor_first_layout() -> None:
    raw = np.zeros((1, 100, 6), dtype=np.float32)
    assert normalise_output(raw).shape == (100, 6)
