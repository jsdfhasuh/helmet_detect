from __future__ import annotations

import threading

import numpy as np
import pytest

from helmet_detect.dynamic_types import SceneObject
from helmet_detect.latest_frame import LatestFrameSlot
from helmet_detect.rider_head import RiderHeadConfig, RiderHeadLogic, associate_heads
from helmet_detect.types import Rect


def test_slot_replaces_old_frames_and_never_duplicates():
    slot = LatestFrameSlot()
    slot.publish("first")
    slot.publish("second")
    item = slot.take(0.01)
    assert item.image == "second"
    assert slot.dropped == 1
    assert slot.take(0.001) is None
    slot.close()
    with pytest.raises(RuntimeError):
        slot.publish("third")


def test_closed_slot_drains_final_frame():
    slot = LatestFrameSlot()
    slot.publish("last")
    slot.close()
    assert slot.take().image == "last"
    assert slot.take() is None


def test_close_wakes_waiter():
    slot = LatestFrameSlot()
    result = []
    thread = threading.Thread(target=lambda: result.append(slot.take(5)))
    thread.start()
    slot.close()
    thread.join(1)
    assert not thread.is_alive()
    assert result == [None]


def frame():
    return np.zeros((300, 400, 3), dtype=np.uint8)


def rows(head_class=2):
    return [[100, 60, 180, 230, 0.9, 0], [125, 62, 153, 95, 0.85, head_class]]


def test_two_distinct_frames_and_one_event_per_session():
    logic = RiderHeadLogic()
    assert logic.update(frame(), rows(), 0)["events"] == 0
    assert logic.update(frame(), rows(), 0.2)["events"] == 1
    assert logic.update(frame(), rows(), 0.4)["events"] == 0
    with pytest.raises(ValueError):
        logic.update(frame(), rows(), 0.4)


def test_helmet_and_missing_head_cannot_trigger():
    logic = RiderHeadLogic()
    for i in range(3):
        assert logic.update(frame(), rows(1), i * 0.2)["events"] == 0
    result = logic.update(frame(), rows()[:1], 1)
    assert result["events"] == 0
    assert result["observations"][0]["state"] == "unknown"


def test_expired_evidence_does_not_accumulate():
    logic = RiderHeadLogic(RiderHeadConfig(evidence_seconds=0.5))
    assert logic.update(frame(), rows(), 0)["events"] == 0
    assert logic.update(frame(), rows(), 1)["events"] == 0


def test_ambiguous_head_not_assigned_to_two_riders():
    riders = [
        SceneObject(0, "rider", 0.9, Rect(100, 60, 180, 230)),
        SceneObject(0, "rider", 0.9, Rect(110, 60, 190, 230)),
    ]
    heads = [SceneObject(2, "no_helmet_head", 0.9, Rect(135, 62, 163, 95))]
    matched = associate_heads(riders, heads)
    assert sum(len(v) for v in matched.values()) == 0


def test_conflicting_classes_are_unknown():
    logic = RiderHeadLogic()
    r = rows() + [[125, 62, 153, 95, 0.8, 1]]
    for i in range(3):
        result = logic.update(frame(), r, i * 0.2)
        assert result["events"] == 0
        assert result["observations"][0]["state"] == "unknown"


def test_two_spatially_different_heads_rejected():
    logic = RiderHeadLogic()
    r = rows() + [[157, 65, 175, 92, 0.8, 2]]
    assert logic.update(frame(), r, 0)["observations"][0]["state"] == "unknown"


def test_validation_rejects_nan():
    with pytest.raises(ValueError):
        RiderHeadConfig(head_threshold=float("nan"))


def test_helmet_flicker_cannot_rearm_a_live_passage():
    logic = RiderHeadLogic()
    states = [2, 2, 1, 1, 1, 2, 2, 2, 2]
    assert sum(logic.update(frame(), rows(c), i * 0.2)["events"] for i, c in enumerate(states)) == 1


def test_a_new_passage_after_absence_can_trigger():
    logic = RiderHeadLogic()
    assert logic.update(frame(), rows(), 0)["events"] == 0
    assert logic.update(frame(), rows(), 0.2)["events"] == 1
    assert logic.update(frame(), [], 3)["events"] == 0
    assert logic.update(frame(), rows(), 3.2)["events"] == 0
    assert logic.update(frame(), rows(), 3.4)["events"] == 1
