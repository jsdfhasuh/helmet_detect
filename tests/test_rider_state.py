from __future__ import annotations

from helmet_detect.dynamic_config import AssociationConfig
from helmet_detect.rider_state import RiderEvidenceTracker


def test_rider_only_mode_uses_current_and_recent_vehicle_evidence() -> None:
    tracker = RiderEvidenceTracker(
        AssociationConfig(
            mode="riders_only",
            vehicle_evidence_window=4,
            minimum_vehicle_hits=1,
            vehicle_grace_seconds=1.0,
        ),
        maximum_age_seconds=3.0,
    )

    pedestrian = tracker.update(7, 0.0, has_vehicle_match=False)
    matched = tracker.update(7, 0.2, has_vehicle_match=True)
    grace = tracker.update(7, 0.8, has_vehicle_match=False)
    expired = tracker.update(7, 1.4, has_vehicle_match=False)

    assert not pedestrian.eligible
    assert matched.eligible
    assert grace.eligible
    assert grace.reason == "recent_vehicle_evidence"
    assert not expired.eligible


def test_all_persons_mode_never_filters_people() -> None:
    tracker = RiderEvidenceTracker(
        AssociationConfig(mode="all_persons"),
        maximum_age_seconds=3.0,
    )

    result = tracker.update(1, 0.0, has_vehicle_match=False)

    assert result.eligible
    assert result.reason == "all_persons_mode"
