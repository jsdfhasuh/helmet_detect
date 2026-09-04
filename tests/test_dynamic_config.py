from __future__ import annotations

import json
from pathlib import Path

import pytest

from helmet_detect.dynamic_config import config_pipeline, load_dynamic_config


def _config() -> dict[str, object]:
    return {
        "pipeline": "dynamic_v2",
        "scene_model": {"path": "scene.pt"},
        "helmet_model": {"path": "helmet.pt"},
    }


def test_load_dynamic_config_resolves_model_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    config = load_dynamic_config(path)

    assert config.scene_model.path == (tmp_path / "scene.pt").resolve()
    assert config.helmet_model.path == (tmp_path / "helmet.pt").resolve()
    assert config.scene_model.inference_class_ids == (0, 1, 3)
    assert config_pipeline(path) == "dynamic_v2"


def test_dynamic_config_rejects_absolute_pixel_alarm_zone(tmp_path: Path) -> None:
    data = _config()
    data["alarm_zone"] = [[100, 100], [200, 100], [200, 200]]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="normalized"):
        load_dynamic_config(path)


def test_dynamic_config_reads_rider_identity_and_dual_threshold(tmp_path: Path) -> None:
    data = _config()
    data["association"] = {
        "mode": "riders_only",
        "vehicle_evidence_window": 6,
        "minimum_vehicle_hits": 2,
        "vehicle_grace_seconds": 2.5,
    }
    data["identity"] = {
        "enabled": True,
        "maximum_gap_seconds": 3.0,
    }
    data["helmet_model"] = {
        "path": "helmet.pt",
        "no_helmet_threshold": 0.2,
        "high_confidence_no_helmet_threshold": 0.5,
    }
    data["temporal"] = {
        "window": 8,
        "minimum_no_helmet_hits": 3,
        "minimum_valid_observations": 3,
        "high_confidence_minimum_hits": 1,
        "helmet_rearm_hits": 3,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    config = load_dynamic_config(path)

    assert config.association.riders_only
    assert config.association.minimum_vehicle_hits == 2
    assert config.identity.maximum_gap_seconds == 3.0
    assert config.helmet_model.high_confidence_no_helmet_threshold == 0.5
    assert config.temporal.high_confidence_minimum_hits == 1


def test_high_confidence_threshold_cannot_be_below_normal_threshold(
    tmp_path: Path,
) -> None:
    data = _config()
    data["helmet_model"] = {
        "path": "helmet.pt",
        "no_helmet_threshold": 0.4,
        "high_confidence_no_helmet_threshold": 0.3,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="high_confidence"):
        load_dynamic_config(path)
