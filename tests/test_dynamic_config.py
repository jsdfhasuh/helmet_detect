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
