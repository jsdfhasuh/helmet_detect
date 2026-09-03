import json
from pathlib import Path

import pytest

from helmet_detect.config import load_config


def test_load_config_resolves_model_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "roi": [0, 0, 100, 100],
                "gate": [10, 10, 90, 90],
                "models": [
                    {
                        "name": "model",
                        "path": "model.onnx",
                        "alarm_threshold": 0.2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.models[0].path == (tmp_path / "model.onnx").resolve()


def test_invalid_temporal_config_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "roi": [0, 0, 100, 100],
                "gate": [10, 10, 90, 90],
                "models": [
                    {
                        "name": "model",
                        "path": "model.onnx",
                        "alarm_threshold": 0.2,
                    }
                ],
                "temporal": {"window": 2, "min_hits": 3},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(config_path)
