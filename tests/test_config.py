"""Tests for configuration loading and overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from football_vision.config import (
    PipelineConfig,
    apply_overrides,
    dump_config,
    load_config,
    parse_override_strings,
)
from football_vision.errors import ConfigError

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_default_yaml_matches_dataclass_defaults() -> None:
    """Checks that configs/default.yaml documents exactly the dataclass defaults."""
    assert load_config(CONFIG_DIR / "default.yaml") == PipelineConfig()


@pytest.mark.parametrize("name", ["walking_skeleton", "fixed_camera", "broadcast"])
def test_shipped_configs_load_on_top_of_default(name: str) -> None:
    """Checks that every shipped config can be layered on the default one."""
    cfg = load_config([CONFIG_DIR / "default.yaml", CONFIG_DIR / f"{name}.yaml"])
    assert cfg.run_name == name


def test_dotted_overrides_and_coercion() -> None:
    """Checks dotted overrides, list-to-tuple coercion and immutability of the input."""
    base = PipelineConfig()
    changed = apply_overrides(
        base, {"detection.imgsz": 960, "video.size": [640, 360], "video.path": "a.mp4"}
    )
    assert changed.detection.imgsz == 960
    assert changed.video.size == (640, 360)
    assert changed.video.path == Path("a.mp4")
    assert base.detection.imgsz == 1280


def test_unknown_key_is_rejected() -> None:
    """Checks that a misspelled key raises ConfigError."""
    with pytest.raises(ConfigError, match="Unknown config key"):
        apply_overrides(PipelineConfig(), {"detection.imgsize": 960})


def test_invalid_choice_and_type_are_rejected() -> None:
    """Checks validation of Literal choices and primitive types."""
    with pytest.raises(ConfigError):
        apply_overrides(PipelineConfig(), {"tracking.space": "sideways"})
    with pytest.raises(ConfigError):
        apply_overrides(PipelineConfig(), {"detection.imgsz": "large"})


def test_parse_override_strings_uses_yaml_types() -> None:
    """Checks that key=value strings are parsed as YAML scalars."""
    parsed = parse_override_strings(
        [
            "detection.imgsz=960",
            "detection.half=true",
            "video.end_frame=null",
            "run_name=x",
        ]
    )
    assert parsed == {
        "detection.imgsz": 960,
        "detection.half": True,
        "video.end_frame": None,
        "run_name": "x",
    }
    with pytest.raises(ConfigError):
        parse_override_strings(["missing_equals_sign"])


def test_dump_and_reload_round_trip(tmp_path: Path) -> None:
    """Checks that a dumped configuration reloads to an equal object."""
    cfg = apply_overrides(
        PipelineConfig(), {"video.path": "clip.mp4", "tracking.space": "pitch"}
    )
    assert load_config(dump_config(cfg, tmp_path / "cfg.yaml")) == cfg
