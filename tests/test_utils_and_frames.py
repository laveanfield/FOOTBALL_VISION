"""Tests for utility helpers and per-frame preprocessing functions."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from football_vision.config import PreprocessConfig
from football_vision.preprocessing.frames import (
    convert_color,
    denoise_frame,
    equalize_lighting,
    preprocess_frame,
    resize_frame,
)
from football_vision.utils import LOGGER_NAME, ensure_dir, set_seed, setup_logging


def test_setup_logging_with_and_without_file(tmp_path: Path) -> None:
    """Configures console and optional file handlers idempotently."""
    logger = setup_logging("DEBUG")
    assert logger.name == LOGGER_NAME
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert len(logger.handlers) == 1

    log_path = tmp_path / "nested" / "run.log"
    logger = setup_logging("INFO", log_file=log_path)
    assert len(logger.handlers) == 2
    logger.info("hello log")
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()
    assert log_path.is_file()
    assert "hello log" in log_path.read_text(encoding="utf-8")


def test_set_seed_with_cuda_mock_and_missing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeds numpy/random/torch including CUDA path and missing-torch fallback."""
    cuda_seeds: list[int] = []
    cpu_seeds: list[int] = []
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda s: cuda_seeds.append(s),
    )
    fake_torch.manual_seed = lambda s: cpu_seeds.append(s)  # type: ignore[attr-defined]
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    set_seed(77)
    assert cpu_seeds == [77]
    assert cuda_seeds == [77]

    monkeypatch.setitem(sys.modules, "torch", None)
    set_seed(88)


def test_ensure_dir_creates_nested_path(tmp_path: Path) -> None:
    """Creates parent directories and returns Path instance."""
    target = ensure_dir(tmp_path / "a" / "b")
    assert isinstance(target, Path) and target.is_dir()


def test_resize_convert_denoise_clahe_and_preprocess_frame() -> None:
    """Covers resize, color conversion, denoise, CLAHE, and error branches."""
    rng = np.random.default_rng(3)
    img = rng.integers(20, 220, size=(60, 80, 3), dtype=np.uint8)

    same = resize_frame(img, (80, 60))
    assert same is img
    smaller = resize_frame(img, (40, 30))
    assert smaller.shape == (30, 40, 3)
    larger = resize_frame(img, (160, 120))
    assert larger.shape == (120, 160, 3)

    assert convert_color(img, "rgb").shape == (60, 80, 3)
    assert convert_color(img, "gray").shape == (60, 80)
    assert convert_color(img, "hsv").shape == (60, 80, 3)
    assert convert_color(img, "lab").shape == (60, 80, 3)
    with pytest.raises(ValueError, match="Unsupported color space"):
        convert_color(img, "bgr")  # type: ignore[arg-type]

    assert denoise_frame(img, "none") is img
    assert denoise_frame(img, "gaussian").shape == img.shape
    assert denoise_frame(img, "bilateral").shape == img.shape
    with pytest.raises(ValueError, match="Unsupported denoise method"):
        denoise_frame(img, "median")  # type: ignore[arg-type]

    eq = equalize_lighting(img)
    assert eq.shape == img.shape and eq.dtype == np.uint8

    out_clahe = preprocess_frame(
        img, PreprocessConfig(denoise="gaussian", use_clahe=True)
    )
    out_plain = preprocess_frame(img, PreprocessConfig(denoise="none", use_clahe=False))
    assert out_clahe.shape == img.shape
    assert out_plain is img
