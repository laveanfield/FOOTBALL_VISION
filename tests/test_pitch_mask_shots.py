"""Tests for grass pitch mask estimation and shot boundary detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from football_vision.config import PreprocessConfig
from football_vision.preprocessing.pitch_mask import (
    compute_pitch_mask,
    estimate_grass_hsv_range,
)
from football_vision.preprocessing.shots import (
    detect_shot_boundaries,
    histogram_distance,
)


def _make_bgr_solid(
    hsv_color: tuple[int, int, int], size: tuple[int, int] = (120, 160)
) -> np.ndarray:
    """Creates a uniform BGR image from an OpenCV HSV color."""
    hsv = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    hsv[:, :] = hsv_color
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_pitch_with_stands() -> np.ndarray:
    """Creates a synthetic frame with bright stands above a green pitch."""
    frame = _make_bgr_solid((60, 180, 140), size=(160, 240))
    frame[:50, :] = (200, 180, 170)
    cv2.circle(frame, (120, 110), 4, (20, 20, 220), -1)
    return frame


def test_compute_pitch_mask_covers_grass_and_excludes_stands() -> None:
    """Checks that the pitch mask keeps grass, drops stands and fills holes."""
    frame = _make_pitch_with_stands()
    mask = compute_pitch_mask(frame, PreprocessConfig())
    assert mask.dtype == np.uint8
    assert np.all(np.isin(mask, [0, 255]))
    assert mask[60:, :].mean() > 250.0
    assert mask[:40, :].mean() == 0.0
    assert mask[110, 120] == 255


def test_compute_pitch_mask_uses_adaptive_fallback_for_yellow_grass() -> None:
    """Checks that adaptive HSV estimation recovers dry yellow-green grass."""
    frame = _make_bgr_solid((26, 170, 150), size=(120, 160))
    frame[:30, :] = (190, 190, 190)
    mask = compute_pitch_mask(frame, PreprocessConfig())
    assert mask[40:, :].mean() > 240.0
    assert mask[:25, :].mean() == 0.0


def test_compute_pitch_mask_handles_dark_frame_and_estimate_range() -> None:
    """Checks fallback on an all-black frame and adaptive hue estimation."""
    black = np.zeros((80, 100, 3), dtype=np.uint8)
    assert not compute_pitch_mask(black, PreprocessConfig()).any()
    low, high = estimate_grass_hsv_range(black)
    assert low == (35, 40, 40) and high == (85, 255, 255)
    yellow = _make_bgr_solid((26, 160, 150))
    low_y, high_y = estimate_grass_hsv_range(yellow)
    assert low_y[0] <= 26 <= high_y[0]


def test_histogram_distance_properties() -> None:
    """Checks symmetry, [0, 1] bounds, spatial invariance and bin configuration."""
    rng = np.random.default_rng(1)
    frame_a = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    frame_b = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_b[:, :32] = (30, 180, 40)
    frame_b[:, 32:] = (200, 40, 20)
    d_ab = histogram_distance(frame_a, frame_b)
    d_ba = histogram_distance(frame_b, frame_a)
    assert d_ab == pytest.approx(d_ba, abs=1e-12)
    assert 0.0 <= d_ab <= 1.0
    flipped = np.flip(frame_b, axis=1)
    resized = cv2.resize(frame_b, (128, 128), interpolation=cv2.INTER_NEAREST)
    assert histogram_distance(frame_b, flipped) == pytest.approx(0.0, abs=1e-9)
    assert histogram_distance(frame_b, resized) == pytest.approx(0.0, abs=1e-9)
    assert histogram_distance(frame_a, frame_b, bins=(4, 4, 4)) >= 0.0


def test_detect_shot_boundaries_handles_empty_single_and_generators() -> None:
    """Checks empty, one-frame, generator and sparse-frame-id sequences."""
    assert detect_shot_boundaries([], threshold=0.3) == []
    green = _make_bgr_solid((60, 180, 140), size=(40, 40))
    red = _make_bgr_solid((0, 200, 200), size=(40, 40))
    assert detect_shot_boundaries([(5, green)], threshold=0.3) == [(5, 5)]
    gen = ((fid, green if i < 3 else red) for i, fid in enumerate([10, 12, 14, 20, 22]))
    assert detect_shot_boundaries(gen, threshold=0.3) == [(10, 14), (20, 22)]


def test_detect_shot_boundaries_ignores_pan_drops_flash_and_keeps_short_cut() -> None:
    """Checks panning stability, flash removal and short real-cut preservation."""
    wide = _make_pitch_with_stands()
    flash = np.full_like(wide, 255)
    close_up = _make_bgr_solid((110, 200, 180), size=wide.shape[:2])

    pan_seq = [(i, np.roll(wide, shift=i * 3, axis=1)) for i in range(10)]
    assert detect_shot_boundaries(pan_seq, threshold=0.3) == [(0, 9)]

    flash_seq = [(i, flash if i == 6 else wide) for i in range(15)]
    assert detect_shot_boundaries(flash_seq, threshold=0.3) == [(0, 14)]

    short_insert_seq = [
        (i, wide if i < 12 else (flash if i < 15 else close_up)) for i in range(27)
    ]
    assert detect_shot_boundaries(short_insert_seq, threshold=0.3) == [
        (0, 11),
        (12, 14),
        (15, 26),
    ]
