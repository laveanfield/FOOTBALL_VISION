"""Tests for smoothing and kinematics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_vision.analytics.kinematics import (
    classify_speed_zones,
    compute_speed,
    distance_covered,
    flag_implausible_speed,
    summarize_track_kinematics,
)
from football_vision.analytics.smoothing import (
    smooth_pitch_tracks,
    smooth_positions,
    split_into_segments,
)
from football_vision.config import AnalyticsConfig

FPS = 25.0
VELOCITY = np.array([4.0, 1.0])  # m/s


def straight_run(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Creates frame ids and positions of a constant-velocity run."""
    frame_ids = np.arange(n)
    return frame_ids, np.outer(frame_ids / FPS, VELOCITY)


def test_speed_and_distance_of_constant_velocity() -> None:
    """Checks speed and distance of a noise-free straight run."""
    frame_ids, xy = straight_run()
    assert compute_speed(xy, frame_ids, FPS) == pytest.approx(np.hypot(4.0, 1.0))
    assert distance_covered(xy) == pytest.approx(np.hypot(4.0, 1.0) * 99 / FPS)


def test_smoothing_reduces_speed_noise() -> None:
    """Checks that smoothing before differentiation tames noisy speeds."""
    frame_ids, xy = straight_run()
    noisy = xy + np.random.default_rng(0).normal(scale=0.3, size=xy.shape)
    true_speed = np.hypot(4.0, 1.0)
    raw_error = np.abs(compute_speed(noisy, frame_ids, FPS) - true_speed).mean()
    smooth_error = np.abs(
        compute_speed(smooth_positions(noisy, 15), frame_ids, FPS) - true_speed
    ).mean()
    assert raw_error > 2.0  # noise of 0.3 m at 25 fps becomes several m/s
    assert smooth_error < raw_error / 2


def test_savgol_derivative_option() -> None:
    """Checks the built-in smoothed derivative on a noise-free run."""
    frame_ids, xy = straight_run()
    speed = compute_speed(xy, frame_ids, FPS, smooth_window=11)
    assert speed == pytest.approx(np.hypot(4.0, 1.0), rel=1e-6)


def test_speed_requires_increasing_frame_ids() -> None:
    """Checks that repeated frame ids are rejected."""
    with pytest.raises(ValueError):
        compute_speed([[0, 0], [1, 1]], [3, 3], FPS)


def test_distance_ignores_steps_across_gaps() -> None:
    """Checks that a jump over an occlusion is not counted as distance."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [51.0, 0.0]])
    assert distance_covered(xy, [0, 1, 20, 21], max_gap_frames=5) == pytest.approx(2.0)


def test_speed_zones_and_outlier_flags() -> None:
    """Checks zone indices (including edge values) and the plausibility flag."""
    speeds = np.array([1.0, 2.0, 4.0, 6.0, 8.0])  # 3.6, 7.2, 14.4, 21.6, 28.8 km/h
    assert classify_speed_zones(speeds, (7.0, 14.0, 20.0, 25.0)).tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert flag_implausible_speed([5.0, 12.5], 12.0).tolist() == [False, True]


def test_segments_split_on_gaps() -> None:
    """Checks that split_into_segments breaks at large frame gaps."""
    assert split_into_segments([0, 1, 2, 10, 11], max_gap=5) == [
        slice(0, 3),
        slice(3, 5),
    ]
    assert split_into_segments([], max_gap=5) == []


def test_short_trajectories_are_left_unchanged() -> None:
    """Checks that smoothing leaves too-short trajectories untouched."""
    xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(smooth_positions(xy, 15), xy)
    np.testing.assert_allclose(smooth_positions(xy, 0), xy)


def test_smooth_pitch_tracks_only_touches_valid_rows() -> None:
    """Checks that invalid rows keep their values and valid rows get smoothed."""
    frame_ids, xy = straight_run(40)
    noisy = xy + np.random.default_rng(1).normal(scale=0.3, size=xy.shape)
    table = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "track_id": 1,
            "X": noisy[:, 0],
            "Y": noisy[:, 1],
            "team": 0,
            "role": "player",
            "valid": 1,
        }
    )
    table.loc[5, "valid"] = 0
    smoothed = smooth_pitch_tracks(table, window=9, polyorder=2, max_gap_frames=5)
    assert smoothed.loc[5, "X"] == table.loc[5, "X"]
    assert not np.allclose(smoothed.loc[10:20, "X"], table.loc[10:20, "X"])


def test_summary_per_track() -> None:
    """Checks the per-track summary of a constant-velocity run."""
    frame_ids, xy = straight_run()
    table = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "track_id": 3,
            "X": xy[:, 0],
            "Y": xy[:, 1],
            "team": 1,
            "role": "player",
            "valid": 1,
        }
    )
    summary = summarize_track_kinematics(table, FPS, AnalyticsConfig()).iloc[0]
    assert summary["track_id"] == 3 and summary["team"] == 1
    assert summary["distance_m"] == pytest.approx(np.hypot(4.0, 1.0) * 99 / FPS)
    assert summary["max_speed_mps"] == pytest.approx(np.hypot(4.0, 1.0))
    assert summary["outlier_fraction"] == 0.0
    assert summary["zone_time_s_2"] == pytest.approx(
        99 / FPS
    )  # 14.8 km/h is in the third zone
