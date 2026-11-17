"""Tests for the report figures, the filmstrip and the A7 speed helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from football_vision.eval.speed import (
    per_sample_speeds,
    stationary_distance_per_minute,
)
from football_vision.viz.figures import (
    plot_heatmap_on_pitch,
    plot_homography_error,
    plot_pipeline_steps,
    plot_speed_distribution,
    plot_team_shape,
)
from football_vision.viz.render import make_filmstrip


def _pitch_xy(tracks: dict[int, np.ndarray], start: int = 0) -> pd.DataFrame:
    """Builds a ``pitch_xy`` table from ``{track_id: (T, 2) positions}``."""
    rows = []
    for track_id, xy in tracks.items():
        for i, (x, y) in enumerate(xy):
            rows.append((start + i, track_id, x, y, 0, "player", 1))
    return pd.DataFrame(
        rows, columns=["frame_id", "track_id", "X", "Y", "team", "role", "valid"]
    )


def test_pipeline_steps_has_four_panels_without_axes() -> None:
    """Checks the 2 x 2 layout and that axes are hidden."""
    frame = np.zeros((90, 160, 3), np.uint8)
    edges = np.zeros((90, 160), np.uint8)
    fig = plot_pipeline_steps(
        frame, edges, np.array([[0, 0, 100, 50.0]]), np.array([[10.0, 20.0]])
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    assert not any(ax.axison for ax in fig.axes)


def test_pipeline_steps_accepts_empty_segments_and_keypoints() -> None:
    """Checks that a frame without detected lines still renders."""
    frame = np.zeros((90, 160, 3), np.uint8)
    fig = plot_pipeline_steps(frame, frame[..., 0], np.zeros((0, 4)), np.zeros((0, 2)))
    assert len(fig.axes) == 4


def test_heatmap_figure_uses_pitch_extent() -> None:
    """Checks that the heatmap spans 105 x 68 m with Y growing downward."""
    fig = plot_heatmap_on_pitch(np.random.default_rng(0).random((68, 105)), title="t")
    ax = fig.axes[0]
    assert ax.get_images()[0].get_extent() == [0.0, 105.0, 68.0, 0.0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    assert ax.get_title() == "t"


def test_team_shape_converts_frames_to_seconds() -> None:
    """Checks the time axis and the two-panel split by unit."""
    shape = pd.DataFrame(
        {
            "frame_id": [0, 25, 50],
            "length_m": [30.0, 31.0, 32.0],
            "width_m": [40.0, 41.0, 42.0],
            "hull_area_m2": [900.0, 950.0, 1000.0],
            "stretch_index_m": [12.0, 12.5, 13.0],
        }
    )
    fig = plot_team_shape(shape, fps=25.0)
    assert len(fig.axes) == 2
    x = fig.axes[0].get_lines()[0].get_xdata()
    np.testing.assert_allclose(x, [0.0, 1.0, 2.0])
    assert fig.axes[1].get_xlabel() == "time (s)"


def test_speed_distribution_overlays_reference_sample() -> None:
    """Checks that the A7 comparison draws two histograms and the limit."""
    rng = np.random.default_rng(1)
    fig = plot_speed_distribution(
        rng.uniform(0, 8, 200), 12.0, reference_mps=rng.uniform(0, 20, 200)
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 3
    assert any("unsmoothed" in label for label in labels)


def test_speed_distribution_ignores_non_finite_values() -> None:
    """Checks that NaN speeds do not break the histogram."""
    fig = plot_speed_distribution(np.array([1.0, np.nan, 3.0, np.inf]))
    assert "n=2" in fig.axes[0].get_legend().get_texts()[0].get_text()


def test_homography_error_shades_invalid_frames() -> None:
    """Checks the valid ratio in the title and the re-anchor marker."""
    quality = pd.DataFrame(
        {
            "frame_id": np.arange(10),
            "valid": [1] * 7 + [0] * 3,
            "n_inliers": 10,
            "reproj_err_m": np.linspace(0.2, 1.0, 10),
            "overlay_score": np.linspace(0.9, 0.3, 10),
        }
    )
    fig = plot_homography_error(quality, reanchor_frames=[5])
    assert fig.axes[0].get_title() == "valid frames: 70.0%"
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert {"invalid H", "re-anchor"} <= set(labels)


def test_filmstrip_tiles_and_pads_last_row() -> None:
    """Checks the output size, the black padding tile and the labels."""
    frames = [np.full((90, 160, 3), 100 + 20 * i, np.uint8) for i in range(5)]
    strip = make_filmstrip(frames, n_cols=3, tile_width=80, labels=list("abcde"))
    assert strip.shape == (2 * 45, 3 * 80, 3)
    assert strip[45:, 160:].max() == 0  # padding tile
    assert strip[30:45, 70:80].mean() == pytest.approx(100, abs=1)


@pytest.mark.parametrize(
    ("frames", "n_cols", "tile_width"),
    [
        ([], 2, 80),
        ([np.zeros((9, 16, 3), np.uint8)], 0, 80),
        ([np.zeros((9, 16, 3), np.uint8), np.zeros((8, 16, 3), np.uint8)], 2, 80),
    ],
)
def test_filmstrip_rejects_bad_input(
    frames: list, n_cols: int, tile_width: int
) -> None:
    """Checks empty input, a non-positive column count and mixed sizes."""
    with pytest.raises(ValueError):
        make_filmstrip(frames, n_cols, tile_width)


def test_per_sample_speeds_do_not_bridge_gaps() -> None:
    """Checks that a teleport across an occlusion is not turned into a speed."""
    walk = np.stack([np.arange(10) * 0.2, np.zeros(10)], axis=1)  # 5 m/s at 25 fps
    table = _pitch_xy({1: walk})
    jump = _pitch_xy({1: walk + [50.0, 0.0]}, start=30)
    speeds = per_sample_speeds(pd.concat([table, jump]), fps=25.0, max_gap_frames=5)
    np.testing.assert_allclose(speeds, 5.0)


def test_stationary_distance_grows_with_noise() -> None:
    """Checks the A7 signal: a noisy static player 'runs', a clean one does not."""
    rng = np.random.default_rng(2)
    still = np.tile([30.0, 30.0], (250, 1))
    clean = stationary_distance_per_minute(_pitch_xy({1: still}), 25.0, 5)
    noisy = stationary_distance_per_minute(
        _pitch_xy({1: still + rng.normal(0, 0.3, still.shape)}), 25.0, 5
    )
    runner = np.stack([np.linspace(0, 50, 250), np.full(250, 30.0)], axis=1)
    assert clean == pytest.approx(0.0)
    assert noisy > 100.0
    assert np.isnan(stationary_distance_per_minute(_pitch_xy({1: runner}), 25.0, 5))
