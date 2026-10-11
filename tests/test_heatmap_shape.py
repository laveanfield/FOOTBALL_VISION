"""Tests for heatmaps and team shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_vision.analytics.heatmap import compute_heatmap
from football_vision.analytics.team_shape import (
    compute_team_shape,
    team_shape_timeseries,
)


def test_heatmap_shape_and_mass() -> None:
    """Checks the grid shape and that a central point keeps unit mass after blurring."""
    heat = compute_heatmap([[52.5, 34.0]], cell_m=1.0, sigma_m=2.0)
    assert heat.shape == (68, 105)
    assert heat.sum() == pytest.approx(1.0)
    assert np.unravel_index(heat.argmax(), heat.shape) in {
        (33, 51),
        (33, 52),
        (34, 51),
        (34, 52),
    }


def test_heatmap_ignores_non_finite_points_and_respects_divisor() -> None:
    """Checks NaN handling and normalization by the number of visible frames."""
    pts = [[10.0, 10.0], [np.nan, 5.0], [10.0, 10.0]]
    assert compute_heatmap(pts, sigma_m=0.0).sum() == pytest.approx(1.0)
    assert compute_heatmap(pts, sigma_m=0.0, visible_frames=4).sum() == pytest.approx(
        0.5
    )


def test_heatmap_rows_follow_y_and_columns_follow_x() -> None:
    """Checks the axis convention of the heatmap."""
    heat = compute_heatmap([[100.5, 5.5]], cell_m=1.0, sigma_m=0.0)
    assert heat[5, 100] > 0 and heat.sum() == pytest.approx(1.0)


def test_team_shape_of_a_square() -> None:
    """Checks centroid, extent, hull area and stretch of a 10 m square."""
    shape = compute_team_shape([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert (shape.centroid_x, shape.centroid_y) == (5.0, 5.0)
    assert shape.length_m == 10.0 and shape.width_m == 10.0
    assert shape.hull_area_m2 == pytest.approx(100.0)
    assert shape.stretch_index_m == pytest.approx(np.hypot(5.0, 5.0))
    assert shape.n_players == 4


def test_team_shape_degenerate_cases() -> None:
    """Checks zero hull area for collinear or too few points, and empty input."""
    assert compute_team_shape([[0, 0], [1, 1], [2, 2]]).hull_area_m2 == 0.0
    assert compute_team_shape([[3, 4]]).hull_area_m2 == 0.0
    with pytest.raises(ValueError):
        compute_team_shape(np.zeros((0, 2)))


def test_team_shape_timeseries_filters_team_goalkeeper_and_count() -> None:
    """Checks team filtering, goalkeeper exclusion and the minimum player count."""
    rows = []
    for frame_id in (0, 1):
        for track_id, (x, y) in enumerate([(0, 0), (10, 0), (10, 10), (0, 10)]):
            rows.append(
                (
                    frame_id,
                    track_id,
                    x,
                    y,
                    0,
                    "goalkeeper" if track_id == 3 else "player",
                    1,
                )
            )
    rows.append((1, 9, 50, 50, 1, "player", 1))  # other team
    table = pd.DataFrame(
        rows, columns=["frame_id", "track_id", "X", "Y", "team", "role", "valid"]
    )
    kept = team_shape_timeseries(table, team=0, exclude_goalkeeper=True, min_players=3)
    assert kept["frame_id"].tolist() == [0, 1] and kept["n_players"].tolist() == [3, 3]
    assert (
        len(
            team_shape_timeseries(table, team=0, exclude_goalkeeper=True, min_players=4)
        )
        == 0
    )
    assert team_shape_timeseries(table, team=0, exclude_goalkeeper=False)[
        "n_players"
    ].tolist() == [4, 4]
