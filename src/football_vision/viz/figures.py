"""Matplotlib figures for the report (stage P8).

Every function returns a :class:`matplotlib.figure.Figure` so the caller decides
whether to show or save it (``fig.savefig("reports/figures/name.png", dpi=200)``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

from football_vision.datatypes import Frame, Mask, Points2D
from football_vision.pitch.template import PitchTemplate


def plot_pipeline_steps(
    frame: Frame,
    edges: Mask,
    segments: npt.NDArray[np.float64],
    keypoints: Points2D,
) -> Figure:
    """Shows the geometric pipeline in a 2 x 2 grid.

    Panels: original frame, edge map, Hough segments over the frame, and keypoints.

    Args:
        frame: BGR frame.
        edges: Binary edge map.
        segments: Merged segments of shape ``(N, 4)``.
        keypoints: Candidate keypoints of shape ``(K, 2)``.

    Returns:
        The figure.

    Raises:
        NotImplementedError: Stub; to be implemented by member D.
    """
    # TODO(D): convert BGR to RGB before imshow; hide axes; add panel titles.
    raise NotImplementedError("TODO(D): implement plot_pipeline_steps")


def plot_heatmap_on_pitch(
    heatmap: npt.NDArray[np.float64],
    template: PitchTemplate | None = None,
    title: str = "",
) -> Figure:
    """Draws a heatmap with the pitch markings on top.

    Args:
        heatmap: Array of shape ``(rows, cols)`` from ``analytics.heatmap``.
        template: Pitch model for the markings; defaults to the 105 x 68 m template.
        title: Figure title.

    Returns:
        The figure.

    Raises:
        NotImplementedError: Stub; to be implemented by member D.
    """
    # TODO(D): imshow with extent=[0, length, width, 0]; overlay template.polylines().
    raise NotImplementedError("TODO(D): implement plot_heatmap_on_pitch")


def plot_team_shape(shape_series: pd.DataFrame, title: str = "") -> Figure:
    """Plots team length, width and hull area over time.

    Args:
        shape_series: Output of ``analytics.team_shape.team_shape_timeseries``.
        title: Figure title.

    Returns:
        The figure.

    Raises:
        NotImplementedError: Stub; to be implemented by member D.
    """
    # TODO(D): x axis in seconds (frame_id / fps); one line per metric.
    raise NotImplementedError("TODO(D): implement plot_team_shape")


def plot_speed_distribution(
    speeds_mps: npt.NDArray[np.float64], max_speed_mps: float = 12.0
) -> Figure:
    """Plots the histogram of player speeds and marks the plausibility limit.

    Args:
        speeds_mps: Speeds in meters per second.
        max_speed_mps: Speeds above this value are drawn as implausible.

    Returns:
        The figure.

    Raises:
        NotImplementedError: Stub; to be implemented by member D.
    """
    # TODO(D): compare smoothed and unsmoothed speeds for ablation A7.
    raise NotImplementedError("TODO(D): implement plot_speed_distribution")


def plot_homography_error(quality: pd.DataFrame) -> Figure:
    """Plots homography quality over time.

    Args:
        quality: Table with the columns of ``HOMOGRAPHY_QUALITY_COLUMNS``.

    Returns:
        The figure with the reprojection error and the line-overlay score.

    Raises:
        NotImplementedError: Stub; to be implemented by member D.
    """
    # TODO(D): shade invalid frames; mark re-anchor events.
    raise NotImplementedError("TODO(D): implement plot_homography_error")
