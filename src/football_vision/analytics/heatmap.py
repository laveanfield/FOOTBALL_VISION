"""Pitch heatmaps (stage P7)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from football_vision.constants import PITCH_LENGTH_M, PITCH_WIDTH_M


def compute_heatmap(
    xy: npt.ArrayLike,
    cell_m: float = 1.0,
    sigma_m: float = 2.0,
    visible_frames: float | None = None,
    length_m: float = PITCH_LENGTH_M,
    width_m: float = PITCH_WIDTH_M,
) -> npt.NDArray[np.float64]:
    """Builds a smoothed occupancy heatmap on the pitch grid.

    The positions are binned into a 2D histogram and blurred with a Gaussian, which
    approximates a kernel density estimate. Choose ``sigma_m`` close to the position
    error: smaller values show noise, larger values hide structure.

    Normalizing by the number of *visible* frames matters: players who stay in the
    camera view longer would otherwise look "hotter" only because they were filmed more.

    Args:
        xy: Positions of shape ``(N, 2)`` as ``(X, Y)`` in meters; non-finite rows
            are ignored.
        cell_m: Size of a heatmap cell in meters.
        sigma_m: Standard deviation of the Gaussian blur in meters (0 disables it).
        visible_frames: Divisor of the counts. ``None`` divides by the number of
            samples, so the map sums to about one (probability of presence); pass
            the number of frames to get the average number of players per cell.
        length_m: Pitch length in meters.
        width_m: Pitch width in meters.

    Returns:
        Array of shape ``(rows, cols)``; rows follow ``Y`` and columns follow
        ``X``.
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    n_cols = int(round(length_m / cell_m))
    n_rows = int(round(width_m / cell_m))
    counts, _, _ = np.histogram2d(
        pts[:, 1],
        pts[:, 0],
        bins=[n_rows, n_cols],
        range=[[0.0, width_m], [0.0, length_m]],
    )
    if sigma_m > 0:
        counts = gaussian_filter(counts, sigma=sigma_m / cell_m, mode="constant")
    divisor = visible_frames if visible_frames else max(len(pts), 1)
    return counts / float(divisor)
