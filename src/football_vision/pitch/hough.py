"""Hough line detection and post-processing (stage P2).

A line is parameterized as ``x*cos(theta) + y*sin(theta) = rho``. Every edge pixel
votes for all ``(rho, theta)`` it could lie on; collinear pixels vote for the same
cell, so accumulator peaks are lines.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from football_vision.config import PitchConfig
from football_vision.datatypes import Mask


def detect_line_segments(edges: Mask, cfg: PitchConfig) -> npt.NDArray[np.float64]:
    """Detects straight segments in an edge map with the probabilistic Hough transform.

    Args:
        edges: Binary edge map of shape ``(H, W)``.
        cfg: Pitch settings with the Hough resolution and thresholds.

    Returns:
        Array of shape ``(N, 4)`` holding ``(x1, y1, x2, y2)`` per segment.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): cv2.HoughLinesP(edges, cfg.hough_rho, np.pi / 180, cfg.hough_threshold,
    #   minLineLength=cfg.hough_min_length, maxLineGap=cfg.hough_max_gap).
    raise NotImplementedError("TODO(A): implement detect_line_segments")


def merge_collinear_segments(
    segments: npt.NDArray[np.float64],
    rho_tolerance_px: float = 10.0,
    theta_tolerance_rad: float = 0.05,
) -> npt.NDArray[np.float64]:
    """Merges segments that lie on the same physical pitch line.

    Hough returns many fragments per marking. Convert each segment to ``(rho, theta)``,
    group those closer than the tolerances, and replace each group by the segment
    spanning its extreme endpoints.

    Args:
        segments: Array of shape ``(N, 4)`` with ``(x1, y1, x2, y2)``.
        rho_tolerance_px: Maximum ``rho`` difference inside a group.
        theta_tolerance_rad: Maximum ``theta`` difference inside a group.

    Returns:
        Array of shape ``(M, 4)`` with ``M <= N`` merged segments.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): handle the theta wrap-around at 0/pi; weight groups by segment length.
    raise NotImplementedError("TODO(A): implement merge_collinear_segments")


def cluster_line_families(
    segments: npt.NDArray[np.float64], n_families: int = 2
) -> npt.NDArray[np.int64]:
    """Assigns segments to families of lines that are parallel in the real world.

    Pitch lines form two families: along the pitch (touchlines) and across it (goal
    lines, halfway line, box edges). Under perspective, each family converges to a
    vanishing point, so cluster by orientation or by a RANSAC vanishing-point fit.

    Args:
        segments: Array of shape ``(N, 4)`` with ``(x1, y1, x2, y2)``.
        n_families: Number of families to find.

    Returns:
        Integer array of shape ``(N,)`` with the family label of each segment
        (``-1`` for outliers).

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): start with KMeans on the segment angle, then refine with
    #   vanishing points.
    raise NotImplementedError("TODO(A): implement cluster_line_families")
