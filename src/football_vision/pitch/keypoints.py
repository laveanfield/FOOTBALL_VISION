"""Pitch keypoints from line intersections and ellipse fitting (stage P2)."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Points2D
from football_vision.homography.geometry import intersect_lines, line_through_points

_MIN_ELLIPSE_POINTS = 5


def find_line_intersections(
    lines_a: npt.NDArray[np.float64],
    lines_b: npt.NDArray[np.float64],
    frame_size: tuple[int, int],
) -> Points2D:
    """Intersects every line of one family with every line of the other.

    Each segment is extended to an infinite line (cross product of two homogeneous
    points, see ``homography.geometry.line_through_points``) and intersected with
    ``homography.geometry.intersect_lines``. Intersections outside the frame are
    discarded.

    Args:
        lines_a: Segments ``(N, 4)`` of the first family.
        lines_b: Segments ``(M, 4)`` of the second family.
        frame_size: ``(width, height)`` of the image, used to keep visible points.

    Returns:
        Array of shape ``(K, 2)`` with candidate keypoints in pixels. It has shape
        ``(0, 2)`` when no pair intersects inside the frame.
    """
    # TODO(A): double loop (or broadcasting) over pairs, skip parallel pairs (None).
    width, height = frame_size
    segments_a = np.asarray(lines_a, dtype=np.float64).reshape(-1, 4)
    segments_b = np.asarray(lines_b, dtype=np.float64).reshape(-1, 4)
    homogeneous_b = [line_through_points(s[:2], s[2:]) for s in segments_b]
    points: list[tuple[float, float]] = []
    for segment in segments_a:
        line_a = line_through_points(segment[:2], segment[2:])
        for line_b in homogeneous_b:
            crossing = intersect_lines(line_a, line_b)
            if crossing is None:
                continue
            x, y = crossing
            if 0.0 <= x < width and 0.0 <= y < height:
                points.append(crossing)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def fit_ellipse_to_edges(
    edge_points: Points2D,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Fits an ellipse to the edge pixels of the center circle.

    Under perspective the center circle is imaged as an ellipse, so Hough circles do
    not apply. The ellipse center and axes give extra correspondences for the
    homography. This step is optional.

    The caller selects the connected edge that belongs to the center circle (for
    example the component nearest to the pitch center); all given points are fitted.

    Args:
        edge_points: Array of shape ``(N, 2)`` with pixels of one connected edge.

    Returns:
        Tuple ``(center, axes, angle_deg)`` as returned by ``cv2.fitEllipse``.

    Raises:
        ValueError: If fewer than five points are given.
    """
    # TODO(A): select the connected component near the pitch center, then
    #   call cv2.fitEllipse.
    points = np.asarray(edge_points, dtype=np.float32).reshape(-1, 2)
    if len(points) < _MIN_ELLIPSE_POINTS:
        raise ValueError(
            f"fit_ellipse_to_edges needs at least {_MIN_ELLIPSE_POINTS} points, "
            f"got {len(points)}"
        )
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(points.reshape(-1, 1, 2))
    return (float(cx), float(cy)), (float(axis_a), float(axis_b)), float(angle)
