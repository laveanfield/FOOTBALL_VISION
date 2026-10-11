"""Pitch keypoints from line intersections and ellipse fitting (stage P2)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Points2D


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
        Array of shape ``(K, 2)`` with candidate keypoints in pixels.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): double loop (or broadcasting) over pairs, skip parallel pairs (None).
    raise NotImplementedError("TODO(A): implement find_line_intersections")


def fit_ellipse_to_edges(
    edge_points: Points2D,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Fits an ellipse to the edge pixels of the center circle.

    Under perspective the center circle is imaged as an ellipse, so Hough circles do
    not apply. The ellipse center and axes give extra correspondences for the
    homography. This step is optional.

    Args:
        edge_points: Array of shape ``(N, 2)`` with pixels of one connected edge.

    Returns:
        Tuple ``(center, axes, angle_deg)`` as returned by ``cv2.fitEllipse``.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): select the connected component near the pitch center, then
    #   call cv2.fitEllipse.
    raise NotImplementedError("TODO(A): implement fit_ellipse_to_edges")
