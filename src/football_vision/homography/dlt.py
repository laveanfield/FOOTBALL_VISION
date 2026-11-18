"""Direct Linear Transform (DLT) estimation of a homography.

Derivation (see Appendix A4 of the project plan): ``x' ~ H x`` is equivalent to
``x' x (H x) = 0``. Each correspondence yields two independent linear equations in
the nine entries of ``H``, so four non-degenerate correspondences determine ``H``.
With more points, the solution of ``min ||A h||`` subject to ``||h|| = 1`` is the
right singular vector of ``A`` with the smallest singular value.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Homography, Points2D
from football_vision.errors import HomographyError
from football_vision.homography.geometry import apply_homography, normalize_homography


def hartley_normalization(points: Points2D) -> tuple[Points2D, npt.NDArray[np.float64]]:
    """Normalizes points for a well-conditioned DLT.

    The transform translates the centroid to the origin and scales the points so
    that their mean distance from the origin is ``sqrt(2)``.

    Args:
        points: Array of shape ``(N, 2)``.

    Returns:
        Tuple ``(normalized_points, transform)`` where ``transform`` is the
        ``(3, 3)`` similarity matrix that produced the normalized points.

    Raises:
        ValueError: If ``points`` is empty or all points coincide.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("Cannot normalize an empty set of points")
    centroid = pts.mean(axis=0)  # (cx, cy): point sent to the origin
    mean_distance = np.linalg.norm(pts - centroid, axis=1).mean()
    if mean_distance < 1e-12:
        raise ValueError("Cannot normalize points that all coincide")
    scale = np.sqrt(2.0) / mean_distance  # s: makes the mean distance sqrt(2)
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return (pts - centroid) * scale, transform


def estimate_homography_dlt(
    src_points: Points2D, dst_points: Points2D, normalize: bool = True
) -> Homography:
    """Estimates the homography mapping ``src_points`` to ``dst_points`` with DLT.

    Steps: optionally normalize both point sets, stack the ``2N x 9`` matrix ``A``
    built from the two independent rows of ``x' x (H x) = 0``, take the right
    singular vector of the smallest singular value, reshape it to ``3 x 3``,
    undo the normalization (``H = T_dst^-1 @ H_hat @ T_src``) and scale so that
    ``H[2, 2] = 1``.

    Args:
        src_points: Array of shape ``(N, 2)`` with ``N >= 4`` source points.
        dst_points: Array of shape ``(N, 2)`` with the matching target points.
        normalize: Whether to apply Hartley normalization first.

    Returns:
        The ``(3, 3)`` homography with ``H[2, 2] = 1``.

    Raises:
        HomographyError: If fewer than four correspondences are given, the two
            point sets differ in length, the points are degenerate (for example
            all identical) or the solution cannot be scaled to ``H[2, 2] = 1``.
    """
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 4:
        raise HomographyError(
            f"DLT needs at least 4 matching points, got {len(src)} and {len(dst)}"
        )
    t_src = np.eye(3)  # similarity applied to the source points (identity if off)
    t_dst = np.eye(3)  # similarity applied to the target points (identity if off)
    if normalize:
        try:
            src, t_src = hartley_normalization(src)
            dst, t_dst = hartley_normalization(dst)
        except ValueError as exc:
            raise HomographyError(f"Degenerate DLT points: {exc}") from exc
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    zeros, ones = np.zeros(len(src)), np.ones(len(src))
    # Two rows per correspondence (from x' x Hx = 0), stacked into A of shape (2N, 9).
    rows_u = np.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], axis=1)
    rows_v = np.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], axis=1)
    matrix = np.concatenate([rows_u, rows_v], axis=0)
    # h = right singular vector of the smallest singular value (last row of Vt).
    h_hat = np.linalg.svd(matrix)[2][-1].reshape(3, 3)
    homography = np.linalg.inv(t_dst) @ h_hat @ t_src
    return normalize_homography(homography)


def reprojection_error(
    homography: Homography, src_points: Points2D, dst_points: Points2D
) -> npt.NDArray[np.float64]:
    """Measures how far mapped source points land from their targets.

    Args:
        homography: Matrix mapping source to target coordinates.
        src_points: Array of shape ``(N, 2)``.
        dst_points: Array of shape ``(N, 2)`` with the expected targets.

    Returns:
        Array of shape ``(N,)`` with Euclidean distances in the units of
        ``dst_points`` (meters when mapping image to pitch).
    """
    mapped = apply_homography(homography, src_points)
    return np.linalg.norm(mapped - np.asarray(dst_points, dtype=np.float64), axis=1)
