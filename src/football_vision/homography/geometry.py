"""Homogeneous-coordinate primitives for points, lines and homographies.

A 2D point ``(x, y)`` is written ``(x, y, 1)``; a line ``a*x + b*y + c = 0`` is
written ``(a, b, c)``. In this representation the line through two points and the
intersection of two lines are both cross products, and a homography is a plain
``3 x 3`` matrix product followed by division by the third coordinate.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Homography, Points2D
from football_vision.errors import HomographyError

_EPS = 1e-12


def to_homogeneous(points: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Appends a coordinate equal to one to every point.

    Args:
        points: Array of shape ``(N, 2)``.

    Returns:
        Array of shape ``(N, 3)``.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.hstack([pts, np.ones((len(pts), 1))])


def from_homogeneous(points_h: npt.ArrayLike, eps: float = _EPS) -> Points2D:
    """Divides homogeneous points by their last coordinate.

    Args:
        points_h: Array of shape ``(N, 3)``.
        eps: Points whose last coordinate has an absolute value below ``eps``
            lie at infinity and are returned as NaN.

    Returns:
        Array of shape ``(N, 2)``.
    """
    ph = np.asarray(points_h, dtype=np.float64).reshape(-1, 3)
    w = ph[:, 2:3]
    return ph[:, :2] / np.where(np.abs(w) < eps, np.nan, w)


def apply_homography(homography: Homography, points: npt.ArrayLike) -> Points2D:
    """Maps points through a homography.

    Args:
        homography: Matrix of shape ``(3, 3)``.
        points: Array of shape ``(N, 2)``.

    Returns:
        Array of shape ``(N, 2)`` with the mapped points (NaN at infinity).
    """
    h = np.asarray(homography, dtype=np.float64)
    return from_homogeneous(to_homogeneous(points) @ h.T)


def normalize_homography(homography: Homography) -> Homography:
    """Scales a homography so that its bottom-right entry equals one.

    Homographies are defined up to scale; fixing ``H[2, 2] = 1`` makes matrices
    comparable and storable.

    Args:
        homography: Matrix of shape ``(3, 3)``.

    Returns:
        The scaled matrix.

    Raises:
        HomographyError: If the bottom-right entry is (numerically) zero.
    """
    h = np.asarray(homography, dtype=np.float64)
    if abs(h[2, 2]) < _EPS:
        raise HomographyError(
            "Cannot normalize a homography with H[2, 2] close to zero"
        )
    return h / h[2, 2]


def invert_homography(homography: Homography) -> Homography:
    """Inverts a homography and normalizes the result.

    Args:
        homography: Matrix of shape ``(3, 3)``.

    Returns:
        The inverse mapping, scaled so that ``H[2, 2] = 1``.

    Raises:
        HomographyError: If the matrix is singular.
    """
    try:
        inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise HomographyError("Singular homography cannot be inverted") from exc
    return normalize_homography(inverse)


def compose_homographies(*matrices: Homography) -> Homography:
    """Chains homographies; the right-most matrix is applied first.

    ``compose_homographies(a, b)`` maps ``x`` to ``a(b(x))``.

    Args:
        *matrices: One or more ``(3, 3)`` matrices.

    Returns:
        The normalized product.

    Raises:
        ValueError: If no matrix is given.
    """
    if not matrices:
        raise ValueError("compose_homographies needs at least one matrix")
    product = np.eye(3, dtype=np.float64)
    for matrix in matrices:
        product = product @ np.asarray(matrix, dtype=np.float64)
    return normalize_homography(product)


def line_through_points(
    p1: npt.ArrayLike, p2: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Computes the homogeneous line through two points.

    Args:
        p1: First point ``(x, y)``.
        p2: Second point ``(x, y)``.

    Returns:
        Line coefficients ``(a, b, c)`` with ``a*x + b*y + c = 0``.
    """
    a = np.append(np.asarray(p1, dtype=np.float64).reshape(2), 1.0)
    b = np.append(np.asarray(p2, dtype=np.float64).reshape(2), 1.0)
    return np.cross(a, b)


def line_from_rho_theta(rho: float, theta: float) -> npt.NDArray[np.float64]:
    """Converts a Hough line ``x*cos(theta) + y*sin(theta) = rho`` to ``(a, b, c)``.

    Args:
        rho: Signed distance of the line from the origin.
        theta: Angle of the line normal in radians.

    Returns:
        Line coefficients ``(cos(theta), sin(theta), -rho)``.
    """
    return np.array([np.cos(theta), np.sin(theta), -rho], dtype=np.float64)


def intersect_lines(
    line_a: npt.ArrayLike, line_b: npt.ArrayLike, eps: float = 1e-9
) -> tuple[float, float] | None:
    """Intersects two homogeneous lines.

    Args:
        line_a: First line ``(a, b, c)``.
        line_b: Second line ``(a, b, c)``.
        eps: Lines whose normals are closer to parallel than ``eps`` (relative to
            their norms) are treated as parallel.

    Returns:
        The intersection ``(x, y)``, or ``None`` if the lines are parallel.
    """
    la = np.asarray(line_a, dtype=np.float64).reshape(3)
    lb = np.asarray(line_b, dtype=np.float64).reshape(3)
    point = np.cross(la, lb)
    scale = np.hypot(la[0], la[1]) * np.hypot(lb[0], lb[1])
    if scale == 0 or abs(point[2]) <= eps * scale:
        return None
    return float(point[0] / point[2]), float(point[1] / point[2])


def point_line_distance(
    line: npt.ArrayLike, points: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Computes the Euclidean distance from points to a line.

    Args:
        line: Line ``(a, b, c)``.
        points: Array of shape ``(N, 2)``.

    Returns:
        Array of shape ``(N,)`` with non-negative distances.
    """
    coeffs = np.asarray(line, dtype=np.float64).reshape(3)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.abs(pts @ coeffs[:2] + coeffs[2]) / np.hypot(coeffs[0], coeffs[1])
