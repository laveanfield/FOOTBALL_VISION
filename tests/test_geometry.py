"""Tests for the homogeneous-coordinate primitives."""

from __future__ import annotations

import numpy as np
import pytest

from football_vision.errors import HomographyError
from football_vision.homography.geometry import (
    apply_homography,
    compose_homographies,
    from_homogeneous,
    intersect_lines,
    invert_homography,
    line_from_rho_theta,
    line_through_points,
    normalize_homography,
    point_line_distance,
    to_homogeneous,
)

H_EXAMPLE = np.array([[2.0, 0.1, 5.0], [0.0, 3.0, -2.0], [0.001, 0.002, 1.0]])


def test_homogeneous_round_trip() -> None:
    """Checks that converting to and from homogeneous coordinates is lossless."""
    pts = np.array([[1.0, 2.0], [-3.0, 4.5]])
    np.testing.assert_allclose(from_homogeneous(to_homogeneous(pts)), pts)


def test_apply_homography_affine_case() -> None:
    """Checks an affine mapping against a hand computation."""
    h = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(apply_homography(h, [[1.0, 1.0]]), [[3.0, 5.0]])


def test_apply_homography_divides_by_third_coordinate() -> None:
    """Checks the perspective division of a projective mapping."""
    x, y = 10.0, 20.0
    p = np.array([x, y, 1.0])
    expected = [
        (H_EXAMPLE[0] @ p) / (H_EXAMPLE[2] @ p),
        (H_EXAMPLE[1] @ p) / (H_EXAMPLE[2] @ p),
    ]
    np.testing.assert_allclose(apply_homography(H_EXAMPLE, [[x, y]])[0], expected)


def test_points_at_infinity_become_nan() -> None:
    """Checks that points mapped to infinity are returned as NaN."""
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.isnan(apply_homography(h, [[0.0, 5.0]])).all()


def test_invert_round_trip_and_normalization() -> None:
    """Checks that the inverse undoes the mapping and is normalized."""
    inverse = invert_homography(H_EXAMPLE)
    assert inverse[2, 2] == pytest.approx(1.0)
    pts = np.array([[3.0, 4.0], [100.0, 50.0]])
    np.testing.assert_allclose(
        apply_homography(inverse, apply_homography(H_EXAMPLE, pts)), pts, atol=1e-9
    )


def test_invert_singular_matrix_raises() -> None:
    """Checks that a singular matrix raises HomographyError."""
    with pytest.raises(HomographyError):
        invert_homography(np.zeros((3, 3)))


def test_normalize_requires_nonzero_corner() -> None:
    """Checks that normalization rejects a zero bottom-right entry."""
    with pytest.raises(HomographyError):
        normalize_homography(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.0]]))


def test_compose_applies_right_matrix_first() -> None:
    """Checks the composition order of compose_homographies."""
    translate = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    scale = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(
        apply_homography(compose_homographies(translate, scale), [[3.0, 0.0]]),
        [[7.0, 0.0]],
    )


def test_line_through_points_and_intersection() -> None:
    """Checks the line through two points and the intersection of two lines."""
    horizontal = line_through_points((0.0, 1.0), (5.0, 1.0))
    vertical = line_through_points((2.0, 0.0), (2.0, 9.0))
    assert intersect_lines(horizontal, vertical) == pytest.approx((2.0, 1.0))


def test_parallel_lines_do_not_intersect() -> None:
    """Checks that parallel lines yield no intersection."""
    first = line_through_points((0.0, 0.0), (1.0, 1.0))
    second = line_through_points((0.0, 2.0), (1.0, 3.0))
    assert intersect_lines(first, second) is None


def test_rho_theta_line_and_point_distance() -> None:
    """Checks the Hough line conversion and the point-to-line distance."""
    line = line_from_rho_theta(5.0, 0.0)  # the vertical line x = 5
    np.testing.assert_allclose(
        point_line_distance(line, [[5.0, 7.0], [7.0, 0.0]]), [0.0, 2.0], atol=1e-12
    )
