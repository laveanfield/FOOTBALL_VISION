"""Contract tests for the algorithms that are still stubs.

Each test states what an implementation must satisfy. They are marked ``xfail``
while the corresponding function raises ``NotImplementedError``. When you implement
a function, delete its ``xfail`` marker so that the test guards your code.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_vision.homography.dlt import (
    estimate_homography_dlt,
    hartley_normalization,
    reprojection_error,
)
from football_vision.homography.geometry import apply_homography
from football_vision.homography.ransac import estimate_homography_ransac
from football_vision.preprocessing.shots import histogram_distance
from football_vision.tracking.association import solve_assignment
from football_vision.tracking.kalman import KalmanFilter, constant_velocity_matrices

H_TRUE = np.array([[0.08, 0.01, -20.0], [0.005, 0.12, -5.0], [1e-4, 2e-4, 1.0]])
FOUR_PIXELS = np.array(
    [[100.0, 100.0], [1100.0, 120.0], [1000.0, 650.0], [150.0, 600.0]]
)

dlt_todo = pytest.mark.xfail(
    raises=NotImplementedError, reason="TODO(B): homography.dlt", strict=False
)
ransac_todo = pytest.mark.xfail(
    raises=NotImplementedError, reason="TODO(B): homography.ransac", strict=False
)
kalman_todo = pytest.mark.xfail(
    raises=NotImplementedError, reason="TODO(C): tracking.kalman", strict=False
)
assign_todo = pytest.mark.xfail(
    raises=NotImplementedError, reason="TODO(C): tracking.association", strict=False
)
shots_todo = pytest.mark.xfail(
    raises=NotImplementedError, reason="TODO(A): preprocessing.shots", strict=False
)


def kalman_model(
    dt: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the constant-velocity model matrices used by the Kalman tests."""
    return constant_velocity_matrices(dt, process_std=0.5, measurement_std=0.3)


@dlt_todo
def test_hartley_normalization_centers_and_scales_points() -> None:
    """Checks zero centroid, mean distance sqrt(2) and a consistent transform."""
    points = np.random.default_rng(0).uniform(0, 1000, size=(10, 2))
    normalized, transform = hartley_normalization(points)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-9)
    assert np.linalg.norm(normalized, axis=1).mean() == pytest.approx(np.sqrt(2))
    np.testing.assert_allclose(
        apply_homography(transform, points), normalized, atol=1e-9
    )


@dlt_todo
@pytest.mark.parametrize("normalize", [True, False])
def test_dlt_recovers_a_homography_from_four_points(normalize: bool) -> None:
    """Checks that four exact correspondences determine the homography."""
    destination = apply_homography(H_TRUE, FOUR_PIXELS)
    estimate = estimate_homography_dlt(FOUR_PIXELS, destination, normalize=normalize)
    assert estimate[2, 2] == pytest.approx(1.0)
    np.testing.assert_allclose(estimate, H_TRUE, rtol=1e-5, atol=1e-8)


@dlt_todo
def test_dlt_is_accurate_with_noisy_overdetermined_data() -> None:
    """Checks the least-squares solution with 30 noisy correspondences."""
    rng = np.random.default_rng(1)
    pixels = rng.uniform([0, 0], [1280, 720], size=(30, 2))
    meters = apply_homography(H_TRUE, pixels)
    estimate = estimate_homography_dlt(
        pixels + rng.normal(scale=1.0, size=pixels.shape), meters
    )
    assert reprojection_error(estimate, pixels, meters).mean() < 0.1


@ransac_todo
def test_ransac_rejects_gross_outliers() -> None:
    """Checks that RANSAC finds the 28 inliers among 40 correspondences."""
    rng = np.random.default_rng(2)
    pixels = rng.uniform([0, 0], [1280, 720], size=(40, 2))
    meters = apply_homography(H_TRUE, pixels)
    outliers = np.arange(28, 40)
    meters[outliers] += rng.uniform(5.0, 20.0, size=(12, 2)) * rng.choice(
        [-1, 1], size=(12, 2)
    )
    estimate, inliers = estimate_homography_ransac(
        pixels, meters, threshold=0.5, seed=0
    )
    assert inliers[:28].mean() > 0.9 and not inliers[outliers].any()
    assert (
        reprojection_error(
            estimate, pixels[:28], apply_homography(H_TRUE, pixels[:28])
        ).max()
        < 0.1
    )


@kalman_todo
def test_kalman_predict_follows_the_motion_model() -> None:
    """Checks that prediction moves the state by velocity * dt and grows uncertainty."""
    f, m, q, r = kalman_model(dt=0.1)
    kalman = KalmanFilter(f, m, q, r, x0=[1.0, 2.0, 3.0, 4.0], p0=np.eye(4))
    np.testing.assert_allclose(kalman.predict(), [1.3, 2.4, 3.0, 4.0])
    assert np.all(np.diag(kalman.p) >= 1.0)


@kalman_todo
def test_kalman_update_pulls_the_estimate_toward_the_measurement() -> None:
    """Checks that a position measurement moves only the measured axis."""
    f, m, q, r = kalman_model()
    kalman = KalmanFilter(f, m, q, r, x0=[0.0, 0.0, 0.0, 0.0], p0=10.0 * np.eye(4))
    kalman.predict()
    updated = kalman.update([1.0, 0.0])
    assert 0.5 < updated[0] <= 1.0
    assert updated[1] == pytest.approx(0.0, abs=1e-9)


@kalman_todo
def test_kalman_tracks_constant_velocity_from_noisy_positions() -> None:
    """Checks that velocity is recovered from noisy position measurements alone."""
    rng = np.random.default_rng(0)
    dt, velocity = 0.04, np.array([4.0, 1.0])
    f, m, q, r = kalman_model(dt)
    kalman = KalmanFilter(
        f, m, q, r, x0=[0.0, 0.0, 0.0, 0.0], p0=np.diag([1.0, 1.0, 25.0, 25.0])
    )
    for step in range(1, 151):
        kalman.predict()
        kalman.update(velocity * step * dt + rng.normal(scale=0.3, size=2))
    assert np.linalg.norm(kalman.x[2:] - velocity) < 0.8
    assert np.linalg.norm(kalman.x[:2] - velocity * 150 * dt) < 0.5


@kalman_todo
def test_kalman_gating_distance_grows_with_distance() -> None:
    """Checks that the squared Mahalanobis distance is zero at the prediction."""
    f, m, q, r = kalman_model()
    kalman = KalmanFilter(f, m, q, r, x0=[0.0, 0.0, 0.0, 0.0], p0=np.eye(4))
    kalman.predict()
    distances = kalman.gating_distance([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    assert distances[0] == pytest.approx(0.0, abs=1e-9)
    assert distances[0] < distances[1] < distances[2]


@assign_todo
def test_assignment_matches_optimally_and_applies_the_gate() -> None:
    """Checks the optimal matching, gating and the unmatched indices."""
    cost = np.array([[0.1, 0.9, 0.9], [0.9, 0.2, 0.9], [0.9, 0.9, 0.95]])
    matches, unmatched_rows, unmatched_cols = solve_assignment(cost, max_cost=0.8)
    assert (
        matches == [(0, 0), (1, 1)] and unmatched_rows == [2] and unmatched_cols == [2]
    )
    assert solve_assignment(np.zeros((0, 3)), 0.5) == ([], [], [0, 1, 2])


@shots_todo
def test_histogram_distance_is_zero_for_equal_frames_and_one_for_disjoint_ones() -> (
    None
):
    """Checks the chi-square histogram distance on black and white frames."""
    black = np.zeros((40, 40, 3), dtype=np.uint8)
    white = np.full((40, 40, 3), 255, dtype=np.uint8)
    assert histogram_distance(black, black) == pytest.approx(0.0, abs=1e-9)
    assert histogram_distance(black, white) == pytest.approx(1.0, abs=1e-6)
