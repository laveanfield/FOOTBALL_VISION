"""Tests for the implemented homography helpers (OpenCV baseline, RANSAC formula)."""

from __future__ import annotations

import numpy as np
import pytest

from football_vision.errors import HomographyError
from football_vision.homography.dlt import reprojection_error
from football_vision.homography.geometry import apply_homography, invert_homography
from football_vision.homography.keyframe import (
    homography_from_landmark_dict,
    homography_from_landmarks,
)
from football_vision.homography.propagation import propagate_homography
from football_vision.homography.ransac import ransac_num_iterations
from football_vision.pitch.template import PitchTemplate

H_PITCH_TO_IMAGE = np.array([[8.0, 1.5, 100.0], [0.5, 6.0, 50.0], [1e-4, 2e-4, 1.0]])
NAMES = [
    "corner_tl",
    "corner_tr",
    "corner_br",
    "corner_bl",
    "center",
    "left_pa_outer_top",
    "right_pa_outer_bottom",
]


def test_landmark_homography_recovers_the_camera() -> None:
    """Checks that clicked landmarks of a synthetic camera give an exact homography."""
    template = PitchTemplate()
    pitch = template.landmark_array(NAMES)
    image = apply_homography(H_PITCH_TO_IMAGE, pitch)
    landmarks = {name: tuple(point) for name, point in zip(NAMES, image, strict=True)}
    homography, n_inliers = homography_from_landmark_dict(landmarks, template)
    assert n_inliers == len(NAMES)
    np.testing.assert_allclose(apply_homography(homography, image), pitch, atol=1e-4)


def test_landmark_homography_needs_four_known_landmarks() -> None:
    """Checks the error for too few or unknown landmarks."""
    with pytest.raises(HomographyError):
        homography_from_landmark_dict(
            {"corner_tl": (0, 0), "not_a_landmark": (1, 1), "center": (5, 5)}
        )
    with pytest.raises(HomographyError):
        homography_from_landmarks(np.zeros((3, 2)), np.zeros((3, 2)))


def test_ransac_iteration_count_matches_the_derivation() -> None:
    """Checks N = ln(1-p) / ln(1-(1-eps)^4) for the values quoted in the plan."""
    assert ransac_num_iterations(0.99, 0.5) == 72
    assert ransac_num_iterations(0.99, 0.7) == 567
    assert ransac_num_iterations(0.99, 0.0) == 1
    with pytest.raises(ValueError):
        ransac_num_iterations(1.0, 0.5)
    with pytest.raises(ValueError):
        ransac_num_iterations(0.99, 1.0)


def test_propagation_composes_frame_alignments() -> None:
    """Checks H_t = H_{t-1} @ G_t and that the result is normalized."""
    h_previous = invert_homography(H_PITCH_TO_IMAGE)
    shift = np.array(
        [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]
    )  # frame t -> frame t-1
    h_current = propagate_homography(h_previous, shift)
    assert h_current[2, 2] == pytest.approx(1.0)
    point_t = np.array([[100.0, 100.0]])
    np.testing.assert_allclose(
        apply_homography(h_current, point_t),
        apply_homography(h_previous, point_t + [5.0, -3.0]),
        atol=1e-9,
    )


def test_reprojection_error_in_target_units() -> None:
    """Checks the Euclidean reprojection error."""
    identity = np.eye(3)
    errors = reprojection_error(
        identity, np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])
    )
    np.testing.assert_allclose(errors, [0.0, 5.0])


def test_landmark_homography_uses_dlt_for_noisy_clicks_and_ransac_for_outliers() -> (
    None
):
    """Uses all points via DLT under ~5 px click jitter and RANSAC on gross outliers."""
    template = PitchTemplate()
    pitch = template.landmark_array(NAMES)
    clean_img = apply_homography(H_PITCH_TO_IMAGE, pitch)
    jitter = np.array(
        [
            [4.0, -3.0],
            [-5.0, 2.0],
            [3.0, 4.0],
            [-4.0, -4.0],
            [5.0, -1.0],
            [-2.0, 5.0],
            [3.0, -3.0],
        ]
    )
    noisy_img = clean_img + jitter
    noisy_lm = {n: tuple(p) for n, p in zip(NAMES, noisy_img, strict=True)}
    h_dlt, n_inliers = homography_from_landmark_dict(
        noisy_lm, template, ransac_threshold_px=3.0
    )
    assert n_inliers == len(NAMES)
    assert reprojection_error(h_dlt, noisy_img, pitch).max() < 1.5

    corrupted_img = noisy_img.copy()
    corrupted_img[0] += [200.0, 200.0]
    corrupt_lm = {n: tuple(p) for n, p in zip(NAMES, corrupted_img, strict=True)}
    h_ransac, n_inliers_corrupt = homography_from_landmark_dict(
        corrupt_lm, template, ransac_threshold_px=12.0
    )
    assert n_inliers_corrupt < len(NAMES)
    assert reprojection_error(h_ransac, corrupted_img[1:], pitch[1:]).max() < 2.0
