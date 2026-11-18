"""Unit tests for Member B's homography estimation, quality, and smoothing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from football_vision.errors import HomographyError
from football_vision.homography.geometry import apply_homography, invert_homography
from football_vision.homography.keyframe import estimate_keyframe_homography_auto
from football_vision.homography.propagation import estimate_frame_alignment
from football_vision.homography.quality import (
    check_homography_validity,
    line_overlay_score,
)
from football_vision.homography.smoothing import smooth_homography_sequence
from football_vision.pitch.template import PitchTemplate

_H_PITCH_TO_IMAGE = np.array(
    [
        [10.0, 0.5, 115.0],
        [0.1, 8.0, 90.0],
        [0.0, 0.0001, 1.0],
    ],
    dtype=np.float64,
)
_FRAME_SIZE = (1280, 720)


def _render_synthetic_pitch_edges(
    homography_img_to_pitch: np.ndarray,
    template: PitchTemplate,
) -> tuple[np.ndarray, np.ndarray]:
    """Draws projected template segments on an edge map and extracts segments."""
    h_pitch_to_img = invert_homography(homography_img_to_pitch)
    width, height = _FRAME_SIZE
    edge_map = np.zeros((height, width), dtype=np.uint8)
    for polyline in template.polylines():
        pts = apply_homography(h_pitch_to_img, polyline)
        for p0, p1 in zip(pts[:-1], pts[1:], strict=True):
            cv2.line(
                edge_map,
                (int(round(p0[0])), int(round(p0[1]))),
                (int(round(p1[0])), int(round(p1[1]))),
                255,
                2,
            )
    segments = []
    for p0, p1 in template.segments():
        pts = apply_homography(h_pitch_to_img, np.array([p0, p1], dtype=np.float64))
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        if (
            0 <= x1 < width
            and 0 <= y1 < height
            and 0 <= x2 < width
            and 0 <= y2 < height
        ):
            segments.append([x1, y1, x2, y2])
    return edge_map, np.asarray(segments, dtype=np.float64)


def test_line_overlay_score_high_on_true_homography_and_zero_on_singular() -> None:
    """Checks that overlay score is near 1.0 for exact H and 0.0 when singular."""
    template = PitchTemplate()
    h_true = invert_homography(_H_PITCH_TO_IMAGE)
    edge_map, _ = _render_synthetic_pitch_edges(h_true, template)
    assert line_overlay_score(edge_map, h_true, template) > 0.9
    assert line_overlay_score(edge_map, np.zeros((3, 3)), template) == 0.0


def test_check_homography_validity_accepts_valid_and_rejects_degenerate() -> None:
    """Checks geometric plausibility tests for valid, mirrored, and singular H."""
    template = PitchTemplate()
    h_true = invert_homography(_H_PITCH_TO_IMAGE)
    assert check_homography_validity(h_true, _FRAME_SIZE, template)
    assert not check_homography_validity(np.zeros((3, 3)), _FRAME_SIZE, template)

    mirrored = h_true.copy()
    mirrored[:, 0] *= -1.0
    assert not check_homography_validity(mirrored, _FRAME_SIZE, template)


def test_estimate_keyframe_homography_auto_recovers_synthetic_pitch() -> None:
    """Checks auto keyframe estimation up to 180-degree pitch symmetry."""
    template = PitchTemplate()
    h_true = invert_homography(_H_PITCH_TO_IMAGE)
    edge_map, segments = _render_synthetic_pitch_edges(h_true, template)
    frame = np.zeros((_FRAME_SIZE[1], _FRAME_SIZE[0], 3), dtype=np.uint8)

    estimate = estimate_keyframe_homography_auto(frame, segments, template)
    assert check_homography_validity(estimate, _FRAME_SIZE, template)
    assert line_overlay_score(edge_map, estimate, template) > 0.85

    with pytest.raises(HomographyError):
        estimate_keyframe_homography_auto(frame, np.zeros((0, 4)), template)


def test_estimate_frame_alignment_recovers_shift_and_rejects_invalid_feature() -> None:
    """Checks inter-frame alignment recovery and feature validation."""
    rng = np.random.default_rng(7)
    frame_prev = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    # Add high-contrast corners so SIFT/ORB finds crisp features.
    for _ in range(40):
        cx, cy = int(rng.integers(20, 280)), int(rng.integers(20, 200))
        cv2.rectangle(frame_prev, (cx, cy), (cx + 12, cy + 12), (255, 255, 255), -1)
    shift = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]])
    frame_curr = cv2.warpPerspective(frame_prev, shift, (320, 240))

    g_curr, n_inliers = estimate_frame_alignment(
        frame_prev, frame_curr, feature="orb", min_inliers=8
    )
    assert n_inliers >= 8
    pt_curr = np.array([[160.0, 120.0]])
    np.testing.assert_allclose(
        apply_homography(g_curr, pt_curr), [[152.0, 124.0]], atol=2.0
    )

    with pytest.raises(ValueError, match="Unsupported feature"):
        estimate_frame_alignment(frame_prev, frame_curr, feature="invalid")  # type: ignore[arg-type]


def test_smooth_homography_sequence_reduces_jitter_and_handles_singular() -> None:
    """Checks temporal smoothing reduces noise and guards against singular frames."""
    h_base = invert_homography(_H_PITCH_TO_IMAGE)
    reference = np.array([[25.0, 15.0], [80.0, 15.0], [80.0, 53.0], [25.0, 53.0]])
    rng = np.random.default_rng(11)
    noisy = []
    for _ in range(15):
        jitter = np.eye(3)
        jitter[0, 2] = rng.normal(scale=0.5)
        jitter[1, 2] = rng.normal(scale=0.5)
        noisy.append(jitter @ h_base)
    mats = np.stack(noisy)
    smoothed = smooth_homography_sequence(mats, reference, window=7)
    assert smoothed.shape == mats.shape

    proj_raw = np.array([apply_homography(m, [[640.0, 360.0]])[0] for m in mats])
    proj_smooth = np.array([apply_homography(m, [[640.0, 360.0]])[0] for m in smoothed])
    assert np.std(np.diff(proj_smooth, axis=0)) < np.std(np.diff(proj_raw, axis=0))

    # Singular frame should not crash smoothing.
    broken = mats.copy()
    broken[3] = 0.0
    fallback = smooth_homography_sequence(broken, reference, window=7)
    np.testing.assert_array_equal(fallback, broken)
