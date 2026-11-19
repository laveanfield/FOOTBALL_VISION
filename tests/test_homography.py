"""Unit tests for Member B's homography estimation, quality, and smoothing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from football_vision.errors import HomographyError
from football_vision.homography.dlt import (
    estimate_homography_dlt,
    hartley_normalization,
)
from football_vision.homography.geometry import (
    apply_homography,
    compose_homographies,
    invert_homography,
)
from football_vision.homography.keyframe import (
    estimate_keyframe_homography_auto,
    homography_from_landmarks,
)
from football_vision.homography.propagation import estimate_frame_alignment
from football_vision.homography.quality import (
    check_homography_validity,
    line_overlay_score,
)
from football_vision.homography.ransac import estimate_homography_ransac
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
    assert smooth_homography_sequence(mats, reference, window=1).shape == mats.shape

    horizon_crossing = np.eye(3)
    horizon_crossing[2, :] = [0.0, 0.0, 1.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        nan_mats = np.repeat(horizon_crossing[None], 5, axis=0)
        nan_mats[:, 2, 1] = np.nan
    np.testing.assert_array_equal(
        smooth_homography_sequence(nan_mats, reference, window=5), nan_mats
    )


def test_homography_validation_and_alignment_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers DLT, RANSAC, geometry, quality, keyframe, and alignment branches."""
    template = PitchTemplate()
    with pytest.raises(ValueError, match="empty"):
        hartley_normalization(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="coincide"):
        hartley_normalization(np.ones((4, 2)))
    with pytest.raises(HomographyError, match="at least 4"):
        estimate_homography_dlt(np.zeros((3, 2)), np.zeros((3, 2)))
    with pytest.raises(HomographyError, match="Degenerate DLT"):
        estimate_homography_dlt(np.ones((4, 2)), np.arange(8).reshape(4, 2))
    with pytest.raises(ValueError, match="at least one matrix"):
        compose_homographies()

    with pytest.raises(HomographyError, match="at least 4"):
        estimate_homography_ransac(np.zeros((3, 2)), np.zeros((3, 2)), threshold=1.0)
    collinear = np.array([[float(i), float(i)] for i in range(6)])
    with pytest.raises(HomographyError, match="no valid homography"):
        estimate_homography_ransac(collinear, collinear, threshold=1.0, seed=1)

    with pytest.raises(HomographyError, match="cv2.findHomography failed"):
        monkeypatch.setattr(cv2, "findHomography", lambda *a, **kw: (None, None))
        homography_from_landmarks(
            np.ones((4, 2)), np.arange(8, dtype=float).reshape(4, 2)
        )
    monkeypatch.undo()

    h_true = invert_homography(_H_PITCH_TO_IMAGE)
    edge_map, _ = _render_synthetic_pitch_edges(h_true, template)
    offscreen = np.array([[1.0, 0.0, 1e5], [0.0, 1.0, 1e5], [0.0, 0.0, 1.0]])
    assert line_overlay_score(edge_map, offscreen, template) == 0.0

    assert not check_homography_validity(np.eye(2), _FRAME_SIZE, template)
    ill_cond = np.diag([1.0, 1e-8, 1.0])
    assert not check_homography_validity(ill_cond, _FRAME_SIZE, template)
    horizon_h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0 / 500.0, -1.0]])
    assert not check_homography_validity(horizon_h, _FRAME_SIZE, template)

    rng = np.random.default_rng(9)
    frame = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    for _ in range(40):
        cx, cy = int(rng.integers(20, 280)), int(rng.integers(20, 200))
        cv2.rectangle(frame, (cx, cy), (cx + 12, cy + 12), (255, 255, 255), -1)
    for feat in ("sift", "akaze"):
        g, n_in = estimate_frame_alignment(
            frame,
            frame,
            exclude_boxes=np.array([[0, 0, 20, 20]], dtype=np.float32),
            feature=feat,
            min_inliers=10,
        )
        assert n_in >= 10 and g.shape == (3, 3)
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(HomographyError, match="Too few features"):
        estimate_frame_alignment(blank, blank, feature="orb")
    with pytest.raises(HomographyError):
        estimate_frame_alignment(frame, frame, feature="orb", min_inliers=50000)
