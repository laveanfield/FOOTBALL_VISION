"""Tests for pitch line extraction, Hough clustering and keypoint detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from football_vision.config import PitchConfig
from football_vision.pitch.edges import (
    detect_edges_canny,
    extract_edges,
    extract_white_lines_tophat,
)
from football_vision.pitch.hough import (
    cluster_line_families,
    detect_line_segments,
    merge_collinear_segments,
)
from football_vision.pitch.keypoints import (
    find_line_intersections,
    fit_ellipse_to_edges,
)


def _draw_synthetic_pitch_gray(width: int = 240, height: int = 160) -> np.ndarray:
    """Creates a synthetic grayscale pitch with mowing stripes and bright lines."""
    gray = np.full((height, width), 70, dtype=np.uint8)
    for x in range(0, width, 40):
        gray[:, x : x + 20] = 95
    cv2.line(gray, (20, 30), (220, 30), 235, 3)
    cv2.line(gray, (120, 20), (120, 140), 235, 3)
    return gray


def test_canny_detects_line_and_erodes_mask_boundary() -> None:
    """Checks Canny edge detection inside an eroded pitch mask."""
    gray = np.full((120, 160), 60, dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[20:100, 20:140] = 255
    gray[20:100, 20:140] = 120
    cv2.line(gray, (30, 60), (130, 60), 240, 2)
    edges = detect_edges_canny(gray, mask, low=50, high=150, blur_sigma=1.0)
    assert edges.dtype == np.uint8
    assert edges[58:63, 40:120].any()
    assert not edges[20, 40:120].any()
    assert not edges[:15, :].any()


def test_canny_rejects_mismatched_mask_shape() -> None:
    """Checks that Canny raises ValueError when the mask shape differs."""
    gray = np.zeros((100, 120), dtype=np.uint8)
    bad_mask = np.zeros((90, 120), dtype=np.uint8)
    with pytest.raises(ValueError, match="mask shape"):
        detect_edges_canny(gray, bad_mask, low=50, high=150, blur_sigma=1.0)


def test_tophat_keeps_thin_line_and_suppresses_mowing_stripes() -> None:
    """Checks that white top-hat isolates thin lines over wide mowing stripes."""
    gray = _draw_synthetic_pitch_gray()
    mask = np.full_like(gray, 255)
    lines = extract_white_lines_tophat(gray, mask, kernel_size=11)
    assert lines[28:33, 40:200].any()
    assert not lines[80:120, 38:42].any()


def test_tophat_returns_empty_on_flat_and_low_noise_frames() -> None:
    """Checks that the minimum top-hat floor prevents false lines on blank images."""
    rng = np.random.default_rng(0)
    flat = np.full((100, 120), 80, dtype=np.uint8)
    noisy = np.clip(flat.astype(np.int16) + rng.integers(-3, 4, flat.shape), 0, 255)
    assert not extract_white_lines_tophat(flat, None, 15).any()
    assert not extract_white_lines_tophat(noisy.astype(np.uint8), None, 15).any()


def test_tophat_rejects_a_mask_of_another_shape() -> None:
    """Checks that top-hat raises ValueError instead of IndexError on shape mismatch."""
    gray = np.zeros((100, 120), dtype=np.uint8)
    bad_mask = np.zeros((90, 120), dtype=np.uint8)
    with pytest.raises(ValueError, match="mask shape"):
        extract_white_lines_tophat(gray, bad_mask, 15)


def test_extract_edges_dispatches_and_rejects_unknown_method() -> None:
    """Checks method selection in extract_edges and invalid method rejection."""
    frame = cv2.cvtColor(_draw_synthetic_pitch_gray(), cv2.COLOR_GRAY2BGR)
    canny_map = extract_edges(frame, None, PitchConfig(edge_method="canny"))
    tophat_map = extract_edges(frame, None, PitchConfig(edge_method="tophat"))
    assert canny_map.shape == frame.shape[:2] and canny_map.any()
    assert tophat_map.shape == frame.shape[:2] and tophat_map.any()
    bad_cfg = PitchConfig(edge_method="sobel")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported edge method"):
        extract_edges(frame, None, bad_cfg)


def test_detect_line_segments_finds_straight_lines_and_empty_map() -> None:
    """Checks probabilistic Hough line detection and empty input handling."""
    empty = np.zeros((120, 160), dtype=np.uint8)
    cfg = PitchConfig(hough_threshold=30, hough_min_length=40, hough_max_gap=5)
    assert detect_line_segments(empty, cfg).shape == (0, 4)
    edges = empty.copy()
    cv2.line(edges, (10, 60), (140, 60), 255, 2)
    segs = detect_line_segments(edges, cfg)
    assert segs.ndim == 2 and segs.shape[1] == 4 and len(segs) >= 1


def test_merge_collinear_segments_combines_fragments_and_thick_edges() -> None:
    """Checks merging of collinear fragments and double edges of a thick line."""
    fragments = np.array(
        [
            [10.0, 50.0, 60.0, 50.0],
            [75.0, 51.0, 130.0, 50.5],
            [10.0, 54.0, 130.0, 54.0],
        ]
    )
    merged = merge_collinear_segments(fragments, rho_tolerance_px=6.0)
    assert len(merged) == 1
    assert min(merged[0, 0], merged[0, 2]) <= 11.0
    assert max(merged[0, 0], merged[0, 2]) >= 129.0


def test_merge_collinear_segments_keeps_parallel_and_crossing_separate() -> None:
    """Checks that distinct parallel and intersecting lines are not merged."""
    segs = np.array(
        [
            [10.0, 20.0, 150.0, 20.0],
            [10.0, 90.0, 150.0, 90.0],
            [80.0, 10.0, 80.0, 120.0],
        ]
    )
    merged = merge_collinear_segments(segs)
    assert len(merged) == 3


def test_merge_collinear_segments_handles_pi_wraparound_and_degenerate_inputs() -> (
    None
):
    """Checks vertical wrap-around near theta=0/pi and zero-length segment removal."""
    vertical = np.array(
        [
            [100.0, 10.0, 100.5, 60.0],
            [100.2, 130.0, 99.8, 70.0],
            [50.0, 50.0, 50.0, 50.0],
        ]
    )
    merged = merge_collinear_segments(vertical, rho_tolerance_px=5.0)
    assert len(merged) == 1
    assert merge_collinear_segments(np.zeros((0, 4))).shape == (0, 4)


def test_cluster_line_families_separates_two_families_order_independently() -> None:
    """Checks horizontal/vertical family labeling independent of input order."""
    segs = np.array(
        [
            [50.0, 0.0, 50.0, 200.0],
            [0.0, 40.0, 200.0, 42.0],
            [140.0, 0.0, 142.0, 200.0],
            [0.0, 120.0, 200.0, 118.0],
        ]
    )
    labels = cluster_line_families(segs)
    assert labels.tolist() == [1, 0, 1, 0]
    rev = cluster_line_families(segs[::-1])
    assert rev.tolist() == [0, 1, 0, 1]


def test_cluster_line_families_handles_theta_wrap_short_and_empty() -> None:
    """Checks wrap-around near pi, short segment filtering and empty input."""
    segs = np.array(
        [
            [0.0, 50.0, 100.0, 51.0],
            [0.0, 100.0, 100.0, 99.0],
            [40.0, 0.0, 40.0, 120.0],
            [0.0, 0.0, 5.0, 5.0],
        ]
    )
    labels = cluster_line_families(segs)
    assert labels[0] == labels[1] == 0
    assert labels[2] == 1
    assert labels[3] == -1
    assert cluster_line_families(np.zeros((0, 4))).shape == (0,)


def test_cluster_does_not_split_a_single_family() -> None:
    """Checks that nearly parallel segments stay in a single horizontal family."""
    horizontal = np.array(
        [
            [0.0, 10.0, 200.0, 10.0],
            [0.0, 60.0, 200.0, 61.0],
            [0.0, 110.0, 200.0, 110.5],
            [0.0, 160.0, 200.0, 159.0],
        ]
    )
    assert cluster_line_families(horizontal).tolist() == [0, 0, 0, 0]


def test_cluster_gives_a_lone_vertical_family_the_label_one() -> None:
    """Checks that a lone near-vertical family receives label 1."""
    vertical = np.array(
        [
            [50.0, 0.0, 52.0, 300.0],
            [150.0, 0.0, 149.0, 300.0],
            [260.0, 0.0, 262.0, 300.0],
        ]
    )
    assert cluster_line_families(vertical).tolist() == [1, 1, 1]


def test_find_line_intersections_extends_segments_and_filters_bounds() -> None:
    """Checks intersection extension, boundary clipping and degenerate inputs."""
    lines_a = np.array([[0.0, 50.0, 40.0, 50.0], [0.0, 50.0, 40.0, 50.0]])
    lines_b = np.array(
        [
            [100.0, 0.0, 100.0, 30.0],
            [200.0, 0.0, 200.0, 100.0],
            [0.0, 50.0, 20.0, 50.0],
        ]
    )
    pts = find_line_intersections(lines_a, lines_b, frame_size=(200, 100))
    np.testing.assert_allclose(pts, [[100.0, 50.0], [100.0, 50.0]])
    assert find_line_intersections(np.zeros((0, 4)), lines_b, (200, 100)).shape == (
        0,
        2,
    )


def test_fit_ellipse_to_edges_recovers_rotated_ellipse_and_checks_count() -> None:
    """Checks rotated ellipse parameter recovery and minimum point validation."""
    t = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    cx, cy, a, b, angle_deg = 120.0, 80.0, 40.0, 20.0, 30.0
    rad = np.deg2rad(angle_deg)
    xs = cx + a * np.cos(t) * np.cos(rad) - b * np.sin(t) * np.sin(rad)
    ys = cy + a * np.cos(t) * np.sin(rad) + b * np.sin(t) * np.cos(rad)
    center, axes, _ = fit_ellipse_to_edges(np.stack([xs, ys], axis=1))
    assert center == pytest.approx((cx, cy), abs=0.5)
    assert sorted(axes) == pytest.approx([2 * b, 2 * a], abs=1.0)
    with pytest.raises(ValueError, match="at least 5 points"):
        fit_ellipse_to_edges(np.zeros((4, 2)))
