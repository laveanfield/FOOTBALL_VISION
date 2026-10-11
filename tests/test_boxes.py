"""Tests for the bounding-box utilities."""

from __future__ import annotations

import numpy as np
import pytest

from football_vision.boxes import (
    box_areas,
    clip_boxes,
    foot_points,
    iou_matrix,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_iou_of_identical_disjoint_and_half_overlapping_boxes() -> None:
    """Checks IoU for identical, disjoint and partially overlapping boxes."""
    a = [[0, 0, 2, 2]]
    b = [[0, 0, 2, 2], [10, 10, 12, 12], [1, 0, 3, 2]]
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 0.0, 1.0 / 3.0]])


def test_iou_with_empty_inputs_has_matching_shape() -> None:
    """Checks the shape of IoU matrices with empty inputs."""
    assert iou_matrix(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_matrix([[0, 0, 1, 1]], np.zeros((0, 4))).shape == (1, 0)


def test_degenerate_box_has_zero_iou_and_area() -> None:
    """Checks that a zero-area box never overlaps anything."""
    assert box_areas([[5, 5, 5, 9]])[0] == 0.0
    assert iou_matrix([[5, 5, 5, 9]], [[5, 5, 5, 9]])[0, 0] == 0.0


def test_foot_points_are_bottom_center() -> None:
    """Checks that the foot point is the middle of the bottom edge."""
    np.testing.assert_allclose(foot_points([[10, 20, 30, 80]]), [[20.0, 80.0]])


def test_box_format_round_trip() -> None:
    """Checks the conversion between corner and (x, y, w, h) formats."""
    boxes = np.array([[1.0, 2.0, 11.0, 22.0]])
    np.testing.assert_allclose(xywh_to_xyxy(xyxy_to_xywh(boxes)), boxes)
    np.testing.assert_allclose(xyxy_to_xywh(boxes), [[1.0, 2.0, 10.0, 20.0]])


def test_clip_boxes_limits_coordinates() -> None:
    """Checks that boxes are clipped to the image rectangle."""
    clipped = clip_boxes([[-5, -5, 700, 400]], width=640, height=360)
    assert clipped[0].tolist() == pytest.approx([0, 0, 640, 360])
