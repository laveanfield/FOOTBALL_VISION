"""Tests for the detection filters."""

from __future__ import annotations

import numpy as np
import pytest

from football_vision.datatypes import Detections
from football_vision.detection.postprocess import (
    filter_by_classes,
    filter_by_confidence,
    filter_by_pitch_mask,
    filter_by_size,
)


def make_detections() -> Detections:
    """Creates three detections: a tall player, a short box and a ball."""
    return Detections(
        boxes_xyxy=np.array(
            [[10, 10, 30, 70], [40, 50, 50, 60], [60, 60, 64, 64]], dtype=np.float32
        ),
        scores=np.array([0.9, 0.3, 0.6], dtype=np.float32),
        class_ids=np.array([0, 0, 32], dtype=np.int64),
    )


def test_filter_by_confidence_size_and_classes() -> None:
    """Checks the three simple filters."""
    dets = make_detections()
    assert filter_by_confidence(dets, 0.5).scores.tolist() == pytest.approx([0.9, 0.6])
    assert len(filter_by_size(dets, 20.0)) == 1
    assert filter_by_classes(dets, [32]).class_ids.tolist() == [32]


def test_filter_by_pitch_mask_uses_feet_and_margin() -> None:
    """Checks that detections whose feet are outside the mask are dropped."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :50] = 255
    dets = Detections(
        boxes_xyxy=np.array([[10, 10, 40, 90], [60, 10, 90, 90]], dtype=np.float32),
        scores=np.array([0.9, 0.9], dtype=np.float32),
        class_ids=np.array([0, 0], dtype=np.int64),
    )
    assert filter_by_pitch_mask(dets, mask).boxes_xyxy[:, 0].tolist() == [10.0]
    assert len(filter_by_pitch_mask(dets, mask, margin_px=30)) == 2
    assert len(filter_by_pitch_mask(Detections.empty(), mask)) == 0
