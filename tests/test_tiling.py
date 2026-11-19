"""Tests for Phase C4 tiled detection (SAHI-style)."""

from __future__ import annotations

import numpy as np
import pytest

from football_vision.datatypes import Detections, Frame
from football_vision.detection.base import Detector
from football_vision.detection.tiling import (
    TiledDetector,
    merge_tile_detections,
    tile_image,
)


def test_tile_image_covers_borders_and_handles_large_tile() -> None:
    """Covers the full frame including the bottom/right edges without out-of-bounds."""
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    tiles = tile_image(frame, tile_size=(64, 64), overlap=0.25)
    offsets = [offset for _, offset in tiles]
    # Last tile must align flush with right (150 - 64 = 86) and bottom (100 - 64 = 36).
    assert (0, 0) in offsets
    assert (86, 36) in offsets
    assert all(tile.shape == (64, 64, 3) for tile, _ in tiles)

    # Tile larger than frame yields a single full-frame tile at (0, 0).
    single = tile_image(frame, tile_size=(300, 200), overlap=0.2)
    assert len(single) == 1
    assert single[0][1] == (0, 0)
    assert single[0][0].shape == (100, 150, 3)


def test_tile_image_validates_inputs() -> None:
    """Rejects empty frames, invalid tile sizes, and out-of-range overlap."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="non-empty"):
        tile_image(empty, (32, 32), 0.2)

    valid = np.zeros((40, 40, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive"):
        tile_image(valid, (0, 32), 0.2)
    with pytest.raises(ValueError, match="overlap"):
        tile_image(valid, (16, 16), 1.0)


def test_merge_tile_detections_shifts_and_runs_class_aware_nms() -> None:
    """Merges overlapping same-class duplicates while keeping distinct classes."""
    det_a = Detections(
        boxes_xyxy=np.array([[10, 10, 30, 50], [12, 12, 32, 52]], dtype=np.float32),
        scores=np.array([0.9, 0.7], dtype=np.float32),
        class_ids=np.array([0, 32], dtype=np.int64),
    )
    # Slightly shifted duplicate of person box from neighbor tile at x_offset=5.
    det_b = Detections(
        boxes_xyxy=np.array([[6, 11, 26, 51]], dtype=np.float32),
        scores=np.array([0.6], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int64),
    )
    merged = merge_tile_detections(
        [det_a, det_b], offsets=[(0, 0), (5, 0)], iou_threshold=0.5
    )
    # Same-class duplicate person (0.6) suppressed by higher-score person (0.9);
    # different-class ball (class 32, 0.7) preserved despite high overlap.
    assert len(merged) == 2
    assert merged.class_ids.tolist() == [0, 32]
    np.testing.assert_allclose(merged.boxes_xyxy[0], [10, 10, 30, 50])

    assert len(merge_tile_detections([], [], 0.5)) == 0


def test_tiled_detector_finds_marked_targets_across_tiles() -> None:
    """Runs TiledDetector with a local bright-spot detector across overlapping tiles."""

    class _ColorSpotDetector(Detector):
        def detect(self, frame: Frame) -> Detections:
            ys, xs = np.where(frame[:, :, 2] > 200)
            if len(xs) == 0:
                return Detections.empty()
            box = np.array(
                [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float32
            )
            return Detections(
                box,
                np.array([0.95], dtype=np.float32),
                np.array([32], dtype=np.int64),
            )

    full = np.zeros((120, 160, 3), dtype=np.uint8)
    full[20:26, 20:26, 2] = 255
    full[90:96, 130:136, 2] = 255
    detector = TiledDetector(
        _ColorSpotDetector(), tile_size=(80, 80), overlap=0.25, iou_threshold=0.5
    )
    dets = detector.detect(full)
    assert len(dets) == 2
    assert dets.class_ids.tolist() == [32, 32]
