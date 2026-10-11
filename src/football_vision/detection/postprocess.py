"""Filters applied to raw detections before tracking (stage P4)."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from football_vision.boxes import foot_points
from football_vision.datatypes import Detections, Mask


def filter_by_confidence(dets: Detections, min_conf: float) -> Detections:
    """Drops detections with a low confidence score.

    Args:
        dets: Detections of one frame.
        min_conf: Minimum score to keep.

    Returns:
        The detections with ``score >= min_conf``.
    """
    return dets.select(dets.scores >= min_conf)


def filter_by_classes(dets: Detections, classes: Sequence[int]) -> Detections:
    """Keeps only the given class ids.

    Args:
        dets: Detections of one frame.
        classes: Class ids to keep.

    Returns:
        The detections whose class id is in ``classes``.
    """
    return dets.select(np.isin(dets.class_ids, list(classes)))


def filter_by_size(dets: Detections, min_height_px: float) -> Detections:
    """Drops boxes that are too short to be a player.

    Args:
        dets: Detections of one frame.
        min_height_px: Minimum box height in pixels.

    Returns:
        The detections with ``y2 - y1 >= min_height_px``.
    """
    heights = dets.boxes_xyxy[:, 3] - dets.boxes_xyxy[:, 1]
    return dets.select(heights >= min_height_px)


def filter_by_pitch_mask(
    dets: Detections, mask: Mask, margin_px: int = 0
) -> Detections:
    """Drops detections whose feet are outside the pitch.

    This removes spectators, coaches and ball boys standing beside the pitch.

    Args:
        dets: Detections of one frame.
        mask: Binary pitch mask of shape ``(H, W)`` (non-zero inside the pitch).
        margin_px: Dilation of the mask in pixels, so players on the lines are kept.

    Returns:
        The detections whose bottom-center point lies inside the (dilated) mask.
    """
    if len(dets) == 0:
        return dets
    region = mask
    if margin_px > 0:
        kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), dtype=np.uint8)
        region = cv2.dilate(mask, kernel)
    feet = foot_points(dets.boxes_xyxy)
    height, width = region.shape[:2]
    xs = np.clip(np.round(feet[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.round(feet[:, 1]).astype(int), 0, height - 1)
    return dets.select(region[ys, xs] > 0)
