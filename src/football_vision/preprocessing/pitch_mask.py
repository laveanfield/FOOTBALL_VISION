"""Grass-colored pitch mask (stage P1).

The mask removes stands, advertising boards and the sky. It is used to filter
edges before Hough, to discard detections outside the pitch and to exclude grass
pixels from jersey-color features.
"""

from __future__ import annotations

import cv2
import numpy as np

from football_vision.config import PreprocessConfig
from football_vision.datatypes import Frame, Mask

_DEFAULT_LOWER = (35, 40, 40)
_DEFAULT_UPPER = (85, 255, 255)
# Grass hues searched in the H histogram (OpenCV scale, 0-179): yellowish to
# bluish green. Restricting the search keeps colorful stands from winning.
_GRASS_HUE_SEARCH = (20, 100)
_HUE_TOLERANCE = 15
_MIN_SATURATION = 30
_MIN_VALUE = 30
# The fixed range from the configuration is replaced by an adaptive one when its
# mask covers less than this fraction of the image.
_MIN_GRASS_FRACTION = 0.05
# White pitch lines (with their blended borders) are not grass, so they cut the
# grass into separate cells. The closing kernel is this many times larger than the
# opening kernel so that it bridges the lines and keeps the pitch in one piece.
_CLOSE_KERNEL_FACTOR = 3


def estimate_grass_hsv_range(
    frame: Frame,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Estimates the HSV range of the grass of a specific frame.

    Fixed thresholds fail under changing light. A robust option is to take the
    dominant hue of the H histogram (or a 2-cluster KMeans on ``(H, S)``) and build
    a range around it.

    The dominant hue is searched among green-ish hues (20-100) after ignoring
    pixels that are nearly gray or nearly black. The range is that hue plus or
    minus 15; the lower saturation and value bounds are low percentiles of the
    pixels around the peak, so that shadowed grass is kept. If the frame has no
    such pixels, the default range ``(35, 40, 40)`` to ``(85, 255, 255)`` is
    returned.

    Args:
        frame: BGR frame.

    Returns:
        Tuple ``(lower, upper)`` of HSV bounds in OpenCV scale.
    """
    # TODO(A): histogram of the H channel -> dominant bin -> +/- tolerance range.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].ravel().astype(np.intp)
    sat = hsv[..., 1].ravel()
    val = hsv[..., 2].ravel()
    colorful = (sat >= _MIN_SATURATION) & (val >= _MIN_VALUE)
    hist = np.bincount(hue[colorful], minlength=180).astype(np.float64)
    # Smooth with a circular box filter so that a single noisy bin cannot win.
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    smooth = np.convolve(padded, np.ones(5) / 5.0, mode="valid")
    low_hue, high_hue = _GRASS_HUE_SEARCH
    band = smooth[low_hue : high_hue + 1]
    if band.max() <= 0:
        return _DEFAULT_LOWER, _DEFAULT_UPPER
    peak = low_hue + int(np.argmax(band))
    near_peak = colorful & (np.abs(hue - peak) <= _HUE_TOLERANCE)
    sat_low = max(_MIN_SATURATION, int(np.percentile(sat[near_peak], 5)))
    val_low = max(_MIN_VALUE, int(np.percentile(val[near_peak], 2)))
    lower = (max(peak - _HUE_TOLERANCE, 0), sat_low, val_low)
    upper = (min(peak + _HUE_TOLERANCE, 179), 255, 255)
    return lower, upper


def _mask_from_range(
    hsv: np.ndarray,
    lower: tuple[int, int, int] | list[int],
    upper: tuple[int, int, int] | list[int],
    kernel_size: int,
) -> Mask:
    """Thresholds an HSV image and keeps the largest, hole-free grass component."""
    raw = cv2.inRange(
        hsv, np.asarray(lower, dtype=np.uint8), np.asarray(upper, dtype=np.uint8)
    )
    size = max(1, int(kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    close_size = (_CLOSE_KERNEL_FACTOR * size) | 1
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if count <= 1:
        return np.zeros(raw.shape, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest, 255, 0).astype(np.uint8)
    # Players inside the pitch are holes of the component; the filled external
    # contour closes them while keeping notches open to the outside.
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def compute_pitch_mask(frame: Frame, cfg: PreprocessConfig) -> Mask:
    """Computes a binary mask of the playing surface.

    Steps: HSV threshold (fixed from ``cfg`` or adaptive), morphological opening and
    closing with ``cfg.morph_kernel``, then keep the largest connected component and
    fill its holes (players are holes in the grass).

    The opening uses ``cfg.morph_kernel``; the closing uses a kernel three times
    larger, because the white pitch lines would otherwise split the grass into
    cells and only one cell would survive. Increase ``cfg.morph_kernel`` if the
    lines are wider than about 15 px.

    The fixed range ``cfg.grass_hsv_low`` to ``cfg.grass_hsv_high`` is tried first.
    If the resulting mask covers less than 5% of the image (different grass color
    or lighting), the range is re-estimated from the frame with
    :func:`estimate_grass_hsv_range` and the larger of the two masks is returned.

    Args:
        frame: BGR frame.
        cfg: Preprocessing settings with the HSV bounds and kernel size.

    Returns:
        ``uint8`` mask of shape ``(H, W)`` with values 0 or 255.
    """
    # TODO(A): cv2.inRange -> cv2.morphologyEx(OPEN, CLOSE) -> largest component
    #   (cv2.connectedComponentsWithStats) -> fill holes (findContours + drawContours).
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _mask_from_range(
        hsv, cfg.grass_hsv_low, cfg.grass_hsv_high, cfg.morph_kernel
    )
    if np.count_nonzero(mask) >= _MIN_GRASS_FRACTION * mask.size:
        return mask
    adaptive_low, adaptive_high = estimate_grass_hsv_range(frame)
    adaptive = _mask_from_range(hsv, adaptive_low, adaptive_high, cfg.morph_kernel)
    return adaptive if np.count_nonzero(adaptive) > np.count_nonzero(mask) else mask
