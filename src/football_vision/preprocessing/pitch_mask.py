"""Grass-colored pitch mask (stage P1).

The mask removes stands, advertising boards and the sky. It is used to filter
edges before Hough, to discard detections outside the pitch and to exclude grass
pixels from jersey-color features.
"""

from __future__ import annotations

from football_vision.config import PreprocessConfig
from football_vision.datatypes import Frame, Mask


def estimate_grass_hsv_range(
    frame: Frame,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Estimates the HSV range of the grass of a specific frame.

    Fixed thresholds fail under changing light. A robust option is to take the
    dominant hue of the H histogram (or a 2-cluster KMeans on ``(H, S)``) and build
    a range around it.

    Args:
        frame: BGR frame.

    Returns:
        Tuple ``(lower, upper)`` of HSV bounds in OpenCV scale.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): histogram of the H channel -> dominant bin -> +/- tolerance range.
    raise NotImplementedError("TODO(A): implement estimate_grass_hsv_range")


def compute_pitch_mask(frame: Frame, cfg: PreprocessConfig) -> Mask:
    """Computes a binary mask of the playing surface.

    Steps: HSV threshold (fixed from ``cfg`` or adaptive), morphological opening and
    closing with ``cfg.morph_kernel``, then keep the largest connected component and
    fill its holes (players are holes in the grass).

    Args:
        frame: BGR frame.
        cfg: Preprocessing settings with the HSV bounds and kernel size.

    Returns:
        ``uint8`` mask of shape ``(H, W)`` with values 0 or 255.

    Raises:
        NotImplementedError: Stub; to be implemented by member A.
    """
    # TODO(A): cv2.inRange -> cv2.morphologyEx(OPEN, CLOSE) -> largest component
    #   (cv2.connectedComponentsWithStats) -> fill holes (findContours + drawContours).
    raise NotImplementedError("TODO(A): implement compute_pitch_mask")
