"""Jersey-color features of tracked players (stage P6)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from football_vision.config import PreprocessConfig
from football_vision.datatypes import Frame

_DEFAULT_PREPROCESS = PreprocessConfig()


def crop_torso(
    frame: Frame,
    box_xyxy: Sequence[float],
    top_ratio: float = 0.15,
    bottom_ratio: float = 0.55,
    side_margin_ratio: float = 0.2,
) -> Frame:
    """Crops the torso of a player, where the jersey color is most visible.

    The crop spans ``top_ratio`` to ``bottom_ratio`` of the box height and drops a
    margin on both sides, which avoids grass, legs and neighboring players.

    Args:
        frame: BGR frame.
        box_xyxy: Player box ``(x1, y1, x2, y2)`` in pixels.
        top_ratio: Top of the crop as a fraction of the box height.
        bottom_ratio: Bottom of the crop as a fraction of the box height.
        side_margin_ratio: Fraction of the box width removed on each side.

    Returns:
        A view of the frame; it is empty if the box lies outside the frame.
    """
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    height, width = frame.shape[:2]
    box_h, box_w = y2 - y1, x2 - x1
    left = int(np.clip(round(x1 + side_margin_ratio * box_w), 0, width))
    right = int(np.clip(round(x2 - side_margin_ratio * box_w), 0, width))
    top = int(np.clip(round(y1 + top_ratio * box_h), 0, height))
    bottom = int(np.clip(round(y1 + bottom_ratio * box_h), 0, height))
    return frame[top:bottom, left:right]


def extract_color_feature(
    crop: Frame,
    color_space: Literal["lab", "hsv"] = "lab",
    exclude_grass: bool = True,
) -> npt.NDArray[np.float64] | None:
    """Summarizes the jersey color of a torso crop as a feature vector.

    Convert the crop to ``color_space``, drop grass-colored pixels when
    ``exclude_grass`` is set, and return a robust summary (median color). Lab
    distances are closer to perceived color differences than RGB distances; HSV
    encodes hue via cosine/sine components to avoid wrap-around at red.

    Args:
        crop: BGR torso crop from :func:`crop_torso`.
        color_space: Color space of the feature (``"lab"`` or ``"hsv"``).
        exclude_grass: Whether to ignore pixels inside the grass HSV range.

    Returns:
        A feature vector of shape ``(D,)``, or ``None`` if no pixel remains.

    Raises:
        ValueError: If ``color_space`` is neither ``"lab"`` nor ``"hsv"``.
    """
    if color_space not in ("lab", "hsv"):
        raise ValueError(f"Unsupported color space '{color_space}'")
    if crop.size == 0 or crop.ndim != 3 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None

    hsv: npt.NDArray[np.uint8] | None = None
    if exclude_grass or color_space == "hsv":
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    if exclude_grass:
        assert hsv is not None
        low = np.asarray(_DEFAULT_PREPROCESS.grass_hsv_low, dtype=np.uint8)
        high = np.asarray(_DEFAULT_PREPROCESS.grass_hsv_high, dtype=np.uint8)
        grass_mask = cv2.inRange(hsv, low, high) > 0
        keep_mask = ~grass_mask
    else:
        keep_mask = np.ones(crop.shape[:2], dtype=bool)

    if not np.any(keep_mask):
        return None

    if color_space == "lab":
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        pixels = lab[keep_mask].astype(np.float64)
        return np.median(pixels, axis=0)

    assert hsv is not None
    hsv_pixels = hsv[keep_mask].astype(np.float64)
    angles = (2.0 * np.pi / 180.0) * hsv_pixels[:, 0]
    encoded = np.column_stack(
        [
            127.5 * np.cos(angles),
            127.5 * np.sin(angles),
            hsv_pixels[:, 1],
            hsv_pixels[:, 2],
        ]
    )
    return np.median(encoded, axis=0)
