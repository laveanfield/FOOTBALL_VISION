"""Jersey-color features of tracked players (stage P6)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from football_vision.datatypes import Frame


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
    ``exclude_grass`` is set, and return a robust summary (median color, optionally
    concatenated with a coarse histogram). Lab distances are closer to perceived
    color differences than RGB distances.

    Args:
        crop: BGR torso crop from :func:`crop_torso`.
        color_space: Color space of the feature.
        exclude_grass: Whether to ignore pixels inside the grass HSV range.

    Returns:
        A feature vector of shape ``(D,)``, or ``None`` if no pixel remains.

    Raises:
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): reuse preprocessing.pitch_mask HSV bounds for the grass test.
    raise NotImplementedError("TODO(C): implement extract_color_feature")
