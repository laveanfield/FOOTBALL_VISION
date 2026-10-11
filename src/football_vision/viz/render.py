"""Composition of video frames for the final demo (stage P8)."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from football_vision.datatypes import Frame


def compose_side_by_side(left: Frame, right: Frame) -> Frame:
    """Places two images next to each other, matching their heights.

    The right image is resized to the height of the left one, keeping its aspect
    ratio. This produces the "original video | top-down map" comparison.

    Args:
        left: BGR image shown on the left.
        right: BGR image shown on the right.

    Returns:
        The combined BGR image.
    """
    height = left.shape[0]
    width = max(int(round(right.shape[1] * height / right.shape[0])), 1)
    resized = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
    return np.hstack([left, resized])


def make_filmstrip(frames: Sequence[Frame], n_cols: int, tile_width: int) -> Frame:
    """Tiles frames sampled over time into one image for the report.

    Args:
        frames: BGR frames in temporal order, all with the same size.
        n_cols: Number of tiles per row.
        tile_width: Width of one tile in pixels; the height keeps the aspect ratio.

    Returns:
        A single BGR image containing the tiled frames.

    Raises:
        NotImplementedError: Stub; to be implemented by member D.
    """
    # TODO(D): resize each frame, optionally stamp its timestamp, then
    #   vstack/hstack the rows.
    raise NotImplementedError("TODO(D): implement make_filmstrip")
