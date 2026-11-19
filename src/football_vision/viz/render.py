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


def make_filmstrip(
    frames: Sequence[Frame],
    n_cols: int,
    tile_width: int,
    labels: Sequence[str] | None = None,
) -> Frame:
    """Tiles frames sampled over time into one image for the report.

    The last row is padded with black tiles when ``len(frames)`` is not a multiple
    of ``n_cols``.

    Args:
        frames: BGR frames in temporal order, all with the same size.
        n_cols: Number of tiles per row.
        tile_width: Width of one tile in pixels; the height keeps the aspect ratio.
        labels: Optional text stamped in the top-left corner of each tile, for
            example the timestamp ``"t=12.4s"``.

    Returns:
        A single BGR image containing the tiled frames.

    Raises:
        ValueError: If there are no frames, the sizes differ, ``n_cols`` or
            ``tile_width`` is not positive, or ``labels`` has the wrong length.
    """
    if not frames:
        raise ValueError("make_filmstrip needs at least one frame")
    if n_cols < 1 or tile_width < 1:
        raise ValueError("n_cols and tile_width must be positive")
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("all frames must have the same size")
    if labels is not None and len(labels) != len(frames):
        raise ValueError("labels must have one entry per frame")
    height, width = frames[0].shape[:2]
    tile_height = max(int(round(height * tile_width / width)), 1)
    tiles = []
    for i, frame in enumerate(frames):
        tile = cv2.resize(
            frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA
        )
        if labels is not None:
            scale = max(tile_width / 640.0, 0.35)
            org = (6, int(22 * scale) + 4)
            for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(
                    tile,
                    labels[i],
                    org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
        tiles.append(tile)
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % n_cols)
    rows = [np.hstack(tiles[r : r + n_cols]) for r in range(0, len(tiles), n_cols)]
    return np.vstack(rows)
