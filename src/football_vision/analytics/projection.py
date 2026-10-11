"""Projection of tracked players onto the pitch plane (stage P7)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_vision.artifacts import PITCH_XY_COLUMNS, HomographySequence
from football_vision.boxes import foot_points
from football_vision.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_vision.homography.geometry import apply_homography


def project_points(
    homography: npt.NDArray[np.float64], points: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Maps image points to pitch coordinates.

    The homography is only exact for points on the ground plane, so pass foot points
    and never box centers.

    Args:
        homography: ``(3, 3)`` matrix mapping image to pitch.
        points: Array of shape ``(N, 2)`` in pixels.

    Returns:
        Array of shape ``(N, 2)`` with ``(X, Y)`` in meters.
    """
    return apply_homography(homography, points)


def project_tracks_to_pitch(
    tracks: pd.DataFrame,
    homographies: HomographySequence,
    margin_m: float = 3.0,
) -> pd.DataFrame:
    """Projects the foot point of every tracked box onto the pitch.

    A row is marked invalid when its frame has no valid homography, when the
    projection is not finite, or when the point lies more than ``margin_m`` outside
    the pitch.

    Args:
        tracks: ``tracks.csv`` table.
        homographies: Per-frame image-to-pitch homographies.
        margin_m: Tolerated distance outside the pitch rectangle, in meters.

    Returns:
        Table with the columns of ``PITCH_XY_COLUMNS``, one row per track row.
    """
    n = len(tracks)
    feet = foot_points(tracks[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64))
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for frame_id, positions in tracks.groupby("frame_id").indices.items():
        if not homographies.is_valid(int(frame_id)):
            continue
        projected = project_points(
            homographies.matrix_for(int(frame_id)), feet[positions]
        )
        xs[positions], ys[positions] = projected[:, 0], projected[:, 1]
        valid[positions] = np.isfinite(projected).all(axis=1)
    inside = (
        (xs >= -margin_m)
        & (xs <= PITCH_LENGTH_M + margin_m)
        & (ys >= -margin_m)
        & (ys <= PITCH_WIDTH_M + margin_m)
    )
    out = tracks[["frame_id", "track_id", "team", "role"]].copy()
    out["X"], out["Y"] = xs, ys
    out["valid"] = (valid & inside).astype(int)
    return out[list(PITCH_XY_COLUMNS)].reset_index(drop=True)
