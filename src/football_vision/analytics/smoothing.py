"""Smoothing of pitch trajectories before differentiation (stage P7).

Differentiating noisy positions amplifies the noise: with independent position noise
of standard deviation ``sigma`` and time step ``dt``, the speed noise per axis is
``sqrt(2) * sigma / dt`` (about 10 m/s for 0.3 m at 25 fps). Smooth first.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import savgol_filter


def split_into_segments(frame_ids: npt.ArrayLike, max_gap: int) -> list[slice]:
    """Splits a sorted frame-id sequence wherever the gap is too large.

    Args:
        frame_ids: Increasing frame ids of one track.
        max_gap: Largest allowed difference between consecutive frame ids.

    Returns:
        Slices into ``frame_ids``, one per continuous segment.
    """
    ids = np.asarray(frame_ids, dtype=np.int64)
    if len(ids) == 0:
        return []
    breaks = np.flatnonzero(np.diff(ids) > max_gap) + 1
    edges = [0, *breaks.tolist(), len(ids)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def smooth_positions(
    xy: npt.ArrayLike, window: int, polyorder: int = 2
) -> npt.NDArray[np.float64]:
    """Smooths one continuous trajectory with a Savitzky-Golay filter.

    The window is shrunk to the longest odd value that fits; trajectories that are
    too short for the polynomial order are returned unchanged.

    Args:
        xy: Positions of shape ``(N, 2)`` sampled uniformly in time.
        window: Window length in samples (``<= 0`` disables smoothing).
        polyorder: Polynomial order of the local fit.

    Returns:
        Smoothed positions of shape ``(N, 2)``.
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if window <= 0:
        return pts
    n = len(pts)
    usable = min(window, n if n % 2 == 1 else n - 1)
    if usable % 2 == 0:
        usable -= 1
    if usable <= polyorder:
        return pts
    return savgol_filter(pts, usable, polyorder, axis=0, mode="interp")


def smooth_pitch_tracks(
    pitch_xy: pd.DataFrame,
    window: int,
    polyorder: int = 2,
    max_gap_frames: int = 5,
) -> pd.DataFrame:
    """Smooths the pitch trajectory of every track segment by segment.

    Only valid rows are smoothed, and a track is split into segments wherever
    consecutive samples are more than ``max_gap_frames`` apart, so smoothing never
    bridges an occlusion.

    Args:
        pitch_xy: ``pitch_xy.csv`` table.
        window: Savitzky-Golay window in samples (``<= 0`` disables smoothing).
        polyorder: Polynomial order of the local fit.
        max_gap_frames: Largest gap inside one continuous segment.

    Returns:
        A copy of the table with smoothed ``X`` and ``Y`` columns.
    """
    out = pitch_xy.copy()
    valid_rows = out[out["valid"] == 1].sort_values(["track_id", "frame_id"])
    for _, group in valid_rows.groupby("track_id"):
        frame_ids = group["frame_id"].to_numpy()
        coords = group[["X", "Y"]].to_numpy(dtype=np.float64)
        smoothed = coords.copy()
        for seg in split_into_segments(frame_ids, max_gap_frames):
            smoothed[seg] = smooth_positions(coords[seg], window, polyorder)
        out.loc[group.index, ["X", "Y"]] = smoothed
    return out
