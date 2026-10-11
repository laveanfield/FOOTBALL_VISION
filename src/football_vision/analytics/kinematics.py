"""Distance, speed and speed zones of tracked players (stage P7).

Distances and speeds are *estimates on the visible part of the pitch*: a player who
leaves the camera view produces a fragmented track, so totals only cover the time
the player was seen.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import savgol_filter

from football_vision.config import AnalyticsConfig


def compute_speed(
    xy: npt.ArrayLike,
    frame_ids: npt.ArrayLike,
    fps: float,
    smooth_window: int = 0,
    polyorder: int = 2,
) -> npt.NDArray[np.float64]:
    """Estimates the speed at every sample of a trajectory.

    Without smoothing the speed is the finite-difference derivative of the positions
    (``numpy.gradient`` on the real timestamps). With ``smooth_window > 0`` the
    derivative of a local Savitzky-Golay polynomial is used instead, which assumes
    roughly uniform sampling.

    Args:
        xy: Positions of shape ``(N, 2)`` in meters.
        frame_ids: Strictly increasing frame ids of shape ``(N,)``.
        fps: Frame rate of the video.
        smooth_window: Savitzky-Golay window in samples (odd; ``0`` disables it).
        polyorder: Polynomial order of the Savitzky-Golay fit.

    Returns:
        Speeds of shape ``(N,)`` in meters per second.

    Raises:
        ValueError: If the frame ids are not strictly increasing.
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    times = np.asarray(frame_ids, dtype=np.float64) / fps
    if len(pts) < 2:
        return np.zeros(len(pts))
    if np.any(np.diff(times) <= 0):
        raise ValueError("frame_ids must be strictly increasing")
    if smooth_window > 0 and len(pts) >= smooth_window and smooth_window > polyorder:
        delta = float(np.median(np.diff(times)))
        velocity = savgol_filter(
            pts, smooth_window, polyorder, deriv=1, delta=delta, axis=0, mode="interp"
        )
    else:
        velocity = np.gradient(pts, times, axis=0)
    return np.linalg.norm(velocity, axis=1)


def distance_covered(
    xy: npt.ArrayLike,
    frame_ids: npt.ArrayLike | None = None,
    max_gap_frames: int | None = None,
) -> float:
    """Sums the step lengths of a trajectory.

    Args:
        xy: Positions of shape ``(N, 2)`` in meters.
        frame_ids: Optional frame ids of shape ``(N,)``.
        max_gap_frames: With ``frame_ids``, steps across a larger gap are ignored,
            because a jump over an occlusion is not distance run.

    Returns:
        The covered distance in meters.
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if frame_ids is not None and max_gap_frames is not None:
        gaps = np.diff(np.asarray(frame_ids, dtype=np.int64))
        steps = steps[gaps <= max_gap_frames]
    return float(steps.sum())


def classify_speed_zones(
    speed_mps: npt.ArrayLike, edges_kmh: tuple[float, ...]
) -> npt.NDArray[np.int64]:
    """Assigns every speed to a speed zone.

    With the default edges ``(7, 14, 20, 25)`` the zones are walking, jogging,
    running, high-speed running and sprinting. Thresholds vary between sources.

    Args:
        speed_mps: Speeds in meters per second.
        edges_kmh: Increasing upper bounds of the zones in km/h.

    Returns:
        Zone indices in ``0..len(edges_kmh)``; a speed equal to an edge belongs to
        the faster zone.
    """
    speeds_kmh = np.asarray(speed_mps, dtype=np.float64) * 3.6
    return np.digitize(speeds_kmh, np.asarray(edges_kmh)).astype(np.int64)


def flag_implausible_speed(
    speed_mps: npt.ArrayLike, max_speed_mps: float
) -> npt.NDArray[np.bool_]:
    """Flags speeds that exceed what a player can run.

    Such samples usually come from ID switches or a jumping homography.

    Args:
        speed_mps: Speeds in meters per second.
        max_speed_mps: Plausibility limit (about 12 m/s).

    Returns:
        Boolean array that is ``True`` where the speed exceeds the limit.
    """
    return np.asarray(speed_mps, dtype=np.float64) > max_speed_mps


def summarize_track_kinematics(
    pitch_xy: pd.DataFrame, fps: float, cfg: AnalyticsConfig
) -> pd.DataFrame:
    """Summarizes distance and speed per track.

    The table is assumed to be smoothed already (see ``analytics.smoothing``), so the
    speed is a plain finite difference of the stored positions. Distance and time
    only accumulate over steps shorter than ``cfg.max_gap_frames``.

    Args:
        pitch_xy: ``pitch_xy.csv`` table.
        fps: Frame rate of the video.
        cfg: Analytics settings (gap limit, speed limit, zone edges).

    Returns:
        One row per track with ``track_id``, ``team``, ``n_samples``,
        ``visible_time_s``, ``distance_m``, ``mean_speed_mps``, ``p95_speed_mps``,
        ``max_speed_mps``, ``outlier_fraction`` and ``zone_time_s_<k>`` columns.
    """
    n_zones = len(cfg.speed_zone_edges_kmh) + 1
    records = []
    valid = pitch_xy[pitch_xy["valid"] == 1].sort_values(["track_id", "frame_id"])
    for track_id, group in valid.groupby("track_id"):
        frame_ids = group["frame_id"].to_numpy(dtype=np.int64)
        xy = group[["X", "Y"]].to_numpy(dtype=np.float64)
        speed = (
            compute_speed(xy, frame_ids, fps)
            if len(group) >= 2
            else np.zeros(len(group))
        )
        gaps = np.diff(frame_ids)
        ok = gaps <= cfg.max_gap_frames
        step_time = gaps[ok] / fps
        zones = classify_speed_zones(speed, cfg.speed_zone_edges_kmh)
        zone_time = np.bincount(zones[:-1][ok], weights=step_time, minlength=n_zones)
        record = {
            "track_id": int(track_id),
            "team": int(group["team"].mode().iloc[0]),
            "n_samples": len(group),
            "visible_time_s": float(step_time.sum()),
            "distance_m": distance_covered(xy, frame_ids, cfg.max_gap_frames),
            "mean_speed_mps": float(speed.mean()),
            "p95_speed_mps": float(np.percentile(speed, 95)),
            "max_speed_mps": float(speed.max()),
            "outlier_fraction": float(
                flag_implausible_speed(speed, cfg.max_speed_mps).mean()
            ),
        }
        record.update({f"zone_time_s_{k}": float(zone_time[k]) for k in range(n_zones)})
        records.append(record)
    columns = [
        "track_id",
        "team",
        "n_samples",
        "visible_time_s",
        "distance_m",
        "mean_speed_mps",
        "p95_speed_mps",
        "max_speed_mps",
        "outlier_fraction",
        *[f"zone_time_s_{k}" for k in range(n_zones)],
    ]
    return pd.DataFrame(records, columns=columns)
