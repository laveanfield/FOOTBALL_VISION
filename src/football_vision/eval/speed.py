"""Sanity statistics for estimated player speeds (stage P9)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_vision.analytics.kinematics import compute_speed, distance_covered
from football_vision.analytics.smoothing import split_into_segments


def speed_sanity_report(
    speeds_mps: npt.ArrayLike, max_speed_mps: float = 12.0
) -> dict[str, float]:
    """Summarizes a speed sample and the share of implausible values.

    Elite players peak at about 10 m/s, so values above ~12 m/s point to ID switches
    or a jumping homography rather than real running.

    Args:
        speeds_mps: Speeds in meters per second.
        max_speed_mps: Plausibility limit.

    Returns:
        Mapping with ``n``, ``mean``, ``p50``, ``p95``, ``p99``, ``max`` and
        ``outlier_fraction`` (NaN statistics if the sample is empty).
    """
    values = np.asarray(speeds_mps, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        keys = ("mean", "p50", "p95", "p99", "max", "outlier_fraction")
        return {"n": 0.0, **dict.fromkeys(keys, np.nan)}
    return {
        "n": float(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "outlier_fraction": float((values > max_speed_mps).mean()),
    }


def _valid_pitch_tracks(pitch_xy: pd.DataFrame) -> pd.DataFrame:
    """Returns finite, valid rows of ``pitch_xy`` sorted by track and frame."""
    if len(pitch_xy) == 0:
        return pitch_xy
    finite = np.isfinite(pitch_xy["X"].to_numpy(dtype=np.float64)) & np.isfinite(
        pitch_xy["Y"].to_numpy(dtype=np.float64)
    )
    return pitch_xy[(pitch_xy["valid"] == 1) & finite].sort_values(
        ["track_id", "frame_id"]
    )


def per_sample_speeds(
    pitch_xy: pd.DataFrame, fps: float, max_gap_frames: int
) -> npt.NDArray[np.float64]:
    """Computes the speed of every valid sample, segment by segment.

    A track is split wherever two samples are more than ``max_gap_frames`` apart, so
    a jump across an occlusion is never turned into a speed.

    Args:
        pitch_xy: ``pitch_xy.csv`` table.
        fps: Frame rate of the video.
        max_gap_frames: Largest gap inside one continuous segment.

    Returns:
        Speeds in meters per second, concatenated over all segments.
    """
    valid = _valid_pitch_tracks(pitch_xy)
    speeds: list[npt.NDArray[np.float64]] = []
    for _, group in valid.groupby("track_id"):
        frame_ids = group["frame_id"].to_numpy(dtype=np.int64)
        xy = group[["X", "Y"]].to_numpy(dtype=np.float64)
        for seg in split_into_segments(frame_ids, max_gap_frames):
            if seg.stop - seg.start >= 2:
                speeds.append(compute_speed(xy[seg], frame_ids[seg], fps))
    return np.concatenate(speeds) if speeds else np.zeros(0)


def stationary_distance_per_minute(
    pitch_xy: pd.DataFrame,
    fps: float,
    max_gap_frames: int,
    radius_m: float = 2.0,
    min_samples: int = 25,
) -> float:
    """Measures the apparent distance covered by players who barely move.

    A track counts as quasi-static when 95 % of its samples stay within
    ``radius_m`` of its median position (a goalkeeper or a referee waiting for a
    restart). Their true distance is close to zero, so whatever distance is measured
    comes from position noise; this is the cleanest ground-truth-free signal for
    ablation A7. The heuristic depends on such tracks existing in the clip.

    Args:
        pitch_xy: ``pitch_xy.csv`` table.
        fps: Frame rate of the video.
        max_gap_frames: Largest gap inside one continuous segment.
        radius_m: Spread below which a track counts as quasi-static.
        min_samples: Shorter tracks are ignored.

    Returns:
        Mean distance in meters per 60 s of visibility over the quasi-static
        tracks, or NaN if the clip has none.
    """
    valid = _valid_pitch_tracks(pitch_xy)
    rates = []
    for _, group in valid.groupby("track_id"):
        if len(group) < min_samples:
            continue
        xy = group[["X", "Y"]].to_numpy(dtype=np.float64)
        spread = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
        if np.percentile(spread, 95) > radius_m:
            continue
        frame_ids = group["frame_id"].to_numpy(dtype=np.int64)
        gaps = np.diff(frame_ids)
        visible_s = float(gaps[gaps <= max_gap_frames].sum()) / fps
        if visible_s <= 0:
            continue
        distance = distance_covered(xy, frame_ids, max_gap_frames)
        rates.append(distance * 60.0 / visible_s)
    return float(np.mean(rates)) if rates else float("nan")
