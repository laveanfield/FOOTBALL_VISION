"""Tracklet post-processing: removing short tracks and stitching fragments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_vision.constants import TEAM_UNKNOWN
from football_vision.tracking.association import solve_assignment


def drop_short_tracks(tracks: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Removes tracks that appear in fewer than ``min_length`` frames.

    Very short tracks are usually false positives or fragments of ID switches.

    Args:
        tracks: Table with at least the columns ``track_id`` and ``frame_id``.
        min_length: Minimum number of rows a track must have.

    Returns:
        The filtered table with a fresh index.
    """
    counts = tracks.groupby("track_id")["frame_id"].transform("size")
    return tracks[counts >= min_length].reset_index(drop=True)


@dataclass(frozen=True)
class _TrackletSummary:
    """Endpoints of one contiguous or fragmented track id.

    Attributes:
        track_id: Track identifier.
        start_frame: First frame id where the track appears.
        end_frame: Last frame id where the track appears.
        start_foot: Bottom-center ``(x, y)`` of the first box.
        end_foot: Bottom-center ``(x, y)`` of the last box.
        team: Majority non-unknown team id, or ``TEAM_UNKNOWN``.
    """

    track_id: int
    start_frame: int
    end_frame: int
    start_foot: tuple[float, float]
    end_foot: tuple[float, float]
    team: int


def _foot_of_row(row: pd.Series) -> tuple[float, float]:
    """Computes bottom-center ``(x, y2)`` of a track table row."""
    return (0.5 * (float(row["x1"]) + float(row["x2"])), float(row["y2"]))


def _summarize_tracklets(tracks: pd.DataFrame) -> list[_TrackletSummary]:
    """Extracts temporal bounds, endpoint foot coordinates, and known team."""
    has_team = "team" in tracks.columns
    summaries: list[_TrackletSummary] = []
    for tid, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values("frame_id")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        team_val = TEAM_UNKNOWN
        if has_team:
            known = ordered["team"][ordered["team"] != TEAM_UNKNOWN]
            if len(known) > 0:
                team_val = int(known.mode().iloc[0])
        summaries.append(
            _TrackletSummary(
                track_id=int(tid),
                start_frame=int(first["frame_id"]),
                end_frame=int(last["frame_id"]),
                start_foot=_foot_of_row(first),
                end_foot=_foot_of_row(last),
                team=team_val,
            )
        )
    return summaries


def stitch_tracklets(
    tracks: pd.DataFrame,
    max_gap_frames: int = 30,
    max_distance_px: float = 80.0,
) -> pd.DataFrame:
    """Joins fragments of the same player that received different track ids.

    Treat each ``track_id`` as a tracklet with a start and an end. Link the end of
    tracklet A to the start of tracklet B when they do not overlap in time, the gap
    is at most ``max_gap_frames``, and the last box of A is close to the first box of
    B (the gate may grow with the gap). Solve the linking as an assignment problem,
    then relabel every linked chain with one id. Same-team and appearance cues make
    the linking more reliable.

    Args:
        tracks: Table with the columns of ``TRACK_COLUMNS``.
        max_gap_frames: Longest temporal gap that may be bridged.
        max_distance_px: Largest distance between the end and start boxes.

    Returns:
        The table with merged track ids; the row count is unchanged.
    """
    if len(tracks) == 0 or tracks["track_id"].nunique() <= 1:
        return tracks.copy()

    summaries = _summarize_tracklets(tracks)
    n = len(summaries)
    cost = np.full((n, n), np.inf, dtype=np.float64)
    scale = max(float(max_distance_px), 1e-6)

    for i, prev in enumerate(summaries):
        for j, nxt in enumerate(summaries):
            if i == j:
                continue
            gap = nxt.start_frame - prev.end_frame
            if gap < 1 or gap > max_gap_frames:
                continue
            if (
                prev.team != TEAM_UNKNOWN
                and nxt.team != TEAM_UNKNOWN
                and prev.team != nxt.team
            ):
                continue
            dist = float(
                np.hypot(
                    nxt.start_foot[0] - prev.end_foot[0],
                    nxt.start_foot[1] - prev.end_foot[1],
                )
            )
            if dist <= max_distance_px:
                cost[i, j] = dist / scale

    matches, _, _ = solve_assignment(cost, max_cost=1.0)
    if not matches:
        return tracks.copy()

    parent_idx = {nxt: prev for prev, nxt in matches}
    relabel: dict[int, int] = {}
    for idx, summary in enumerate(summaries):
        root = idx
        seen = {root}
        while root in parent_idx and parent_idx[root] not in seen:
            root = parent_idx[root]
            seen.add(root)
        relabel[summary.track_id] = summaries[root].track_id

    out = tracks.copy()
    out["track_id"] = out["track_id"].map(relabel).astype(out["track_id"].dtype)
    return out
