"""Ball track, possession and passes (stage P6, optional Level 4).

These steps depend on ball detection, which is the least reliable part of the
pipeline. Treat them as stretch goals and report their limitations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_vision.artifacts import BALL_COLUMNS
from football_vision.constants import (
    COCO_SPORTS_BALL_ID,
    ROLE_REFEREE,
    TEAM_UNKNOWN,
)

POSSESSION_COLUMNS = ("frame_id", "track_id", "team")
PASS_COLUMNS = ("frame_id", "from_track", "to_track", "team")
NETWORK_COLUMNS = ("from_track", "to_track", "count")


def select_ball_track(
    detections: pd.DataFrame,
    max_gap_frames: int = 10,
    max_jump_px: float = 150.0,
) -> pd.DataFrame:
    """Builds one ball position per frame from raw ball detections.

    Keep the best-scoring ball detection per frame, reject jumps larger than
    ``max_jump_px`` between consecutive frames, and linearly interpolate gaps of at
    most ``max_gap_frames`` (marking those rows as interpolated).

    Args:
        detections: ``detections.csv`` table, filtered to the ball class.
        max_gap_frames: Longest gap that is interpolated.
        max_jump_px: Largest plausible displacement between consecutive frames.

    Returns:
        Table with the columns of ``BALL_COLUMNS``.

    Raises:
        ValueError: If ``max_gap_frames`` is negative or ``max_jump_px`` is not
            positive.
    """
    if max_gap_frames < 0:
        raise ValueError(f"max_gap_frames must be >= 0, got {max_gap_frames}")
    if max_jump_px <= 0.0:
        raise ValueError(f"max_jump_px must be > 0, got {max_jump_px}")

    if len(detections) == 0:
        return pd.DataFrame(columns=list(BALL_COLUMNS))

    df = detections.copy()
    if "cls" in df.columns:
        df = df[df["cls"] == COCO_SPORTS_BALL_ID]
    if len(df) == 0:
        return pd.DataFrame(columns=list(BALL_COLUMNS))

    if "x" not in df.columns or "y" not in df.columns:
        df["x"] = 0.5 * (df["x1"].astype(float) + df["x2"].astype(float))
        df["y"] = 0.5 * (df["y1"].astype(float) + df["y2"].astype(float))

    best = (
        df.sort_values("conf", ascending=False, kind="mergesort")
        .drop_duplicates(subset=["frame_id"], keep="first")
        .sort_values("frame_id", kind="mergesort")
    )

    accepted: list[tuple[int, float, float, float, bool]] = []
    for row in best.itertuples(index=False):
        fid = int(row.frame_id)
        bx, by, bconf = float(row.x), float(row.y), float(row.conf)
        if not accepted:
            accepted.append((fid, bx, by, bconf, False))
            continue
        prev_fid, prev_x, prev_y, _, _ = accepted[-1]
        span = fid - prev_fid
        gap = span - 1
        dist = float(np.hypot(bx - prev_x, by - prev_y))
        # Scale jump budget by frame span so steady ball motion over gaps passes.
        if gap <= max_gap_frames and dist > max_jump_px * span:
            continue
        accepted.append((fid, bx, by, bconf, False))

    rows: list[tuple[int, float, float, float, bool]] = []
    for idx, curr in enumerate(accepted):
        if idx > 0:
            prev_fid, prev_x, prev_y, _, _ = accepted[idx - 1]
            curr_fid, curr_x, curr_y, _, _ = curr
            span = curr_fid - prev_fid
            gap = span - 1
            if 1 <= gap <= max_gap_frames:
                for step in range(1, span):
                    alpha = float(step) / float(span)
                    ix = prev_x + alpha * (curr_x - prev_x)
                    iy = prev_y + alpha * (curr_y - prev_y)
                    rows.append((prev_fid + step, ix, iy, 0.0, True))
        rows.append(curr)

    out = pd.DataFrame(rows, columns=list(BALL_COLUMNS))
    out["frame_id"] = out["frame_id"].astype(np.int64)
    out["x"] = out["x"].astype(np.float64)
    out["y"] = out["y"].astype(np.float64)
    out["conf"] = out["conf"].astype(np.float64)
    out["interpolated"] = out["interpolated"].astype(bool)
    return out


def assign_possession(
    pitch_xy: pd.DataFrame,
    ball_xy: pd.DataFrame,
    radius_m: float = 2.5,
    min_frames: int = 5,
) -> pd.DataFrame:
    """Assigns the ball to the nearest player and derives team possession.

    A player holds the ball when it is the nearest player and closer than
    ``radius_m`` for at least ``min_frames`` consecutive frames (this avoids flicker).
    The ball in the air violates the ground-plane assumption, so its projected
    position is unreliable.

    Args:
        pitch_xy: ``pitch_xy.csv`` table.
        ball_xy: Ball positions in pitch coordinates with the columns
            ``frame_id``, ``X`` and ``Y``.
        radius_m: Maximum player-ball distance for possession.
        min_frames: Minimum number of consecutive frames to confirm possession.

    Returns:
        Table with the columns ``frame_id``, ``track_id`` and ``team`` of the holder
        (``-1`` when nobody holds the ball).

    Raises:
        ValueError: If ``radius_m`` is not positive or ``min_frames`` is smaller
            than 1.
    """
    if radius_m <= 0.0:
        raise ValueError(f"radius_m must be > 0, got {radius_m}")
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}")

    if len(pitch_xy) == 0 and len(ball_xy) == 0:
        return pd.DataFrame(columns=list(POSSESSION_COLUMNS))

    players = pitch_xy.copy()
    if "valid" in players.columns:
        players = players[players["valid"].astype(bool)]
    if "role" in players.columns:
        players = players[players["role"] != ROLE_REFEREE]
    players = players[np.isfinite(players["X"]) & np.isfinite(players["Y"])]

    balls = ball_xy[np.isfinite(ball_xy["X"]) & np.isfinite(ball_xy["Y"])]
    ball_by_frame = {
        int(row.frame_id): (float(row.X), float(row.Y))
        for row in balls.drop_duplicates("frame_id").itertuples(index=False)
    }
    player_by_frame = {int(f): g for f, g in players.groupby("frame_id", sort=True)}

    frame_ids = sorted(
        {int(f) for f in pitch_xy["frame_id"].unique()}
        | {int(f) for f in ball_xy["frame_id"].unique()}
    )

    raw: list[tuple[int, int, int]] = []
    for fid in frame_ids:
        ball_pos = ball_by_frame.get(fid)
        p_df = player_by_frame.get(fid)
        if ball_pos is None or p_df is None or len(p_df) == 0:
            raw.append((fid, -1, TEAM_UNKNOWN))
            continue
        bx, by = ball_pos
        px = p_df["X"].to_numpy(dtype=np.float64)
        py = p_df["Y"].to_numpy(dtype=np.float64)
        dists = np.hypot(px - bx, py - by)
        best_idx = int(np.argmin(dists))
        if dists[best_idx] <= radius_m:
            best_row = p_df.iloc[best_idx]
            raw.append((fid, int(best_row["track_id"]), int(best_row["team"])))
        else:
            raw.append((fid, -1, TEAM_UNKNOWN))

    # Apply temporal hysteresis: only contiguous runs >= min_frames confirm possession.
    confirmed = [(-1, TEAM_UNKNOWN)] * len(raw)
    idx = 0
    while idx < len(raw):
        _, tid, team = raw[idx]
        end = idx + 1
        while end < len(raw) and raw[end][1] == tid:
            end += 1
        if tid != -1 and (end - idx) >= min_frames:
            for pos in range(idx, end):
                confirmed[pos] = (tid, team)
        idx = end

    out = pd.DataFrame(
        [(raw[i][0], confirmed[i][0], confirmed[i][1]) for i in range(len(raw))],
        columns=list(POSSESSION_COLUMNS),
    )
    out["frame_id"] = out["frame_id"].astype(np.int64)
    out["track_id"] = out["track_id"].astype(np.int64)
    out["team"] = out["team"].astype(np.int64)
    return out


def detect_passes(possession: pd.DataFrame) -> pd.DataFrame:
    """Detects passes as possession changes between players of the same team.

    Args:
        possession: Output of :func:`assign_possession`.

    Returns:
        Table with the columns ``frame_id``, ``from_track``, ``to_track`` and ``team``.
    """
    if len(possession) == 0:
        return pd.DataFrame(columns=list(PASS_COLUMNS))

    ordered = possession.sort_values("frame_id", kind="mergesort")
    passes: list[tuple[int, int, int, int]] = []
    prev_tid: int | None = None
    prev_team: int | None = None

    for row in ordered.itertuples(index=False):
        tid = int(row.track_id)
        team = int(row.team)
        if tid < 0:
            continue
        if team == TEAM_UNKNOWN:
            prev_tid = None
            prev_team = None
            continue
        if prev_tid is not None and tid != prev_tid and team == prev_team:
            passes.append((int(row.frame_id), prev_tid, tid, team))
        prev_tid = tid
        prev_team = team

    out = pd.DataFrame(passes, columns=list(PASS_COLUMNS))
    for col in PASS_COLUMNS:
        out[col] = out[col].astype(np.int64)
    return out


def build_passing_network(passes: pd.DataFrame, team: int) -> pd.DataFrame:
    """Aggregates passes of one team into a weighted directed graph.

    Args:
        passes: Output of :func:`detect_passes`.
        team: Team id to build the network for.

    Returns:
        Edge table with the columns ``from_track``, ``to_track`` and ``count``.
    """
    if len(passes) == 0:
        return pd.DataFrame(columns=list(NETWORK_COLUMNS))

    subset = passes[passes["team"] == team]
    if len(subset) == 0:
        return pd.DataFrame(columns=list(NETWORK_COLUMNS))

    edges = (
        subset.groupby(["from_track", "to_track"], sort=True)
        .size()
        .reset_index(name="count")
    )
    for col in NETWORK_COLUMNS:
        edges[col] = edges[col].astype(np.int64)
    return edges
