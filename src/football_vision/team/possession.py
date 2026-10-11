"""Ball track, possession and passes (stage P6, optional Level 4).

These steps depend on ball detection, which is the least reliable part of the
pipeline. Treat them as stretch goals and report their limitations.
"""

from __future__ import annotations

import pandas as pd


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
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): the ball is in the air at times; positions are image positions here.
    raise NotImplementedError("TODO(C): implement select_ball_track")


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
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): nearest neighbor per frame, then temporal hysteresis.
    raise NotImplementedError("TODO(C): implement assign_possession")


def detect_passes(possession: pd.DataFrame) -> pd.DataFrame:
    """Detects passes as possession changes between players of the same team.

    Args:
        possession: Output of :func:`assign_possession`.

    Returns:
        Table with the columns ``frame_id``, ``from_track``, ``to_track`` and ``team``.

    Raises:
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): a pass is a holder change inside one team; a turnover changes team.
    raise NotImplementedError("TODO(C): implement detect_passes")


def build_passing_network(passes: pd.DataFrame, team: int) -> pd.DataFrame:
    """Aggregates passes of one team into a weighted directed graph.

    Args:
        passes: Output of :func:`detect_passes`.
        team: Team id to build the network for.

    Returns:
        Edge table with the columns ``from_track``, ``to_track`` and ``count``.

    Raises:
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): groupby(['from_track', 'to_track']).size(); plot with networkx
    #   if desired.
    raise NotImplementedError("TODO(C): implement build_passing_network")
