"""Tracklet post-processing: removing short tracks and stitching fragments."""

from __future__ import annotations

import pandas as pd


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

    Raises:
        NotImplementedError: Stub; to be implemented by member C.
    """
    # TODO(C): can also operate in pitch coordinates once homographies are available.
    raise NotImplementedError("TODO(C): implement stitch_tracklets")
