"""Tests for the projection of tracks onto the pitch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_vision.analytics.projection import project_tracks_to_pitch
from football_vision.artifacts import HOMOGRAPHY_QUALITY_COLUMNS, HomographySequence


def make_sequence(frame_ids: list[int], valid: list[int]) -> HomographySequence:
    """Builds a sequence whose homography scales pixels by 0.1 to meters."""
    h = np.diag([0.1, 0.1, 1.0])
    quality = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "valid": valid,
            "n_inliers": 4,
            "reproj_err_m": 0.0,
            "overlay_score": np.nan,
        }
    )
    return HomographySequence(
        np.repeat(h[None], len(frame_ids), axis=0),
        quality[list(HOMOGRAPHY_QUALITY_COLUMNS)],
    )


def make_tracks(foot_x: float, frame_id: int = 0) -> pd.DataFrame:
    """Creates one track row whose foot point is ``(foot_x, 300)``."""
    return pd.DataFrame(
        {
            "frame_id": [frame_id],
            "track_id": [1],
            "x1": [foot_x - 20],
            "y1": [200.0],
            "x2": [foot_x + 20],
            "y2": [300.0],
            "conf": [0.9],
            "team": [0],
            "role": ["player"],
        }
    )


def test_projection_uses_the_foot_point() -> None:
    """Checks that the bottom-center of the box is projected."""
    out = project_tracks_to_pitch(make_tracks(120.0), make_sequence([0], [1]))
    assert out.loc[0, "X"] == pytest.approx(12.0)
    assert out.loc[0, "Y"] == pytest.approx(30.0)
    assert out.loc[0, "valid"] == 1


def test_points_far_outside_the_pitch_are_invalid() -> None:
    """Checks that projections beyond the margin are flagged invalid."""
    out = project_tracks_to_pitch(
        make_tracks(2000.0), make_sequence([0], [1]), margin_m=3.0
    )
    assert out.loc[0, "valid"] == 0


def test_frames_without_valid_homography_are_invalid() -> None:
    """Checks invalid and missing homographies both invalidate the rows."""
    flagged = project_tracks_to_pitch(make_tracks(120.0), make_sequence([0], [0]))
    missing = project_tracks_to_pitch(
        make_tracks(120.0, frame_id=7), make_sequence([0], [1])
    )
    assert flagged.loc[0, "valid"] == 0 and np.isnan(flagged.loc[0, "X"])
    assert missing.loc[0, "valid"] == 0
