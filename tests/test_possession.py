"""Tests for Phase C5 ball tracking, possession, passes, and networks."""

from __future__ import annotations

import pandas as pd
import pytest

from football_vision.artifacts import BALL_COLUMNS
from football_vision.constants import (
    COCO_PERSON_ID,
    COCO_SPORTS_BALL_ID,
    ROLE_PLAYER,
    ROLE_REFEREE,
    TEAM_UNKNOWN,
)
from football_vision.team.possession import (
    assign_possession,
    build_passing_network,
    detect_passes,
    select_ball_track,
)


def test_select_ball_track_filters_jumps_and_interpolates() -> None:
    """Keeps top confidence per frame, drops outliers, and interpolates gaps."""
    dets = pd.DataFrame(
        [
            # Two detections on frame 0 -> keep higher conf around (100, 100)
            {
                "frame_id": 0,
                "x1": 96.0,
                "y1": 96.0,
                "x2": 104.0,
                "y2": 104.0,
                "conf": 0.9,
                "cls": COCO_SPORTS_BALL_ID,
            },
            {
                "frame_id": 0,
                "x1": 196.0,
                "y1": 196.0,
                "x2": 204.0,
                "y2": 204.0,
                "conf": 0.3,
                "cls": COCO_SPORTS_BALL_ID,
            },
            # Person box on frame 0 must be ignored when cls column is present.
            {
                "frame_id": 0,
                "x1": 50.0,
                "y1": 50.0,
                "x2": 80.0,
                "y2": 120.0,
                "conf": 0.99,
                "cls": COCO_PERSON_ID,
            },
            # Implausible jump outlier at frame 1 -> dropped and then interpolated.
            {
                "frame_id": 1,
                "x1": 496.0,
                "y1": 496.0,
                "x2": 504.0,
                "y2": 504.0,
                "conf": 0.8,
                "cls": COCO_SPORTS_BALL_ID,
            },
            # Valid ball at frame 2 -> (120, 110)
            {
                "frame_id": 2,
                "x1": 116.0,
                "y1": 106.0,
                "x2": 124.0,
                "y2": 114.0,
                "conf": 0.85,
                "cls": COCO_SPORTS_BALL_ID,
            },
            # After gap > max_gap_frames (=2), ball reappears elsewhere -> re-anchor.
            {
                "frame_id": 8,
                "x1": 600.0,
                "y1": 300.0,
                "x2": 608.0,
                "y2": 308.0,
                "conf": 0.75,
                "cls": COCO_SPORTS_BALL_ID,
            },
        ]
    )
    ball = select_ball_track(dets, max_gap_frames=2, max_jump_px=150.0)
    assert list(ball.columns) == list(BALL_COLUMNS)
    assert ball["frame_id"].tolist() == [0, 1, 2, 8]
    assert ball["interpolated"].tolist() == [False, True, False, False]
    assert ball.loc[ball["frame_id"] == 1, "x"].iloc[0] == pytest.approx(110.0)
    assert ball.loc[ball["frame_id"] == 1, "y"].iloc[0] == pytest.approx(105.0)
    assert ball.loc[ball["frame_id"] == 8, "x"].iloc[0] == pytest.approx(604.0)

    empty = select_ball_track(dets.iloc[0:0])
    assert list(empty.columns) == list(BALL_COLUMNS) and empty.empty


def test_select_ball_track_validates_params() -> None:
    """Rejects negative gaps and non-positive jump distances."""
    empty = pd.DataFrame(columns=["frame_id", "x1", "y1", "x2", "y2", "conf", "cls"])
    with pytest.raises(ValueError, match="max_gap_frames"):
        select_ball_track(empty, max_gap_frames=-1)
    with pytest.raises(ValueError, match="max_jump_px"):
        select_ball_track(empty, max_jump_px=0.0)


def test_assign_possession_detect_passes_and_network() -> None:
    """Covers hysteresis, referee exclusion, transit gaps, turnovers, and edges."""
    # Frames:
    # 0..3: track 1 (team 0) holds ball
    # 4: brief 1-frame brush past track 99 (team 1) -> must NOT count (min_frames=3)
    # 5: ball in empty space (> radius_m) -> unknown (-1)
    # 6..9: track 2 (team 0) holds ball -> pass 1 -> 2 (team 0) at frame 6
    # 10..12: track 10 (team 1) intercepts -> turnover (NOT a pass)
    # 13..15: track 11 (team 1) receives -> pass 10 -> 11 (team 1) at frame 13
    player_rows = []
    ball_rows = []
    for f in range(16):
        # Referee stands directly on the ball in all frames -> must be ignored.
        bx = 20.0 + f
        if f == 5:
            bx = 90.0  # nobody near ball on frame 5
        ball_rows.append({"frame_id": f, "X": bx, "Y": 30.0})
        player_rows.append(
            {
                "frame_id": f,
                "track_id": 900,
                "X": bx,
                "Y": 30.0,
                "team": TEAM_UNKNOWN,
                "role": ROLE_REFEREE,
                "valid": 1,
            }
        )
        # Track 1 close in 0..3
        player_rows.append(
            {
                "frame_id": f,
                "track_id": 1,
                "X": (20.5 + f) if f <= 3 else 5.0,
                "Y": 30.0,
                "team": 0,
                "role": ROLE_PLAYER,
                "valid": 1,
            }
        )
        # Track 99 close only at frame 4 (flicker)
        player_rows.append(
            {
                "frame_id": f,
                "track_id": 99,
                "X": 24.5 if f == 4 else 80.0,
                "Y": 30.0,
                "team": 1,
                "role": ROLE_PLAYER,
                "valid": 1,
            }
        )
        # Track 2 close in 6..9
        player_rows.append(
            {
                "frame_id": f,
                "track_id": 2,
                "X": (20.5 + f) if 6 <= f <= 9 else 10.0,
                "Y": 30.0,
                "team": 0,
                "role": ROLE_PLAYER,
                "valid": 1,
            }
        )
        # Track 10 close in 10..12
        player_rows.append(
            {
                "frame_id": f,
                "track_id": 10,
                "X": (20.5 + f) if 10 <= f <= 12 else 100.0,
                "Y": 30.0,
                "team": 1,
                "role": ROLE_PLAYER,
                "valid": 1,
            }
        )
        # Track 11 close in 13..15
        player_rows.append(
            {
                "frame_id": f,
                "track_id": 11,
                "X": (20.5 + f) if 13 <= f <= 15 else 102.0,
                "Y": 30.0,
                "team": 1,
                "role": ROLE_PLAYER,
                "valid": 1,
            }
        )

    pitch_xy = pd.DataFrame(player_rows)
    ball_xy = pd.DataFrame(ball_rows)
    poss = assign_possession(pitch_xy, ball_xy, radius_m=2.0, min_frames=3)

    by_f = poss.set_index("frame_id")
    assert all(by_f.loc[0:3, "track_id"] == 1)
    assert by_f.loc[4, "track_id"] == -1
    assert by_f.loc[5, "track_id"] == -1
    assert all(by_f.loc[6:9, "track_id"] == 2)
    assert all(by_f.loc[10:12, "track_id"] == 10)
    assert all(by_f.loc[13:15, "track_id"] == 11)

    passes = detect_passes(poss)
    assert len(passes) == 2
    assert passes.iloc[0].to_dict() == {
        "frame_id": 6,
        "from_track": 1,
        "to_track": 2,
        "team": 0,
    }
    assert passes.iloc[1].to_dict() == {
        "frame_id": 13,
        "from_track": 10,
        "to_track": 11,
        "team": 1,
    }

    net_t0 = build_passing_network(passes, team=0)
    assert net_t0.to_dict(orient="records") == [
        {"from_track": 1, "to_track": 2, "count": 1}
    ]
    assert build_passing_network(passes, team=2).empty


def test_select_ball_track_scales_jump_gate_across_short_gap() -> None:
    """Allows steady ball speed across an occluded multi-frame gap."""
    dets = pd.DataFrame(
        [
            {"frame_id": 0, "x": 100.0, "y": 100.0, "conf": 0.9, "cls": 32},
            {"frame_id": 4, "x": 300.0, "y": 100.0, "conf": 0.9, "cls": 32},
            {"frame_id": 5, "x": 350.0, "y": 100.0, "conf": 0.9, "cls": 32},
        ]
    )
    ball = select_ball_track(dets, max_gap_frames=10, max_jump_px=150.0)
    assert ball["frame_id"].tolist() == [0, 1, 2, 3, 4, 5]
    assert ball["interpolated"].tolist() == [False, True, True, True, False, False]


def test_detect_passes_resets_across_unknown_team_holder() -> None:
    """Does not fabricate a same-team pass across an unknown-team ball holder."""
    poss = pd.DataFrame(
        [
            {"frame_id": 0, "track_id": 1, "team": 0},
            {"frame_id": 1, "track_id": 50, "team": TEAM_UNKNOWN},
            {"frame_id": 2, "track_id": 2, "team": 0},
        ]
    )
    assert detect_passes(poss).empty
