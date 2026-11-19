"""Tests for Phase C3 team and role classification modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_vision.artifacts import TEAM_LABEL_COLUMNS
from football_vision.config import TeamConfig
from football_vision.constants import (
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    ROLE_UNKNOWN,
    TEAM_UNKNOWN,
)
from football_vision.eval.team import team_accuracy
from football_vision.team.classifier import (
    TeamClassifier,
    assign_teams_and_roles,
    vote_per_track,
)
from football_vision.team.color_features import crop_torso, extract_color_feature


def _solid_bgr_patch(bgr: tuple[int, int, int], size: int = 32) -> np.ndarray:
    """Builds a solid BGR image patch."""
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[:, :] = bgr
    return patch


def test_crop_torso_and_extract_lab_and_hsv_features() -> None:
    """Verifies torso cropping, Lab/HSV feature shapes, and grass filtering."""
    frame = _solid_bgr_patch((35, 160, 35), size=100)  # grass-green background
    # Red jersey drawn inside torso band of [20, 10, 60, 90] (height 80 -> y=22..54)
    frame[22:55, 28:52] = (20, 20, 220)

    crop = crop_torso(frame, [20, 10, 60, 90])
    assert crop.shape[0] > 0 and crop.shape[1] > 0

    lab_feat = extract_color_feature(crop, color_space="lab", exclude_grass=True)
    hsv_feat = extract_color_feature(crop, color_space="hsv", exclude_grass=True)
    assert lab_feat is not None and lab_feat.shape == (3,)
    assert hsv_feat is not None and hsv_feat.shape == (4,)

    # Red near OpenCV HSV H=0/179 should have positive cos(2*pi*H/180)
    assert hsv_feat[0] > 0.8


def test_extract_color_feature_returns_none_for_empty_or_all_grass() -> None:
    """Returns None when crop is empty or consists only of pitch grass pixels."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert extract_color_feature(empty) is None

    grass = _solid_bgr_patch((35, 160, 35), size=24)
    assert extract_color_feature(grass, exclude_grass=True) is None
    assert extract_color_feature(grass, exclude_grass=False) is not None


def test_extract_color_feature_rejects_unknown_color_space() -> None:
    """Raises ValueError for unsupported color spaces."""
    patch = _solid_bgr_patch((20, 20, 220))
    with pytest.raises(ValueError, match="Unsupported color space"):
        extract_color_feature(patch, color_space="rgb")  # type: ignore[arg-type]


def test_team_classifier_clusters_synthetic_kits() -> None:
    """Separates two team kits and a third referee kit cleanly in Lab space."""
    red_feat = extract_color_feature(_solid_bgr_patch((20, 20, 220)), "lab", False)
    blue_feat = extract_color_feature(_solid_bgr_patch((220, 40, 20)), "lab", False)
    yellow_feat = extract_color_feature(_solid_bgr_patch((20, 220, 220)), "lab", False)
    assert red_feat is not None and blue_feat is not None and yellow_feat is not None

    rng = np.random.default_rng(7)
    red_samples = red_feat + rng.normal(0.0, 1.5, size=(12, 3))
    blue_samples = blue_feat + rng.normal(0.0, 1.5, size=(12, 3))
    yellow_samples = yellow_feat + rng.normal(0.0, 1.5, size=(6, 3))

    features = np.vstack([red_samples, blue_samples, yellow_samples])
    clf = TeamClassifier(TeamConfig(n_clusters=3, color_space="lab"))
    clf.fit(features)
    preds = clf.predict(features)

    assert len(np.unique(preds[:12])) == 1
    assert len(np.unique(preds[12:24])) == 1
    assert len(np.unique(preds[24:])) == 1
    assert len({int(preds[0]), int(preds[12]), int(preds[24])}) == 3


def test_team_classifier_handles_small_samples_and_errors() -> None:
    """Clamps cluster count on tiny inputs and validates fit/shape contracts."""
    clf = TeamClassifier(TeamConfig(n_clusters=3))
    with pytest.raises(RuntimeError, match="must be fitted"):
        clf.predict(np.zeros((2, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="empty"):
        clf.fit(np.zeros((0, 3), dtype=np.float64))

    identical = np.ones((4, 3), dtype=np.float64) * 50.0
    clf.fit(identical)
    preds = clf.predict(identical)
    assert np.array_equal(preds, np.zeros(4, dtype=np.int64))


def test_vote_per_track_and_assign_teams_and_roles() -> None:
    """Checks majority voting, team mapping, GK/referee outlier roles, and eval."""
    samples = pd.DataFrame(
        [
            {"track_id": 1, "cluster": 0},
            {"track_id": 1, "cluster": 0},
            {"track_id": 1, "cluster": 1},  # outlier frame
            {"track_id": 2, "cluster": 0},
            {"track_id": 2, "cluster": 0},
            {"track_id": 3, "cluster": 1},
            {"track_id": 3, "cluster": 1},
            {"track_id": 4, "cluster": 1},
            {"track_id": 4, "cluster": 1},
            {"track_id": 5, "cluster": 2},  # extreme left -> goalkeeper
            {"track_id": 6, "cluster": 2},  # central -> referee
        ]
    )
    voted = vote_per_track(samples)
    assert dict(zip(voted["track_id"], voted["cluster"], strict=True)) == {
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
    }

    # Build tracks with distinct x positions: players middle, track 5 far left edge
    track_rows = [
        (1, 400.0, 200.0, 430.0, 280.0),
        (2, 500.0, 200.0, 530.0, 280.0),
        (3, 650.0, 200.0, 680.0, 280.0),
        (4, 750.0, 200.0, 780.0, 280.0),
        (5, 40.0, 220.0, 70.0, 300.0),
        (6, 590.0, 210.0, 620.0, 290.0),
        (7, 550.0, 210.0, 580.0, 290.0),
    ]
    tracks = pd.DataFrame(
        [
            {"frame_id": 0, "track_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            for tid, x1, y1, x2, y2 in track_rows
        ]
    )
    labels = assign_teams_and_roles(voted, tracks, TeamConfig(n_clusters=3))
    assert list(labels.columns) == list(TEAM_LABEL_COLUMNS)

    by_id = labels.set_index("track_id")
    assert by_id.loc[1, "team"] == by_id.loc[2, "team"]
    assert by_id.loc[3, "team"] == by_id.loc[4, "team"]
    assert {int(by_id.loc[1, "team"]), int(by_id.loc[3, "team"])} == {0, 1}
    assert by_id.loc[1, "role"] == ROLE_PLAYER
    assert by_id.loc[3, "role"] == ROLE_PLAYER
    assert by_id.loc[5, "role"] == ROLE_GOALKEEPER
    assert by_id.loc[5, "team"] == TEAM_UNKNOWN
    assert by_id.loc[6, "role"] == ROLE_REFEREE
    assert by_id.loc[6, "team"] == TEAM_UNKNOWN
    assert by_id.loc[7, "role"] == ROLE_UNKNOWN
    assert by_id.loc[7, "team"] == TEAM_UNKNOWN

    pred_teams = by_id.loc[[1, 2, 3, 4], "team"].to_numpy()
    true_teams = np.array([1, 1, 0, 0])  # deliberately flipped IDs to test matching
    assert team_accuracy(pred_teams, true_teams) == pytest.approx(1.0)


def test_vote_and_assign_handle_empty_tables() -> None:
    """Handles empty sample and track DataFrames gracefully."""
    empty_samples = pd.DataFrame(columns=["track_id", "cluster"])
    voted = vote_per_track(empty_samples)
    assert voted.empty

    empty_tracks = pd.DataFrame(
        columns=["frame_id", "track_id", "x1", "y1", "x2", "y2"]
    )
    labels = assign_teams_and_roles(voted, empty_tracks, TeamConfig())
    assert list(labels.columns) == list(TEAM_LABEL_COLUMNS)
    assert labels.empty
