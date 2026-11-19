"""Focused edge-case tests closing remaining branch coverage gaps across modules."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from football_vision.analytics.kinematics import compute_speed, distance_covered
from football_vision.analytics.smoothing import smooth_positions
from football_vision.artifacts import DETECTIONS, TRACK_COLUMNS, ArtifactStore
from football_vision.config import TrackingConfig, apply_overrides, load_config
from football_vision.constants import ROLE_GOALKEEPER, ROLE_REFEREE, TEAM_UNKNOWN
from football_vision.datasets.sportsmot import (
    frames_to_video,
    read_mot_gt,
    read_seqinfo,
)
from football_vision.datatypes import Detections
from football_vision.detection.base import Detector
from football_vision.detection.tiling import merge_tile_detections
from football_vision.errors import ConfigError, SchemaError
from football_vision.eval.detection import _compute_ap_101, evaluate_detections
from football_vision.eval.homography import evaluate_homography_sequence
from football_vision.eval.speed import per_sample_speeds, stationary_distance_per_minute
from football_vision.eval.tracking import evaluate_tracking
from football_vision.homography.geometry import invert_homography
from football_vision.pitch.edges import extract_white_lines_tophat
from football_vision.pitch.template import PitchTemplate
from football_vision.team.classifier import (
    _classify_outlier_role,
    assign_teams_and_roles,
)
from football_vision.team.possession import (
    assign_possession,
    build_passing_network,
    detect_passes,
    select_ball_track,
)
from football_vision.tracking.association import solve_assignment
from football_vision.tracking.bytetrack import ByteTrackTracker
from football_vision.video import VideoReader, VideoWriter
from football_vision.viz.overlay import color_for, draw_pitch_template
from football_vision.viz.render import make_filmstrip
from football_vision.viz.topdown import TrailBuffer


def test_config_artifacts_datatypes_and_video_validation_edges(tmp_path: Path) -> None:
    """Covers config errors, ArtifactStore checks, Detections, and Video I/O."""
    bad_yaml = tmp_path / "list.yaml"
    bad_yaml.write_text("- item1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML mapping"):
        load_config(bad_yaml)

    str_path_cfg = tmp_path / "empty.yaml"
    str_path_cfg.write_text("", encoding="utf-8")
    assert load_config(str_path_cfg).run_name == "debug"

    cfg = load_config()
    with pytest.raises(ConfigError, match="Conflicting override"):
        apply_overrides(cfg, {"video": 1, "video.stride": 2})
    with pytest.raises(ConfigError, match="must be a list"):
        apply_overrides(cfg, {"video.size": 1280})
    with pytest.raises(ConfigError, match="must be a list"):
        apply_overrides(cfg, {"stages": "preprocess"})
    with pytest.raises(ConfigError, match="Section"):
        apply_overrides(cfg, {"video": "not_a_mapping"})
    assert (
        apply_overrides(cfg, {"analytics.pitch_margin_m": 4}).analytics.pitch_margin_m
        == 4.0
    )

    with pytest.raises(ValueError, match="class_ids"):
        Detections(
            np.zeros((1, 4), np.float32),
            np.ones((1,), np.float32),
            np.zeros((2,), np.int64),
        )
    with pytest.raises(ValueError, match="track_ids"):
        Detections(
            np.zeros((1, 4), np.float32),
            np.ones((1,), np.float32),
            np.zeros((1,), np.int64),
            track_ids=np.zeros((2,), np.int64),
        )

    store = ArtifactStore(tmp_path / "store")
    bad_csv = store.path(DETECTIONS)
    bad_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"wrong_col": [1]}).to_csv(bad_csv, index=False)
    with pytest.raises(SchemaError, match="missing columns"):
        store.read_table(DETECTIONS)

    vid_path = tmp_path / "short.mp4"
    with VideoWriter(vid_path, 10.0, (64, 48)) as writer:
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    with VideoReader(vid_path, stride=3) as reader:
        frames = list(reader)
        assert len(frames) == 1
    with pytest.raises(ValueError, match="stride"):
        with VideoReader(vid_path, stride=0):
            pass
    with pytest.raises(OSError, match="Cannot open video writer"):
        VideoWriter(tmp_path / "bad.avi", 10.0, (64, 48), codec="ZZZZ")


def test_detection_eval_and_tracking_eval_edge_cases() -> None:
    """Covers Detector.detect_batch, merge errors, COCO AP, and tracking eval guards."""

    class _DummyDetector(Detector):
        def detect(self, frame: np.ndarray) -> Detections:
            return Detections.empty()

    batch = _DummyDetector().detect_batch(
        [np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)]
    )
    assert len(batch) == 2

    with pytest.raises(ValueError, match="mismatch"):
        merge_tile_detections([Detections.empty()], [])
    with pytest.raises(ValueError, match="iou_threshold"):
        merge_tile_detections([], [], iou_threshold=1.5)

    assert _compute_ap_101(np.array([]), np.array([])) == 0.0
    empty_df = pd.DataFrame(columns=["frame_id", "x1", "y1", "x2", "y2", "conf", "cls"])
    assert evaluate_detections(empty_df, empty_df)["map50"] == 0.0
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_detections(empty_df, empty_df, iou_threshold=0.0)

    gt_det = pd.DataFrame(
        [{"frame_id": 0, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "cls": 0}]
    )
    preds_extra = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "conf": 0.9,
                "cls": 0,
            },
            {
                "frame_id": 0,
                "x1": 1.0,
                "y1": 1.0,
                "x2": 11.0,
                "y2": 11.0,
                "conf": 0.8,
                "cls": 0,
            },
            {
                "frame_id": 1,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "conf": 0.7,
                "cls": 0,
            },
        ]
    )
    res_det = evaluate_detections(preds_extra, gt_det)
    assert 0.0 < res_det["map50"] <= 1.0

    # Tracking evaluation with unmatched frames & classes
    empty_tracks = pd.DataFrame(columns=list(TRACK_COLUMNS))
    gt_tracks = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "track_id": 1,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
            },
            {
                "frame_id": 1,
                "track_id": 1,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
            },
        ]
    )
    assert evaluate_tracking(empty_tracks, gt_tracks)["hota"] == 0.0
    bad_preds = pd.DataFrame(
        [
            {
                "frame_id": 1,
                "track_id": 2,
                "x1": 200.0,
                "y1": 200.0,
                "x2": 210.0,
                "y2": 210.0,
            }
        ]
    )
    assert evaluate_tracking(bad_preds, gt_tracks)["hota"] == 0.0
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_tracking(empty_tracks, gt_tracks, iou_threshold=1.5)


def test_team_possession_tracking_and_sportsmot_edge_cases(tmp_path: Path) -> None:
    """Covers role heuristics, possession guards, association errors, and SportsMOT."""
    assert _classify_outlier_role(1, pd.DataFrame(), 0.0, 0.0, 2) == ROLE_REFEREE
    tracks = pd.DataFrame(
        [{"frame_id": 0, "track_id": 2, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 20.0}]
        * 5
    )
    assert _classify_outlier_role(99, tracks, 0.0, 100.0, 5) == ROLE_REFEREE
    assert (
        _classify_outlier_role(2, tracks, 0.0, 100.0, 5, pitch_x_by_track={2: 100.0})
        == ROLE_GOALKEEPER
    )
    assert (
        _classify_outlier_role(2, tracks, 0.0, 100.0, 5, pitch_x_by_track={2: 50.0})
        == ROLE_REFEREE
    )
    voted = pd.DataFrame({"track_id": [1], "cluster": [0]})
    tiny_tracks = pd.DataFrame(
        [{"frame_id": 0, "track_id": 1, "x1": 5.0, "y1": 0.0, "x2": 5.0, "y2": 10.0}]
    )
    assert (
        assign_teams_and_roles(voted, tiny_tracks, team_cfg := load_config().team).loc[
            0, "role"
        ]
        == "player"
    )
    del team_cfg

    # Possession error and empty-input branches
    with pytest.raises(ValueError, match="radius_m"):
        assign_possession(pd.DataFrame(), pd.DataFrame(), radius_m=0.0)
    with pytest.raises(ValueError, match="min_frames"):
        assign_possession(pd.DataFrame(), pd.DataFrame(), min_frames=0)
    assert assign_possession(pd.DataFrame(), pd.DataFrame()).empty
    no_ball = select_ball_track(
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "x1": 0,
                    "y1": 0,
                    "x2": 10,
                    "y2": 10,
                    "conf": 0.9,
                    "cls": 0,
                }
            ]
        )
    )
    assert no_ball.empty
    assert detect_passes(pd.DataFrame()).empty
    assert build_passing_network(pd.DataFrame(), 0).empty

    # Association validation & empty ByteTrack frame
    with pytest.raises(ValueError, match="2D"):
        solve_assignment(np.zeros((2, 2, 2)), max_cost=1.0)
    with pytest.raises(ValueError, match="NaN"):
        solve_assignment(np.array([[np.nan]]), max_cost=1.0)
    with pytest.raises(ValueError, match="-inf"):
        solve_assignment(np.array([[-np.inf]]), max_cost=1.0)
    m, ur, uc = solve_assignment(np.array([[50.0]]), max_cost=1.0)
    assert m == [] and ur == [0] and uc == [0]
    bt = ByteTrackTracker(TrackingConfig(tracker="bytetrack"), fps=25.0)
    assert len(bt.update(Detections.empty(), 0)) == 0

    # SportsMOT missing ini / missing image / resize / short mot_gt
    with pytest.raises(FileNotFoundError):
        read_seqinfo(tmp_path / "missing_seq")
    seq_dir = tmp_path / "seq"
    (seq_dir / "img1").mkdir(parents=True)
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nframeRate=25\nimWidth=32\nimHeight=24\nseqLength=1\n",
        encoding="utf-8",
    )
    info = read_seqinfo(seq_dir)
    with pytest.raises(FileNotFoundError):
        frames_to_video(info, tmp_path / "out.mp4")
    cv2.imwrite(str(seq_dir / "img1" / "000001.jpg"), np.zeros((16, 16, 3), np.uint8))
    assert frames_to_video(info, tmp_path / "out.mp4").is_file()
    mot_short = tmp_path / "mot.txt"
    mot_short.write_text("1,1,10,10,20,40\n", encoding="utf-8")
    assert len(read_mot_gt(mot_short).columns) == 9


def test_analytics_pitch_eval_and_viz_edge_cases() -> None:
    """Covers speed/kinematics, homography eval, tophat empty mask, and viz helpers."""
    np.testing.assert_array_equal(
        compute_speed(np.array([[0.0, 0.0]]), [0], 25.0), [0.0]
    )
    assert distance_covered(np.array([[0.0, 0.0]])) == 0.0
    pts4 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert smooth_positions(pts4, window=4, polyorder=2).shape == (4, 2)

    empty_pxy = pd.DataFrame(columns=["frame_id", "track_id", "X", "Y", "valid"])
    assert len(per_sample_speeds(empty_pxy, 25.0, 5)) == 0
    zero_span = pd.DataFrame(
        [{"frame_id": 0, "track_id": 1, "X": 10.0, "Y": 10.0, "valid": 1}] * 30
    )
    assert np.isnan(stationary_distance_per_minute(zero_span, 25.0, 5, min_samples=10))

    from football_vision.artifacts import HOMOGRAPHY_QUALITY_COLUMNS, HomographySequence

    seq = HomographySequence(
        np.eye(3)[None],
        pd.DataFrame([[0, 1, 4, 0.0, 1.0]], columns=list(HOMOGRAPHY_QUALITY_COLUMNS)),
    )
    assert evaluate_homography_sequence(seq, {0: {"unknown_lm": (1.0, 2.0)}}).empty

    gray = np.zeros((32, 32), dtype=np.uint8)
    assert (
        extract_white_lines_tophat(
            gray, mask=np.zeros((32, 32), np.uint8), kernel_size=15
        ).max()
        == 0
    )

    assert color_for(TEAM_UNKNOWN, ROLE_REFEREE) == (0, 220, 255)
    assert color_for(TEAM_UNKNOWN, ROLE_GOALKEEPER) == (60, 220, 60)
    h_inv_horizon = np.array(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [-1.0 / 105.0, 0.0, 1.0]]
    )
    horizon_overlay = draw_pitch_template(
        np.zeros((60, 80, 3), np.uint8),
        invert_homography(h_inv_horizon),
        PitchTemplate(),
    )
    assert horizon_overlay.shape == (60, 80, 3)

    with pytest.raises(ValueError, match="labels"):
        make_filmstrip([np.zeros((20, 20, 3), np.uint8)], 1, 20, labels=["a", "b"])
    tb = TrailBuffer(length=2)
    tb.update(0, pd.DataFrame([{"track_id": 1, "X": 10.0, "Y": 10.0, "valid": 1}]))
    tb.update(5, None)
    assert tb.trails() == {}
