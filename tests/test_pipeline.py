"""Integration tests of the stages and the runner on a synthetic video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from football_vision.artifacts import (
    ANALYTICS,
    DETECTIONS,
    HOMOGRAPHY_QUALITY_COLUMNS,
    META,
    OUTPUTS,
    TEAM_LABEL_COLUMNS,
    TEAM_LABELS,
    TRACK_COLUMNS,
    TRACKS,
    HomographySequence,
    save_landmarks_json,
)
from football_vision.config import PipelineConfig, apply_overrides
from football_vision.datatypes import Detections, Frame
from football_vision.detection.base import Detector
from football_vision.errors import PipelineError
from football_vision.eval.ablation import Ablation, run_ablation
from football_vision.homography.geometry import apply_homography
from football_vision.pipeline.runner import PipelineRunner
from football_vision.pitch.template import PitchTemplate
from football_vision.tracking.base import Tracker
from tests.helpers import N_FRAMES


class FakeDetector(Detector):
    """Detector that returns one fixed person box in every frame."""

    def detect(self, frame: Frame) -> Detections:
        """Returns a single person box."""
        return Detections(
            np.array([[20, 20, 40, 70]], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            np.array([0], dtype=np.int64),
        )


class FakeTracker(Tracker):
    """Tracker that gives every detection of a frame its index as track id."""

    def update(
        self,
        detections: Detections,
        frame_id: int,
        homography: np.ndarray | None = None,
    ) -> Detections:
        """Returns the detections with ids 1..N."""
        return detections.with_track_ids(np.arange(1, len(detections) + 1))

    def reset(self) -> None:
        """Does nothing."""


def frames_in_video(path: Path) -> int:
    """Counts the frames of a video file."""
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def test_walking_skeleton_end_to_end_and_caching(base_cfg: PipelineConfig) -> None:
    """Checks video -> detections -> overlay video, then caching and forcing."""
    runner = PipelineRunner(base_cfg, services={"detector": FakeDetector()})
    results = runner.run(
        ["render", "preprocess", "detect"]
    )  # order of the request is irrelevant
    assert [(r.name, r.status) for r in results] == [
        ("preprocess", "ran"),
        ("detect", "ran"),
        ("render", "ran"),
    ]
    store = runner.ctx.store
    table = store.read_table(DETECTIONS)
    assert sorted(table["frame_id"].unique()) == list(range(N_FRAMES))
    assert frames_in_video(store.path(OUTPUTS) / "overlay.mp4") == N_FRAMES
    assert (store.root / "config.yaml").is_file() and (
        store.root / "timings.json"
    ).is_file()

    again = runner.run(["preprocess", "detect", "render"])
    assert {r.status for r in again} == {"skipped"}
    forced = runner.run(["preprocess", "detect", "render"], force=["detect"])
    assert [r.status for r in forced] == ["skipped", "ran", "skipped"]


def test_runner_reports_missing_prerequisites_and_unknown_stages(
    base_cfg: PipelineConfig,
) -> None:
    """Checks the errors for a stage without its inputs and for unknown names."""
    runner = PipelineRunner(base_cfg, services={"detector": FakeDetector()})
    with pytest.raises(PipelineError, match="needs artifacts"):
        runner.run(["detect"])
    with pytest.raises(PipelineError, match="Unknown stage"):
        runner.run(["fly"])


def test_missing_video_path_is_a_clear_error(tmp_path: Path) -> None:
    """Checks the message when video.path was not configured."""
    cfg = apply_overrides(PipelineConfig(), {"output_dir": str(tmp_path)})
    with pytest.raises(PipelineError, match="video.path"):
        PipelineRunner(cfg).run(["preprocess"])


def test_tracking_stage_glue_with_a_fake_tracker(
    base_cfg: PipelineConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks that the tracking stage writes tracks and drops short ones."""
    monkeypatch.setattr(
        "football_vision.pipeline.stages.make_tracker", lambda cfg, fps: FakeTracker()
    )
    runner = PipelineRunner(base_cfg, services={"detector": FakeDetector()})
    runner.run(["preprocess", "detect", "track"])
    tracks = runner.ctx.store.read_table(TRACKS)
    assert list(tracks.columns) == list(TRACK_COLUMNS)
    assert set(tracks["track_id"]) == {1} and len(tracks) == N_FRAMES


@pytest.mark.parametrize("tracker_name", ["sort", "bytetrack"])
def test_tracking_stage_with_real_c2_trackers(
    base_cfg: PipelineConfig, tracker_name: str
) -> None:
    """Checks real SortTracker and ByteTrackTracker inside TrackingStage."""
    cfg = apply_overrides(
        base_cfg,
        {
            "run_name": f"real_{tracker_name}",
            "tracking.tracker": tracker_name,
            "tracking.stitch": True,
        },
    )
    runner = PipelineRunner(cfg, services={"detector": FakeDetector()})
    runner.run(["preprocess", "detect", "track"])
    tracks = runner.ctx.store.read_table(TRACKS)
    assert list(tracks.columns) == list(TRACK_COLUMNS)
    assert set(tracks["track_id"]) == {1} and len(tracks) == N_FRAMES


def test_team_stage_colors_distinct_players(tmp_path: Path) -> None:
    """Checks TeamStage groups distinct jersey colors and updates TRACKS."""
    video_path = tmp_path / "team_clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 files in this environment")

    boxes = np.array(
        [
            [40, 40, 60, 110],
            [90, 40, 110, 110],
            [190, 40, 210, 110],
            [240, 40, 260, 110],
        ],
        dtype=np.float32,
    )
    colors = [(20, 20, 220), (20, 20, 220), (220, 40, 20), (220, 40, 20)]
    for _ in range(6):
        frame = np.full((180, 320, 3), (60, 140, 60), dtype=np.uint8)
        for box, bgr in zip(boxes, colors, strict=True):
            cv2.rectangle(
                frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), bgr, -1
            )
        writer.write(frame)
    writer.release()

    class _FourPlayerDetector(Detector):
        def detect(self, frame: Frame) -> Detections:
            """Returns four stationary person boxes."""
            return Detections(
                boxes,
                np.full(4, 0.95, dtype=np.float32),
                np.zeros(4, dtype=np.int64),
            )

    cfg = apply_overrides(
        PipelineConfig(),
        {
            "video.path": str(video_path),
            "video.size": [320, 180],
            "output_dir": str(tmp_path / "out"),
            "run_name": "team_c3",
            "tracking.tracker": "bytetrack",
            "tracking.min_track_len": 2,
            "team.n_clusters": 2,
            "team.color_space": "lab",
        },
    )
    runner = PipelineRunner(cfg, services={"detector": _FourPlayerDetector()})
    runner.run(["preprocess", "detect", "track", "team"])

    labels = runner.ctx.store.read_table(TEAM_LABELS)
    tracks = runner.ctx.store.read_table(TRACKS)
    assert list(labels.columns) == list(TEAM_LABEL_COLUMNS)
    assert len(labels) == 4
    assert set(labels["role"]) == {"player"}

    # Map each track to its average x center to match red (left) vs blue (right).
    centers = tracks.groupby("track_id")["x1"].mean().to_dict()
    red_tids = [tid for tid, x1 in centers.items() if x1 < 150.0]
    blue_tids = [tid for tid, x1 in centers.items() if x1 > 150.0]
    label_map = dict(zip(labels["track_id"], labels["team"], strict=True))
    assert len({label_map[t] for t in red_tids}) == 1
    assert len({label_map[t] for t in blue_tids}) == 1
    assert label_map[red_tids[0]] != label_map[blue_tids[0]]
    assert set(tracks["role"]) == {"player"}


def test_fixed_homography_stage_from_clicked_landmarks(
    base_cfg: PipelineConfig, tmp_path: Path
) -> None:
    """Checks the fixed-camera homography stage against a synthetic camera."""
    template = PitchTemplate()
    names = [
        "corner_tl",
        "corner_tr",
        "corner_br",
        "corner_bl",
        "center",
        "left_pa_outer_top",
    ]
    h_pitch_to_image = np.array(
        [[8.0, 1.5, 100.0], [0.5, 6.0, 50.0], [1e-4, 2e-4, 1.0]]
    )
    image = apply_homography(h_pitch_to_image, template.landmark_array(names))
    landmarks_path = save_landmarks_json(
        {n: tuple(p) for n, p in zip(names, image, strict=True)}, tmp_path / "lm.json"
    )
    cfg = apply_overrides(base_cfg, {"homography.landmarks_path": str(landmarks_path)})
    runner = PipelineRunner(cfg)
    runner.run(["preprocess", "homography"])
    sequence = runner.ctx.store.read_homographies()
    assert (
        len(sequence.matrices) == N_FRAMES
        and sequence.is_valid(0)
        and sequence.is_valid(N_FRAMES - 1)
    )
    np.testing.assert_allclose(
        apply_homography(sequence.matrix_for(3), image),
        template.landmark_array(names),
        atol=1e-4,
    )
    assert sequence.quality["reproj_err_m"].max() < 1e-4


def test_projection_analytics_and_side_by_side_rendering(
    base_cfg: PipelineConfig,
) -> None:
    """Checks projection, analytics and rendering from hand-made inputs."""
    runner = PipelineRunner(base_cfg)
    runner.run(["preprocess"])
    store = runner.ctx.store
    rows = []
    for frame_id in range(N_FRAMES):
        for track_id, (team, x1) in enumerate(
            [(0, 10 + 5 * frame_id), (1, 100 - 4 * frame_id)], start=1
        ):
            rows.append(
                (frame_id, track_id, x1, 20.0, x1 + 20.0, 70.0, 0.9, team, "player")
            )
    store.write_table(TRACKS, pd.DataFrame(rows, columns=list(TRACK_COLUMNS)))
    quality = pd.DataFrame(
        {
            "frame_id": range(N_FRAMES),
            "valid": 1,
            "n_inliers": 4,
            "reproj_err_m": 0.0,
            "overlay_score": np.nan,
        }
    )
    h = np.diag([105 / 160, 68 / 90, 1.0])
    store.write_homographies(
        HomographySequence(
            np.repeat(h[None], N_FRAMES, axis=0),
            quality[list(HOMOGRAPHY_QUALITY_COLUMNS)],
        )
    )
    runner.run(["project", "analytics", "render"])
    pitch_xy = store.read_table("pitch_xy.csv")
    assert len(pitch_xy) == 2 * N_FRAMES and pitch_xy["valid"].all()
    analytics = store.path(ANALYTICS)
    assert (analytics / "kinematics.csv").is_file() and (
        analytics / "heatmap_team_0.npy"
    ).is_file()
    assert (analytics / "heatmap_team_1.npy").is_file() and (
        analytics / "heatmap_all.npy"
    ).is_file()
    assert np.load(analytics / "heatmap_all.npy").shape == (68, 105)
    assert frames_in_video(store.path(OUTPUTS) / "side_by_side.mp4") == N_FRAMES


def test_ablation_runner_copies_upstream_artifacts_and_collects_metrics(
    base_cfg: PipelineConfig,
) -> None:
    """Checks that each variant re-runs its stage in its own folder."""
    PipelineRunner(base_cfg).run(["preprocess"])
    ablation = Ablation(
        "T1",
        "Does the denoise setting change the frame count?",
        ("preprocess",),
        ("n_frames",),
        {
            "none": {"preprocess.denoise": "none"},
            "gaussian": {"preprocess.denoise": "gaussian"},
        },
    )

    def evaluate(cfg: PipelineConfig, run_dir: Path) -> dict[str, float]:
        from football_vision.artifacts import ArtifactStore

        return {"n_frames": float(ArtifactStore(run_dir).read_meta().n_frames)}

    table = run_ablation(
        ablation,
        base_cfg,
        evaluate,
        output_csv=base_cfg.run_dir.parent / "ablation_T1.csv",
    )
    assert table["variant"].tolist() == ["none", "gaussian"]
    assert table["n_frames"].tolist() == [float(N_FRAMES)] * 2
    assert (base_cfg.run_dir.parent / "test_T1_gaussian" / META).is_file()
    assert (base_cfg.run_dir.parent / "ablation_T1.csv").is_file()


def test_fixed_homography_stage_marks_degenerate_landmarks_invalid(
    base_cfg: PipelineConfig, tmp_path: Path
) -> None:
    """Checks that HomographyStage records invalid flag for degenerate landmarks."""
    bad_landmarks = {
        "left_pa_outer_bottom": (434.0, 499.0),
        "left_ga_outer_top": (720.0, 245.0),
        "left_ga_goalline_top": (752.0, 185.0),
        "left_penalty_spot": (699.0, 316.0),
    }
    landmarks_path = save_landmarks_json(bad_landmarks, tmp_path / "bad_lm.json")
    cfg = apply_overrides(
        base_cfg,
        {
            "run_name": "bad_hom",
            "homography.landmarks_path": str(landmarks_path),
        },
    )
    runner = PipelineRunner(cfg)
    runner.run(["preprocess", "homography"])
    sequence = runner.ctx.store.read_homographies()
    assert len(sequence.matrices) == N_FRAMES
    assert not sequence.is_valid(0)


def test_preprocess_pitch_lines_detection_tiling_and_stage_errors(
    base_cfg: PipelineConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers mask preprocess, pitch_lines, tiled detect, pitch track, and errors."""
    cfg = apply_overrides(
        base_cfg,
        {
            "run_name": "full_opts",
            "preprocess.detect_shots": True,
            "preprocess.compute_pitch_mask": True,
            "preprocess.denoise": "gaussian",
            "preprocess.use_clahe": True,
            "detection.use_pitch_mask": True,
            "detection.use_tiling": True,
            "tracking.space": "pitch",
        },
    )
    monkeypatch.setattr(
        "football_vision.pipeline.stages.YoloDetector", lambda _cfg: FakeDetector()
    )
    runner = PipelineRunner(cfg)
    runner.run(["preprocess", "pitch_lines", "detect"])
    store = runner.ctx.store
    assert store.exists("pitch_lines.csv")
    assert store.path("pitch_mask").is_dir()

    with pytest.raises(PipelineError, match="homography.npy"):
        runner.run(["track"])

    # Run ablation copying pitch_mask subdirectory and with output_csv=None
    ab = Ablation("T2", "No-op pitch_lines", ("pitch_lines",), ("rows",), {"a": {}})
    table = run_ablation(ab, cfg, lambda _c, d: {"rows": float(len(d.name))})
    assert len(table) == 1
    assert (cfg.run_dir.parent / "full_opts_T2_a" / "pitch_mask").is_dir()

    # Render without detections raises clear PipelineError
    cfg_nodet = apply_overrides(base_cfg, {"run_name": "nodet"})
    r2 = PipelineRunner(cfg_nodet)
    r2.run(["preprocess"])
    with pytest.raises(PipelineError, match="detect"):
        r2.run(["render"])


def test_propagate_homography_and_pitch_x_team_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers _run_propagate (alignment, re-anchor, smoothing) and pitch-X TeamStage."""
    video_path = tmp_path / "prop.mp4"
    size = (320, 180)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 files in this environment")
    rng = np.random.default_rng(42)
    base = np.full((180, 320, 3), (50, 130, 50), dtype=np.uint8)
    for _ in range(30):
        cx, cy = int(rng.integers(20, 290)), int(rng.integers(20, 150))
        cv2.rectangle(base, (cx, cy), (cx + 10, cy + 10), (240, 240, 240), -1)
    # Two players: GK near left goal (x=30) and ref midfield (x=200)
    for i in range(6):
        shift = np.float32([[1, 0, i], [0, 1, 0]])
        frame = cv2.warpAffine(base, shift, size)
        cv2.rectangle(frame, (25, 60), (45, 130), (220, 40, 20), -1)
        cv2.rectangle(frame, (190, 60), (210, 130), (20, 220, 220), -1)
        writer.write(frame)
    writer.release()

    template = PitchTemplate()
    names = ["corner_tl", "corner_tr", "corner_br", "corner_bl", "center"]
    h_p2i = np.array([[2.5, 0.2, 10.0], [0.1, 2.2, 10.0], [0.0, 0.0, 1.0]])
    img_pts = apply_homography(h_p2i, template.landmark_array(names))
    lm_path = save_landmarks_json(
        {n: tuple(p) for n, p in zip(names, img_pts, strict=True)},
        tmp_path / "lm_prop.json",
    )

    class _TwoDet(Detector):
        def detect(self, frame: Frame) -> Detections:
            return Detections(
                np.array([[25, 60, 45, 130], [190, 60, 210, 130]], dtype=np.float32),
                np.array([0.95, 0.95], dtype=np.float32),
                np.zeros(2, dtype=np.int64),
            )

    cfg = apply_overrides(
        PipelineConfig(),
        {
            "video.path": str(video_path),
            "video.size": list(size),
            "output_dir": str(tmp_path / "out"),
            "run_name": "prop_run",
            "homography.mode": "propagate",
            "homography.feature": "orb",
            "homography.min_inliers": 6,
            "homography.landmarks_path": str(lm_path),
            "homography.reanchor_overlay_threshold": 0.99,
            "homography.reanchor_every_n": 2,
            "homography.smoothing_window": 3,
            "tracking.space": "pitch",
            "tracking.min_track_len": 2,
        },
    )
    monkeypatch.setattr(
        "football_vision.pipeline.stages.estimate_keyframe_homography_auto",
        lambda *_a, **_kw: np.linalg.inv(h_p2i),
    )
    runner = PipelineRunner(cfg, services={"detector": _TwoDet()})
    runner.run(["preprocess", "homography", "detect", "track", "team", "render"])
    seq = runner.ctx.store.read_homographies()
    assert len(seq.matrices) == 6
    assert runner.ctx.store.exists(TEAM_LABELS)
    assert runner.ctx.store.exists(OUTPUTS)
