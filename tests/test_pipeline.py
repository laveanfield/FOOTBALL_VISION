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
