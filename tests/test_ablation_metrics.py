"""Tests for the ablation metrics of ``scripts/run_ablation.py`` and the figure CLI.

Each test rebuilds a small run directory and reproduces one review finding (D1-D5).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from football_vision.artifacts import (
    DETECTIONS,
    PITCH_LINES,
    PITCH_XY,
    TEAM_LABELS,
    TRACKS,
    ArtifactStore,
    HomographySequence,
    VideoMeta,
)
from football_vision.config import load_config
from football_vision.pitch.template import PitchTemplate

ROOT = Path(__file__).resolve().parents[1]
CLIP_SIZE = (1280, 720)


def _load_script(name: str) -> ModuleType:
    """Imports a module from ``scripts/`` without making it a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while decorating
    spec.loader.exec_module(module)
    return module


ablation = _load_script("run_ablation")


def _store(tmp_path: Path, n_frames: int = 100, fps: float = 25.0) -> ArtifactStore:
    """Creates a run directory with metadata and stage timings."""
    store = ArtifactStore(tmp_path)
    store.write_meta(VideoMeta("clip.mp4", fps, *CLIP_SIZE, n_frames))
    timings = {"detect": {"seconds": 2.0}, "homography": {"seconds": 0.5}}
    (tmp_path / "timings.json").write_text(json.dumps(timings), encoding="utf-8")
    return store


def _quality(valid: list[int]) -> pd.DataFrame:
    """Quality table with the given validity flags."""
    n = len(valid)
    return pd.DataFrame(
        {
            "frame_id": np.arange(n),
            "valid": valid,
            "n_inliers": 10,
            "reproj_err_m": 0.3,
            "overlay_score": 0.8,
        }
    )


def _shift(dx: float) -> np.ndarray:
    """Image-to-pitch homography of a pure image translation by ``dx`` pixels."""
    return np.array([[1.0, 0.0, -dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_a1_reports_line_metrics(tmp_path: Path) -> None:
    """D1: A1 now reports stable_frame_ratio, mean_segments and line_accuracy."""
    store = _store(tmp_path, n_frames=4)
    # Frame 0 has both families, frame 1 only one, frames 2-3 nothing.
    lines = pd.DataFrame(
        {
            "frame_id": [0, 0, 1],
            "x1": [100.0, 200.0, 100.0],
            "y1": [100.0, 50.0, 100.0],
            "x2": [600.0, 210.0, 600.0],
            "y2": [100.0, 650.0, 100.0],
            "orient": [0, 1, 0],
            "score": 1.0,
        }
    )
    store.write_table(PITCH_LINES, lines)
    out = ablation.evaluate_variant(load_config(), tmp_path, "A1")
    assert out["stable_frame_ratio"] == pytest.approx(1 / 4)
    assert out["mean_segments"] == pytest.approx(3 / 4)
    assert np.isnan(out["line_accuracy"])


def test_a1_line_accuracy_with_landmarks(tmp_path: Path) -> None:
    """D1: a segment drawn on a true marking is accurate, a random one is not."""
    store = _store(tmp_path, n_frames=1)
    template = PitchTemplate()
    names = ["corner_tl", "halfway_top", "center", "left_pa_outer_top"]
    scale = 10.0  # 10 px per meter, identity orientation
    clicks = {
        n: [template.landmarks()[n][0] * scale, template.landmarks()[n][1] * scale]
        for n in names
    }
    (tmp_path / "gt.json").write_text(json.dumps({"0": clicks}), encoding="utf-8")
    lines = pd.DataFrame(
        {
            "frame_id": [0, 0],
            "x1": [0.0, 200.0],
            "y1": [0.0, 200.0],  # on the top touchline
            "x2": [400.0, 300.0],
            "y2": [0.0, 250.0],  # off any marking
            "orient": [0, 1],
            "score": 1.0,
        }
    )
    store.write_table(PITCH_LINES, lines)
    gt = ablation.GroundTruthFiles(landmarks=tmp_path / "gt.json")
    out = ablation.evaluate_variant(load_config(), tmp_path, "A1", gt)
    assert out["line_accuracy"] == pytest.approx(0.5)


def test_detection_fps_counts_frames_without_boxes(tmp_path: Path) -> None:
    """D2: 100 frames in 2 s is 50 fps even if only 2 frames have detections."""
    store = _store(tmp_path, n_frames=100)
    dets = pd.DataFrame(
        {
            "frame_id": [3, 70],
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 20.0,
            "conf": 0.9,
            "cls": 0,
        }
    )
    store.write_table(DETECTIONS, dets)
    out = ablation.evaluate_variant(load_config(), tmp_path, "A4")
    assert out["fps"] == pytest.approx(50.0)


def test_detection_fps_respects_stride(tmp_path: Path) -> None:
    """D2: with stride 2 only half of the frames are processed."""
    store = _store(tmp_path, n_frames=100)
    store.write_table(
        DETECTIONS,
        pd.DataFrame(
            {
                "frame_id": [0],
                "x1": 0.0,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
                "conf": 0.9,
                "cls": 0,
            }
        ),
    )
    cfg = load_config(None, {"video.stride": 2})
    assert ablation.evaluate_variant(cfg, tmp_path, "A4")["fps"] == pytest.approx(25.0)


def test_jitter_ignores_jump_across_invalid_frames(tmp_path: Path) -> None:
    """D3: two still stretches separated by an invalid frame have zero jitter."""
    store = _store(tmp_path, n_frames=5)
    matrices = np.stack([_shift(0), _shift(0), _shift(250), _shift(500), _shift(500)])
    store.write_homographies(HomographySequence(matrices, _quality([1, 1, 0, 1, 1])))
    out = ablation.evaluate_variant(load_config(), tmp_path, "A2")
    assert out["jitter_px"] == pytest.approx(0.0, abs=1e-9)
    assert out["valid_ratio"] == pytest.approx(0.8)


def _write_tracks(store: ArtifactStore) -> None:
    """Two predicted tracks (ids 7, 8) that overlap ground-truth tracks 1, 2."""
    rows = []
    for f in range(5):
        rows.append((f, 7, 100.0, 100.0, 140.0, 200.0, 0.9, 0, "player"))
        rows.append((f, 8, 400.0, 100.0, 440.0, 200.0, 0.9, 1, "player"))
    store.write_table(
        TRACKS,
        pd.DataFrame(
            rows,
            columns=[
                "frame_id",
                "track_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "conf",
                "team",
                "role",
            ],
        ),
    )
    store.write_table(
        TEAM_LABELS,
        pd.DataFrame({"track_id": [7, 8], "team": [0, 1], "role": "player"}),
    )


def test_team_accuracy_ignores_unlabeled_rows(tmp_path: Path) -> None:
    """D4: an empty team cell in the template does not count as a wrong answer."""
    store = _store(tmp_path, n_frames=5)
    _write_tracks(store)
    store.write_table(  # track 9 is predicted but left unlabeled in the template
        TEAM_LABELS,
        pd.DataFrame({"track_id": [7, 8, 9], "team": [0, 1, 0], "role": "player"}),
    )
    teams = tmp_path / "teams.csv"
    teams.write_text("track_id,team\n7,0\n8,1\n9,\n", encoding="utf-8")
    gt = ablation.GroundTruthFiles(teams=teams)
    assert ablation.evaluate_variant(load_config(), tmp_path, "A8", gt)[
        "team_accuracy"
    ] == pytest.approx(1.0)


def test_team_accuracy_matches_ground_truth_ids(tmp_path: Path) -> None:
    """Labels keyed by ground-truth ids are matched to predicted tracks by IoU."""
    store = _store(tmp_path, n_frames=5)
    _write_tracks(store)
    mot = tmp_path / "mot.txt"
    mot.write_text(
        "".join(
            f"{f + 1},1,100,100,40,100,1,1,1\n{f + 1},2,400,100,40,100,1,1,1\n"
            for f in range(5)
        ),
        encoding="utf-8",
    )
    teams = tmp_path / "teams.csv"
    teams.write_text("track_id,team\n1,1\n2,0\n", encoding="utf-8")  # swapped ids
    gt = ablation.GroundTruthFiles(mot=mot, teams=teams)
    assert ablation.evaluate_variant(load_config(), tmp_path, "A8", gt)[
        "team_accuracy"
    ] == pytest.approx(1.0)
    no_match = ablation.GroundTruthFiles(teams=tmp_path / "teams.csv")
    # Without the MOT file the ids 1, 2 never meet the predicted ids 7, 8.
    assert np.isnan(
        ablation.evaluate_variant(load_config(), tmp_path, "A8", no_match)[
            "team_accuracy"
        ]
    )


def test_make_figures_uses_run_config_and_skips_missing(tmp_path: Path) -> None:
    """D5a and CLI smoke test: figures are written, the run's config is used."""
    figures = _load_script("make_figures")
    run = tmp_path / "run"
    store = _store(run, n_frames=50)
    store.write_homographies(
        HomographySequence(np.stack([np.eye(3)] * 50), _quality([1] * 45 + [0] * 5))
    )
    t = np.arange(50)
    store.write_table(
        PITCH_XY,
        pd.DataFrame(
            {
                "frame_id": t,
                "track_id": 1,
                "X": 30 + 0.1 * t,
                "Y": 30.0,
                "team": 0,
                "role": "player",
                "valid": 1,
            }
        ),
    )
    analytics = store.directory("analytics")
    np.save(analytics / "heatmap_all.npy", np.ones((68, 105)))
    (run / "config.yaml").write_text("analytics:\n  max_speed_mps: 7.5\n", "utf-8")
    assert figures.resolve_config_paths([], run) == [run / "config.yaml"]
    assert figures.resolve_config_paths(["x.yaml"], run) == [Path("x.yaml")]
    out = tmp_path / "figs"
    assert figures.main(["--run-dir", str(run), "--output-dir", str(out)]) == 0
    assert {p.name for p in out.glob("*.png")} == {
        "homography_error.png",
        "heatmap_all.png",
        "speed_distribution.png",
    }
