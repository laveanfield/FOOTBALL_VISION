"""Tests for the report figures, the filmstrip and the A7 speed helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from football_vision.artifacts import (
    ANALYTICS,
    DETECTIONS,
    HOMOGRAPHY_QUALITY,
    PITCH_LINES,
    PITCH_MASK,
    PITCH_XY,
    TEAM_LABELS,
    ArtifactStore,
    HomographySequence,
    VideoMeta,
)
from football_vision.config import PipelineConfig
from football_vision.eval.speed import (
    per_sample_speeds,
    stationary_distance_per_minute,
)
from football_vision.homography.geometry import apply_homography, invert_homography
from football_vision.pitch.template import PitchTemplate
from football_vision.viz.figures import (
    plot_heatmap_on_pitch,
    plot_homography_error,
    plot_pipeline_steps,
    plot_speed_distribution,
    plot_team_shape,
)
from football_vision.viz.render import make_filmstrip

_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    """Loads a CLI module from ``scripts/<name>.py``."""
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pitch_xy(tracks: dict[int, np.ndarray], start: int = 0) -> pd.DataFrame:
    """Builds a ``pitch_xy`` table from ``{track_id: (T, 2) positions}``."""
    rows = []
    for track_id, xy in tracks.items():
        for i, (x, y) in enumerate(xy):
            rows.append((start + i, track_id, x, y, 0, "player", 1))
    return pd.DataFrame(
        rows, columns=["frame_id", "track_id", "X", "Y", "team", "role", "valid"]
    )


def test_pipeline_steps_has_four_panels_without_axes() -> None:
    """Checks the 2 x 2 layout and that axes are hidden."""
    frame = np.zeros((90, 160, 3), np.uint8)
    edges = np.zeros((90, 160), np.uint8)
    fig = plot_pipeline_steps(
        frame, edges, np.array([[0, 0, 100, 50.0]]), np.array([[10.0, 20.0]])
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4
    assert not any(ax.axison for ax in fig.axes)


def test_pipeline_steps_accepts_empty_segments_and_keypoints() -> None:
    """Checks that a frame without detected lines still renders."""
    frame = np.zeros((90, 160, 3), np.uint8)
    fig = plot_pipeline_steps(frame, frame[..., 0], np.zeros((0, 4)), np.zeros((0, 2)))
    assert len(fig.axes) == 4


def test_heatmap_figure_uses_pitch_extent() -> None:
    """Checks that the heatmap spans 105 x 68 m with Y growing downward."""
    fig = plot_heatmap_on_pitch(np.random.default_rng(0).random((68, 105)), title="t")
    ax = fig.axes[0]
    assert ax.get_images()[0].get_extent() == [0.0, 105.0, 68.0, 0.0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    assert ax.get_title() == "t"


def test_team_shape_converts_frames_to_seconds() -> None:
    """Checks the time axis and the two-panel split by unit."""
    shape = pd.DataFrame(
        {
            "frame_id": [0, 25, 50],
            "length_m": [30.0, 31.0, 32.0],
            "width_m": [40.0, 41.0, 42.0],
            "hull_area_m2": [900.0, 950.0, 1000.0],
            "stretch_index_m": [12.0, 12.5, 13.0],
        }
    )
    fig = plot_team_shape(shape, fps=25.0)
    assert len(fig.axes) == 2
    x = fig.axes[0].get_lines()[0].get_xdata()
    np.testing.assert_allclose(x, [0.0, 1.0, 2.0])
    assert fig.axes[1].get_xlabel() == "time (s)"


def test_speed_distribution_overlays_reference_sample() -> None:
    """Checks that the A7 comparison draws two histograms and the limit."""
    rng = np.random.default_rng(1)
    fig = plot_speed_distribution(
        rng.uniform(0, 8, 200), 12.0, reference_mps=rng.uniform(0, 20, 200)
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 3
    assert any("unsmoothed" in label for label in labels)


def test_speed_distribution_ignores_non_finite_values() -> None:
    """Checks that NaN speeds do not break the histogram."""
    fig = plot_speed_distribution(np.array([1.0, np.nan, 3.0, np.inf]))
    assert "n=2" in fig.axes[0].get_legend().get_texts()[0].get_text()


def test_homography_error_shades_invalid_frames() -> None:
    """Checks the valid ratio in the title and the re-anchor marker."""
    quality = pd.DataFrame(
        {
            "frame_id": np.arange(10),
            "valid": [1] * 7 + [0] * 3,
            "n_inliers": 10,
            "reproj_err_m": np.linspace(0.2, 1.0, 10),
            "overlay_score": np.linspace(0.9, 0.3, 10),
        }
    )
    fig = plot_homography_error(quality, reanchor_frames=[5])
    assert fig.axes[0].get_title() == "valid frames: 70.0%"
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert {"invalid H", "re-anchor"} <= set(labels)


def test_filmstrip_tiles_and_pads_last_row() -> None:
    """Checks the output size, the black padding tile and the labels."""
    frames = [np.full((90, 160, 3), 100 + 20 * i, np.uint8) for i in range(5)]
    strip = make_filmstrip(frames, n_cols=3, tile_width=80, labels=list("abcde"))
    assert strip.shape == (2 * 45, 3 * 80, 3)
    assert strip[45:, 160:].max() == 0  # padding tile
    assert strip[30:45, 70:80].mean() == pytest.approx(100, abs=1)


@pytest.mark.parametrize(
    ("frames", "n_cols", "tile_width"),
    [
        ([], 2, 80),
        ([np.zeros((9, 16, 3), np.uint8)], 0, 80),
        ([np.zeros((9, 16, 3), np.uint8), np.zeros((8, 16, 3), np.uint8)], 2, 80),
    ],
)
def test_filmstrip_rejects_bad_input(
    frames: list, n_cols: int, tile_width: int
) -> None:
    """Checks empty input, a non-positive column count and mixed sizes."""
    with pytest.raises(ValueError):
        make_filmstrip(frames, n_cols, tile_width)


def test_per_sample_speeds_do_not_bridge_gaps() -> None:
    """Checks that a teleport across an occlusion is not turned into a speed."""
    walk = np.stack([np.arange(10) * 0.2, np.zeros(10)], axis=1)  # 5 m/s at 25 fps
    table = _pitch_xy({1: walk})
    jump = _pitch_xy({1: walk + [50.0, 0.0]}, start=30)
    speeds = per_sample_speeds(pd.concat([table, jump]), fps=25.0, max_gap_frames=5)
    np.testing.assert_allclose(speeds, 5.0)


def test_stationary_distance_grows_with_noise() -> None:
    """Checks the A7 signal: a noisy static player 'runs', a clean one does not."""
    rng = np.random.default_rng(2)
    still = np.tile([30.0, 30.0], (250, 1))
    clean = stationary_distance_per_minute(_pitch_xy({1: still}), 25.0, 5)
    noisy = stationary_distance_per_minute(
        _pitch_xy({1: still + rng.normal(0, 0.3, still.shape)}), 25.0, 5
    )
    runner = np.stack([np.linspace(0, 50, 250), np.full(250, 30.0)], axis=1)
    assert clean == pytest.approx(0.0)
    assert noisy > 100.0
    assert np.isnan(stationary_distance_per_minute(_pitch_xy({1: runner}), 25.0, 5))


def test_speed_distribution_empty_emits_no_warning() -> None:
    """Checks that empty speeds produce a valid histogram without RuntimeWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_speed_distribution(np.array([], dtype=np.float64))
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    label = fig.axes[0].get_legend().get_texts()[0].get_text()
    assert "(n=0)" in label and "nan%" not in label


def test_per_sample_speeds_ignores_non_finite_coordinates() -> None:
    """Checks that non-finite coordinates marked valid do not poison speed stats."""
    walk = np.stack([np.arange(6) * 0.2, np.zeros(6)], axis=1)
    df = _pitch_xy({1: walk})
    df.loc[df.index[2], "X"] = np.nan
    speeds = per_sample_speeds(df, fps=25.0, max_gap_frames=5)
    assert np.all(np.isfinite(speeds))
    assert speeds.mean() == pytest.approx(5.0)
    still = np.tile([30.0, 30.0], (30, 1))
    still[10, 1] = np.inf
    rate = stationary_distance_per_minute(_pitch_xy({1: still}), 25.0, 5)
    assert rate == pytest.approx(0.0)


def test_ablation_a1_and_fixed_metrics_helpers(tmp_path: Path) -> None:
    """Checks A1 pitch-line metrics, FPS count, valid-run jitter, and team NaN."""
    ra = _load_script("run_ablation")
    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir)
    store.write_meta(
        VideoMeta(
            source="x.mp4",
            fps=25.0,
            width=1280,
            height=720,
            n_frames=100,
            shots=[(0, 99)],
        )
    )
    (run_dir / "timings.json").write_text(
        json.dumps({"detect": {"seconds": 2.0}}), encoding="utf-8"
    )

    # D1: A1 metrics with and without landmarks
    h_pitch_to_img = np.array([[10.0, 0.0, 100.0], [0.0, 8.0, 50.0], [0.0, 0.0, 1.0]])
    template = PitchTemplate()
    lm_names = ["corner_tl", "corner_tr", "corner_br", "corner_bl"]
    lm_img = apply_homography(h_pitch_to_img, template.landmark_array(lm_names))
    lm_path = tmp_path / "heldout.json"
    lm_payload = {
        "0": {
            n: [float(p[0]), float(p[1])] for n, p in zip(lm_names, lm_img, strict=True)
        }
    }
    lm_path.write_text(json.dumps(lm_payload), encoding="utf-8")
    seg0_img = apply_homography(
        h_pitch_to_img, np.array([[0.0, 0.0], [105.0, 0.0]], dtype=np.float64)
    )
    seg1_img = apply_homography(
        h_pitch_to_img, np.array([[0.0, 0.0], [0.0, 68.0]], dtype=np.float64)
    )
    store.write_table(
        PITCH_LINES,
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "x1": seg0_img[0, 0],
                    "y1": seg0_img[0, 1],
                    "x2": seg0_img[1, 0],
                    "y2": seg0_img[1, 1],
                    "orient": 0,
                    "score": 1.0,
                },
                {
                    "frame_id": 0,
                    "x1": seg1_img[0, 0],
                    "y1": seg1_img[0, 1],
                    "x2": seg1_img[1, 0],
                    "y2": seg1_img[1, 1],
                    "orient": 1,
                    "score": 1.0,
                },
            ]
        ),
    )
    cfg = PipelineConfig()
    a1_no_gt = ra.evaluate_variant(cfg, run_dir, ablation_id="A1")
    assert a1_no_gt["stable_frame_ratio"] == pytest.approx(0.01)
    assert a1_no_gt["mean_segments"] == pytest.approx(0.02)
    assert np.isnan(a1_no_gt["line_accuracy"])

    a1_gt = ra.evaluate_variant(
        cfg, run_dir, ablation_id="A1", gt=ra.GroundTruthFiles(landmarks=lm_path)
    )
    assert a1_gt["line_accuracy"] == pytest.approx(1.0)

    # D2: detection_metrics uses processed frame count (100 / 2.0 = 50.0 fps)
    store.write_table(
        DETECTIONS,
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "x1": 10.0,
                    "y1": 10.0,
                    "x2": 30.0,
                    "y2": 50.0,
                    "conf": 0.9,
                    "cls": 0,
                },
                {
                    "frame_id": 50,
                    "x1": 10.0,
                    "y1": 10.0,
                    "x2": 30.0,
                    "y2": 50.0,
                    "conf": 0.9,
                    "cls": 0,
                },
            ]
        ),
    )
    det_res = ra.detection_metrics(store, cfg, ra.GroundTruthFiles(), 2.0)
    assert det_res["fps"] == pytest.approx(50.0)

    # D3: homography_metrics does not measure jitter across an invalid gap
    h1 = invert_homography(h_pitch_to_img)
    h2 = invert_homography(
        np.array([[10.0, 0.0, 250.0], [0.0, 8.0, 50.0], [0.0, 0.0, 1.0]])
    )
    mats = np.stack([h1, h1, h1, h2, h2])
    qual = pd.DataFrame(
        [
            {
                "frame_id": i,
                "valid": 0 if i == 2 else 1,
                "n_inliers": 4,
                "reproj_err_m": 0.1,
                "overlay_score": 0.9,
            }
            for i in range(5)
        ]
    )
    store.write_homographies(HomographySequence(mats, qual))
    hom_res = ra.homography_metrics(store, ra.GroundTruthFiles())
    assert hom_res["jitter_px"] == pytest.approx(0.0, abs=1e-9)

    # D4: team_metrics ignores blank/NaN ground-truth rows
    store.write_table(
        TEAM_LABELS,
        pd.DataFrame(
            [
                {"track_id": 0, "team": 0, "role": "player"},
                {"track_id": 1, "team": 1, "role": "player"},
                {"track_id": 2, "team": 0, "role": "player"},
            ]
        ),
    )
    teams_partial = tmp_path / "teams_partial.csv"
    teams_partial.write_text(
        "track_id,first_frame_id,n_frames,team\n0,0,10,0\n1,0,10,1\n2,0,10,\n",
        encoding="utf-8",
    )
    assert ra.team_metrics(store, ra.GroundTruthFiles(teams=teams_partial))[
        "team_accuracy"
    ] == pytest.approx(1.0)
    teams_blank = tmp_path / "teams_blank.csv"
    teams_blank.write_text(
        "track_id,first_frame_id,n_frames,team\n0,0,10,\n", encoding="utf-8"
    )
    assert np.isnan(
        ra.team_metrics(store, ra.GroundTruthFiles(teams=teams_blank))["team_accuracy"]
    )


def test_make_figures_auto_loads_run_config_and_uses_pitch_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks run config auto-loading and pitch-mask clipping in make_figures."""
    mf = _load_script("make_figures")
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 90)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 files in this environment")
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    # Bright line in upper half (masked out) and lower half (kept inside mask).
    frame[20:22, 20:140] = 255
    frame[60:62, 20:140] = 255
    writer.write(frame)
    writer.release()

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir)
    store.write_meta(
        VideoMeta(
            source=str(video_path),
            fps=25.0,
            width=160,
            height=90,
            n_frames=1,
            shots=[(0, 0)],
        )
    )
    store.write_table(
        PITCH_LINES,
        pd.DataFrame(columns=["frame_id", "x1", "y1", "x2", "y2", "orient", "score"]),
    )
    store.write_table(
        HOMOGRAPHY_QUALITY,
        pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "valid": 1,
                    "n_inliers": 4,
                    "reproj_err_m": 0.1,
                    "overlay_score": 0.9,
                }
            ]
        ),
    )
    store.write_table(
        PITCH_XY,
        pd.DataFrame(
            columns=["frame_id", "track_id", "X", "Y", "team", "role", "valid"]
        ),
    )
    analytics_dir = store.path(ANALYTICS)
    analytics_dir.mkdir(parents=True, exist_ok=True)
    np.save(analytics_dir / "heatmap_all.npy", np.zeros((10, 10)))
    mask_dir = store.path(PITCH_MASK)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((90, 160), dtype=np.uint8)
    mask[45:, :] = 255
    cv2.imwrite(str(mask_dir / "000000.png"), mask)

    cfg = PipelineConfig()
    cfg.analytics.max_speed_mps = 9.0
    (run_dir / "config.yaml").write_text(
        "analytics:\n  max_speed_mps: 9.0\n", encoding="utf-8"
    )

    fig_steps = mf.pipeline_steps_figure(store, cfg, frame_id=0)
    edge_img = fig_steps.axes[1].get_images()[0].get_array()
    assert np.asarray(edge_img)[:40, :].max() == 0

    seen_limits: list[float] = []
    orig_plot = mf.plot_speed_distribution

    def _spy_speed(
        speeds: np.ndarray, max_speed_mps: float = 12.0, **kwargs: object
    ) -> Figure:
        seen_limits.append(max_speed_mps)
        return orig_plot(speeds, max_speed_mps=max_speed_mps, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mf, "plot_speed_distribution", _spy_speed)
    assert (
        mf.main(["--run-dir", str(run_dir), "--output-dir", str(tmp_path / "figs")])
        == 0
    )
    assert seen_limits == [9.0]
