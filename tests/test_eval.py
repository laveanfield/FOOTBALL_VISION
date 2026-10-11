"""Tests for the evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_vision.artifacts import HOMOGRAPHY_QUALITY_COLUMNS, HomographySequence
from football_vision.eval.homography import (
    evaluate_homography_sequence,
    reference_point_jitter_px,
    reprojection_error_m,
    summarize_errors,
)
from football_vision.eval.profiling import StageProfiler, Timer
from football_vision.eval.speed import speed_sanity_report
from football_vision.eval.team import confusion_table, team_accuracy
from football_vision.eval.tracking import load_mot_ground_truth
from football_vision.homography.geometry import apply_homography, invert_homography
from football_vision.pitch.template import PitchTemplate

H_PITCH_TO_IMAGE = np.array([[8.0, 1.5, 100.0], [0.5, 6.0, 50.0], [1e-4, 2e-4, 1.0]])


def test_reprojection_error_is_zero_for_the_true_homography() -> None:
    """Checks that the true homography has zero landmark error."""
    template = PitchTemplate()
    names = ["corner_tl", "center", "right_pa_outer_top"]
    pitch = template.landmark_array(names)
    image = apply_homography(H_PITCH_TO_IMAGE, pitch)
    errors = reprojection_error_m(invert_homography(H_PITCH_TO_IMAGE), image, pitch)
    np.testing.assert_allclose(errors, 0.0, atol=1e-6)


def test_summarize_errors_ignores_non_finite_values() -> None:
    """Checks the statistics and the handling of NaN and empty samples."""
    summary = summarize_errors([1.0, 2.0, 3.0, np.nan])
    assert summary["n"] == 3 and summary["median"] == 2.0 and summary["max"] == 3.0
    assert np.isnan(summarize_errors([])["mean"])


def test_evaluate_homography_sequence_skips_invalid_frames() -> None:
    """Checks held-out evaluation, including a frame with an invalid homography."""
    template = PitchTemplate()
    names = ["corner_tl", "corner_br", "center"]
    image = apply_homography(H_PITCH_TO_IMAGE, template.landmark_array(names))
    annotations = {
        0: {n: tuple(p) for n, p in zip(names, image, strict=True)},
        1: {"center": (0.0, 0.0)},
    }
    quality = pd.DataFrame(
        {
            "frame_id": [0, 1],
            "valid": [1, 0],
            "n_inliers": 4,
            "reproj_err_m": 0.0,
            "overlay_score": np.nan,
        }
    )
    h = invert_homography(H_PITCH_TO_IMAGE)
    sequence = HomographySequence(
        np.stack([h, h]), quality[list(HOMOGRAPHY_QUALITY_COLUMNS)]
    )
    table = evaluate_homography_sequence(sequence, annotations, template)
    assert set(table["frame_id"]) == {0}
    assert table["error_m"].max() < 1e-6


def test_reference_point_jitter() -> None:
    """Checks that identical matrices have zero jitter and shifted ones do not."""
    h = invert_homography(H_PITCH_TO_IMAGE)
    reference = np.array([[10.0, 10.0], [50.0, 30.0]])
    assert reference_point_jitter_px(np.stack([h, h, h]), reference).tolist() == [
        0.0,
        0.0,
    ]
    shift = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert reference_point_jitter_px(np.stack([h, h, shift @ h]), reference)[1] > 0


def test_team_accuracy_is_invariant_to_label_permutation() -> None:
    """Checks that swapped cluster ids still count as correct."""
    truth = [0, 0, 1, 1, 1]
    assert team_accuracy([1, 1, 0, 0, 0], truth) == 1.0
    assert team_accuracy([1, 1, 0, 0, 1], truth) == pytest.approx(0.8)
    assert team_accuracy([], []) == 0.0
    with pytest.raises(ValueError):
        team_accuracy([0], [0, 1])


def test_confusion_table_counts() -> None:
    """Checks the confusion table layout."""
    table = confusion_table([0, 1, 1], [0, 1, 0])
    assert table.loc[0, 0] == 1 and table.loc[0, 1] == 1 and table.loc[1, 1] == 1


def test_speed_sanity_report() -> None:
    """Checks the outlier fraction and the empty-sample behavior."""
    report = speed_sanity_report([2.0, 4.0, 13.0, 3.0], max_speed_mps=12.0)
    assert report["outlier_fraction"] == pytest.approx(0.25) and report["max"] == 13.0
    assert np.isnan(speed_sanity_report([])["mean"])


def test_load_mot_ground_truth_converts_format_and_frame_base(tmp_path: Path) -> None:
    """Checks the MOT parser: corner boxes and 1-based to 0-based frames."""
    path = tmp_path / "gt.txt"
    path.write_text(
        "1,7,100,200,40,80,1,-1,-1,-1\n2,7,110,200,40,80,1,-1,-1,-1\n", encoding="utf-8"
    )
    table = load_mot_ground_truth(path)
    assert table["frame_id"].tolist() == [0, 1]
    assert table.loc[0, ["x1", "y1", "x2", "y2"]].tolist() == [100, 200, 140, 280]
    assert load_mot_ground_truth(path, one_based_frames=False)["frame_id"].tolist() == [
        1,
        2,
    ]


def test_timer_and_stage_profiler() -> None:
    """Checks that timers measure and the profiler aggregates per stage."""
    with Timer() as timer:
        sum(range(1000))
    assert timer.elapsed >= 0.0
    profiler = StageProfiler()
    profiler.record("detect", 2.0, items=50)
    with profiler.measure("track", items=10):
        pass
    summary = profiler.summary().set_index("stage")
    assert summary.loc["detect", "fps"] == pytest.approx(25.0)
    assert summary.loc["detect", "ms_per_item"] == pytest.approx(40.0)
    assert summary.loc["track", "calls"] == 1
