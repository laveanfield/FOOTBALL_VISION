"""Tests for the evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_vision.artifacts import HOMOGRAPHY_QUALITY_COLUMNS, HomographySequence
from football_vision.eval.detection import evaluate_detections
from football_vision.eval.homography import (
    evaluate_homography_sequence,
    reference_point_jitter_px,
    reprojection_error_m,
    summarize_errors,
)
from football_vision.eval.profiling import StageProfiler, Timer
from football_vision.eval.speed import speed_sanity_report
from football_vision.eval.team import confusion_table, team_accuracy
from football_vision.eval.tracking import evaluate_tracking, load_mot_ground_truth
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


def test_evaluate_detections_precision_recall_and_map() -> None:
    """Verifies detection metrics, unannotated frame filtering, and COCO 101-pt AP."""
    gt = pd.DataFrame(
        [
            {"frame_id": 0, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 50.0, "cls": 0},
            {"frame_id": 0, "x1": 60.0, "y1": 20.0, "x2": 68.0, "y2": 28.0, "cls": 32},
        ]
    )
    preds = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "x1": 10.0,
                "y1": 10.0,
                "x2": 30.0,
                "y2": 50.0,
                "conf": 0.95,
                "cls": 0,
            },
            {
                "frame_id": 0,
                "x1": 60.0,
                "y1": 20.0,
                "x2": 68.0,
                "y2": 28.0,
                "conf": 0.85,
                "cls": 32,
            },
            # Unannotated frame must be ignored by evaluate_detections.
            {
                "frame_id": 5,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "conf": 0.90,
                "cls": 0,
            },
        ]
    )
    perfect = evaluate_detections(preds, gt, iou_threshold=0.5)
    assert perfect["precision"] == pytest.approx(1.0)
    assert perfect["recall"] == pytest.approx(1.0)
    assert perfect["map50"] == pytest.approx(1.0)
    assert perfect["map50_95"] == pytest.approx(1.0)
    assert perfect["precision_cls0"] == pytest.approx(1.0)

    # Single FP after 1 TP out of 2 GT on cls 0 yields precision=0.5, recall=0.5.
    gt_two = pd.DataFrame(
        [
            {"frame_id": 0, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 50.0, "cls": 0},
            {"frame_id": 0, "x1": 80.0, "y1": 10.0, "x2": 100.0, "y2": 50.0, "cls": 0},
        ]
    )
    preds_mixed = pd.DataFrame(
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
                "frame_id": 0,
                "x1": 120.0,
                "y1": 10.0,
                "x2": 140.0,
                "y2": 50.0,
                "conf": 0.5,
                "cls": 0,
            },
        ]
    )
    mixed = evaluate_detections(preds_mixed, gt_two, iou_threshold=0.5)
    assert mixed["precision"] == pytest.approx(0.5)
    assert mixed["recall"] == pytest.approx(0.5)
    assert mixed["map50"] == pytest.approx(51.0 / 101.0)


def test_evaluate_tracking_perfect_and_id_switch() -> None:
    """Checks MOTA, IDF1, HOTA, id_switches, and fragmentations."""
    gt = pd.DataFrame(
        [
            {
                "frame_id": fid,
                "track_id": 1,
                "x1": 10.0 + fid,
                "y1": 20.0,
                "x2": 30.0 + fid,
                "y2": 60.0,
            }
            for fid in range(4)
        ]
    )
    perfect = evaluate_tracking(gt.copy(), gt, iou_threshold=0.5)
    assert perfect["mota"] == pytest.approx(1.0)
    assert perfect["idf1"] == pytest.approx(1.0)
    assert perfect["hota"] == pytest.approx(1.0)
    assert perfect["id_switches"] == 0.0
    assert perfect["fragmentations"] == 0.0

    # Flip predicted track_id halfway -> MOTA drops for IDSW, IDF1/HOTA drop too.
    switched = gt.copy()
    switched.loc[switched["frame_id"] >= 2, "track_id"] = 2
    res = evaluate_tracking(switched, gt, iou_threshold=0.5)
    assert res["id_switches"] >= 1.0
    assert res["mota"] < 1.0
    assert res["idf1"] == pytest.approx(0.5)
    assert 0.5 < res["hota"] < 1.0

    empty_res = evaluate_tracking(gt, gt.iloc[0:0], iou_threshold=0.5)
    assert empty_res["mota"] == 0.0 and empty_res["hota"] == 0.0
