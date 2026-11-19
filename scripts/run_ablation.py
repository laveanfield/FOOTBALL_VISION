"""Runs one ablation experiment (A1-A8) on top of a finished base run.

Typical usage example:

    python scripts/run_ablation.py --ablation A5 --config configs/fixed_camera.yaml
        --set video.path=data/raw/clip.mp4 --mot-gt data/annotations/clip_mot.txt

The base run (named by the configuration) must already contain the artifacts that
the ablation's stages depend on. Metrics that need ground truth are reported as NaN
when the matching ``--*-gt`` file is not given.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_vision.artifacts import (
    DETECTIONS,
    META,
    PITCH_LINES,
    PITCH_XY,
    TEAM_LABELS,
    TRACKS,
    ArtifactStore,
    load_landmark_annotations,
)
from football_vision.boxes import iou_matrix
from football_vision.config import PipelineConfig, load_config, parse_override_strings
from football_vision.eval.ablation import ABLATIONS, run_ablation
from football_vision.eval.detection import evaluate_detections
from football_vision.eval.homography import (
    evaluate_homography_sequence,
    reference_point_jitter_px,
    summarize_errors,
)
from football_vision.eval.speed import (
    per_sample_speeds,
    speed_sanity_report,
    stationary_distance_per_minute,
)
from football_vision.eval.team import team_accuracy
from football_vision.eval.tracking import evaluate_tracking, load_mot_ground_truth
from football_vision.pitch.template import PitchTemplate
from football_vision.utils import setup_logging
from football_vision.video import frame_id_range

NAN = float("nan")
JITTER_LANDMARKS = ("center", "halfway_top", "halfway_bottom", "corner_tl", "corner_bl")
LINE_TOLERANCE_M = 1.0


@dataclass(frozen=True)
class GroundTruthFiles:
    """Ground-truth files used by the ablation metrics; ``None`` when missing.

    Attributes:
        landmarks: Held-out landmark JSON (A1 line accuracy, A2, A3).
        detections: Detection CSV with ``frame_id, x1, y1, x2, y2, cls`` (A4).
        mot: MOT tracking ground truth (A5, A6; A8 track matching).
        teams: CSV with ``track_id, team`` (A8). Its track ids are ground-truth ids
            when ``mot`` is given, otherwise ids of the predicted tracks.
    """

    landmarks: Path | None = None
    detections: Path | None = None
    mot: Path | None = None
    teams: Path | None = None


def processed_frame_count(store: ArtifactStore, cfg: PipelineConfig) -> int | None:
    """Counts the frames a run processed, including frames with no output rows.

    Args:
        store: Artifact store of the run.
        cfg: Configuration of the run.

    Returns:
        The number of processed frames, or ``None`` if ``meta.json`` is missing.
    """
    if not store.exists(META):
        return None
    meta = store.read_meta()
    video = cfg.video
    return len(
        frame_id_range(meta.n_frames, video.start_frame, video.end_frame, video.stride)
    )


def _point_to_polyline_m(
    points: npt.NDArray[np.float64], polyline: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Distance from each point to the closest segment of a polyline."""
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    denom = np.maximum((ab**2).sum(axis=1), 1e-12)
    t = ((points[:, None, :] - a[None]) * ab[None]).sum(axis=2) / denom[None]
    closest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def segment_line_errors_m(
    segments: npt.NDArray[np.float64],
    landmarks: dict[str, tuple[float, float]],
    template: PitchTemplate,
) -> npt.NDArray[np.float64]:
    """Measures how far detected segments lie from the true pitch markings.

    The homography of the frame is fitted to its clicked landmarks, each segment's
    end points and midpoint are projected to the pitch, and the largest distance to
    the nearest template marking is reported.

    Args:
        segments: Array of shape ``(N, 4)`` in pixels.
        landmarks: Clicked landmarks of the frame (at least four).
        template: Pitch model.

    Returns:
        Array of shape ``(N,)`` in meters (empty if fewer than four landmarks).
    """
    known = template.landmarks()
    names = [n for n in landmarks if n in known]
    if len(names) < 4 or len(segments) == 0:
        return np.zeros(0)
    image = np.float32([landmarks[n] for n in names])
    pitch = np.float32([known[n] for n in names])
    homography, _ = cv2.findHomography(image, pitch, 0)
    if homography is None:
        return np.zeros(0)
    segs = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    samples = np.concatenate(
        [segs[:, :2], segs[:, 2:], (segs[:, :2] + segs[:, 2:]) / 2], axis=0
    )
    projected = cv2.perspectiveTransform(samples[None], homography)[0]
    distances = np.min(
        [
            _point_to_polyline_m(projected, np.asarray(line, dtype=np.float64))
            for line in template.polylines()
        ],
        axis=0,
    )
    return distances.reshape(3, -1).max(axis=0)


def pitch_lines_metrics(
    store: ArtifactStore, cfg: PipelineConfig, gt: GroundTruthFiles
) -> dict[str, float]:
    """Computes the line-extraction metrics of ablation A1.

    Args:
        store: Artifact store of the variant.
        cfg: Configuration of the variant.
        gt: Ground-truth files.

    Returns:
        ``stable_frame_ratio`` (frames with segments of both line families),
        ``mean_segments`` and ``line_accuracy`` (share of segments within
        ``LINE_TOLERANCE_M`` of a true marking; NaN without landmarks).
    """
    lines = store.read_table(PITCH_LINES)
    n_frames = processed_frame_count(store, cfg) or lines["frame_id"].nunique()
    both = lines.groupby("frame_id")["orient"].agg(lambda o: {0, 1} <= set(o))
    out = {
        "stable_frame_ratio": float(both.sum() / n_frames) if n_frames else NAN,
        "mean_segments": float(len(lines) / n_frames) if n_frames else NAN,
        "line_accuracy": NAN,
    }
    if gt.landmarks is not None:
        template = PitchTemplate()
        errors = [
            segment_line_errors_m(
                lines.loc[lines["frame_id"] == f, ["x1", "y1", "x2", "y2"]].to_numpy(),
                clicks,
                template,
            )
            for f, clicks in load_landmark_annotations(gt.landmarks).items()
        ]
        flat = np.concatenate(errors) if errors else np.zeros(0)
        if len(flat):
            out["line_accuracy"] = float((flat <= LINE_TOLERANCE_M).mean())
    return out


def _valid_runs(valid: npt.NDArray[np.bool_]) -> list[npt.NDArray[np.int64]]:
    """Index arrays of the maximal runs of consecutive ``True`` entries."""
    index = np.flatnonzero(valid)
    if len(index) == 0:
        return []
    return np.split(index, np.flatnonzero(np.diff(index) > 1) + 1)


def homography_metrics(store: ArtifactStore, gt: GroundTruthFiles) -> dict[str, float]:
    """Computes reprojection error, validity and jitter of the homographies.

    Jitter is measured only between consecutive valid frames, so the jump across an
    invalid stretch is never counted as camera motion.

    Args:
        store: Artifact store of the variant.
        gt: Ground-truth files.

    Returns:
        ``reproj_err_m_median``, ``reproj_err_m_p95``, ``valid_ratio`` and
        ``jitter_px`` (NaN entries when their inputs are missing).
    """
    sequence = store.read_homographies()
    valid = sequence.quality["valid"].to_numpy() == 1
    out = {"valid_ratio": float(valid.mean()) if len(valid) else NAN}
    reference = PitchTemplate().landmark_array(JITTER_LANDMARKS)
    steps = [
        reference_point_jitter_px(sequence.matrices[run], reference)
        for run in _valid_runs(valid)
        if len(run) >= 2
    ]
    out["jitter_px"] = float(np.concatenate(steps).mean()) if steps else NAN
    out["reproj_err_m_median"] = out["reproj_err_m_p95"] = NAN
    if gt.landmarks is not None:
        errors = evaluate_homography_sequence(
            sequence, load_landmark_annotations(gt.landmarks)
        )["error_m"]
        summary = summarize_errors(errors)
        out["reproj_err_m_median"] = summary["median"]
        out["reproj_err_m_p95"] = summary["p95"]
    return out


def detection_metrics(
    store: ArtifactStore,
    cfg: PipelineConfig,
    gt: GroundTruthFiles,
    detect_seconds: float,
) -> dict[str, float]:
    """Computes mAP on the annotated frames and the detection throughput.

    Args:
        store: Artifact store of the variant.
        cfg: Configuration of the variant.
        gt: Ground-truth files.
        detect_seconds: Wall time of the ``detect`` stage.

    Returns:
        ``map50``, ``map50_95`` and ``fps``. The frame count includes frames
        without any detection.
    """
    preds = store.read_table(DETECTIONS)
    n_frames = processed_frame_count(store, cfg) or preds["frame_id"].nunique()
    out = {
        "fps": n_frames / detect_seconds if detect_seconds > 0 else NAN,
        "map50": NAN,
        "map50_95": NAN,
    }
    if gt.detections is not None:
        truth = pd.read_csv(gt.detections)
        preds = preds[preds["frame_id"].isin(truth["frame_id"].unique())]
        result = evaluate_detections(preds, truth)
        out["map50"] = float(result.get("map50", NAN))
        out["map50_95"] = float(result.get("map50_95", NAN))
    return out


def tracking_metrics(store: ArtifactStore, gt: GroundTruthFiles) -> dict[str, float]:
    """Computes identity metrics against the MOT ground truth.

    Args:
        store: Artifact store of the variant.
        gt: Ground-truth files.

    Returns:
        ``idf1``, ``mota``, ``hota`` and ``id_switches`` (NaN without ground truth).
    """
    keys = ("idf1", "mota", "hota", "id_switches")
    if gt.mot is None:
        return dict.fromkeys(keys, NAN)
    result = evaluate_tracking(store.read_table(TRACKS), load_mot_ground_truth(gt.mot))
    return {k: float(result.get(k, NAN)) for k in keys}


def match_tracks_to_ground_truth(
    tracks: pd.DataFrame, truth: pd.DataFrame, iou_threshold: float = 0.5
) -> dict[int, int]:
    """Maps every predicted track id to the ground-truth id it overlaps most.

    In each frame boxes are matched one to one (Hungarian on IoU); a predicted track
    takes the ground-truth id it was matched to most often.

    Args:
        tracks: Predicted tracks with ``frame_id, track_id, x1, y1, x2, y2``.
        truth: Ground truth in the same columns (``load_mot_ground_truth``).
        iou_threshold: Minimum IoU for a match.

    Returns:
        Mapping from predicted track id to ground-truth track id.
    """
    votes: dict[tuple[int, int], int] = {}
    cols = ["x1", "y1", "x2", "y2"]
    truth_by_frame = dict(tuple(truth.groupby("frame_id")))
    for frame_id, pred in tracks.groupby("frame_id"):
        gt_frame = truth_by_frame.get(frame_id)
        if gt_frame is None:
            continue
        iou = iou_matrix(pred[cols].to_numpy(), gt_frame[cols].to_numpy())
        rows, matched = linear_sum_assignment(-iou)
        for r, c in zip(rows, matched, strict=True):
            if iou[r, c] >= iou_threshold:
                key = (int(pred["track_id"].iat[r]), int(gt_frame["track_id"].iat[c]))
                votes[key] = votes.get(key, 0) + 1
    best: dict[int, tuple[int, int]] = {}
    for (pred_id, gt_id), count in votes.items():
        if count > best.get(pred_id, (-1, 0))[1]:
            best[pred_id] = (gt_id, count)
    return {pred_id: gt_id for pred_id, (gt_id, _) in best.items()}


def team_metrics(store: ArtifactStore, gt: GroundTruthFiles) -> dict[str, float]:
    """Computes team-assignment accuracy on the labeled tracks.

    Rows whose team is missing or negative (unlabeled, referee) are ignored. With a
    MOT ground truth, predicted tracks are first matched to ground-truth tracks so
    that labels written for ground-truth ids (the SportsMOT template) can be used.

    Args:
        store: Artifact store of the variant.
        gt: Ground-truth files.

    Returns:
        ``team_accuracy`` (NaN without labels or without any matched track).
    """
    if gt.teams is None:
        return {"team_accuracy": NAN}
    truth = pd.read_csv(gt.teams)[["track_id", "team"]]
    truth = truth.assign(team=pd.to_numeric(truth["team"], errors="coerce"))
    truth = truth[truth["team"] >= 0].astype({"track_id": int, "team": int})
    pred = store.read_table(TEAM_LABELS)[["track_id", "team"]]
    pred = pred[pred["team"] >= 0].copy()
    if gt.mot is not None:
        mapping = match_tracks_to_ground_truth(
            store.read_table(TRACKS), load_mot_ground_truth(gt.mot)
        )
        pred["track_id"] = pred["track_id"].map(mapping)
        pred = pred.dropna(subset=["track_id"]).astype({"track_id": int})
    merged = pred.merge(truth, on="track_id", suffixes=("_pred", "_true"))
    if merged.empty:
        return {"team_accuracy": NAN}
    return {"team_accuracy": team_accuracy(merged["team_pred"], merged["team_true"])}


def analytics_metrics(store: ArtifactStore, cfg: PipelineConfig) -> dict[str, float]:
    """Computes the speed sanity statistics used by ablation A7.

    No ground truth is needed: implausible speeds and the apparent motion of
    quasi-static players both measure how much position noise leaks into speed.

    Args:
        store: Artifact store of the variant.
        cfg: Configuration of the variant.

    Returns:
        ``speed_outlier_fraction``, ``p95_speed_mps`` and ``stationary_distance_m``
        (meters per minute of quasi-static players).
    """
    pitch_xy = store.read_table(PITCH_XY)
    fps = store.read_meta().fps
    max_gap = cfg.analytics.max_gap_frames * cfg.video.stride
    report = speed_sanity_report(
        per_sample_speeds(pitch_xy, fps, max_gap), cfg.analytics.max_speed_mps
    )
    return {
        "speed_outlier_fraction": report["outlier_fraction"],
        "p95_speed_mps": report["p95"],
        "stationary_distance_m": stationary_distance_per_minute(pitch_xy, fps, max_gap),
    }


def evaluate_variant(
    cfg: PipelineConfig,
    run_dir: Path,
    ablation_id: str = "",
    gt: GroundTruthFiles | None = None,
) -> dict[str, float]:
    """Measures one ablation variant with the metrics its ablation names.

    Args:
        cfg: Configuration of the variant run.
        run_dir: Directory with the variant's artifacts.
        ablation_id: Id of the running ablation, which selects the metrics.
        gt: Ground-truth files; ``None`` means none are available.

    Returns:
        Mapping from metric name to value; ``stage_seconds`` is always present.
    """
    gt = gt or GroundTruthFiles()
    store = ArtifactStore(run_dir)
    timings = json.loads((Path(run_dir) / "timings.json").read_text(encoding="utf-8"))
    metrics = {"stage_seconds": float(sum(t["seconds"] for t in timings.values()))}
    if ablation_id == "A1":
        metrics |= pitch_lines_metrics(store, cfg, gt)
    elif ablation_id in ("A2", "A3"):
        metrics |= homography_metrics(store, gt)
    elif ablation_id == "A4":
        metrics |= detection_metrics(
            store, cfg, gt, float(timings.get("detect", {}).get("seconds", 0.0))
        )
    elif ablation_id in ("A5", "A6"):
        metrics |= tracking_metrics(store, gt)
    elif ablation_id == "A7":
        metrics |= analytics_metrics(store, cfg)
    elif ablation_id == "A8":
        metrics |= team_metrics(store, gt)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the selected ablation and writes ``reports/tables/ablation_<id>.csv``.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Run an ablation experiment.")
    parser.add_argument("--ablation", required=True, choices=sorted(ABLATIONS))
    parser.add_argument("--config", "-c", action="append", default=[])
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--landmarks-gt", type=Path, default=None)
    parser.add_argument("--detections-gt", type=Path, default=None)
    parser.add_argument("--mot-gt", type=Path, default=None)
    parser.add_argument("--teams-gt", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, parse_override_strings(args.overrides))
    setup_logging("INFO")
    ablation = ABLATIONS[args.ablation]
    gt = GroundTruthFiles(
        args.landmarks_gt, args.detections_gt, args.mot_gt, args.teams_gt
    )
    print(f"{ablation.ablation_id}: {ablation.question}")
    table = run_ablation(
        ablation,
        cfg,
        functools.partial(evaluate_variant, ablation_id=ablation.ablation_id, gt=gt),
        args.output_dir / f"ablation_{ablation.ablation_id}.csv",
    )
    missing = [
        m
        for m in ablation.metrics
        if m not in table or table[m].apply(lambda v: math.isnan(float(v))).all()
    ]
    if missing:
        print(f"[note] no value for {missing}: pass the matching --*-gt file")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0 if np.isfinite(table["stage_seconds"]).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
