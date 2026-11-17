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

import numpy as np
import pandas as pd

from football_vision.artifacts import (
    DETECTIONS,
    PITCH_XY,
    TEAM_LABELS,
    TRACKS,
    ArtifactStore,
    load_landmark_annotations,
)
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

NAN = float("nan")
JITTER_LANDMARKS = ("center", "halfway_top", "halfway_bottom", "corner_tl", "corner_bl")


@dataclass(frozen=True)
class GroundTruthFiles:
    """Ground-truth files used by the ablation metrics; ``None`` when missing.

    Attributes:
        landmarks: Held-out landmark JSON (A2, A3).
        detections: Detection CSV with ``frame_id, x1, y1, x2, y2, cls`` (A4).
        mot: MOT tracking ground truth (A5, A6).
        teams: CSV with ``track_id, team`` for the predicted track ids (A8).
    """

    landmarks: Path | None = None
    detections: Path | None = None
    mot: Path | None = None
    teams: Path | None = None


def homography_metrics(store: ArtifactStore, gt: GroundTruthFiles) -> dict[str, float]:
    """Computes reprojection error, validity and jitter of the homographies.

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
    matrices = sequence.matrices[valid]
    reference = PitchTemplate().landmark_array(JITTER_LANDMARKS)
    out["jitter_px"] = (
        float(reference_point_jitter_px(matrices, reference).mean())
        if len(matrices) >= 2
        else NAN
    )
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
    store: ArtifactStore, gt: GroundTruthFiles, detect_seconds: float
) -> dict[str, float]:
    """Computes mAP on the annotated frames and the detection throughput.

    Args:
        store: Artifact store of the variant.
        gt: Ground-truth files.
        detect_seconds: Wall time of the ``detect`` stage.

    Returns:
        ``map50``, ``map50_95`` and ``fps``.
    """
    preds = store.read_table(DETECTIONS)
    n_frames = preds["frame_id"].nunique()
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


def team_metrics(store: ArtifactStore, gt: GroundTruthFiles) -> dict[str, float]:
    """Computes team-assignment accuracy on the annotated tracks.

    Args:
        store: Artifact store of the variant.
        gt: Ground-truth files.

    Returns:
        ``team_accuracy`` (NaN without ground truth).
    """
    if gt.teams is None:
        return {"team_accuracy": NAN}
    merged = store.read_table(TEAM_LABELS).merge(
        pd.read_csv(gt.teams)[["track_id", "team"]],
        on="track_id",
        suffixes=("_pred", "_true"),
    )
    return {"team_accuracy": team_accuracy(merged["team_pred"], merged["team_true"])}


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
    timings = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
    metrics = {"stage_seconds": float(sum(t["seconds"] for t in timings.values()))}
    if ablation_id in ("A2", "A3"):
        metrics |= homography_metrics(store, gt)
    elif ablation_id == "A4":
        metrics |= detection_metrics(
            store, gt, float(timings.get("detect", {}).get("seconds", 0.0))
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
