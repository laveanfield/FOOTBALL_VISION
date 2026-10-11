"""Ablation experiments A1-A8 and a runner that executes them.

An ablation changes one factor, re-runs the affected stages and compares a metric.
Each experiment is a hypothesis, a variable, a metric and a conclusion, including
the conclusion "the hypothesis was wrong".
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from football_vision.artifacts import OUTPUTS
from football_vision.config import PipelineConfig, apply_overrides
from football_vision.pipeline.runner import PipelineRunner
from football_vision.pipeline.stages import STAGE_REGISTRY


@dataclass(frozen=True)
class Ablation:
    """Definition of one ablation experiment.

    Attributes:
        ablation_id: Short id such as ``"A1"``.
        question: Question that the experiment answers.
        stages: Stages that must be re-run for every variant.
        metrics: Names of the metrics to report.
        variants: Mapping from variant name to dotted configuration overrides.
    """

    ablation_id: str
    question: str
    stages: tuple[str, ...]
    metrics: tuple[str, ...]
    variants: Mapping[str, Mapping[str, Any]]


ABLATIONS: dict[str, Ablation] = {
    "A1": Ablation(
        "A1",
        "Does a white top-hat extract pitch lines better than raw Canny?",
        ("pitch_lines",),
        ("line_accuracy", "stable_frame_ratio"),
        {
            "canny": {"pitch.edge_method": "canny"},
            "tophat": {"pitch.edge_method": "tophat"},
        },
    ),
    "A2": Ablation(
        "A2",
        "Which way to initialize and maintain H is most stable?",
        ("homography",),
        ("reproj_err_m_median", "reproj_err_m_p95", "valid_ratio"),
        {
            "fixed": {"homography.mode": "fixed"},
            "propagate_no_reanchor": {
                "homography.mode": "propagate",
                "homography.reanchor_overlay_threshold": 0.0,
            },
            "propagate_reanchor": {
                "homography.mode": "propagate",
                "homography.reanchor_overlay_threshold": 0.5,
            },
        },
    ),
    "A3": Ablation(
        "A3",
        "Does temporal smoothing of H reduce jitter without adding error?",
        ("homography",),
        ("jitter_px", "reproj_err_m_median"),
        {f"window_{w}": {"homography.smoothing_window": w} for w in (0, 9, 21)},
    ),
    "A4": Ablation(
        "A4",
        "What is the speed/accuracy trade-off of model size and input resolution?",
        ("detect",),
        ("map50", "map50_95", "fps"),
        {
            f"{weights.split('.')[0]}_{size}": {
                "detection.weights": weights,
                "detection.imgsz": size,
            }
            for weights in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt")
            for size in (640, 960, 1280)
        },
    ),
    "A5": Ablation(
        "A5",
        "Does ByteTrack beat our own SORT in identity consistency?",
        ("track",),
        ("idf1", "mota", "hota", "id_switches"),
        {
            "sort": {"tracking.tracker": "sort"},
            "bytetrack": {"tracking.tracker": "bytetrack"},
        },
    ),
    "A6": Ablation(
        "A6",
        "Does tracking in pitch space cause fewer ID switches than in the image?",
        ("track",),
        ("idf1", "id_switches"),
        {"image": {"tracking.space": "image"}, "pitch": {"tracking.space": "pitch"}},
    ),
    "A7": Ablation(
        "A7",
        "How much does trajectory smoothing change speed and distance estimates?",
        ("project", "analytics"),
        ("speed_outlier_fraction", "stationary_distance_m", "p95_speed_mps"),
        {f"savgol_{w}": {"analytics.smooth_window": w} for w in (0, 15, 25)},
    ),
    "A8": Ablation(
        "A8",
        "Which color space separates the teams best?",
        ("team",),
        ("team_accuracy",),
        {"lab": {"team.color_space": "lab"}, "hsv": {"team.color_space": "hsv"}},
    ),
}


def _prepare_variant_dir(
    base_dir: Path, variant_dir: Path, rerun_artifacts: set[str]
) -> None:
    """Copies the upstream artifacts of the base run into a fresh variant directory."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    skipped = rerun_artifacts | {"config.yaml", "timings.json", OUTPUTS}
    for item in base_dir.iterdir():
        if item.name in skipped:
            continue
        target = variant_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def run_ablation(
    ablation: Ablation,
    base_cfg: PipelineConfig,
    evaluate: Callable[[PipelineConfig, Path], Mapping[str, float]],
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Runs every variant of an ablation and collects its metrics.

    For each variant, the artifacts of the base run are copied (except those that the
    affected stages produce), the stages in ``ablation.stages`` are forced to re-run
    with the overridden configuration, and ``evaluate`` measures the result.

    Args:
        ablation: Experiment definition.
        base_cfg: Configuration of the completed base run; its run directory must
            contain the upstream artifacts.
        evaluate: Callback that receives the variant configuration and its run
            directory and returns metric values.
        output_csv: Optional file that receives the result table.

    Returns:
        Table with one row per variant: ``variant`` followed by the metrics.
    """
    rerun = {art for name in ablation.stages for art in STAGE_REGISTRY[name].produces}
    records = []
    for variant, overrides in ablation.variants.items():
        run_name = f"{base_cfg.run_name}_{ablation.ablation_id}_{variant}"
        cfg = apply_overrides(base_cfg, {**overrides, "run_name": run_name})
        _prepare_variant_dir(base_cfg.run_dir, cfg.run_dir, rerun)
        PipelineRunner(cfg).run(stages=list(ablation.stages), force=True)
        records.append({"variant": variant, **evaluate(cfg, cfg.run_dir)})
    table = pd.DataFrame(records)
    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
    return table
