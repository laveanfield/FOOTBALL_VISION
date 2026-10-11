"""Runs one ablation experiment (A1-A8) on top of a finished base run.

Typical usage example:

    python scripts/run_ablation.py --ablation A5 --config configs/fixed_camera.yaml
        --set video.path=data/raw/clip.mp4

The base run (named by the configuration) must already contain the artifacts that
the ablation's stages depend on. Replace :func:`evaluate_variant` with real metric
computation (see ``football_vision.eval``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from football_vision.config import PipelineConfig, load_config, parse_override_strings
from football_vision.eval.ablation import ABLATIONS, run_ablation
from football_vision.utils import setup_logging


def evaluate_variant(cfg: PipelineConfig, run_dir: Path) -> dict[str, float]:
    """Measures one ablation variant.

    The default implementation only reports the stage timings. Extend it with the
    metrics named by the ablation (mAP, IDF1, reprojection error, ...), loading the
    ground truth from ``data/annotations``.

    Args:
        cfg: Configuration of the variant run.
        run_dir: Directory with the variant's artifacts.

    Returns:
        Mapping from metric name to value.
    """
    # TODO(D): call eval.tracking / eval.homography / eval.team functions per ablation.
    timings = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
    return {"stage_seconds": float(sum(t["seconds"] for t in timings.values()))}


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
    args = parser.parse_args(argv)
    cfg = load_config(args.config, parse_override_strings(args.overrides))
    setup_logging("INFO")
    ablation = ABLATIONS[args.ablation]
    print(f"{ablation.ablation_id}: {ablation.question}")
    table = run_ablation(
        ablation,
        cfg,
        evaluate_variant,
        args.output_dir / f"ablation_{ablation.ablation_id}.csv",
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
