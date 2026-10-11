"""Runs the Football Vision pipeline from the command line.

Typical usage example:

    python scripts/run_pipeline.py --config configs/walking_skeleton.yaml
        --set video.path=data/raw/clip.mp4

Stages are cached: a stage whose artifacts already exist is skipped unless it is
listed after ``--force`` (use ``--force`` alone to re-run every selected stage).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from football_vision.config import load_config, parse_override_strings
from football_vision.pipeline.runner import PipelineRunner
from football_vision.pipeline.stages import STAGE_ORDER
from football_vision.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Creates the command-line parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description="Run the Football Vision pipeline.")
    parser.add_argument(
        "--config",
        "-c",
        action="append",
        default=[],
        help="YAML configuration file; repeat to layer files (later files win).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted override, for example detection.imgsz=960 (repeatable).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        help="Stages to run (default: the stages listed in the configuration).",
    )
    parser.add_argument(
        "--force",
        nargs="*",
        choices=STAGE_ORDER,
        default=None,
        help=(
            "Re-run cached stages: name them, or give no names to re-run "
            "all selected stages."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the pipeline.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, parse_override_strings(args.overrides))
    logger = setup_logging(args.log_level, cfg.run_dir / "run.log")
    force: bool | list[str] = False if args.force is None else (args.force or True)
    results = PipelineRunner(cfg, logger=logger).run(args.stages, force)
    for result in results:
        print(f"{result.name:12s} {result.status:8s} {result.seconds:8.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
