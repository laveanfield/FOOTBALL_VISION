"""Evaluates stored homographies against held-out landmark annotations.

Typical usage example:

    python scripts/eval_homography.py --run-dir outputs/debug
        --annotations data/annotations/homography_gt.json

The annotation file maps frame ids to clicked landmarks, for example
``{"120": {"corner_tl": [10.0, 20.0], "center": [640.0, 360.0]}}``. These landmarks
must not be the ones used to estimate the homography.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from football_vision.artifacts import ArtifactStore, load_landmark_annotations
from football_vision.eval.homography import (
    evaluate_homography_sequence,
    summarize_errors,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Computes and prints the reprojection error in meters.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate homographies on held-out landmarks."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV with per-landmark errors.",
    )
    args = parser.parse_args(argv)
    sequence = ArtifactStore(args.run_dir).read_homographies()
    table = evaluate_homography_sequence(
        sequence, load_landmark_annotations(args.annotations)
    )
    for key, value in summarize_errors(table["error_m"]).items():
        print(f"{key:>7s}: {value:.3f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
