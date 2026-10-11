"""Evaluates tracks against a MOT-format ground truth (MOTA, IDF1, HOTA).

Typical usage example:

    python scripts/eval_tracking.py --run-dir outputs/debug
        --ground-truth data/annotations/tracking_gt.txt
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from football_vision.artifacts import TRACKS, ArtifactStore
from football_vision.eval.tracking import evaluate_tracking, load_mot_ground_truth


def main(argv: Sequence[str] | None = None) -> int:
    """Computes and prints the tracking metrics on the annotated frame range.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate tracking on an annotated clip."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--ground-truth", type=Path, required=True, help="MOT-format text file."
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Ground-truth frames start at 0 (default: 1).",
    )
    args = parser.parse_args(argv)
    ground_truth = load_mot_ground_truth(
        args.ground_truth, one_based_frames=not args.zero_based
    )
    predictions = ArtifactStore(args.run_dir).read_table(TRACKS)
    predictions = predictions[
        predictions["frame_id"].between(
            ground_truth["frame_id"].min(), ground_truth["frame_id"].max()
        )
    ]
    for name, value in evaluate_tracking(predictions, ground_truth, args.iou).items():
        print(f"{name:>16s}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
