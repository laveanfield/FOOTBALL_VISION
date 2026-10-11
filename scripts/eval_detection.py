"""Evaluates detections against annotated frames (precision, recall, mAP).

Typical usage example:

    python scripts/eval_detection.py --run-dir outputs/debug
        --ground-truth data/annotations/detection_gt.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from football_vision.artifacts import DETECTIONS, ArtifactStore
from football_vision.eval.detection import evaluate_detections


def main(argv: Sequence[str] | None = None) -> int:
    """Computes and prints the detection metrics.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Evaluate detections on annotated frames."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="CSV with frame_id, x1, y1, x2, y2, cls.",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args(argv)
    predictions = ArtifactStore(args.run_dir).read_table(DETECTIONS)
    ground_truth = pd.read_csv(args.ground_truth)
    predictions = predictions[
        predictions["frame_id"].isin(ground_truth["frame_id"].unique())
    ]
    for name, value in evaluate_detections(predictions, ground_truth, args.iou).items():
        print(f"{name:>20s}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
