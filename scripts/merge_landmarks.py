"""Merges per-frame landmark files into one held-out annotation file.

``scripts/annotate_landmarks.py`` saves the clicks of one frame. For evaluation,
``scripts/eval_homography.py`` reads one file that maps frame ids to landmarks. Name
the per-frame files ``<prefix>_f<frame_id>.json`` and merge them:

    python scripts/merge_landmarks.py --inputs data/annotations/heldout/clip_f*.json
        --output data/annotations/clip_heldout.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

FRAME_PATTERN = re.compile(r"_f(\d+)\.json$")


def merge_landmark_files(paths: Sequence[Path]) -> dict[str, dict[str, list[float]]]:
    """Builds the held-out annotation mapping from per-frame files.

    Args:
        paths: Files named ``<prefix>_f<frame_id>.json``.

    Returns:
        Mapping from frame id (as a string) to landmark name to ``[x, y]``.

    Raises:
        ValueError: If a file name carries no frame id.
    """
    merged: dict[str, dict[str, list[float]]] = {}
    for path in sorted(paths):
        match = FRAME_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"{path.name}: expected a name ending in _f<frame>.json")
        clicks = json.loads(path.read_text(encoding="utf-8"))
        merged[str(int(match.group(1)))] = {
            name: [float(v) for v in xy] for name, xy in clicks.items()
        }
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    """Writes the merged annotation file.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code (0 on success, 1 if there was no input).
    """
    parser = argparse.ArgumentParser(description="Merge per-frame landmark files.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    merged = merge_landmark_files(args.inputs)
    if not merged:
        print("No input file.")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    n_points = sum(len(v) for v in merged.values())
    print(f"Wrote {len(merged)} frames / {n_points} landmarks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
