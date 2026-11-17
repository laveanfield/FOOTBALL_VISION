"""Converts extracted SportsMOT sequences into ``data/raw`` and ``data/annotations``.

Typical usage example:

    python scripts/prepare_sportsmot.py --root ../datasets/sportsmot/train
        --names v_1yHWGw8DH4A_c029 v_HdiyOtliFiw_c003

Without ``--names`` every sequence below ``--root`` is converted. ``--football``
keeps only the sequences listed in SportsMOT's ``splits_txt/football.txt``. The
script prints the rows to paste into ``data/README.md`` and suggests frames for the
held-out landmark annotation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from football_vision.datasets.sportsmot import (
    convert_sequence,
    evenly_spaced_frames,
    find_sequences,
    read_seqinfo,
    readme_row,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Converts the selected sequences and prints a summary.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code (0 on success, 1 if nothing was converted).
    """
    parser = argparse.ArgumentParser(description="Convert SportsMOT sequences.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument(
        "--football",
        type=Path,
        default=None,
        metavar="FOOTBALL_TXT",
        help="splits_txt/football.txt; keep only the football sequences",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("data/annotations")
    )
    parser.add_argument("--detection-every", type=int, default=5)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--landmark-frames", type=int, default=30)
    args = parser.parse_args(argv)

    names = args.names
    if args.football is not None:
        football = args.football.read_text(encoding="utf-8").split()
        names = [n for n in (names or football) if n in set(football)]
    sequences = find_sequences(args.root, names)
    if not sequences:
        print(f"No sequence found below {args.root} (names: {names})")
        return 1

    rows = []
    for folder in sequences:
        info = read_seqinfo(folder)
        try:
            outputs = convert_sequence(
                info,
                args.raw_dir,
                args.annotations_dir,
                args.detection_every,
                write_video=not args.no_video,
            )
        except FileNotFoundError as exc:
            print(f"[skip] {info.name}: {exc}")
            continue
        rows.append(readme_row(info))
        print(f"[ok]   {info.name}: " + ", ".join(str(p) for p in outputs.values()))
        frames = evenly_spaced_frames(info.n_frames, args.landmark_frames)
        print(f"       held-out landmark frames: {' '.join(map(str, frames))}")

    if rows:
        print("\nRows for data/README.md:")
        print("\n".join(rows))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
