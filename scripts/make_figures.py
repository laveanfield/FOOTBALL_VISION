"""Generates the report figures from the artifacts of a run.

Typical usage example:

    python scripts/make_figures.py --run-dir outputs/debug

Figures whose plotting function is still a stub are reported as ``[todo]``; figures
whose input artifacts are missing are reported as ``[skip]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from football_vision.artifacts import ANALYTICS, HOMOGRAPHY_QUALITY, ArtifactStore
from football_vision.utils import ensure_dir
from football_vision.viz.figures import plot_heatmap_on_pitch, plot_homography_error


def main(argv: Sequence[str] | None = None) -> int:
    """Creates every figure that can be produced and saves it as PNG.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args(argv)
    store = ArtifactStore(args.run_dir)
    out_dir = ensure_dir(args.output_dir)
    jobs: dict[str, Callable[[], Figure]] = {
        "homography_error": lambda: plot_homography_error(
            store.read_table(HOMOGRAPHY_QUALITY)
        ),
        "heatmap_all": lambda: plot_heatmap_on_pitch(
            np.load(store.path(ANALYTICS) / "heatmap_all.npy"), title="All players"
        ),
    }
    # TODO(D): add pipeline-step, team-shape and speed-distribution figures here.
    for name, make in jobs.items():
        try:
            figure = make()
        except NotImplementedError as exc:
            print(f"[todo] {name}: {exc}")
            continue
        except FileNotFoundError as exc:
            print(f"[skip] {name}: missing artifact ({exc})")
            continue
        figure.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        print(f"[ok]   {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
