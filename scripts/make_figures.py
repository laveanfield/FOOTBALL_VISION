"""Generates the report figures from the artifacts of a run.

Typical usage example:

    python scripts/make_figures.py --run-dir outputs/debug
    python scripts/make_figures.py --run-dir outputs/debug --frame-id 120 \
        --compare-run-dir outputs/debug_A7_savgol_0

Figures whose plotting function is still a stub are reported as ``[todo]``; figures
whose input artifacts are missing are reported as ``[skip]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

from football_vision.artifacts import (
    ANALYTICS,
    HOMOGRAPHY_QUALITY,
    OUTPUTS,
    PITCH_LINES,
    PITCH_XY,
    ArtifactStore,
)
from football_vision.config import PipelineConfig, load_config
from football_vision.eval.speed import per_sample_speeds
from football_vision.pitch.edges import extract_edges
from football_vision.pitch.keypoints import find_line_intersections
from football_vision.preprocessing.frames import preprocess_frame
from football_vision.utils import ensure_dir
from football_vision.video import VideoReader
from football_vision.viz.figures import (
    plot_heatmap_on_pitch,
    plot_homography_error,
    plot_pipeline_steps,
    plot_speed_distribution,
    plot_team_shape,
)
from football_vision.viz.render import make_filmstrip


def pipeline_steps_figure(
    store: ArtifactStore, cfg: PipelineConfig, frame_id: int
) -> Figure:
    """Rebuilds the P2 intermediates of one frame and plots them.

    Edges are recomputed with the configured settings; segments and their families
    come from ``pitch_lines.csv``; keypoints are the intersections of the two line
    families.

    Args:
        store: Artifact store of the run.
        cfg: Configuration used for the run (preprocessing and pitch settings).
        frame_id: Frame to show.

    Returns:
        The figure.

    Raises:
        FileNotFoundError: If the frame cannot be read from the source video.
    """
    meta = store.read_meta()
    with VideoReader(
        meta.source, size=(meta.width, meta.height), start=frame_id, stop=frame_id + 1
    ) as reader:
        frames = [frame for _, frame in reader]
    if not frames:
        raise FileNotFoundError(f"frame {frame_id} of {meta.source}")
    frame = preprocess_frame(frames[0], cfg.preprocess)
    edges = extract_edges(frame, None, cfg.pitch)
    lines = store.read_table(PITCH_LINES)
    lines = lines[lines["frame_id"] == frame_id]
    segments = lines[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
    orient = lines["orient"].to_numpy()
    keypoints = find_line_intersections(
        segments[orient == 0], segments[orient == 1], (frame.shape[1], frame.shape[0])
    )
    return plot_pipeline_steps(frame, edges, segments, keypoints)


def save_filmstrip(
    video: Path, out_path: Path, n_tiles: int = 8, n_cols: int = 4
) -> None:
    """Samples evenly spaced frames from a rendered video and tiles them.

    Args:
        video: Rendered video, for example ``outputs/side_by_side.mp4``.
        out_path: PNG file to write.
        n_tiles: Number of frames to sample.
        n_cols: Number of tiles per row.

    Raises:
        FileNotFoundError: If the video is missing or no frame can be read.
    """
    if not video.exists():
        raise FileNotFoundError(video)
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames, labels = [], []
    for index in np.linspace(0, max(total - 1, 0), n_tiles).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
            labels.append(f"t={index / fps:.1f}s")
    capture.release()
    if not frames:
        raise FileNotFoundError(f"no readable frame in {video}")
    strip = make_filmstrip(frames, n_cols=n_cols, tile_width=480, labels=labels)
    cv2.imwrite(str(out_path), strip)


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
    parser.add_argument("--config", "-c", action="append", default=[])
    parser.add_argument(
        "--frame-id", type=int, default=None, help="frame for the pipeline figure"
    )
    parser.add_argument(
        "--compare-run-dir",
        type=Path,
        default=None,
        help="run without trajectory smoothing, overlaid on the speed histogram (A7)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config or None)
    store = ArtifactStore(args.run_dir)
    out_dir = ensure_dir(args.output_dir)
    analytics_dir = store.path(ANALYTICS)
    max_gap = cfg.analytics.max_gap_frames * cfg.video.stride

    def speeds(run: ArtifactStore) -> npt.NDArray[np.float64]:
        return per_sample_speeds(run.read_table(PITCH_XY), run.read_meta().fps, max_gap)

    def speed_figure() -> Figure:
        reference = (
            speeds(ArtifactStore(args.compare_run_dir))
            if args.compare_run_dir
            else None
        )
        return plot_speed_distribution(
            speeds(store), cfg.analytics.max_speed_mps, reference_mps=reference
        )

    jobs: dict[str, Callable[[], Figure]] = {
        "homography_error": lambda: plot_homography_error(
            store.read_table(HOMOGRAPHY_QUALITY)
        ),
        "heatmap_all": lambda: plot_heatmap_on_pitch(
            np.load(analytics_dir / "heatmap_all.npy"), title="All players"
        ),
        "speed_distribution": speed_figure,
    }
    for path in sorted(analytics_dir.glob("heatmap_team_*.npy")):
        team = path.stem.removeprefix("heatmap_team_")
        jobs[f"heatmap_team_{team}"] = lambda p=path, t=team: plot_heatmap_on_pitch(
            np.load(p), title=f"Team {t}"
        )
    for path in sorted(analytics_dir.glob("team_shape_team_*.csv")):
        team = path.stem.removeprefix("team_shape_team_")
        jobs[f"team_shape_team_{team}"] = lambda p=path, t=team: plot_team_shape(
            pd.read_csv(p), title=f"Team {t} shape", fps=store.read_meta().fps
        )
    if args.frame_id is not None:
        jobs[f"pipeline_steps_{args.frame_id}"] = lambda: pipeline_steps_figure(
            store, cfg, args.frame_id
        )

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

    outputs_dir = store.path(OUTPUTS)
    for video in ("side_by_side.mp4", "overlay.mp4"):
        try:
            save_filmstrip(outputs_dir / video, out_dir / "filmstrip.png")
        except FileNotFoundError:
            continue
        print(f"[ok]   filmstrip ({video})")
        break
    else:
        print("[skip] filmstrip: no rendered video in outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
