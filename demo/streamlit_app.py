"""Streamlit demo that explores the artifacts of a pipeline run.

Typical usage example:

    uv run streamlit run demo/streamlit_app.py
    uv run streamlit run demo/streamlit_app.py -- --run-dir outputs/base_c004_f675

By default the app opens the bundled ``demo/sample_run`` folder so it works
immediately after cloning, without needing raw videos in ``data/raw/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import streamlit as st

from football_vision.artifacts import (
    ANALYTICS,
    DETECTIONS,
    META,
    OUTPUTS,
    PITCH_XY,
    TRACKS,
    ArtifactStore,
    table_to_detections,
)
from football_vision.config import PipelineConfig, load_config
from football_vision.pitch.template import PitchTemplate
from football_vision.video import VideoReader
from football_vision.viz.overlay import (
    draw_detections,
    draw_pitch_template,
    draw_tracks,
)
from football_vision.viz.render import compose_side_by_side
from football_vision.viz.topdown import TopDownRenderer


def parse_args() -> argparse.Namespace:
    """Parses the arguments given after ``--`` on the ``streamlit run`` command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("demo/sample_run"))
    args, _ = parser.parse_known_args()
    return args


def read_frame(cfg: PipelineConfig, frame_id: int):  # noqa: ANN201 (numpy image)
    """Reads one processed frame of the configured video.

    Args:
        cfg: Configuration of the run (video path and processing size).
        frame_id: 0-based source frame id.

    Returns:
        The BGR frame, or ``None`` if the video is unavailable or lacks the frame.
    """
    if cfg.video.path is None or not Path(cfg.video.path).is_file():
        return None
    with VideoReader(
        cfg.video.path, size=cfg.video.size, start=frame_id, stop=frame_id + 1
    ) as reader:
        for _, frame in reader:
            return frame
    return None


def read_video_frame(video_path: Path, frame_index: int):  # noqa: ANN201 (numpy image)
    """Reads one frame from a rendered MP4 video by 0-based offset.

    Args:
        video_path: Path to a rendered MP4 file.
        frame_index: 0-based index inside ``video_path``.

    Returns:
        The BGR frame, or ``None`` if the video or offset cannot be read.
    """
    if not video_path.is_file():
        return None
    idx = max(0, int(frame_index))
    with VideoReader(video_path, size=None, start=idx, stop=idx + 1) as reader:
        for _, frame in reader:
            return frame
    return None


def _frame_bounds(store: ArtifactStore, cfg: PipelineConfig) -> tuple[int, int]:
    """Determines the inclusive ``(min_frame, max_frame)`` covered by a run."""
    for artifact in (TRACKS, PITCH_XY, DETECTIONS):
        if store.exists(artifact):
            table = store.read_table(artifact)
            if len(table) > 0:
                return int(table["frame_id"].min()), int(table["frame_id"].max())
    meta = store.read_meta()
    start = int(cfg.video.start_frame)
    end = (
        min(meta.n_frames, cfg.video.end_frame)
        if cfg.video.end_frame is not None
        else meta.n_frames
    )
    return start, max(start, end - 1)


def main() -> None:
    """Builds the Streamlit page."""
    args = parse_args()
    st.set_page_config(page_title="Football Vision", layout="wide")
    st.title("Football Tactical Vision Analyzer")
    store = ArtifactStore(args.run_dir)
    if not store.exists(META):
        st.error(
            f"No pipeline artifacts found in {args.run_dir}. "
            "Run scripts/run_pipeline.py first."
        )
        return
    cfg = load_config(args.run_dir / "config.yaml")
    min_frame, max_frame = _frame_bounds(store, cfg)

    st.sidebar.header("Layers")
    show_boxes = st.sidebar.checkbox("Boxes and ids", value=True)
    show_lines = st.sidebar.checkbox(
        "Pitch lines", value=store.exists("homography.npy")
    )
    show_topdown = st.sidebar.checkbox("Top-down map", value=store.exists(PITCH_XY))
    if max_frame > min_frame:
        frame_id = st.sidebar.slider("Frame", min_frame, max_frame, min_frame)
    else:
        frame_id = min_frame

    videos = sorted(store.path(OUTPUTS).glob("*.mp4")) if store.exists(OUTPUTS) else []
    if videos:
        st.subheader("Rendered video")
        st.video(str(videos[0]))

    st.subheader(f"Frame {frame_id}")
    frame = read_frame(cfg, frame_id)
    if frame is not None:
        canvas = frame
        source = TRACKS if store.exists(TRACKS) else DETECTIONS
        if show_boxes and store.exists(source):
            rows = store.read_table(source)
            rows = rows[rows["frame_id"] == frame_id]
            canvas = (
                draw_tracks(canvas, rows)
                if source == TRACKS
                else draw_detections(canvas, table_to_detections(rows))
            )
        if show_lines and store.exists("homography.npy"):
            sequence = store.read_homographies()
            if sequence.is_valid(frame_id):
                canvas = draw_pitch_template(
                    canvas, sequence.matrix_for(frame_id), PitchTemplate()
                )
        if show_topdown and store.exists(PITCH_XY):
            pitch_xy = store.read_table(PITCH_XY)
            rows = pitch_xy[pitch_xy["frame_id"] == frame_id]
            canvas = compose_side_by_side(canvas, TopDownRenderer().render(rows))
        st.image(canvas[:, :, ::-1], channels="RGB")
    else:
        st.info(
            "Source video file is not available locally; showing rendered output "
            "and 2D tactical projection from stored artifacts."
        )
        rendered = read_video_frame(videos[0], frame_id - min_frame) if videos else None
        if rendered is not None:
            st.image(rendered[:, :, ::-1], channels="RGB")
        elif show_topdown and store.exists(PITCH_XY):
            pitch_xy = store.read_table(PITCH_XY)
            rows = pitch_xy[pitch_xy["frame_id"] == frame_id]
            topdown = TopDownRenderer().render(rows)
            st.image(topdown[:, :, ::-1], channels="RGB")
        else:
            st.warning("Neither source video nor rendered frames are available.")

    kinematics = store.path(ANALYTICS) / "kinematics.csv"
    if kinematics.is_file():
        st.subheader("Distance and speed per player")
        st.dataframe(pd.read_csv(kinematics))


if __name__ == "__main__":
    main()
