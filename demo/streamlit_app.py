"""Streamlit demo that explores the artifacts of a pipeline run.

Typical usage example:

    streamlit run demo/streamlit_app.py -- --run-dir outputs/debug

The app shows the rendered video, lets you step through frames with selectable
layers, and displays the stored analytics tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import streamlit as st

from football_vision.artifacts import (
    ANALYTICS,
    DETECTIONS,
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
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/debug"))
    args, _ = parser.parse_known_args()
    return args


def read_frame(cfg: PipelineConfig, frame_id: int):  # noqa: ANN201 (numpy image)
    """Reads one processed frame of the configured video.

    Args:
        cfg: Configuration of the run (video path and processing size).
        frame_id: 0-based source frame id.

    Returns:
        The BGR frame, or ``None`` if the video has no such frame.
    """
    with VideoReader(
        cfg.video.path, size=cfg.video.size, start=frame_id, stop=frame_id + 1
    ) as reader:
        for _, frame in reader:
            return frame
    return None


def main() -> None:
    """Builds the Streamlit page."""
    args = parse_args()
    st.set_page_config(page_title="Football Vision", layout="wide")
    st.title("Football Tactical Vision Analyzer")
    store = ArtifactStore(args.run_dir)
    if not store.exists("meta.json"):
        st.error(
            f"No pipeline artifacts found in {args.run_dir}. "
            "Run scripts/run_pipeline.py first."
        )
        return
    meta = store.read_meta()
    cfg = load_config(args.run_dir / "config.yaml")

    st.sidebar.header("Layers")
    show_boxes = st.sidebar.checkbox("Boxes and ids", value=True)
    show_lines = st.sidebar.checkbox(
        "Pitch lines", value=store.exists("homography.npy")
    )
    show_topdown = st.sidebar.checkbox("Top-down map", value=store.exists(PITCH_XY))
    frame_id = st.sidebar.slider("Frame", 0, max(meta.n_frames - 1, 0), 0)

    videos = sorted(store.path(OUTPUTS).glob("*.mp4")) if store.exists(OUTPUTS) else []
    if videos:
        st.subheader("Rendered video")
        st.video(str(videos[0]))

    st.subheader(f"Frame {frame_id}")
    frame = read_frame(cfg, frame_id)
    if frame is None:
        st.warning("Frame not available (check video.path in the run configuration).")
    else:
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

    kinematics = store.path(ANALYTICS) / "kinematics.csv"
    if kinematics.is_file():
        st.subheader("Distance and speed per player")
        st.dataframe(pd.read_csv(kinematics))


if __name__ == "__main__":
    main()
