"""Matplotlib figures for the report (stage P8).

Every function returns a :class:`matplotlib.figure.Figure` so the caller decides
whether to show or save it (``fig.savefig("reports/figures/name.png", dpi=200)``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from football_vision.datatypes import Frame, Mask, Points2D
from football_vision.pitch.template import PitchTemplate


def _draw_pitch_lines(ax: Axes, template: PitchTemplate, color: str = "white") -> None:
    """Draws the pitch markings of ``template`` on ``ax`` in meter coordinates."""
    for line in template.polylines():
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.0)


def plot_pipeline_steps(
    frame: Frame,
    edges: Mask,
    segments: npt.NDArray[np.float64],
    keypoints: Points2D,
) -> Figure:
    """Shows the geometric pipeline in a 2 x 2 grid.

    Panels: original frame, edge map, Hough segments over the frame, and keypoints.

    Args:
        frame: BGR frame.
        edges: Binary edge map.
        segments: Merged segments of shape ``(N, 4)``.
        keypoints: Candidate keypoints of shape ``(K, 2)``.

    Returns:
        The figure.
    """
    rgb = np.ascontiguousarray(frame[..., ::-1])
    segs = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    pts = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    fig = Figure(figsize=(12, 7), layout="constrained")
    axes = fig.subplots(2, 2)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Input frame")
    axes[0, 1].imshow(edges, cmap="gray")
    axes[0, 1].set_title("2. Line edges")
    axes[1, 0].imshow(rgb)
    for x1, y1, x2, y2 in segs:
        axes[1, 0].plot([x1, x2], [y1, y2], color="yellow", linewidth=1.5)
    axes[1, 0].set_title(f"3. Hough segments (N={len(segs)})")
    axes[1, 1].imshow(rgb)
    if len(pts):
        axes[1, 1].scatter(pts[:, 0], pts[:, 1], s=30, c="red", edgecolors="white")
    axes[1, 1].set_title(f"4. Keypoints (K={len(pts)})")
    height, width = rgb.shape[:2]
    for ax in axes.flat:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_axis_off()
    return fig


def plot_heatmap_on_pitch(
    heatmap: npt.NDArray[np.float64],
    template: PitchTemplate | None = None,
    title: str = "",
) -> Figure:
    """Draws a heatmap with the pitch markings on top.

    Rows of the heatmap follow ``Y`` and columns follow ``X`` (the convention of
    ``analytics.heatmap``), and ``Y`` grows downward as in the template.

    Args:
        heatmap: Array of shape ``(rows, cols)`` from ``analytics.heatmap``.
        template: Pitch model for the markings; defaults to the 105 x 68 m template.
        title: Figure title.

    Returns:
        The figure.
    """
    model = template or PitchTemplate()
    fig = Figure(figsize=(10.5, 7.2), layout="constrained")
    ax = fig.add_subplot()
    image = ax.imshow(
        np.asarray(heatmap, dtype=np.float64),
        extent=(0.0, model.length, model.width, 0.0),
        origin="upper",
        cmap="magma",
        interpolation="bilinear",
    )
    _draw_pitch_lines(ax, model)
    ax.set_xlim(0, model.length)
    ax.set_ylim(model.width, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.colorbar(image, ax=ax, shrink=0.8, label="share of visible time")
    if title:
        ax.set_title(title)
    return fig


def plot_team_shape(
    shape_series: pd.DataFrame, title: str = "", fps: float | None = None
) -> Figure:
    """Plots team length, width and hull area over time.

    Length and width share one panel (meters); the hull area gets its own panel
    (square meters) so the two units are never mixed on one axis.

    Args:
        shape_series: Output of ``analytics.team_shape.team_shape_timeseries``.
        title: Figure title.
        fps: Frame rate; when given the x axis is in seconds, otherwise in frames.

    Returns:
        The figure.
    """
    data = shape_series.sort_values("frame_id")
    x = data["frame_id"].to_numpy(dtype=np.float64)
    x_label = "frame"
    if fps:
        x = x / fps
        x_label = "time (s)"
    fig = Figure(figsize=(10, 6), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True)
    top.plot(x, data["length_m"], label="length")
    top.plot(x, data["width_m"], label="width")
    if "stretch_index_m" in data:
        top.plot(x, data["stretch_index_m"], label="stretch index", linestyle="--")
    top.set_ylabel("meters")
    top.legend(loc="upper right")
    top.grid(alpha=0.3)
    bottom.plot(x, data["hull_area_m2"], color="tab:green")
    bottom.set_ylabel("hull area (m²)")
    bottom.set_xlabel(x_label)
    bottom.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    return fig


def plot_speed_distribution(
    speeds_mps: npt.NDArray[np.float64],
    max_speed_mps: float = 12.0,
    reference_mps: npt.ArrayLike | None = None,
    labels: Sequence[str] = ("smoothed", "unsmoothed"),
) -> Figure:
    """Plots the histogram of player speeds and marks the plausibility limit.

    Pass the unsmoothed speeds as ``reference_mps`` to overlay both distributions
    for ablation A7: without smoothing, position noise inflates the speed tail.

    Args:
        speeds_mps: Speeds in meters per second.
        max_speed_mps: Speeds above this value are drawn as implausible.
        reference_mps: Optional second sample drawn on the same bins.
        labels: Legend labels of ``speeds_mps`` and ``reference_mps``.

    Returns:
        The figure.
    """
    main = np.asarray(speeds_mps, dtype=np.float64)
    main = main[np.isfinite(main)]
    samples = [(main, labels[0])]
    if reference_mps is not None:
        ref = np.asarray(reference_mps, dtype=np.float64)
        samples.append((ref[np.isfinite(ref)], labels[1]))
    top = max([max_speed_mps * 1.5, *(float(s.max()) for s, _ in samples if len(s))])
    bins = np.linspace(0.0, top, 61)
    fig = Figure(figsize=(9, 5), layout="constrained")
    ax = fig.add_subplot()
    for sample, label in samples:
        outliers = float((sample > max_speed_mps).mean()) if len(sample) else np.nan
        ax.hist(
            sample,
            bins=bins,
            density=True,
            alpha=0.55,
            label=f"{label} (n={len(sample)}, >{max_speed_mps:g} m/s: {outliers:.1%})",
        )
    ax.axvline(max_speed_mps, color="red", linestyle="--", label="plausibility limit")
    ax.axvspan(max_speed_mps, top, color="red", alpha=0.08)
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("density")
    ax.set_xlim(0, top)
    ax.legend()
    return fig


def plot_homography_error(
    quality: pd.DataFrame, reanchor_frames: Sequence[int] = ()
) -> Figure:
    """Plots homography quality over time.

    Frames whose homography is invalid are shaded. The quality table has no
    re-anchor column, so re-anchor events are drawn only when the caller passes
    them explicitly.

    Args:
        quality: Table with the columns of ``HOMOGRAPHY_QUALITY_COLUMNS``.
        reanchor_frames: Frame ids at which the homography was re-anchored.

    Returns:
        The figure with the reprojection error and the line-overlay score.
    """
    data = quality.sort_values("frame_id")
    frames = data["frame_id"].to_numpy()
    invalid = data["valid"].to_numpy() == 0
    fig = Figure(figsize=(11, 6), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True)
    top.plot(frames, data["reproj_err_m"], color="tab:blue", linewidth=1.0)
    top.set_ylabel("reprojection error (m)")
    bottom.plot(frames, data["overlay_score"], color="tab:orange", linewidth=1.0)
    bottom.set_ylabel("line-overlay score")
    bottom.set_xlabel("frame")
    for ax in (top, bottom):
        if invalid.any():
            ax.fill_between(
                frames,
                0,
                1,
                where=invalid,
                transform=ax.get_xaxis_transform(),
                color="red",
                alpha=0.15,
                step="mid",
                label="invalid H",
            )
        for i, frame_id in enumerate(reanchor_frames):
            ax.axvline(
                frame_id,
                color="green",
                linestyle=":",
                label="re-anchor" if i == 0 else None,
            )
        ax.grid(alpha=0.3)
    top.set_title(f"valid frames: {1 - invalid.mean():.1%}" if len(data) else "")
    if invalid.any() or len(reanchor_frames):
        top.legend(loc="upper right")
    return fig
