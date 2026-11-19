"""Pipeline stages (P1-P8).

Each stage reads the artifacts of earlier stages from the
:class:`~football_vision.artifacts.ArtifactStore`, does its work, and writes its own
artifacts. Because stages only communicate through files, any stage can be re-run
alone (for example after changing a threshold) without recomputing the others.

Stage names, in execution order: ``preprocess``, ``pitch_lines``, ``homography``,
``detect``, ``track``, ``team``, ``project``, ``analytics`` and ``render``.
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pandas as pd

from football_vision.analytics.heatmap import compute_heatmap
from football_vision.analytics.kinematics import summarize_track_kinematics
from football_vision.analytics.projection import project_tracks_to_pitch
from football_vision.analytics.smoothing import smooth_pitch_tracks
from football_vision.analytics.team_shape import team_shape_timeseries
from football_vision.artifacts import (
    ANALYTICS,
    DETECTIONS,
    HOMOGRAPHY,
    HOMOGRAPHY_QUALITY,
    META,
    OUTPUTS,
    PITCH_LINE_COLUMNS,
    PITCH_LINES,
    PITCH_MASK,
    PITCH_XY,
    TEAM_LABELS,
    TRACK_COLUMNS,
    TRACKS,
    ArtifactStore,
    HomographySequence,
    detections_to_table,
    load_landmarks_json,
    table_to_detections,
    tracks_to_table,
)
from football_vision.config import PipelineConfig
from football_vision.constants import COCO_PERSON_ID, TEAM_UNKNOWN
from football_vision.datatypes import Detections, Mask
from football_vision.detection.base import Detector
from football_vision.detection.postprocess import filter_by_pitch_mask, filter_by_size
from football_vision.detection.tiling import TiledDetector
from football_vision.detection.yolo import YoloDetector
from football_vision.errors import HomographyError, PipelineError
from football_vision.homography.dlt import reprojection_error
from football_vision.homography.keyframe import (
    estimate_keyframe_homography_auto,
    homography_from_landmark_dict,
)
from football_vision.homography.propagation import (
    estimate_frame_alignment,
    propagate_homography,
)
from football_vision.homography.quality import (
    check_homography_validity,
    line_overlay_score,
)
from football_vision.homography.smoothing import smooth_homography_sequence
from football_vision.pitch.edges import extract_edges
from football_vision.pitch.hough import (
    cluster_line_families,
    detect_line_segments,
    merge_collinear_segments,
)
from football_vision.pitch.template import PitchTemplate
from football_vision.preprocessing.frames import preprocess_frame
from football_vision.preprocessing.pitch_mask import compute_pitch_mask
from football_vision.preprocessing.shots import detect_shot_boundaries
from football_vision.team.classifier import (
    TeamClassifier,
    assign_teams_and_roles,
    vote_per_track,
)
from football_vision.team.color_features import crop_torso, extract_color_feature
from football_vision.tracking.base import make_tracker
from football_vision.tracking.stitching import drop_short_tracks, stitch_tracklets
from football_vision.video import VideoReader, VideoWriter, frame_id_range, probe_video
from football_vision.viz.overlay import (
    draw_detections,
    draw_pitch_template,
    draw_tracks,
)
from football_vision.viz.render import compose_side_by_side
from football_vision.viz.topdown import TopDownRenderer, TrailBuffer

_VIDEO_PATH_HINT = (
    "video.path is not set; pass --set video.path=<file> or edit the configuration."
)


@dataclass
class RunContext:
    """Everything a stage needs to run.

    Attributes:
        cfg: Pipeline configuration.
        store: Artifact store of the run.
        logger: Logger for progress messages.
        services: Optional injected dependencies, for example ``{"detector": obj}``
            to replace the YOLO detector in tests or notebooks.
    """

    cfg: PipelineConfig
    store: ArtifactStore
    logger: logging.Logger
    services: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Base class of a pipeline stage.

    Attributes:
        name: Name used on the command line and in the configuration.
        requires: Artifacts that must exist before the stage can run.
        produces: Artifacts the stage writes; when all exist the runner skips the stage.
        ctx: Run context the stage is bound to.
    """

    name: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()
    produces: ClassVar[tuple[str, ...]] = ()

    def __init__(self, ctx: RunContext) -> None:
        """Binds the stage to a run context.

        Args:
            ctx: Context with the configuration, store and logger.
        """
        self.ctx = ctx

    def outputs_exist(self) -> bool:
        """Tells whether every artifact of this stage has already been produced.

        Returns:
            ``True`` if the stage declares outputs and all of them exist.
        """
        return bool(self.produces) and all(
            self.ctx.store.exists(a) for a in self.produces
        )

    @abstractmethod
    def run(self) -> None:
        """Executes the stage and writes its artifacts."""


# ------------------------------------------------------------------------- helpers
def _open_video(cfg: PipelineConfig) -> VideoReader:
    """Opens the configured video with the configured frame range and stride."""
    if cfg.video.path is None:
        raise PipelineError(_VIDEO_PATH_HINT)
    return VideoReader(
        cfg.video.path,
        size=cfg.video.size,
        start=cfg.video.start_frame,
        stop=cfg.video.end_frame,
        stride=cfg.video.stride,
    )


def _processed_frame_ids(cfg: PipelineConfig, n_frames: int) -> range:
    """Lists the source frame ids selected by the video settings."""
    return frame_id_range(
        n_frames, cfg.video.start_frame, cfg.video.end_frame, cfg.video.stride
    )


def _group_by_frame(table: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Splits a table with a ``frame_id`` column into per-frame tables."""
    return {int(frame_id): group for frame_id, group in table.groupby("frame_id")}


def _load_mask(mask_dir: Path | None, frame_id: int) -> Mask | None:
    """Reads the stored pitch mask of a frame, if it exists."""
    if mask_dir is None:
        return None
    mask = cv2.imread(str(mask_dir / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return mask


# ---------------------------------------------------------------------- P1 stage
class PreprocessStage(Stage):
    """P1: probes the video, splits it into shots and optionally stores pitch masks."""

    name = "preprocess"
    requires = ()
    produces = (META,)

    def run(self) -> None:
        """Writes ``meta.json`` and, if enabled, ``pitch_mask/`` PNG files."""
        cfg, store = self.ctx.cfg, self.ctx.store
        if cfg.video.path is None:
            raise PipelineError(_VIDEO_PATH_HINT)
        meta = probe_video(cfg.video.path, size=cfg.video.size)
        frame_ids = _processed_frame_ids(cfg, meta.n_frames)
        if cfg.preprocess.detect_shots:
            with _open_video(cfg) as reader:
                prepared = (
                    (fid, preprocess_frame(frame, cfg.preprocess))
                    for fid, frame in reader
                )
                meta.shots = detect_shot_boundaries(
                    prepared, cfg.preprocess.shot_threshold
                )
        else:
            meta.shots = [(frame_ids[0], frame_ids[-1])] if len(frame_ids) else []
        store.write_meta(meta)
        if cfg.preprocess.compute_pitch_mask:
            mask_dir = store.directory(PITCH_MASK)
            with _open_video(cfg) as reader:
                for frame_id, frame in reader:
                    cv2.imwrite(
                        str(mask_dir / f"{frame_id:06d}.png"),
                        compute_pitch_mask(frame, cfg.preprocess),
                    )
        self.ctx.logger.info(
            "Probed %s: %d frames, %d shot(s)",
            cfg.video.path,
            meta.n_frames,
            len(meta.shots),
        )


# ---------------------------------------------------------------------- P2 stage
class PitchLinesStage(Stage):
    """P2: extracts pitch line segments from every frame."""

    name = "pitch_lines"
    requires = (META,)
    produces = (PITCH_LINES,)

    def run(self) -> None:
        """Writes ``pitch_lines.csv`` with merged segments and their families."""
        cfg, store = self.ctx.cfg, self.ctx.store
        mask_dir = store.path(PITCH_MASK) if store.exists(PITCH_MASK) else None
        tables = []
        with _open_video(cfg) as reader:
            for frame_id, frame in reader:
                frame = preprocess_frame(frame, cfg.preprocess)
                edges = extract_edges(frame, _load_mask(mask_dir, frame_id), cfg.pitch)
                segments = merge_collinear_segments(
                    detect_line_segments(edges, cfg.pitch)
                )
                segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
                families = (
                    cluster_line_families(segments)
                    if len(segments)
                    else np.zeros(0, dtype=np.int64)
                )
                lengths = np.hypot(
                    segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]
                )
                tables.append(
                    pd.DataFrame(
                        {
                            "frame_id": frame_id,
                            "x1": segments[:, 0],
                            "y1": segments[:, 1],
                            "x2": segments[:, 2],
                            "y2": segments[:, 3],
                            "orient": families,
                            "score": lengths,
                        }
                    )
                )
        table = (
            pd.concat(tables, ignore_index=True)
            if tables
            else pd.DataFrame(columns=list(PITCH_LINE_COLUMNS))
        )
        store.write_table(PITCH_LINES, table)


# ---------------------------------------------------------------------- P3 stage
class HomographyStage(Stage):
    """P3: estimates the image-to-pitch homography of every processed frame."""

    name = "homography"
    requires = (META,)
    produces = (HOMOGRAPHY, HOMOGRAPHY_QUALITY)

    def run(self) -> None:
        """Writes ``homography.npy`` and ``homography_quality.csv``."""
        cfg, store = self.ctx.cfg, self.ctx.store
        meta = store.read_meta()
        frame_ids = list(_processed_frame_ids(cfg, meta.n_frames))
        if cfg.homography.mode == "fixed":
            sequence = self._run_fixed(frame_ids)
        else:
            sequence = self._run_propagate(frame_ids, meta.shots)
        store.write_homographies(sequence)
        self.ctx.logger.info(
            "Homography: %d frames, mode=%s", len(frame_ids), cfg.homography.mode
        )

    def _run_fixed(self, frame_ids: list[int]) -> HomographySequence:
        """Static camera: one homography from clicked landmarks for the whole clip."""
        hcfg = self.ctx.cfg.homography
        if hcfg.landmarks_path is None:
            raise PipelineError(
                "homography.landmarks_path is required in 'fixed' mode "
                "(create it with scripts/annotate_landmarks.py)."
            )
        template = PitchTemplate()
        landmarks = load_landmarks_json(hcfg.landmarks_path)
        matrix, n_inliers = homography_from_landmark_dict(
            landmarks, template, hcfg.ransac_threshold_px
        )
        names = [n for n in landmarks if n in template.landmarks()]
        errors = reprojection_error(
            matrix,
            np.array([landmarks[n] for n in names]),
            template.landmark_array(names),
        )
        is_valid = int(
            check_homography_validity(matrix, self.ctx.cfg.video.size, template)
        )
        quality = pd.DataFrame(
            {
                "frame_id": frame_ids,
                "valid": is_valid,
                "n_inliers": n_inliers,
                "reproj_err_m": float(errors.mean()),
                "overlay_score": np.nan,
            }
        )
        return HomographySequence(
            np.repeat(matrix[None], len(frame_ids), axis=0), quality
        )

    def _anchor(
        self,
        frame_id: int,
        frame: np.ndarray,
        first_frame_id: int,
        template: PitchTemplate,
        mask: Mask | None = None,
    ) -> np.ndarray:
        """Estimates a keyframe homography from clicked landmarks or automatically."""
        cfg = self.ctx.cfg
        if frame_id == first_frame_id and cfg.homography.landmarks_path is not None:
            landmarks = load_landmarks_json(cfg.homography.landmarks_path)
            return homography_from_landmark_dict(
                landmarks, template, cfg.homography.ransac_threshold_px
            )[0]
        edges = extract_edges(frame, mask, cfg.pitch)
        segments = merge_collinear_segments(detect_line_segments(edges, cfg.pitch))
        return estimate_keyframe_homography_auto(frame, segments, template)

    def _run_propagate(
        self, frame_ids: list[int], shots: list[tuple[int, int]]
    ) -> HomographySequence:
        """Moving camera: anchor at shot starts, chain frame-to-frame alignments."""
        cfg, store = self.ctx.cfg, self.ctx.store
        hcfg = cfg.homography
        template = PitchTemplate()
        shot_starts = {first for first, _ in shots}
        mask_dir = store.path(PITCH_MASK) if store.exists(PITCH_MASK) else None
        size = cfg.video.size
        matrices: list[np.ndarray] = []
        rows: list[dict[str, float]] = []
        previous_frame, h_previous, since_anchor = None, None, 0
        with _open_video(cfg) as reader:
            for frame_id, frame in reader:
                mask = _load_mask(mask_dir, frame_id)
                n_inliers = 0
                if h_previous is None or frame_id in shot_starts:
                    h_current = self._anchor(
                        frame_id, frame, frame_ids[0], template, mask
                    )
                    since_anchor = 0
                else:
                    try:
                        g_current, n_inliers = estimate_frame_alignment(
                            previous_frame,
                            frame,
                            None,
                            hcfg.feature,
                            hcfg.ratio_test,
                            hcfg.ransac_threshold_px,
                            hcfg.min_inliers,
                        )
                        h_current = propagate_homography(h_previous, g_current)
                    except HomographyError:
                        self.ctx.logger.warning(
                            "Frame alignment failed at frame %d; keeping previous H",
                            frame_id,
                        )
                        h_current = h_previous
                        n_inliers = 0
                    since_anchor += 1
                edges = extract_edges(frame, mask, cfg.pitch)
                score = line_overlay_score(edges, h_current, template)
                periodic = (
                    hcfg.reanchor_every_n > 0 and since_anchor >= hcfg.reanchor_every_n
                )
                if score < hcfg.reanchor_overlay_threshold or periodic:
                    try:
                        h_cand = self._anchor(
                            frame_id, frame, frame_ids[0], template, mask
                        )
                        cand_score = line_overlay_score(edges, h_cand, template)
                        if (
                            check_homography_validity(h_cand, size, template)
                            and cand_score > score
                        ):
                            h_current, score, since_anchor = h_cand, cand_score, 0
                        else:
                            self.ctx.logger.warning(
                                "Rejected re-anchor at frame %d "
                                "(score %.3f <= %.3f); keeping propagated H",
                                frame_id,
                                cand_score,
                                score,
                            )
                    except HomographyError:
                        self.ctx.logger.warning(
                            "Re-anchor failed at frame %d; keeping propagated H",
                            frame_id,
                        )
                matrices.append(h_current)
                rows.append(
                    {
                        "frame_id": frame_id,
                        "valid": int(
                            check_homography_validity(h_current, size, template)
                        ),
                        "n_inliers": n_inliers,
                        "reproj_err_m": np.nan,
                        "overlay_score": score,
                    }
                )
                previous_frame, h_previous = frame, h_current
        stacked = np.stack(matrices)
        if hcfg.smoothing_window >= 3:
            reference = np.array(
                [[25.0, 15.0], [80.0, 15.0], [80.0, 53.0], [25.0, 53.0]]
            )
            frame_arr = np.asarray(frame_ids)
            shot_ranges = shots or [(frame_ids[0], frame_ids[-1])]
            for first, last in shot_ranges:
                in_shot = (frame_arr >= first) & (frame_arr <= last)
                if np.any(in_shot):
                    stacked[in_shot] = smooth_homography_sequence(
                        stacked[in_shot], reference, hcfg.smoothing_window
                    )
        return HomographySequence(stacked, pd.DataFrame(rows))


# ---------------------------------------------------------------------- P4 stage
class DetectionStage(Stage):
    """P4: detects people and the ball in every processed frame."""

    name = "detect"
    requires = (META,)
    produces = (DETECTIONS,)

    def _make_detector(self) -> Detector:
        """Returns the injected detector or builds the configured YOLO detector."""
        injected = self.ctx.services.get("detector")
        if injected is not None:
            return injected
        base: Detector = YoloDetector(self.ctx.cfg.detection)
        return TiledDetector(base) if self.ctx.cfg.detection.use_tiling else base

    def run(self) -> None:
        """Writes ``detections.csv`` after size and pitch-mask filtering."""
        cfg, store = self.ctx.cfg, self.ctx.store
        dcfg = cfg.detection
        detector = self._make_detector()
        mask_dir = (
            store.path(PITCH_MASK)
            if dcfg.use_pitch_mask and store.exists(PITCH_MASK)
            else None
        )
        tables = []
        with _open_video(cfg) as reader:
            for frame_id, frame in reader:
                dets = filter_by_size(detector.detect(frame), dcfg.min_box_height_px)
                mask = _load_mask(mask_dir, frame_id)
                if mask is not None:
                    dets = filter_by_pitch_mask(dets, mask, dcfg.pitch_mask_margin_px)
                tables.append(detections_to_table(frame_id, dets))
        table = (
            pd.concat(tables, ignore_index=True)
            if tables
            else detections_to_table(0, Detections.empty())
        )
        store.write_table(DETECTIONS, table)
        self.ctx.logger.info("Detected %d boxes in %d frames", len(table), len(tables))


# ---------------------------------------------------------------------- P5 stage
class TrackingStage(Stage):
    """P5: links person detections into tracks."""

    name = "track"
    requires = (META, DETECTIONS)
    produces = (TRACKS,)

    def run(self) -> None:
        """Writes ``tracks.csv`` after optional stitching and short-track removal."""
        cfg, store = self.ctx.cfg, self.ctx.store
        meta = store.read_meta()
        detections = store.read_table(DETECTIONS)
        people = _group_by_frame(detections[detections["cls"] == COCO_PERSON_ID])
        sequence = None
        if cfg.tracking.space == "pitch":
            if not store.exists(HOMOGRAPHY):
                raise PipelineError(
                    "Tracking in pitch space needs homography.npy; "
                    "run the 'homography' stage first."
                )
            sequence = store.read_homographies()
        tracker = make_tracker(cfg.tracking, fps=meta.fps / cfg.video.stride)
        tables = []
        for frame_id in _processed_frame_ids(cfg, meta.n_frames):
            rows = people.get(frame_id)
            dets = table_to_detections(rows) if rows is not None else Detections.empty()
            h = (
                sequence.matrix_for(frame_id)
                if sequence is not None and sequence.is_valid(frame_id)
                else None
            )
            tracked = tracker.update(dets, frame_id, h)
            if len(tracked):
                tables.append(tracks_to_table(frame_id, tracked))
        tracks = (
            pd.concat(tables, ignore_index=True)
            if tables
            else pd.DataFrame(columns=list(TRACK_COLUMNS))
        )
        if cfg.tracking.stitch and len(tracks):
            tracks = stitch_tracklets(tracks, cfg.tracking.max_stitch_gap)
        tracks = drop_short_tracks(tracks, cfg.tracking.min_track_len)
        store.write_table(TRACKS, tracks)
        self.ctx.logger.info("Tracked %d players", tracks["track_id"].nunique())


# ---------------------------------------------------------------------- P6 stage
class TeamStage(Stage):
    """P6: labels every track with a team and a role."""

    name = "team"
    requires = (META, TRACKS)
    produces = (TEAM_LABELS,)

    def run(self) -> None:
        """Writes ``team_labels.csv`` and fills team/role in ``tracks.csv``."""
        cfg, store = self.ctx.cfg, self.ctx.store
        tcfg = cfg.team
        tracks = store.read_table(TRACKS)
        by_frame = _group_by_frame(tracks)
        samples: list[dict[str, Any]] = []
        features: list[np.ndarray] = []
        with _open_video(cfg) as reader:
            for frame_id, frame in reader:
                for row in by_frame.get(frame_id, pd.DataFrame()).itertuples(
                    index=False
                ):
                    crop = crop_torso(
                        frame,
                        (row.x1, row.y1, row.x2, row.y2),
                        tcfg.torso_top_ratio,
                        tcfg.torso_bottom_ratio,
                    )
                    feature = extract_color_feature(
                        crop, tcfg.color_space, tcfg.exclude_grass
                    )
                    if feature is not None:
                        samples.append(
                            {"frame_id": frame_id, "track_id": int(row.track_id)}
                        )
                        features.append(feature)
        labels = pd.DataFrame(
            {
                "track_id": tracks["track_id"].unique(),
                "team": TEAM_UNKNOWN,
                "role": "unknown",
            }
        )
        if features:
            matrix = np.vstack(features)
            classifier = TeamClassifier(tcfg)
            classifier.fit(matrix)
            samples_table = pd.DataFrame(samples)
            samples_table["cluster"] = classifier.predict(matrix)
            labels = assign_teams_and_roles(vote_per_track(samples_table), tracks, tcfg)
        store.write_table(TEAM_LABELS, labels)
        merged = tracks.drop(columns=["team", "role"]).merge(
            labels, on="track_id", how="left"
        )
        store.write_table(TRACKS, merged)
        self.ctx.logger.info("Labeled %d tracks", len(labels))


# --------------------------------------------------------------------- P7 stages
class ProjectionStage(Stage):
    """P7a: projects tracked players onto the pitch and smooths their trajectories."""

    name = "project"
    requires = (TRACKS, HOMOGRAPHY, HOMOGRAPHY_QUALITY)
    produces = (PITCH_XY,)

    def run(self) -> None:
        """Writes ``pitch_xy.csv``."""
        cfg, store = self.ctx.cfg, self.ctx.store
        acfg = cfg.analytics
        pitch_xy = project_tracks_to_pitch(
            store.read_table(TRACKS), store.read_homographies(), acfg.pitch_margin_m
        )
        pitch_xy = smooth_pitch_tracks(
            pitch_xy,
            acfg.smooth_window,
            acfg.smooth_polyorder,
            acfg.max_gap_frames * cfg.video.stride,
        )
        store.write_table(PITCH_XY, pitch_xy)
        self.ctx.logger.info(
            "Projected %d rows (%d valid)", len(pitch_xy), int(pitch_xy["valid"].sum())
        )


class AnalyticsStage(Stage):
    """P7b: heatmaps, team shape time series and per-player kinematics."""

    name = "analytics"
    requires = (META, PITCH_XY)
    produces = (ANALYTICS,)

    def run(self) -> None:
        """Writes heatmaps, team shape tables and kinematics into ``analytics/``."""
        cfg, store = self.ctx.cfg, self.ctx.store
        meta = store.read_meta()
        acfg = dataclasses.replace(
            cfg.analytics,
            max_gap_frames=cfg.analytics.max_gap_frames * cfg.video.stride,
        )
        pitch_xy = store.read_table(PITCH_XY)
        out_dir = store.directory(ANALYTICS)
        valid = pitch_xy[pitch_xy["valid"] == 1]
        n_frames = max(int(valid["frame_id"].nunique()), 1)

        def heat(points: np.ndarray) -> np.ndarray:
            return compute_heatmap(
                points,
                acfg.heatmap_cell_m,
                acfg.heatmap_sigma_m,
                visible_frames=n_frames,
            )

        np.save(out_dir / "heatmap_all.npy", heat(valid[["X", "Y"]].to_numpy()))
        for team_id in sorted(
            int(t) for t in valid["team"].unique() if t != TEAM_UNKNOWN
        ):
            np.save(
                out_dir / f"heatmap_team_{team_id}.npy",
                heat(valid[valid["team"] == team_id][["X", "Y"]].to_numpy()),
            )
            shape = team_shape_timeseries(
                pitch_xy, team_id, acfg.exclude_goalkeeper_from_shape
            )
            shape.to_csv(out_dir / f"team_shape_team_{team_id}.csv", index=False)
        summarize_track_kinematics(pitch_xy, meta.fps, acfg).to_csv(
            out_dir / "kinematics.csv", index=False
        )
        self.ctx.logger.info("Analytics written to %s", out_dir)


# ---------------------------------------------------------------------- P8 stage
class RenderStage(Stage):
    """P8: renders the overlay video or the side-by-side video with the map."""

    name = "render"
    requires = (META,)
    produces = (OUTPUTS,)

    def run(self) -> None:
        """Writes ``outputs/overlay.mp4`` or ``outputs/side_by_side.mp4``.

        The richest video that the available artifacts allow is produced: boxes only
        after detection, boxes with ids after tracking, pitch lines when homographies
        exist, and a top-down panel when ``pitch_xy.csv`` exists.
        """
        cfg, store = self.ctx.cfg, self.ctx.store
        meta = store.read_meta()
        has_tracks = store.exists(TRACKS)
        if not has_tracks and not store.exists(DETECTIONS):
            raise PipelineError("Nothing to render: run the 'detect' stage first.")
        boxes = _group_by_frame(store.read_table(TRACKS if has_tracks else DETECTIONS))
        pitch_xy = (
            _group_by_frame(store.read_table(PITCH_XY))
            if store.exists(PITCH_XY)
            else None
        )
        sequence = None
        if store.exists(HOMOGRAPHY) and store.exists(HOMOGRAPHY_QUALITY):
            sequence = store.read_homographies()
        template = PitchTemplate()
        renderer = (
            TopDownRenderer(template, cfg.viz.topdown_scale_px_per_m)
            if pitch_xy is not None
            else None
        )
        trails = TrailBuffer(cfg.viz.trail_length)
        out_path = store.directory(OUTPUTS) / (
            "side_by_side.mp4" if renderer else "overlay.mp4"
        )
        with ExitStack() as stack:
            reader = stack.enter_context(_open_video(cfg))
            writer: VideoWriter | None = None
            for frame_id, frame in reader:
                canvas = frame
                rows = boxes.get(frame_id)
                if rows is not None:
                    canvas = (
                        draw_tracks(canvas, rows)
                        if has_tracks
                        else draw_detections(canvas, table_to_detections(rows))
                    )
                if (
                    cfg.viz.draw_pitch_lines
                    and sequence is not None
                    and sequence.is_valid(frame_id)
                ):
                    canvas = draw_pitch_template(
                        canvas, sequence.matrix_for(frame_id), template
                    )
                if renderer is not None and pitch_xy is not None:
                    xy_rows = pitch_xy.get(frame_id)
                    trails.update(frame_id, xy_rows)
                    empty = pd.DataFrame(
                        columns=["track_id", "X", "Y", "team", "role", "valid"]
                    )
                    canvas = compose_side_by_side(
                        canvas,
                        renderer.render(
                            xy_rows if xy_rows is not None else empty, trails.trails()
                        ),
                    )
                if writer is None:
                    size = (canvas.shape[1], canvas.shape[0])
                    writer = stack.enter_context(
                        VideoWriter(
                            out_path, meta.fps / cfg.video.stride, size, cfg.viz.codec
                        )
                    )
                writer.write(canvas)
        self.ctx.logger.info("Wrote %s", out_path)


STAGE_ORDER: tuple[str, ...] = (
    "preprocess",
    "pitch_lines",
    "homography",
    "detect",
    "track",
    "team",
    "project",
    "analytics",
    "render",
)

STAGE_REGISTRY: dict[str, type[Stage]] = {
    cls.name: cls
    for cls in (
        PreprocessStage,
        PitchLinesStage,
        HomographyStage,
        DetectionStage,
        TrackingStage,
        TeamStage,
        ProjectionStage,
        AnalyticsStage,
        RenderStage,
    )
}
