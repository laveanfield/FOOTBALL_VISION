"""Typed configuration dataclasses and YAML loading helpers.

The configuration is a tree of dataclasses rooted at :class:`PipelineConfig`.
Every field has a default, so an empty YAML file is a valid configuration.
Several YAML files can be layered (later files win) and single values can be
overridden with dotted keys such as ``detection.imgsz``.

Typical usage example:

    cfg = load_config(
        ["configs/default.yaml", "configs/fixed_camera.yaml"],
        overrides={"video.path": "data/raw/clip.mp4"},
    )
    print(cfg.detection.imgsz)
"""

from __future__ import annotations

import copy
import dataclasses
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from football_vision.errors import ConfigError


@dataclass
class VideoConfig:
    """Input video selection.

    Attributes:
        path: Path of the input video. It must be set before running the pipeline.
        size: Target ``(width, height)`` in pixels that all frames are resized to.
        start_frame: First source frame (0-based) to process.
        end_frame: Exclusive end frame. ``None`` processes until the video ends.
        stride: Processes every ``stride``-th frame (1 processes every frame).
    """

    path: Path | None = None
    size: tuple[int, int] = (1280, 720)
    start_frame: int = 0
    end_frame: int | None = None
    stride: int = 1


@dataclass
class PreprocessConfig:
    """Settings of stage P1 (preprocessing).

    Attributes:
        detect_shots: Whether to run shot boundary detection.
        shot_threshold: Histogram distance above which a cut is declared.
        compute_pitch_mask: Whether to compute and store grass masks per frame.
        denoise: Denoising method applied before edge extraction.
        use_clahe: Whether to equalize lighting with CLAHE.
        grass_hsv_low: Lower HSV bound of the grass color (OpenCV scale).
        grass_hsv_high: Upper HSV bound of the grass color (OpenCV scale).
        morph_kernel: Kernel size (pixels) of the open/close operations on the mask.
    """

    detect_shots: bool = False
    shot_threshold: float = 0.5
    compute_pitch_mask: bool = False
    denoise: Literal["none", "gaussian", "bilateral"] = "none"
    use_clahe: bool = False
    grass_hsv_low: tuple[int, int, int] = (35, 40, 40)
    grass_hsv_high: tuple[int, int, int] = (85, 255, 255)
    morph_kernel: int = 7


@dataclass
class PitchConfig:
    """Settings of stage P2 (edges, lines and keypoints).

    Attributes:
        edge_method: ``"canny"`` for Canny edges or ``"tophat"`` for white top-hat.
        blur_sigma: Gaussian sigma applied before Canny.
        canny_low: Lower Canny hysteresis threshold.
        canny_high: Upper Canny hysteresis threshold.
        tophat_kernel: Structuring element size of the white top-hat transform.
        hough_rho: Distance resolution of the Hough accumulator (pixels).
        hough_threshold: Minimum number of votes for a Hough line.
        hough_min_length: Minimum segment length (pixels) for probabilistic Hough.
        hough_max_gap: Maximum gap (pixels) bridged inside one Hough segment.
    """

    edge_method: Literal["canny", "tophat"] = "canny"
    blur_sigma: float = 1.4
    canny_low: int = 50
    canny_high: int = 150
    tophat_kernel: int = 15
    hough_rho: float = 1.0
    hough_threshold: int = 80
    hough_min_length: int = 60
    hough_max_gap: int = 10


@dataclass
class HomographyConfig:
    """Settings of stage P3 (homography and temporal alignment).

    Attributes:
        mode: ``"fixed"`` estimates one homography for the whole clip (static
            camera); ``"propagate"`` chains frame-to-frame homographies.
        landmarks_path: JSON file with clicked landmarks of the keyframe.
        ransac_threshold_px: RANSAC inlier threshold in pixels.
        feature: Feature type used for frame-to-frame matching.
        ratio_test: Lowe ratio-test threshold for feature matches.
        min_inliers: Minimum number of inliers for an accepted alignment.
        reanchor_overlay_threshold: Line-overlay score below which a new keyframe
            homography is estimated.
        reanchor_every_n: Forces a re-anchor every N frames (0 disables it).
        smoothing_window: Window (frames) of the temporal smoothing.
    """

    mode: Literal["fixed", "propagate"] = "fixed"
    landmarks_path: Path | None = None
    ransac_threshold_px: float = 3.0
    feature: Literal["orb", "sift", "akaze"] = "sift"
    ratio_test: float = 0.75
    min_inliers: int = 15
    reanchor_overlay_threshold: float = 0.5
    reanchor_every_n: int = 0
    smoothing_window: int = 9


@dataclass
class DetectionConfig:
    """Settings of stage P4 (object detection).

    Attributes:
        weights: Weights name or path understood by Ultralytics YOLO.
        imgsz: Inference resolution; players are small, so use a large value.
        conf: Minimum confidence of kept detections.
        iou: IoU threshold of non-maximum suppression.
        classes: COCO class ids to keep (0 = person, 32 = sports ball).
        device: Torch device string; ``None`` lets Ultralytics choose.
        half: Whether to run inference in FP16.
        min_box_height_px: Boxes shorter than this are discarded.
        use_pitch_mask: Whether to discard boxes whose feet lie outside the pitch.
        pitch_mask_margin_px: Dilation (pixels) applied to the pitch mask.
        use_tiling: Whether to use tiled inference for small objects.
    """

    weights: str = "yolov8m.pt"
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.5
    classes: tuple[int, ...] = (0, 32)
    device: str | None = None
    half: bool = False
    min_box_height_px: float = 20.0
    use_pitch_mask: bool = False
    pitch_mask_margin_px: int = 10
    use_tiling: bool = False


@dataclass
class TrackingConfig:
    """Settings of stage P5 (multi-object tracking).

    Attributes:
        tracker: ``"bytetrack"`` (library baseline) or ``"sort"`` (own SORT).
        space: ``"image"`` tracks boxes in pixels; ``"pitch"`` tracks foot points
            in meters using the per-frame homography.
        high_thresh: Score threshold of the first association round.
        low_thresh: Lowest score used in ByteTrack's second association round.
        match_thresh: Association gate (maximum cost) for matching.
        track_buffer: Frames a lost track is kept before deletion.
        min_track_len: Tracks shorter than this many frames are dropped.
        stitch: Whether to stitch fragmented tracklets (enable once implemented).
        max_stitch_gap: Maximum frame gap bridged by stitching.
    """

    tracker: Literal["bytetrack", "sort"] = "bytetrack"
    space: Literal["image", "pitch"] = "image"
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_track_len: int = 5
    stitch: bool = False
    max_stitch_gap: int = 30


@dataclass
class TeamConfig:
    """Settings of stage P6 (team and role classification).

    Attributes:
        n_clusters: KMeans clusters (2 teams, plus goalkeepers and referees).
        torso_top_ratio: Top of the torso crop as a fraction of box height.
        torso_bottom_ratio: Bottom of the torso crop as a fraction of box height.
        color_space: Color space of the appearance feature.
        exclude_grass: Whether to mask out grass-colored pixels before clustering.
    """

    n_clusters: int = 3
    torso_top_ratio: float = 0.15
    torso_bottom_ratio: float = 0.55
    color_space: Literal["lab", "hsv"] = "lab"
    exclude_grass: bool = True


@dataclass
class AnalyticsConfig:
    """Settings of stage P7 (projection and tactical analytics).

    Attributes:
        pitch_margin_m: Points further than this outside the pitch are invalid.
        smooth_window: Savitzky-Golay window in samples (odd; 0 disables).
        smooth_polyorder: Savitzky-Golay polynomial order.
        max_gap_frames: Largest frame gap treated as one continuous segment.
        heatmap_cell_m: Heatmap cell size in meters.
        heatmap_sigma_m: Gaussian blur of the heatmap in meters.
        max_speed_mps: Speeds above this value are flagged as implausible.
        speed_zone_edges_kmh: Upper bounds (km/h) of the speed zones.
        exclude_goalkeeper_from_shape: Whether team shape ignores goalkeepers.
    """

    pitch_margin_m: float = 3.0
    smooth_window: int = 15
    smooth_polyorder: int = 2
    max_gap_frames: int = 5
    heatmap_cell_m: float = 1.0
    heatmap_sigma_m: float = 2.0
    max_speed_mps: float = 12.0
    speed_zone_edges_kmh: tuple[float, ...] = (7.0, 14.0, 20.0, 25.0)
    exclude_goalkeeper_from_shape: bool = True


@dataclass
class VizConfig:
    """Settings of stage P8 (visualization).

    Attributes:
        trail_length: Number of past positions drawn as a trail.
        draw_pitch_lines: Whether to overlay the pitch template on the video.
        topdown_scale_px_per_m: Resolution of the top-down map in pixels per meter.
        codec: FourCC code of the output video.
    """

    trail_length: int = 50
    draw_pitch_lines: bool = True
    topdown_scale_px_per_m: float = 8.0
    codec: str = "mp4v"


@dataclass
class PipelineConfig:
    """Root configuration of a pipeline run.

    Attributes:
        run_name: Name of the run; artifacts go to ``output_dir / run_name``.
        output_dir: Directory that contains all runs.
        seed: Seed for all random number generators.
        stages: Names of the stages to execute, see ``pipeline.stages``.
        video: Input video settings.
        preprocess: Stage P1 settings.
        pitch: Stage P2 settings.
        homography: Stage P3 settings.
        detection: Stage P4 settings.
        tracking: Stage P5 settings.
        team: Stage P6 settings.
        analytics: Stage P7 settings.
        viz: Stage P8 settings.
    """

    run_name: str = "debug"
    output_dir: Path = Path("outputs")
    seed: int = 42
    stages: list[str] = field(
        default_factory=lambda: ["preprocess", "detect", "render"]
    )
    video: VideoConfig = field(default_factory=VideoConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    @property
    def run_dir(self) -> Path:
        """Directory that receives every artifact of this run."""
        return Path(self.output_dir) / self.run_name


# --------------------------------------------------------------------------- helpers
def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merges ``update`` into a copy of ``base``."""
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_dotted(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Turns ``{"a.b": 1}`` into ``{"a": {"b": 1}}``."""
    tree: dict[str, Any] = {}
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Conflicting override keys around '{dotted}'")
        node[parts[-1]] = value
    return tree


def _coerce(value: Any, hint: Any, key: str) -> Any:
    """Converts a YAML value to the annotated type and validates it."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if value is None:
        return None
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        return _coerce(value, non_none[0], key) if len(non_none) == 1 else value
    if hint is Path:
        return Path(value)
    if origin is typing.Literal:
        if value not in args:
            raise ConfigError(f"'{key}' must be one of {list(args)}, got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"'{key}' must be a list, got {value!r}")
        return tuple(value)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"'{key}' must be a list, got {value!r}")
        return list(value)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint in (int, float, str, bool) and not isinstance(value, hint):
        raise ConfigError(f"'{key}' must be {hint.__name__}, got {value!r}")
    return value


def _build(cls: type, data: Mapping[str, Any], prefix: str = "") -> Any:
    """Builds a (nested) dataclass instance from a mapping, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Section '{prefix.rstrip('.')}' must be a mapping, got {data!r}"
        )
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in names:
            raise ConfigError(f"Unknown config key '{prefix}{key}'")
        hint = hints[key]
        if dataclasses.is_dataclass(hint):
            kwargs[key] = _build(hint, value or {}, f"{prefix}{key}.")
        else:
            kwargs[key] = _coerce(value, hint, f"{prefix}{key}")
    return cls(**kwargs)


def _plain(value: Any) -> Any:
    """Converts paths and tuples to YAML-friendly built-ins."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


# ------------------------------------------------------------------------- public API
def config_from_dict(data: Mapping[str, Any]) -> PipelineConfig:  # noqa: DOC502
    """Builds a validated configuration from a nested mapping.

    Args:
        data: Nested mapping that mirrors the dataclass tree. Missing keys fall
            back to the dataclass defaults.

    Returns:
        The resulting :class:`PipelineConfig`.

    Raises:
        ConfigError: If a key is unknown or a value has the wrong type.
    """
    return _build(PipelineConfig, data)


def config_to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Converts a configuration to a YAML-friendly nested dictionary.

    Args:
        cfg: Configuration to convert.

    Returns:
        A nested dictionary containing only built-in types.
    """
    return _plain(dataclasses.asdict(cfg))


def load_config(  # noqa: DOC503
    paths: str | Path | Sequence[str | Path] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """Loads and merges YAML configuration files.

    Args:
        paths: One YAML file or a sequence of files; later files override
            earlier ones. ``None`` returns the default configuration.
        overrides: Optional dotted-key overrides applied last, for example
            ``{"detection.imgsz": 960}``.

    Returns:
        The merged and validated :class:`PipelineConfig`.

    Raises:
        ConfigError: If a file is not a YAML mapping, or a key or value is invalid.
        FileNotFoundError: If a given file does not exist.
    """
    if paths is None:
        file_list: list[str | Path] = []
    elif isinstance(paths, (str, Path)):
        file_list = [paths]
    else:
        file_list = list(paths)
    data: dict[str, Any] = {}
    for file_path in file_list:
        loaded = yaml.safe_load(Path(file_path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{file_path} must contain a YAML mapping")
        data = _deep_update(data, loaded)
    if overrides:
        data = _deep_update(data, _expand_dotted(overrides))
    return config_from_dict(data)


def apply_overrides(  # noqa: DOC502
    cfg: PipelineConfig, overrides: Mapping[str, Any]
) -> PipelineConfig:
    """Returns a copy of ``cfg`` with dotted-key overrides applied.

    Args:
        cfg: Configuration to start from; it is not modified.
        overrides: Dotted-key overrides, for example ``{"tracking.space": "pitch"}``.

    Returns:
        A new validated :class:`PipelineConfig`.

    Raises:
        ConfigError: If an override uses an unknown key or an invalid value.
    """
    data = _deep_update(config_to_dict(cfg), _expand_dotted(overrides))
    return config_from_dict(data)


def parse_override_strings(items: Sequence[str]) -> dict[str, Any]:
    """Parses ``key=value`` strings, as given on the command line.

    Values are interpreted as YAML, so ``960`` becomes an integer, ``true`` a
    boolean and ``null`` ``None``.

    Args:
        items: Strings such as ``"detection.imgsz=960"``.

    Returns:
        A dictionary of dotted keys to parsed values.

    Raises:
        ConfigError: If an item does not contain ``=``.
    """
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like key=value")
        key, raw = item.split("=", 1)
        parsed[key.strip()] = yaml.safe_load(raw)
    return parsed


def dump_config(cfg: PipelineConfig, path: str | Path) -> Path:
    """Writes a configuration to a YAML file.

    Args:
        cfg: Configuration to save.
        path: Destination file; parent directories are created.

    Returns:
        The path that was written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config_to_dict(cfg), sort_keys=False, allow_unicode=True)
    destination.write_text(text, encoding="utf-8")
    return destination
