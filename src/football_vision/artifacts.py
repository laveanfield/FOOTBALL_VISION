"""Artifact names, table schemas and the on-disk artifact store.

Every pipeline stage communicates through files in the run directory. The names
and column layouts below are the *data contract* between stages (see section 3.2
of the project plan), so modules can be developed in parallel.

Typical usage example:

    store = ArtifactStore("outputs/debug")
    store.write_table(DETECTIONS, detections_table)
    table = store.read_table(DETECTIONS)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from football_vision.constants import ROLE_UNKNOWN, TEAM_UNKNOWN
from football_vision.datatypes import Detections, Homography
from football_vision.errors import SchemaError

# ------------------------------------------------------------------ artifact names
META = "meta.json"
PITCH_MASK = "pitch_mask"  # directory of per-frame PNG masks
PITCH_LINES = "pitch_lines.csv"
HOMOGRAPHY = "homography.npy"
HOMOGRAPHY_QUALITY = "homography_quality.csv"
DETECTIONS = "detections.csv"
TRACKS = "tracks.csv"
TEAM_LABELS = "team_labels.csv"
BALL = "ball.csv"
PITCH_XY = "pitch_xy.csv"
ANALYTICS = "analytics"  # directory
OUTPUTS = "outputs"  # directory

# ----------------------------------------------------------------- table schemas
DETECTION_COLUMNS = ("frame_id", "x1", "y1", "x2", "y2", "conf", "cls")
TRACK_COLUMNS = ("frame_id", "track_id", "x1", "y1", "x2", "y2", "conf", "team", "role")
TEAM_LABEL_COLUMNS = ("track_id", "team", "role")
PITCH_LINE_COLUMNS = ("frame_id", "x1", "y1", "x2", "y2", "orient", "score")
HOMOGRAPHY_QUALITY_COLUMNS = (
    "frame_id",
    "valid",
    "n_inliers",
    "reproj_err_m",
    "overlay_score",
)
PITCH_XY_COLUMNS = ("frame_id", "track_id", "X", "Y", "team", "role", "valid")
BALL_COLUMNS = ("frame_id", "x", "y", "conf", "interpolated")

SCHEMAS: dict[str, tuple[str, ...]] = {
    DETECTIONS: DETECTION_COLUMNS,
    TRACKS: TRACK_COLUMNS,
    TEAM_LABELS: TEAM_LABEL_COLUMNS,
    PITCH_LINES: PITCH_LINE_COLUMNS,
    HOMOGRAPHY_QUALITY: HOMOGRAPHY_QUALITY_COLUMNS,
    PITCH_XY: PITCH_XY_COLUMNS,
    BALL: BALL_COLUMNS,
}


# ----------------------------------------------------------------------- metadata
@dataclass
class VideoMeta:
    """Metadata of the processed video, stored in ``meta.json``.

    Attributes:
        source: Path of the source video as given by the user.
        fps: Frames per second of the source video.
        width: Width in pixels of the frames the pipeline processes.
        height: Height in pixels of the frames the pipeline processes.
        n_frames: Number of frames reported by the container.
        shots: Inclusive ``(first_frame, last_frame)`` ranges, one per shot.
    """

    source: str
    fps: float
    width: int
    height: int
    n_frames: int
    shots: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class HomographySequence:
    """Per-frame homographies together with their quality table.

    Row ``i`` of ``matrices`` belongs to row ``i`` of ``quality``.

    Attributes:
        matrices: Array of shape ``(T, 3, 3)`` mapping image to pitch coordinates.
        quality: Table with the columns of ``HOMOGRAPHY_QUALITY_COLUMNS``.
    """

    matrices: npt.NDArray[np.float64]
    quality: pd.DataFrame
    _index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validates shapes and builds the frame-id lookup.

        Raises:
            SchemaError: If the matrices are not ``(T, 3, 3)`` or the quality table
                has a different length or lacks the required columns.
        """
        if self.matrices.ndim != 3 or self.matrices.shape[1:] != (3, 3):
            raise SchemaError(
                f"matrices must have shape (T, 3, 3), got {self.matrices.shape}"
            )
        missing = [
            c for c in HOMOGRAPHY_QUALITY_COLUMNS if c not in self.quality.columns
        ]
        if missing:
            raise SchemaError(f"quality table is missing columns {missing}")
        if len(self.quality) != len(self.matrices):
            raise SchemaError("matrices and quality table must have the same length")
        self._index = {int(f): i for i, f in enumerate(self.quality["frame_id"])}

    @property
    def frame_ids(self) -> npt.NDArray[np.int64]:
        """Frame ids covered by the sequence, in storage order."""
        return self.quality["frame_id"].to_numpy(dtype=np.int64)

    def has_frame(self, frame_id: int) -> bool:
        """Tells whether a frame has an entry in the sequence.

        Args:
            frame_id: Frame id to look up.

        Returns:
            ``True`` if the frame is present, valid or not.
        """
        return int(frame_id) in self._index

    def matrix_for(self, frame_id: int) -> Homography:  # noqa: DOC502
        """Returns the homography of a frame.

        Args:
            frame_id: Frame id to look up.

        Returns:
            The ``(3, 3)`` matrix mapping image to pitch coordinates.

        Raises:
            KeyError: If the frame is not part of the sequence.
        """
        return self.matrices[self._index[int(frame_id)]]

    def is_valid(self, frame_id: int) -> bool:
        """Tells whether a frame has a usable homography.

        Args:
            frame_id: Frame id to look up.

        Returns:
            ``True`` only if the frame is present and flagged valid.
        """
        idx = self._index.get(int(frame_id))
        return idx is not None and bool(self.quality["valid"].iloc[idx])


# ------------------------------------------------------------------ conversions
def detections_to_table(frame_id: int, dets: Detections) -> pd.DataFrame:
    """Converts the detections of one frame into rows of ``detections.csv``.

    Args:
        frame_id: Frame the detections belong to.
        dets: Detections of that frame.

    Returns:
        A table with the columns of ``DETECTION_COLUMNS``.
    """
    return pd.DataFrame(
        {
            "frame_id": np.full(len(dets), frame_id, dtype=np.int64),
            "x1": dets.boxes_xyxy[:, 0],
            "y1": dets.boxes_xyxy[:, 1],
            "x2": dets.boxes_xyxy[:, 2],
            "y2": dets.boxes_xyxy[:, 3],
            "conf": dets.scores,
            "cls": dets.class_ids,
        },
        columns=list(DETECTION_COLUMNS),
    )


def tracks_to_table(frame_id: int, dets: Detections) -> pd.DataFrame:
    """Converts tracked detections of one frame into rows of ``tracks.csv``.

    Team and role are initialised to their "unknown" values; stage P6 fills them.

    Args:
        frame_id: Frame the detections belong to.
        dets: Detections carrying ``track_ids``.

    Returns:
        A table with the columns of ``TRACK_COLUMNS``.

    Raises:
        ValueError: If ``dets`` has no track ids.
    """
    if dets.track_ids is None:
        raise ValueError(
            "Detections must carry track_ids before they become track rows"
        )
    return pd.DataFrame(
        {
            "frame_id": np.full(len(dets), frame_id, dtype=np.int64),
            "track_id": dets.track_ids,
            "x1": dets.boxes_xyxy[:, 0],
            "y1": dets.boxes_xyxy[:, 1],
            "x2": dets.boxes_xyxy[:, 2],
            "y2": dets.boxes_xyxy[:, 3],
            "conf": dets.scores,
            "team": np.full(len(dets), TEAM_UNKNOWN, dtype=np.int64),
            "role": [ROLE_UNKNOWN] * len(dets),
        },
        columns=list(TRACK_COLUMNS),
    )


def table_to_detections(rows: pd.DataFrame) -> Detections:
    """Converts detection or track rows of one frame back into :class:`Detections`.

    Args:
        rows: Rows of ``detections.csv`` or ``tracks.csv`` for a single frame.
            Track rows have no ``cls`` column; their class id defaults to 0.

    Returns:
        The detections, with ``track_ids`` set if the rows have a ``track_id`` column.
    """
    boxes = rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).reshape(-1, 4)
    scores = rows["conf"].to_numpy(dtype=np.float32)
    if "cls" in rows.columns:
        class_ids = rows["cls"].to_numpy(dtype=np.int64)
    else:
        class_ids = np.zeros(len(rows), dtype=np.int64)
    track_ids = (
        rows["track_id"].to_numpy(dtype=np.int64)
        if "track_id" in rows.columns
        else None
    )
    return Detections(boxes, scores, class_ids, track_ids)


# ----------------------------------------------------------------------- landmarks
def save_landmarks_json(
    landmarks: dict[str, tuple[float, float]], path: str | Path
) -> Path:
    """Saves clicked landmarks of one frame to JSON.

    Args:
        landmarks: Mapping from landmark name (see ``PitchTemplate.landmarks``) to
            the ``(x, y)`` pixel position in the image.
        path: Destination file; parent directories are created.

    Returns:
        The path that was written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [float(x), float(y)] for name, (x, y) in landmarks.items()}
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_landmarks_json(path: str | Path) -> dict[str, tuple[float, float]]:
    """Loads clicked landmarks of one frame from JSON.

    Args:
        path: JSON file written by :func:`save_landmarks_json`.

    Returns:
        Mapping from landmark name to ``(x, y)`` pixel position.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: (float(xy[0]), float(xy[1])) for name, xy in raw.items()}


def load_landmark_annotations(
    path: str | Path,
) -> dict[int, dict[str, tuple[float, float]]]:
    """Loads held-out landmark annotations for several frames.

    The file maps frame ids to landmark dictionaries, for example
    ``{"120": {"corner_tl": [10.0, 20.0]}}``.

    Args:
        path: JSON annotation file.

    Returns:
        Mapping from frame id to a mapping from landmark name to pixel position.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(frame_id): {
            name: (float(xy[0]), float(xy[1])) for name, xy in frame.items()
        }
        for frame_id, frame in raw.items()
    }


# --------------------------------------------------------------------------- store
class ArtifactStore:
    """Reads and writes the artifacts of one pipeline run.

    Attributes:
        root: Run directory that holds every artifact.
    """

    def __init__(self, root: str | Path) -> None:
        """Creates a store and its root directory.

        Args:
            root: Run directory; it is created if it does not exist.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        """Returns the absolute location of an artifact.

        Args:
            name: Artifact name, for example ``DETECTIONS``.

        Returns:
            The path inside the run directory (it may not exist yet).
        """
        return self.root / name

    def exists(self, name: str) -> bool:
        """Tells whether an artifact has been produced.

        Args:
            name: Artifact name. A directory artifact counts as present only if
                it contains at least one entry.

        Returns:
            ``True`` if the artifact exists.
        """
        target = self.path(name)
        if target.is_dir():
            return any(target.iterdir())
        return target.is_file()

    def directory(self, name: str) -> Path:
        """Returns a directory artifact, creating it if necessary.

        Args:
            name: Directory artifact name, for example ``OUTPUTS``.

        Returns:
            The existing directory.
        """
        target = self.path(name)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_table(self, name: str, table: pd.DataFrame) -> Path:
        """Writes a table as CSV, enforcing its schema when one is declared.

        Args:
            name: Artifact name.
            table: Table to store. Extra columns are dropped for schema-bound
                artifacts, and columns are written in schema order.

        Returns:
            The path that was written.

        Raises:
            SchemaError: If a required column is missing.
        """
        schema = SCHEMAS.get(name)
        if schema is not None:
            missing = [c for c in schema if c not in table.columns]
            if missing:
                raise SchemaError(f"{name}: missing columns {missing}")
            table = table.loc[:, list(schema)]
        destination = self.path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        return destination

    def read_table(self, name: str) -> pd.DataFrame:
        """Reads a CSV artifact and validates its schema when one is declared.

        Args:
            name: Artifact name.

        Returns:
            The stored table.

        Raises:
            FileNotFoundError: If the artifact has not been written.
            SchemaError: If a required column is missing.
        """
        source = self.path(name)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact '{name}' not found in {self.root}")
        table = pd.read_csv(source)
        schema = SCHEMAS.get(name)
        if schema is not None:
            missing = [c for c in schema if c not in table.columns]
            if missing:
                raise SchemaError(f"{name}: missing columns {missing}")
        return table

    def write_array(self, name: str, array: npt.NDArray[Any]) -> Path:
        """Writes a NumPy array in ``.npy`` format.

        Args:
            name: Artifact name, ending with ``.npy``.
            array: Array to store.

        Returns:
            The path that was written.
        """
        destination = self.path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, array)
        return destination

    def read_array(self, name: str) -> npt.NDArray[Any]:
        """Reads a ``.npy`` artifact.

        Args:
            name: Artifact name, ending with ``.npy``.

        Returns:
            The stored array.
        """
        return np.load(self.path(name))

    def write_meta(self, meta: VideoMeta) -> Path:
        """Writes ``meta.json``.

        Args:
            meta: Video metadata to store.

        Returns:
            The path that was written.
        """
        destination = self.path(META)
        destination.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        return destination

    def read_meta(self) -> VideoMeta:
        """Reads ``meta.json``.

        Returns:
            The stored video metadata.
        """
        raw = json.loads(self.path(META).read_text(encoding="utf-8"))
        raw["shots"] = [(int(a), int(b)) for a, b in raw.get("shots", [])]
        return VideoMeta(**raw)

    def write_homographies(self, sequence: HomographySequence) -> None:
        """Writes ``homography.npy`` and ``homography_quality.csv``.

        Args:
            sequence: Per-frame homographies and their quality table.
        """
        self.write_array(HOMOGRAPHY, sequence.matrices)
        self.write_table(HOMOGRAPHY_QUALITY, sequence.quality)

    def read_homographies(self) -> HomographySequence:
        """Reads ``homography.npy`` and ``homography_quality.csv``.

        Returns:
            The stored per-frame homographies.
        """
        return HomographySequence(
            self.read_array(HOMOGRAPHY), self.read_table(HOMOGRAPHY_QUALITY)
        )
