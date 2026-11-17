"""Converts SportsMOT sequences into this project's video and ground-truth layout.

SportsMOT (Cui et al., ICCV 2023; CC BY-NC 4.0) stores every clip as a MOTChallenge
sequence: ``<seq>/img1/000001.jpg ...``, ``<seq>/gt/gt.txt`` and ``<seq>/seqinfo.ini``.
The pipeline reads videos and the evaluation code reads MOT text files and a
detection CSV, so each sequence becomes:

* ``data/raw/<seq>.mp4``: the frames encoded at the sequence frame rate;
* ``data/annotations/<seq>_mot.txt``: the tracking ground truth (MOT format,
  1-based frames, read by ``eval.tracking.load_mot_ground_truth``);
* ``data/annotations/<seq>_detections.csv``: boxes of evenly spaced frames with the
  columns ``frame_id, x1, y1, x2, y2, cls`` (0-based frames, ``cls`` = COCO person);
* ``data/annotations/<seq>_teams.csv``: one row per ground-truth track with an empty
  ``team`` column to fill in by hand (ablation A8).

Only players are annotated in SportsMOT (no referees, coaches or ball), so detection
precision measured on these files is a lower bound: correct referee detections count
as false positives.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_vision.constants import COCO_PERSON_ID
from football_vision.video import VideoWriter

MOT_COLUMNS = ("frame", "id", "left", "top", "width", "height", "conf", "cls", "vis")


@dataclass(frozen=True)
class SequenceInfo:
    """Metadata of one MOTChallenge sequence, read from ``seqinfo.ini``.

    Attributes:
        name: Sequence name, for example ``v_1yHWGw8DH4A_c029``.
        directory: Folder that contains ``img1/`` and ``seqinfo.ini``.
        fps: Frame rate.
        width: Image width in pixels.
        height: Image height in pixels.
        n_frames: Number of frames.
        image_dir: Name of the image folder (``img1``).
        image_ext: Image extension including the dot (``.jpg``).
    """

    name: str
    directory: Path
    fps: float
    width: int
    height: int
    n_frames: int
    image_dir: str = "img1"
    image_ext: str = ".jpg"

    @property
    def duration_s(self) -> float:
        """Returns the clip length in seconds."""
        return self.n_frames / self.fps


def read_seqinfo(directory: str | Path) -> SequenceInfo:
    """Reads ``seqinfo.ini`` of a MOTChallenge sequence.

    Args:
        directory: Sequence folder.

    Returns:
        The parsed metadata.

    Raises:
        FileNotFoundError: If ``seqinfo.ini`` is missing.
    """
    folder = Path(directory)
    ini = folder / "seqinfo.ini"
    if not ini.exists():
        raise FileNotFoundError(ini)
    parser = configparser.ConfigParser()
    parser.read(ini, encoding="utf-8")
    seq = parser["Sequence"]
    return SequenceInfo(
        name=seq.get("name", folder.name),
        directory=folder,
        fps=float(seq["frameRate"]),
        width=int(seq["imWidth"]),
        height=int(seq["imHeight"]),
        n_frames=int(seq["seqLength"]),
        image_dir=seq.get("imDir", "img1"),
        image_ext=seq.get("imExt", ".jpg"),
    )


def find_sequences(root: str | Path, names: list[str] | None = None) -> list[Path]:
    """Finds sequence folders (those containing ``seqinfo.ini``) below ``root``.

    Args:
        root: Folder where the SportsMOT archive was extracted.
        names: Optional sequence names to keep; ``None`` keeps all.

    Returns:
        Sorted sequence folders.
    """
    found = sorted(p.parent for p in Path(root).rglob("seqinfo.ini"))
    if names is not None:
        wanted = set(names)
        found = [p for p in found if p.name in wanted]
    return found


def frames_to_video(info: SequenceInfo, output: str | Path) -> Path:
    """Encodes the image folder of a sequence as an MP4 video.

    Args:
        info: Sequence metadata.
        output: Destination video file.

    Returns:
        The path that was written.

    Raises:
        FileNotFoundError: If a frame image is missing or unreadable.
    """
    image_dir = info.directory / info.image_dir
    with VideoWriter(output, info.fps, (info.width, info.height)) as writer:
        for index in range(1, info.n_frames + 1):
            path = image_dir / f"{index:06d}{info.image_ext}"
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(path)
            if (frame.shape[1], frame.shape[0]) != (info.width, info.height):
                frame = cv2.resize(frame, (info.width, info.height))
            writer.write(frame)
    return Path(output)


def read_mot_gt(path: str | Path) -> pd.DataFrame:
    """Reads a MOTChallenge ``gt.txt`` file.

    Args:
        path: Ground-truth file.

    Returns:
        Table with the columns of ``MOT_COLUMNS`` (missing trailing columns are
        filled with 1).
    """
    raw = pd.read_csv(path, header=None)
    for i in range(raw.shape[1], len(MOT_COLUMNS)):
        raw[i] = 1
    raw = raw.iloc[:, : len(MOT_COLUMNS)]
    raw.columns = list(MOT_COLUMNS)
    return raw


def mot_to_detection_gt(mot: pd.DataFrame, n_frames: int, every: int) -> pd.DataFrame:
    """Turns MOT boxes of every ``every``-th frame into a detection ground truth.

    Args:
        mot: Output of ``read_mot_gt``.
        n_frames: Number of frames in the sequence.
        every: Sampling step in frames (keeps frames 0, every, 2*every, ...).

    Returns:
        Table with the columns ``frame_id, x1, y1, x2, y2, cls`` (0-based frames).
    """
    keep = set(range(0, n_frames, max(every, 1)))
    rows = mot.assign(frame_id=mot["frame"] - 1)
    rows = rows[rows["frame_id"].isin(keep) & (rows["conf"] != 0)]
    return pd.DataFrame(
        {
            "frame_id": rows["frame_id"].astype(int),
            "x1": rows["left"],
            "y1": rows["top"],
            "x2": rows["left"] + rows["width"],
            "y2": rows["top"] + rows["height"],
            "cls": COCO_PERSON_ID,
        }
    ).reset_index(drop=True)


def team_template(mot: pd.DataFrame) -> pd.DataFrame:
    """Lists ground-truth tracks so their team can be filled in by hand.

    Args:
        mot: Output of ``read_mot_gt``.

    Returns:
        Table with ``track_id``, ``first_frame_id`` (0-based, where to look),
        ``n_frames`` and an empty ``team`` column (to fill with 0 or 1).
    """
    grouped = mot.groupby("id")["frame"]
    return pd.DataFrame(
        {
            "track_id": grouped.min().index.astype(int),
            "first_frame_id": (grouped.min() - 1).to_numpy(dtype=int),
            "n_frames": grouped.size().to_numpy(dtype=int),
            "team": "",
        }
    )


def evenly_spaced_frames(n_frames: int, count: int) -> list[int]:
    """Picks ``count`` evenly spaced 0-based frame ids, for held-out landmarks.

    Args:
        n_frames: Number of frames in the clip.
        count: Number of frames to pick.

    Returns:
        Sorted unique frame ids.
    """
    if n_frames <= 0 or count <= 0:
        return []
    return sorted({int(i) for i in np.linspace(0, n_frames - 1, count)})


def convert_sequence(
    info: SequenceInfo,
    raw_dir: str | Path,
    annotations_dir: str | Path,
    detection_every: int = 5,
    write_video: bool = True,
) -> dict[str, Path]:
    """Converts one sequence into the project layout.

    Args:
        info: Sequence metadata.
        raw_dir: Folder for the video (``data/raw``).
        annotations_dir: Folder for the ground truth (``data/annotations``).
        detection_every: Sampling step of the detection ground truth.
        write_video: Whether to encode the video (skip to refresh only the labels).

    Returns:
        Mapping from output kind to the written path.

    Raises:
        FileNotFoundError: If the sequence has no ``gt/gt.txt`` (test split).
    """
    gt_path = info.directory / "gt" / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"{gt_path} (test sequences have no ground truth)")
    ann = Path(annotations_dir)
    ann.mkdir(parents=True, exist_ok=True)
    mot = read_mot_gt(gt_path)
    out = {
        "mot": ann / f"{info.name}_mot.txt",
        "detections": ann / f"{info.name}_detections.csv",
    }
    mot.to_csv(out["mot"], header=False, index=False)
    mot_to_detection_gt(mot, info.n_frames, detection_every).to_csv(
        out["detections"], index=False
    )
    teams = ann / f"{info.name}_teams.csv"
    if not teams.exists():  # never overwrite labels filled in by hand
        team_template(mot).to_csv(teams, index=False)
    out["teams"] = teams
    if write_video:
        out["video"] = frames_to_video(info, Path(raw_dir) / f"{info.name}.mp4")
    return out


def readme_row(info: SequenceInfo, role: str = "dev") -> str:
    """Formats the ``data/README.md`` table row of a converted sequence.

    Args:
        info: Sequence metadata.
        role: Role of the clip in the experiments (dev, test, hard case).

    Returns:
        One Markdown table row.
    """
    return (
        f"| {info.name} | SportsMOT (MultiSports, YouTube) | CC BY-NC 4.0 | "
        f"{info.fps:g} | {info.width}x{info.height} | {info.duration_s:.1f} s | "
        f"{role} |"
    )
