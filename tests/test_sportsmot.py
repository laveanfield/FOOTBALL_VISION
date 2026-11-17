"""Tests for the SportsMOT converter and the landmark merge helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_vision.artifacts import load_landmark_annotations
from football_vision.datasets.sportsmot import (
    convert_sequence,
    evenly_spaced_frames,
    find_sequences,
    read_seqinfo,
    readme_row,
)
from football_vision.eval.tracking import load_mot_ground_truth
from football_vision.video import probe_video

ROOT = Path(__file__).resolve().parents[1]


def _make_sequence(root: Path, name: str = "v_test_c001", n: int = 12) -> Path:
    """Writes a tiny MOTChallenge sequence with two tracks."""
    seq = root / "train" / name
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    for i in range(1, n + 1):
        img = np.full((48, 64, 3), 10 * i, np.uint8)
        cv2.imwrite(str(seq / "img1" / f"{i:06d}.jpg"), img)
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=25\nseqLength={n}\n"
        "imWidth=64\nimHeight=48\nimExt=.jpg\n",
        encoding="utf-8",
    )
    rows = []
    for f in range(1, n + 1):
        for tid in (1, 2):
            rows.append(f"{f},{tid},{5 * tid},4,10,20,1,1,1")
    (seq / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return seq


def _load_script(name: str) -> object:
    """Imports a module from ``scripts/`` without making it a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seqinfo_and_discovery(tmp_path: Path) -> None:
    """Checks that sequences are found and their metadata parsed."""
    seq = _make_sequence(tmp_path)
    assert find_sequences(tmp_path) == [seq]
    assert find_sequences(tmp_path, ["other"]) == []
    info = read_seqinfo(seq)
    assert (info.fps, info.width, info.height, info.n_frames) == (25.0, 64, 48, 12)
    assert "CC BY-NC 4.0" in readme_row(info)


def test_convert_sequence_writes_readable_outputs(tmp_path: Path) -> None:
    """Checks the video, the MOT file, the detection CSV and the team template."""
    info = read_seqinfo(_make_sequence(tmp_path))
    out = convert_sequence(info, tmp_path / "raw", tmp_path / "ann", detection_every=5)

    meta = probe_video(out["video"])
    assert (meta.width, meta.height, meta.n_frames) == (64, 48, 12)

    mot = load_mot_ground_truth(out["mot"])
    assert mot["frame_id"].min() == 0
    assert mot["track_id"].nunique() == 2

    dets = pd.read_csv(out["detections"])
    assert sorted(dets["frame_id"].unique()) == [0, 5, 10]
    assert list(dets.columns) == ["frame_id", "x1", "y1", "x2", "y2", "cls"]
    np.testing.assert_allclose(dets.loc[0, ["x1", "y1", "x2", "y2"]], [5, 4, 15, 24])

    teams = pd.read_csv(out["teams"])
    assert list(teams["track_id"]) == [1, 2]
    assert teams["team"].isna().all()


def test_team_labels_are_never_overwritten(tmp_path: Path) -> None:
    """Checks that re-running the converter keeps hand-filled team labels."""
    info = read_seqinfo(_make_sequence(tmp_path))
    out = convert_sequence(info, tmp_path / "raw", tmp_path / "ann", write_video=False)
    filled = pd.read_csv(out["teams"]).assign(team=[0, 1])
    filled.to_csv(out["teams"], index=False)
    convert_sequence(info, tmp_path / "raw", tmp_path / "ann", write_video=False)
    assert list(pd.read_csv(out["teams"])["team"]) == [0, 1]
    assert "video" not in out


def test_evenly_spaced_frames() -> None:
    """Checks the held-out frame suggestion."""
    assert evenly_spaced_frames(101, 5) == [0, 25, 50, 75, 100]
    assert evenly_spaced_frames(0, 5) == []


def test_prepare_script_skips_sequences_without_gt(tmp_path: Path) -> None:
    """Checks the CLI end to end, including a test-split sequence without GT."""
    _make_sequence(tmp_path, "v_a_c001")
    no_gt = _make_sequence(tmp_path, "v_b_c001")
    (no_gt / "gt" / "gt.txt").unlink()
    football = tmp_path / "football.txt"
    football.write_text("v_a_c001\nv_b_c001\n", encoding="utf-8")
    script = _load_script("prepare_sportsmot")
    code = script.main(
        [
            "--root",
            str(tmp_path / "train"),
            "--football",
            str(football),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--annotations-dir",
            str(tmp_path / "ann"),
        ]
    )
    assert code == 0
    assert (tmp_path / "raw" / "v_a_c001.mp4").exists()
    assert not (tmp_path / "raw" / "v_b_c001.mp4").exists()


def test_merge_landmarks_round_trip(tmp_path: Path) -> None:
    """Checks that merged files load with the evaluation loader."""
    for frame in (0, 120):
        path = tmp_path / f"clip_f{frame}.json"
        path.write_text(json.dumps({"center": [640, 360]}), encoding="utf-8")
    script = _load_script("merge_landmarks")
    output = tmp_path / "heldout.json"
    assert (
        script.main(
            ["--inputs", *map(str, tmp_path.glob("*.json")), "--output", str(output)]
        )
        == 0
    )
    loaded = load_landmark_annotations(output)
    assert sorted(loaded) == [0, 120]
    assert loaded[120]["center"] == (640.0, 360.0)
