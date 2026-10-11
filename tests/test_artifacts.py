"""Tests for the artifact store, schemas and conversions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_vision.artifacts import (
    DETECTIONS,
    HOMOGRAPHY_QUALITY_COLUMNS,
    TRACKS,
    ArtifactStore,
    HomographySequence,
    VideoMeta,
    detections_to_table,
    load_landmark_annotations,
    load_landmarks_json,
    save_landmarks_json,
    table_to_detections,
    tracks_to_table,
)
from football_vision.datatypes import Detections
from football_vision.errors import SchemaError


def make_detections() -> Detections:
    """Creates two detections of different classes."""
    return Detections(
        boxes_xyxy=np.array([[10, 20, 30, 80], [50, 60, 70, 120]], dtype=np.float32),
        scores=np.array([0.9, 0.4], dtype=np.float32),
        class_ids=np.array([0, 32], dtype=np.int64),
    )


def test_detections_round_trip_through_csv(tmp_path: Path) -> None:
    """Checks that detections survive a write/read cycle."""
    store = ArtifactStore(tmp_path)
    store.write_table(DETECTIONS, detections_to_table(7, make_detections()))
    table = store.read_table(DETECTIONS)
    assert table["frame_id"].tolist() == [7, 7]
    restored = table_to_detections(table)
    np.testing.assert_allclose(restored.boxes_xyxy, make_detections().boxes_xyxy)
    assert restored.class_ids.tolist() == [0, 32]
    assert restored.track_ids is None


def test_tracks_table_has_unknown_team_and_role(tmp_path: Path) -> None:
    """Checks the defaults written by tracks_to_table and the round trip."""
    dets = make_detections().with_track_ids([4, 9])
    store = ArtifactStore(tmp_path)
    store.write_table(TRACKS, tracks_to_table(3, dets))
    table = store.read_table(TRACKS)
    assert table["team"].tolist() == [-1, -1]
    assert table["role"].tolist() == ["unknown", "unknown"]
    assert table_to_detections(table).track_ids.tolist() == [4, 9]


def test_tracks_to_table_requires_track_ids() -> None:
    """Checks that rows cannot be built without track ids."""
    with pytest.raises(ValueError):
        tracks_to_table(0, make_detections())


def test_schema_violations_are_reported(tmp_path: Path) -> None:
    """Checks that a missing column raises SchemaError when writing."""
    store = ArtifactStore(tmp_path)
    with pytest.raises(SchemaError):
        store.write_table(DETECTIONS, pd.DataFrame({"frame_id": [0]}))
    with pytest.raises(FileNotFoundError):
        store.read_table(DETECTIONS)


def test_detections_validate_shapes() -> None:
    """Checks that Detections rejects inconsistent array shapes."""
    with pytest.raises(ValueError):
        Detections(
            np.zeros((2, 4), np.float32), np.zeros(3, np.float32), np.zeros(3, np.int64)
        )


def test_detections_select_and_empty() -> None:
    """Checks selection by mask and the empty container."""
    dets = make_detections().with_track_ids([1, 2])
    kept = dets.select(dets.scores > 0.5)
    assert len(kept) == 1
    assert kept.track_ids.tolist() == [1]
    assert len(Detections.empty()) == 0


def test_meta_round_trip(tmp_path: Path) -> None:
    """Checks that video metadata round-trips, including shot tuples."""
    store = ArtifactStore(tmp_path)
    meta = VideoMeta("clip.mp4", 25.0, 1280, 720, 100, shots=[(0, 49), (50, 99)])
    store.write_meta(meta)
    assert store.read_meta() == meta


def test_homography_sequence_lookup_and_round_trip(tmp_path: Path) -> None:
    """Checks lookup, validity flags and persistence of a homography sequence."""
    quality = pd.DataFrame(
        {
            "frame_id": [0, 5],
            "valid": [1, 0],
            "n_inliers": [8, 0],
            "reproj_err_m": [0.1, np.nan],
            "overlay_score": [np.nan, np.nan],
        }
    )
    sequence = HomographySequence(
        np.stack([np.eye(3), 2 * np.eye(3)]), quality[list(HOMOGRAPHY_QUALITY_COLUMNS)]
    )
    assert (
        sequence.is_valid(0) and not sequence.is_valid(5) and not sequence.is_valid(99)
    )
    assert sequence.has_frame(5) and not sequence.has_frame(1)
    np.testing.assert_allclose(sequence.matrix_for(5), 2 * np.eye(3))
    store = ArtifactStore(tmp_path)
    store.write_homographies(sequence)
    restored = store.read_homographies()
    np.testing.assert_allclose(restored.matrices, sequence.matrices)
    assert restored.frame_ids.tolist() == [0, 5]


def test_homography_sequence_rejects_bad_shapes() -> None:
    """Checks validation of matrix shape and table length."""
    quality = pd.DataFrame({c: [0] for c in HOMOGRAPHY_QUALITY_COLUMNS})
    with pytest.raises(SchemaError):
        HomographySequence(np.zeros((1, 2, 2)), quality)
    with pytest.raises(SchemaError):
        HomographySequence(np.zeros((2, 3, 3)), quality)


def test_directory_artifact_exists_only_when_non_empty(tmp_path: Path) -> None:
    """Checks that an empty directory artifact does not count as produced."""
    store = ArtifactStore(tmp_path)
    folder = store.directory("outputs")
    assert not store.exists("outputs")
    (folder / "video.mp4").write_bytes(b"x")
    assert store.exists("outputs")


def test_landmark_json_round_trip(tmp_path: Path) -> None:
    """Checks saving and loading of clicked landmarks and per-frame annotations."""
    landmarks = {"corner_tl": (10.5, 20.0), "center": (640.0, 360.0)}
    path = save_landmarks_json(landmarks, tmp_path / "lm.json")
    assert load_landmarks_json(path) == landmarks
    annotated = tmp_path / "gt.json"
    annotated.write_text('{"120": {"center": [1.0, 2.0]}}', encoding="utf-8")
    assert load_landmark_annotations(annotated) == {120: {"center": (1.0, 2.0)}}
