"""Tests for video reading and writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_vision.video import VideoReader, VideoWriter, frame_id_range, probe_video
from tests.helpers import FPS, FRAME_SIZE, N_FRAMES


def test_reader_properties_and_all_frames(synthetic_video: Path) -> None:
    """Checks container properties and that every frame is yielded in order."""
    with VideoReader(synthetic_video) as reader:
        assert reader.size == FRAME_SIZE
        assert reader.n_frames == N_FRAMES
        assert reader.fps == pytest.approx(FPS)
        frames = list(reader)
    assert [frame_id for frame_id, _ in frames] == list(range(N_FRAMES))
    assert frames[0][1].shape == (FRAME_SIZE[1], FRAME_SIZE[0], 3)


def test_reader_start_stop_stride_and_resize(synthetic_video: Path) -> None:
    """Checks frame selection and resizing."""
    with VideoReader(
        synthetic_video, size=(80, 45), start=2, stop=10, stride=3
    ) as reader:
        items = list(reader)
    assert [frame_id for frame_id, _ in items] == [2, 5, 8]
    assert items[0][1].shape == (45, 80, 3)


def test_reader_missing_file_raises(tmp_path: Path) -> None:
    """Checks that opening a missing file raises OSError."""
    with pytest.raises(OSError):
        VideoReader(tmp_path / "nope.mp4")


def test_writer_rejects_wrong_frame_size(tmp_path: Path) -> None:
    """Checks that frames of the wrong size are refused."""
    with VideoWriter(tmp_path / "out.mp4", FPS, FRAME_SIZE) as writer:
        with pytest.raises(ValueError):
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))


def test_probe_video(synthetic_video: Path) -> None:
    """Checks the metadata read from a video."""
    meta = probe_video(synthetic_video)
    assert (meta.width, meta.height, meta.n_frames) == (*FRAME_SIZE, N_FRAMES)
    assert meta.shots == [(0, N_FRAMES - 1)]


def test_frame_id_range() -> None:
    """Checks the processed frame ids and the stride validation."""
    assert frame_id_range(100, 10, 50, 5) == range(10, 50, 5)
    assert frame_id_range(30, 0, None, 1) == range(0, 30)
    assert frame_id_range(30, 0, 99, 1) == range(0, 30)
    with pytest.raises(ValueError):
        frame_id_range(30, stride=0)
