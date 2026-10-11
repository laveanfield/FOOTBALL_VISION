"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from football_vision.config import PipelineConfig, apply_overrides
from tests.helpers import FPS, FRAME_SIZE, N_FRAMES


@pytest.fixture()
def synthetic_video(tmp_path: Path) -> Path:
    """Writes a short synthetic video of a white rectangle moving over grass."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 files in this environment")
    for i in range(N_FRAMES):
        frame = np.full(
            (FRAME_SIZE[1], FRAME_SIZE[0], 3), (60, 140, 60), dtype=np.uint8
        )
        cv2.rectangle(frame, (10 + 6 * i, 30), (30 + 6 * i, 70), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def base_cfg(tmp_path: Path, synthetic_video: Path) -> PipelineConfig:
    """Builds a configuration that points at the synthetic video and a temp folder."""
    return apply_overrides(
        PipelineConfig(),
        {
            "video.path": str(synthetic_video),
            "video.size": list(FRAME_SIZE),
            "output_dir": str(tmp_path / "out"),
            "run_name": "test",
        },
    )
