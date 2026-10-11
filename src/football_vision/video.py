"""Video input and output helpers built on OpenCV.

Typical usage example:

    with VideoReader("clip.mp4", size=(1280, 720)) as reader:
        for frame_id, frame in reader:
            ...
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from pathlib import Path

import cv2

from football_vision.artifacts import VideoMeta
from football_vision.datatypes import Frame
from football_vision.preprocessing.frames import resize_frame

_DEFAULT_FPS = 25.0


def frame_id_range(
    n_frames: int, start: int = 0, stop: int | None = None, stride: int = 1
) -> range:
    """Lists the source frame ids a run processes.

    Args:
        n_frames: Number of frames in the source video.
        start: First frame id.
        stop: Exclusive end frame id; ``None`` means ``n_frames``.
        stride: Distance between processed frames (at least 1).

    Returns:
        The range of processed frame ids.

    Raises:
        ValueError: If ``stride`` is smaller than 1.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    end = n_frames if stop is None else min(stop, n_frames)
    return range(start, end, stride)


class VideoReader:
    """Iterates over the frames of a video with optional resizing and subsampling.

    Attributes:
        path: Path of the video file.
        fps: Frames per second reported by the container (25 if unknown).
        n_frames: Frame count reported by the container; it may be approximate.
        source_size: ``(width, height)`` of the stored video.
        size: ``(width, height)`` of the frames produced by iteration.
    """

    def __init__(
        self,
        path: str | Path,
        size: tuple[int, int] | None = None,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
    ) -> None:
        """Opens a video file.

        Args:
            path: Video file to read.
            size: Optional ``(width, height)`` that frames are resized to.
            start: First frame id to yield.
            stop: Exclusive end frame id; ``None`` reads to the end.
            stride: Yields every ``stride``-th frame.

        Raises:
            OSError: If the file cannot be opened.
            ValueError: If ``stride`` is smaller than 1.
        """
        if stride < 1:
            raise ValueError("stride must be at least 1")
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"Cannot open video: {self.path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else _DEFAULT_FPS
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.size = size or self.source_size
        self._start, self._stop, self._stride = start, stop, stride

    def __iter__(self) -> Iterator[tuple[int, Frame]]:
        """Yields ``(frame_id, frame)`` pairs.

        Yields:
            The 0-based source frame id and the BGR frame, resized if requested.
        """
        if self._start > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)
        frame_id = self._start
        while self._stop is None or frame_id < self._stop:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame_id, resize_frame(frame, self.size)
            for _ in range(self._stride - 1):
                if not self._cap.grab():
                    return
            frame_id += self._stride

    def close(self) -> None:
        """Releases the underlying capture handle."""
        self._cap.release()

    def __enter__(self) -> VideoReader:
        """Enters the context manager.

        Returns:
            The reader itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        """Closes the reader when leaving the context.

        Args:
            exc_type: Exception type raised inside the block, if any.
            exc: Exception instance raised inside the block, if any.
            tb: Traceback of the exception, if any.
        """
        self.close()


class VideoWriter:
    """Writes frames to a video file.

    Attributes:
        path: Destination file.
        size: ``(width, height)`` every written frame must have.
    """

    def __init__(
        self, path: str | Path, fps: float, size: tuple[int, int], codec: str = "mp4v"
    ) -> None:
        """Opens a video file for writing.

        Args:
            path: Destination file; parent directories are created.
            fps: Frame rate of the output video.
            size: ``(width, height)`` of the frames that will be written.
            codec: Four-character codec code, for example ``"mp4v"``.

        Raises:
            OSError: If the writer cannot be opened (missing codec or bad path).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (int(size[0]), int(size[1]))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), self.size)
        if not self._writer.isOpened():
            raise OSError(
                f"Cannot open video writer for {self.path} with codec '{codec}'"
            )

    def write(self, frame: Frame) -> None:
        """Appends one frame.

        Args:
            frame: BGR frame whose width and height equal ``size``.

        Raises:
            ValueError: If the frame size differs from the writer size.
        """
        height, width = frame.shape[:2]
        if (width, height) != self.size:
            raise ValueError(
                f"Frame size {(width, height)} does not match writer size {self.size}"
            )
        self._writer.write(frame)

    def close(self) -> None:
        """Finalizes the file."""
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        """Enters the context manager.

        Returns:
            The writer itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        """Closes the writer when leaving the context.

        Args:
            exc_type: Exception type raised inside the block, if any.
            exc: Exception instance raised inside the block, if any.
            tb: Traceback of the exception, if any.
        """
        self.close()


def probe_video(path: str | Path, size: tuple[int, int] | None = None) -> VideoMeta:
    """Reads container metadata without decoding frames.

    Args:
        path: Video file to inspect.
        size: Processing size ``(width, height)``; defaults to the stored size.

    Returns:
        The metadata with a single shot covering the whole video.
    """
    with VideoReader(path, size=size) as reader:
        width, height = reader.size
        return VideoMeta(
            source=str(path),
            fps=reader.fps,
            width=width,
            height=height,
            n_frames=reader.n_frames,
            shots=[(0, max(reader.n_frames - 1, 0))],
        )
