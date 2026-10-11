"""Wall-clock profiling of pipeline stages (stage P9)."""

from __future__ import annotations

import types
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

import pandas as pd


class Timer:
    """Context manager that measures elapsed wall-clock time.

    Attributes:
        elapsed: Duration of the last ``with`` block in seconds.
    """

    def __init__(self) -> None:
        """Creates a timer with zero elapsed time."""
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> Timer:
        """Starts the clock.

        Returns:
            The timer itself.
        """
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        """Stops the clock.

        Args:
            exc_type: Exception type raised inside the block, if any.
            exc: Exception instance raised inside the block, if any.
            tb: Traceback of the exception, if any.
        """
        self.elapsed = perf_counter() - self._start


class StageProfiler:
    """Accumulates timings per named stage and reports ms/frame and FPS."""

    def __init__(self) -> None:
        """Creates an empty profiler."""
        self._seconds: dict[str, float] = defaultdict(float)
        self._items: dict[str, int] = defaultdict(int)
        self._calls: dict[str, int] = defaultdict(int)

    def record(self, stage: str, seconds: float, items: int = 1) -> None:
        """Adds a measurement.

        Args:
            stage: Name of the stage.
            seconds: Duration in seconds.
            items: Number of items (frames) processed in that time.
        """
        self._seconds[stage] += seconds
        self._items[stage] += items
        self._calls[stage] += 1

    @contextmanager
    def measure(self, stage: str, items: int = 1) -> Iterator[None]:
        """Times the body of a ``with`` block and records it.

        Args:
            stage: Name of the stage.
            items: Number of items (frames) processed inside the block.

        Yields:
            Nothing; the timed code runs at this point.
        """
        start = perf_counter()
        try:
            yield
        finally:
            self.record(stage, perf_counter() - start, items)

    def summary(self) -> pd.DataFrame:
        """Summarizes all recorded stages.

        Returns:
            Table with ``stage``, ``calls``, ``total_s``, ``items``, ``ms_per_item``
            and ``fps`` (items per second), one row per stage.
        """
        rows = []
        for stage, seconds in self._seconds.items():
            items = self._items[stage]
            rows.append(
                {
                    "stage": stage,
                    "calls": self._calls[stage],
                    "total_s": seconds,
                    "items": items,
                    "ms_per_item": 1000.0 * seconds / max(items, 1),
                    "fps": items / seconds if seconds > 0 else float("inf"),
                }
            )
        return pd.DataFrame(
            rows, columns=["stage", "calls", "total_s", "items", "ms_per_item", "fps"]
        )
