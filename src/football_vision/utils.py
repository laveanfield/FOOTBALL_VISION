"""Small utilities: logging setup, reproducibility and filesystem helpers."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

LOGGER_NAME = "football_vision"


def setup_logging(
    level: int | str = "INFO", log_file: str | Path | None = None
) -> logging.Logger:
    """Configures and returns the package logger.

    Any handler installed by a previous call is replaced, so the function is safe
    to call repeatedly.

    Args:
        level: Logging level, either numeric or a name such as ``"DEBUG"``.
        log_file: Optional file that also receives the log records.

    Returns:
        The configured ``football_vision`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        path = ensure_dir(Path(log_file).parent) / Path(log_file).name
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy and (if installed) PyTorch for reproducible runs.

    Args:
        seed: Seed value shared by all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415  (optional dependency)
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Creates a directory (and parents) if it does not exist.

    Args:
        path: Directory to create.

    Returns:
        The directory as a :class:`~pathlib.Path`.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
