"""Custom exception hierarchy for the Football Vision package."""

from __future__ import annotations


class FootballVisionError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(FootballVisionError, ValueError):
    """Raised when a configuration file or override is invalid."""


class SchemaError(FootballVisionError, ValueError):
    """Raised when a tabular artifact does not match its declared schema."""


class HomographyError(FootballVisionError, RuntimeError):
    """Raised when a homography cannot be estimated or is degenerate."""


class PipelineError(FootballVisionError, RuntimeError):
    """Raised when a pipeline stage cannot run, for example missing inputs."""
