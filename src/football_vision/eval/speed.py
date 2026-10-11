"""Sanity statistics for estimated player speeds (stage P9)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def speed_sanity_report(
    speeds_mps: npt.ArrayLike, max_speed_mps: float = 12.0
) -> dict[str, float]:
    """Summarizes a speed sample and the share of implausible values.

    Elite players peak at about 10 m/s, so values above ~12 m/s point to ID switches
    or a jumping homography rather than real running.

    Args:
        speeds_mps: Speeds in meters per second.
        max_speed_mps: Plausibility limit.

    Returns:
        Mapping with ``n``, ``mean``, ``p50``, ``p95``, ``p99``, ``max`` and
        ``outlier_fraction`` (NaN statistics if the sample is empty).
    """
    values = np.asarray(speeds_mps, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        keys = ("mean", "p50", "p95", "p99", "max", "outlier_fraction")
        return {"n": 0.0, **dict.fromkeys(keys, np.nan)}
    return {
        "n": float(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "outlier_fraction": float((values > max_speed_mps).mean()),
    }
