"""Hough line detection and post-processing (stage P2).

A line is parameterized as ``x*cos(theta) + y*sin(theta) = rho``. Every edge pixel
votes for all ``(rho, theta)`` it could lie on; collinear pixels vote for the same
cell, so accumulator peaks are lines.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from football_vision.config import PitchConfig
from football_vision.datatypes import Mask

_MIN_SEGMENT_LENGTH = 1e-6
# Segments shorter than this are too noisy to give a reliable orientation.
_MIN_FAMILY_SEGMENT_PX = 20.0
# A segment deviating more than this from the mean direction of its family is
# reported as an outlier (perspective makes a family fan out, so this is generous).
_MAX_FAMILY_SPREAD_RAD = np.deg2rad(45.0)
_KMEANS_MAX_ITERATIONS = 50


def _connected_components(
    adjacency: npt.NDArray[np.bool_],
) -> tuple[int, npt.NDArray[np.int64]]:
    """Labels the connected components of a symmetric boolean adjacency matrix."""
    n = len(adjacency)
    labels = np.full(n, -1, dtype=np.int64)
    count = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        labels[start] = count
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in np.flatnonzero(adjacency[current] & (labels < 0)):
                labels[neighbor] = count
                stack.append(int(neighbor))
        count += 1
    return count, labels


def _weighted_kmeans(
    points: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], k: int
) -> npt.NDArray[np.int64]:
    """Runs a deterministic weighted k-means and returns the cluster of each point.

    Centers start at the heaviest point, then repeatedly at the point farthest from
    the centers chosen so far; Lloyd iterations follow. No random numbers are used.
    """
    centers = [points[int(np.argmax(weights))]]
    while len(centers) < k:
        nearest = np.min(
            [np.sum((points - center) ** 2, axis=1) for center in centers], axis=0
        )
        centers.append(points[int(np.argmax(nearest))])
    center_array = np.asarray(centers, dtype=np.float64)
    for _ in range(_KMEANS_MAX_ITERATIONS):
        assignment = np.argmin(
            np.sum((points[:, None, :] - center_array[None]) ** 2, axis=2), axis=1
        )
        updated = center_array.copy()
        for cluster in range(k):
            member = assignment == cluster
            if member.any():
                updated[cluster] = np.average(
                    points[member], axis=0, weights=weights[member]
                )
        if np.allclose(updated, center_array):
            break
        center_array = updated
    return np.argmin(
        np.sum((points[:, None, :] - center_array[None]) ** 2, axis=2), axis=1
    ).astype(np.int64)


def detect_line_segments(edges: Mask, cfg: PitchConfig) -> npt.NDArray[np.float64]:
    """Detects straight segments in an edge map with the probabilistic Hough transform.

    Args:
        edges: Binary edge map of shape ``(H, W)``.
        cfg: Pitch settings with the Hough resolution and thresholds.

    Returns:
        Array of shape ``(N, 4)`` holding ``(x1, y1, x2, y2)`` per segment. It has
        shape ``(0, 4)`` when nothing is found.
    """
    # TODO(A): cv2.HoughLinesP(edges, cfg.hough_rho, np.pi / 180, cfg.hough_threshold,
    #   minLineLength=cfg.hough_min_length, maxLineGap=cfg.hough_max_gap).
    lines = cv2.HoughLinesP(
        edges,
        cfg.hough_rho,
        np.pi / 180.0,
        int(cfg.hough_threshold),
        minLineLength=cfg.hough_min_length,
        maxLineGap=cfg.hough_max_gap,
    )
    if lines is None:
        return np.zeros((0, 4), dtype=np.float64)
    return lines.reshape(-1, 4).astype(np.float64)


def _merge_group(
    group: npt.NDArray[np.float64], lengths: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Replaces segments of one physical line by the segment spanning them all.

    The line direction is the principal axis of the endpoints, weighted by the
    length of the segment each endpoint belongs to.
    """
    if len(group) == 1:
        return group[0]
    points = group.reshape(-1, 2)
    weights = np.repeat(lengths, 2)
    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    covariance = (centered.T * weights) @ centered / weights.sum()
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
        direction = -direction
    along = centered @ direction
    start = centroid + along.min() * direction
    end = centroid + along.max() * direction
    return np.concatenate([start, end])


def merge_collinear_segments(
    segments: npt.NDArray[np.float64],
    rho_tolerance_px: float = 10.0,
    theta_tolerance_rad: float = 0.05,
) -> npt.NDArray[np.float64]:
    """Merges segments that lie on the same physical pitch line.

    Hough returns many fragments per marking. Convert each segment to ``(rho, theta)``,
    group those closer than the tolerances, and replace each group by the segment
    spanning its extreme endpoints.

    Two segments are in the same group when their directions differ by at most
    ``theta_tolerance_rad`` and the midpoint of each lies within ``rho_tolerance_px``
    of the other's line. Measuring the distance to the line, rather than comparing
    ``rho`` values, makes the test independent of the wrap-around of ``theta`` at
    0 and pi, where ``rho`` changes sign. Grouping is transitive. Each group becomes
    one segment on the length-weighted best-fit line, spanning the extreme
    endpoints; its endpoints are ordered by increasing ``x``. Zero-length segments
    are dropped.

    Args:
        segments: Array of shape ``(N, 4)`` with ``(x1, y1, x2, y2)``.
        rho_tolerance_px: Maximum ``rho`` difference inside a group.
        theta_tolerance_rad: Maximum ``theta`` difference inside a group.

    Returns:
        Array of shape ``(M, 4)`` with ``M <= N`` merged segments.
    """
    # TODO(A): handle the theta wrap-around at 0/pi; weight groups by segment length.
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    delta = seg[:, 2:] - seg[:, :2]
    length = np.hypot(delta[:, 0], delta[:, 1])
    keep = length > _MIN_SEGMENT_LENGTH
    seg, delta, length = seg[keep], delta[keep], length[keep]
    if len(seg) <= 1:
        return seg.copy()
    direction = np.arctan2(delta[:, 1], delta[:, 0]) % np.pi
    angle_gap = np.abs(direction[:, None] - direction[None, :])
    angle_gap = np.minimum(angle_gap, np.pi - angle_gap)
    normal = np.stack([-delta[:, 1], delta[:, 0]], axis=1) / length[:, None]
    offset = -np.sum(normal * seg[:, :2], axis=1)
    midpoint = (seg[:, :2] + seg[:, 2:]) / 2.0
    # distance[i, j]: distance from the midpoint of segment j to the line of segment i
    distance = np.abs(normal @ midpoint.T + offset[:, None])
    same_line = (angle_gap <= theta_tolerance_rad) & (
        np.maximum(distance, distance.T) <= rho_tolerance_px
    )
    count, labels = _connected_components(same_line)
    merged = [
        _merge_group(seg[labels == label], length[labels == label])
        for label in range(count)
    ]
    return np.asarray(merged, dtype=np.float64).reshape(-1, 4)


def cluster_line_families(
    segments: npt.NDArray[np.float64], n_families: int = 2
) -> npt.NDArray[np.int64]:
    """Assigns segments to families of lines that are parallel in the real world.

    Pitch lines form two families: along the pitch (touchlines) and across it (goal
    lines, halfway line, box edges). Under perspective, each family converges to a
    vanishing point, so cluster by orientation or by a RANSAC vanishing-point fit.

    This implementation clusters by orientation only. Angles are embedded as
    ``(cos 2a, sin 2a)`` so that directions just below pi and just above 0 fall in
    the same cluster, and a small deterministic k-means (written with NumPy) is run
    with the segment lengths as weights. Labels
    are ordered by orientation: ``0`` is the family closest to horizontal in the
    image, so the label keeps its meaning from frame to frame. Segments shorter
    than 20 px, or more than 45 degrees away from the mean direction of their
    cluster, get ``-1``.

    Args:
        segments: Array of shape ``(N, 4)`` with ``(x1, y1, x2, y2)``.
        n_families: Number of families to find.

    Returns:
        Integer array of shape ``(N,)`` with the family label of each segment
        (``-1`` for outliers).
    """
    # TODO(A): start with KMeans on the segment angle, then refine with
    #   vanishing points.
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    labels = np.full(len(seg), -1, dtype=np.int64)
    delta = seg[:, 2:] - seg[:, :2]
    length = np.hypot(delta[:, 0], delta[:, 1])
    usable = np.flatnonzero(length >= _MIN_FAMILY_SEGMENT_PX)
    if usable.size == 0:
        return labels
    angle = np.arctan2(delta[usable, 1], delta[usable, 0]) % np.pi
    weight = length[usable]
    embedding = np.stack([np.cos(2 * angle), np.sin(2 * angle)], axis=1)
    if usable.size <= n_families:
        raw = np.arange(usable.size)
    else:
        raw = _weighted_kmeans(embedding, weight, n_families)
    # Mean direction of each cluster in (-pi/2, pi/2]; 0 is horizontal.
    cluster_ids = np.unique(raw)
    mean_direction = {}
    for cluster in cluster_ids:
        member = raw == cluster
        mean_direction[cluster] = 0.5 * np.arctan2(
            np.sum(weight[member] * np.sin(2 * angle[member])),
            np.sum(weight[member] * np.cos(2 * angle[member])),
        )
    ranked = sorted(cluster_ids, key=lambda c: abs(mean_direction[c]))
    rank_of = {cluster: rank for rank, cluster in enumerate(ranked)}
    for position, index in enumerate(usable):
        cluster = raw[position]
        deviation = abs(
            (angle[position] - mean_direction[cluster] + np.pi / 2) % np.pi - np.pi / 2
        )
        if deviation <= _MAX_FAMILY_SPREAD_RAD:
            labels[index] = rank_of[cluster]
    return labels
