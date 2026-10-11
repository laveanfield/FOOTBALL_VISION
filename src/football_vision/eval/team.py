"""Team classification metrics (stage P9)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linear_sum_assignment


def team_accuracy(predicted: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """Computes accuracy after the best one-to-one relabeling of clusters.

    KMeans cluster ids are arbitrary (cluster 0 may be the team labeled 1), so
    predicted labels are first matched to true labels with the Hungarian algorithm on
    the contingency table.

    Args:
        predicted: Predicted labels of shape ``(N,)``.
        truth: True labels of shape ``(N,)``.

    Returns:
        The fraction of correct samples under the best matching (0 if empty).

    Raises:
        ValueError: If the inputs have different lengths.
    """
    pred = np.asarray(predicted)
    true = np.asarray(truth)
    if len(pred) != len(true):
        raise ValueError("predicted and truth must have the same length")
    if len(pred) == 0:
        return 0.0
    pred_labels, true_labels = np.unique(pred), np.unique(true)
    contingency = np.array(
        [[np.sum((pred == p) & (true == t)) for t in true_labels] for p in pred_labels]
    )
    rows, cols = linear_sum_assignment(-contingency)
    return float(contingency[rows, cols].sum() / len(pred))


def confusion_table(predicted: npt.ArrayLike, truth: npt.ArrayLike) -> pd.DataFrame:
    """Builds a confusion table of true labels (rows) against predictions (columns).

    Args:
        predicted: Predicted labels of shape ``(N,)``.
        truth: True labels of shape ``(N,)``.

    Returns:
        The contingency table as a DataFrame.
    """
    return pd.crosstab(
        pd.Series(np.asarray(truth), name="truth"),
        pd.Series(np.asarray(predicted), name="predicted"),
    )
