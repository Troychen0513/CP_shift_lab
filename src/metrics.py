"""Evaluation metrics for prediction intervals and importance weights."""

from __future__ import annotations

import numpy as np


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Return the fraction of labels covered by [lower, upper]."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((lower <= y) & (y <= upper)))


def interval_length(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return interval lengths U-L for each test point."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return upper - lower


def mean_interval_length(lower: np.ndarray, upper: np.ndarray) -> float:
    """Return the average interval length."""
    return float(np.mean(interval_length(lower, upper)))


def interval_score(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return the Winkler interval score for each test point."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    length = upper - lower
    below_penalty = (2 / alpha) * (lower - y) * (y < lower)
    above_penalty = (2 / alpha) * (y - upper) * (y > upper)
    return length + below_penalty + above_penalty


def effective_sample_size(weights: np.ndarray) -> float:
    """Return ESS = (sum w)^2 / sum(w^2)."""
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0:
        raise ValueError("weights must not be empty")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")

    squared_sum = float(np.sum(weights**2))
    if squared_sum == 0:
        return 0.0

    return float(np.sum(weights) ** 2 / squared_sum)
