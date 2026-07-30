"""Conformal quantile rules used in the CP shift lab."""

from __future__ import annotations

import math

import numpy as np


def conformal_order_statistic(scores: np.ndarray, alpha: float) -> float:
    """Return the split conformal finite-sample order statistic."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    if n == 0:
        raise ValueError("scores must not be empty")

    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return np.inf

    return float(np.sort(scores)[k - 1])


def weighted_conformal_threshold(
    scores: np.ndarray,
    weights_cal: np.ndarray,
    weight_test: float,
    alpha: float,
) -> float:
    """Return the WCP threshold for one test point.

    The test point contributes weight at +infinity. If the target cumulative
    weight is larger than all finite calibration weight, the threshold is inf.
    """
    scores = np.asarray(scores, dtype=float)
    weights_cal = np.asarray(weights_cal, dtype=float)

    if len(scores) == 0:
        raise ValueError("scores must not be empty")
    if scores.shape != weights_cal.shape:
        raise ValueError("scores and weights_cal must have the same shape")
    if np.any(weights_cal < 0) or weight_test < 0:
        raise ValueError("weights must be nonnegative")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = weights_cal[order]

    total_cal_weight = float(np.sum(sorted_weights))
    target_weight = (1 - alpha) * (total_cal_weight + float(weight_test))

    if target_weight > total_cal_weight:
        return np.inf

    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, target_weight, side="left"))
    return float(sorted_scores[index])
