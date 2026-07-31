"""Evaluation metrics for prediction intervals and importance weights."""

import numpy as np


# 覆盖率
def coverage(y:np.ndarray,lower:np.ndarray,upper:np.ndarray) -> float:
    """Compute the coverage of prediction intervals."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    
    is_covered = (y >= lower) & (y <= upper)
    return float(np.mean(is_covered))


# 分箱覆盖率
def binned_coverage(x, y, lower, upper, n_bins: int = 5) -> dict:
    """Compute coverage in equal-sized X bins."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    order = np.argsort(x)
    bins = np.array_split(order, n_bins)
    
    result = {}
    
    for i, idx in enumerate(bins, start=1):
        result[f"bin_{i}_coverage"] = coverage(y[idx], lower[idx], upper[idx])
        
    return result
    


def interval_length(lower:np.ndarray, upper:np.ndarray) -> np.ndarray:
    """Compute the length of prediction intervals."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    
    return upper - lower

# 平均区间长度
def mean_interval_length(lower:np.ndarray, upper:np.ndarray) -> float:
    """Compute the mean length of prediction intervals."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    
    
    lengths = interval_length(lower, upper)
    return float(np.mean(lengths))


# Winkler interval score 区间评分
def interval_score(y:np.ndarray, lower:np.ndarray, upper:np.ndarray, alpha:float) -> float:
    """Compute the Winkler interval score for prediction intervals."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    
    lengths = interval_length(lower, upper)
    penalty = (2 / alpha) * ((lower - y) * (y < lower) + (y - upper) * (y > upper))
    
    return lengths + penalty

def mean_interval_score(y, lower, upper, alpha) -> float:
    scores = interval_score(y, lower, upper, alpha)
    return float(np.mean(scores))


# ESS 有效样本量
def effective_sample_size(weights: np.ndarray) -> float:
    """Return ESS = (sum w)^2 / sum(w^2)."""
    weights = np.asarray(weights,dtype=float)
    
    if len(weights) == 0:
        raise ValueError("weights must not be empty")

    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    
    denominator = np.sum(weights**2)
    
    if denominator == 0:
        return 0.0
    
    return float((np.sum(weights))**2 / denominator )

