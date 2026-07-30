"""Conformal quantile rules used in the CP shift lab."""

import math
import numpy as np


def conformal_order_statistic(scores: np.ndarray, alpha: float) -> float:
    """Return the split conformal finite-sample order statistic.
    scores:加权非符合性得分； alpha：计算覆盖率
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    if n == 0:
        raise ValueError("scores must not be empty")

    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return np.inf

    return float(np.sort(scores)[k - 1])



def weighted_conformal_threshold(scores,weights_cal,weight_test,alpha):
    """Return the WCP threshold for one test point.
    weights_cal:每个校准点的密度比权重; alpha:错误率；weight_test：当前测试点 x_* 的权重，影响阈值
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
    
    total_cal_weight = np.sum(sorted_weights)
    target_weight = (1 - alpha) * (total_cal_weight + weight_test) 
    
    if target_weight > total_cal_weight:
        return np.inf
    
    cumulative_weight = np.cumsum(sorted_weights)
    # 找到第一个累计权重大于等于 target_weight 的位置
    index = np.searchsorted(cumulative_weight,target_weight,side = 'left')
    
    # 返回对应位置的分数，也就是 WCP 阈值
    return float(sorted_scores[index])
    