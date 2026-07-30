import numpy as np

from src.conformal import conformal_order_statistic,weighted_conformal_threshold

def test_conformal_order_statistic_basic():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    q = conformal_order_statistic(scores, alpha=0.2)
    assert q == 4.0


def test_conformal_order_statistic_infinity():
    scores = np.array([1.0, 2.0])
    q = conformal_order_statistic(scores, alpha=0.1)
    assert np.isinf(q)
    
def test_weighted_conformal_threshold_basic():
    scores = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])

    q = weighted_conformal_threshold(
        scores=scores,
        weights_cal=weights,
        weight_test=0.0,
        alpha=1 / 3,
    )

    assert q == 2.0
    
    
def test_wcp_test_mass_inf():
    scores = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])

    q = weighted_conformal_threshold(
        scores=scores,
        weights_cal=weights,
        weight_test=10.0,
        alpha=0.1,
    )

    assert np.isinf(q)