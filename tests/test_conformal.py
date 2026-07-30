import numpy as np

from src.conformal import conformal_order_statistic

def test_conformal_order_statistic_basic():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    q = conformal_order_statistic(scores, alpha=0.2)
    assert q == 4.0


def test_conformal_order_statistic_infinity():
    scores = np.array([1.0, 2.0])
    q = conformal_order_statistic(scores, alpha=0.1)
    assert np.isinf(q)
    
