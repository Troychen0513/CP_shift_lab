"""测试数据形状"""

import numpy as np

from src.data import f_true, sample_source_xy, sample_target_xy, sigma_true


# pytest 自动运行所有名字以 test_ 开头的函数
def test_f_true_shape():
    x = np.array([-2, 0, 2])
    y = f_true(x)
    
    assert y.shape == x.shape
    
def test_sigma_true_shape():
    x = np.array([-2, 0, 2])
    sigma = sigma_true(x)
    
    assert np.all(sigma>= 0)
    
def test_sample_source_xy_shape():
    n = 10
    rng = np.random.default_rng(42)
    x, y = sample_source_xy(n=n, rng=rng)
    
    assert x.shape == (n,)
    assert y.shape == (n,)   
    
def test_sample_target_xy_shape():
    n = 10
    rng = np.random.default_rng(42)
    for scenario in ["S0", "S1", "S2", "S3", "S4"]:
        x, y = sample_target_xy(scenario=scenario, n=n, rng=rng)
        
        assert x.shape == (n,)
        assert y.shape == (n,)
        
