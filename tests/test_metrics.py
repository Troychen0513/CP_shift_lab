import numpy as np

from src.metrics import (
    coverage,
    interval_length,
    interval_score,
    mean_interval_length,
    mean_interval_score,
    effective_sample_size,
)


def test_coverage():
    y = np.array([0.0, 2.0, 5.0])
    lower = np.array([-1.0, 1.0, 3.0])
    upper = np.array([1.0, 3.0, 4.0])

    result = coverage(y, lower, upper)

    assert result == 2 / 3


def test_interval_length():
    lower = np.array([-1.0, 1.0, 3.0])
    upper = np.array([1.0, 3.0, 4.0])

    lengths = interval_length(lower, upper)

    np.testing.assert_allclose(lengths, np.array([2.0, 2.0, 1.0]))


def test_mean_interval_length():
    lower = np.array([-1.0, 1.0, 3.0])
    upper = np.array([1.0, 3.0, 4.0])

    result = mean_interval_length(lower, upper)

    assert result == 5 / 3


def test_interval_score():
    y = np.array([0.0, 5.0])
    lower = np.array([-1.0, 1.0])
    upper = np.array([1.0, 3.0])

    scores = interval_score(y, lower, upper, alpha=0.1)

    np.testing.assert_allclose(scores, np.array([2.0, 42.0]))


def test_mean_interval_score():
    y = np.array([0.0, 5.0])
    lower = np.array([-1.0, 1.0])
    upper = np.array([1.0, 3.0])

    result = mean_interval_score(y, lower, upper, alpha=0.1)

    assert result == 22.0


def test_effective_sample_size_equal_weights():
    weights = np.ones(5)

    result = effective_sample_size(weights)

    assert result == 5.0


def test_effective_sample_size_one_dominant_weight():
    weights = np.array([1.0, 0.0, 0.0, 0.0])

    result = effective_sample_size(weights)

    assert result == 1.0