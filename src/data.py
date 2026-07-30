"""Data generation utilities for the CP shift lab."""

from __future__ import annotations

import numpy as np


SCENARIOS = ("S0", "S1", "S2", "S3", "S4")


def f_true(x: np.ndarray) -> np.ndarray:
    """Return the true conditional mean E[Y | X=x]."""
    x = np.asarray(x, dtype=float)
    return np.sin(2 * x) + 0.3 * x


def sigma_true(x: np.ndarray) -> np.ndarray:
    """Return the true noise standard deviation at x."""
    x = np.asarray(x, dtype=float)
    return 0.20 + 0.50 * (x + 2) / 4


def _sample_exponential_tilted_x(
    n: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample X on [-2, 2] from a density proportional to exp(beta * x)."""
    u = rng.uniform(0, 1, size=n)
    left = np.exp(-2 * beta)
    right = np.exp(2 * beta)
    return np.log(left + u * (right - left)) / beta


def sample_source_xy(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample labeled source-domain data."""
    x = rng.uniform(-2, 2, size=n)
    epsilon = rng.normal(0, 1, size=n)
    y = f_true(x) + sigma_true(x) * epsilon
    return x, y


def sample_target_x(
    scenario: str,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample target-domain covariates X under scenario S0-S4."""
    if scenario == "S0":
        return rng.uniform(-2, 2, size=n)

    if scenario in {"S1", "S4"}:
        return _sample_exponential_tilted_x(n=n, beta=1.2, rng=rng)

    if scenario == "S2":
        return _sample_exponential_tilted_x(n=n, beta=2.5, rng=rng)

    if scenario == "S3":
        return rng.uniform(1, 3, size=n)

    raise ValueError(f"Unknown scenario: {scenario}")


def sample_target_xy(
    scenario: str,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample labeled target-domain test data under scenario S0-S4."""
    x = sample_target_x(scenario=scenario, n=n, rng=rng)
    epsilon = rng.normal(0, 1, size=n)
    y = f_true(x) + sigma_true(x) * epsilon

    if scenario == "S4":
        y = y + 0.8 * (x > 0)

    return x, y


def true_density_ratio(x: np.ndarray, scenario: str) -> np.ndarray:
    """Return the oracle density ratio w(x), up to a constant factor."""
    x = np.asarray(x, dtype=float)

    if scenario == "S0":
        return np.ones_like(x)

    if scenario in {"S1", "S4"}:
        return np.exp(1.2 * x)

    if scenario == "S2":
        return np.exp(2.5 * x)

    if scenario == "S3":
        return np.where((x >= 1) & (x <= 2), 1.0, 0.0)

    raise ValueError(f"Unknown scenario: {scenario}")
