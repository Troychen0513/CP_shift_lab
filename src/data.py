"""生成实验所需要的数据。"""

import numpy as np


def f_true(x: np.ndarray) -> np.ndarray:
    """真实均值函数 f(x)。"""
    x = np.asarray(x, dtype=float)
    return np.sin(2 * x) + 0.3 * x


def sigma_true(x: np.ndarray) -> np.ndarray:
    """真实噪声标准差函数 sigma(x)。"""
    x = np.asarray(x, dtype=float)
    return 0.2 + 0.5 * (x + 2) / 4


# 源域labeled数据生成。
def sample_source_xy(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample labeled source-domain data."""
    x = rng.uniform(-2, 2, size=n)
    y = f_true(x) + sigma_true(x) * rng.normal(0, 1, size=n)

    return x, y


def sample_target_x(scenario: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample unlabeled target-domain data with various distribution shifts."""
    if scenario == "S0":
        return rng.uniform(-2, 2, size=n)

    elif scenario == "S1":
        beta = 1.2
        u = rng.uniform(0, 1, size=n)
        return np.log(np.exp(-2 * beta) + u * (np.exp(2 * beta) - np.exp(-2 * beta))) / beta

    elif scenario == "S2":
        beta = 2.5
        u = rng.uniform(0, 1, size=n)
        return np.log(np.exp(-2 * beta) + u * (np.exp(2 * beta) - np.exp(-2 * beta))) / beta

    elif scenario == "S3":
        return rng.uniform(1, 3, size=n)

    elif scenario == "S4":
        # S4 的 X 分布使用 S1，但响应 Y 的生成机制会改变。
        beta = 1.2
        u = rng.uniform(0, 1, size=n)
        return np.log(np.exp(-2 * beta) + u * (np.exp(2 * beta) - np.exp(-2 * beta))) / beta

    raise ValueError(f"Unknown scenario: {scenario}")


# 这个函数用于最终评估 coverage。
def sample_target_xy(scenario: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample labeled target-domain data with distribution shifts."""
    x = sample_target_x(scenario, n, rng)
    y = f_true(x) + sigma_true(x) * rng.normal(0, 1, size=n)

    if scenario == "S4":
        y += 0.8 * (x > 0)

    return x, y


# 真实密度比。S1/S2/S4 的密度比正比于 exp(beta*x)，S0 的密度比为 1。
def true_density_ratio(x: np.ndarray, scenario: str) -> np.ndarray:
    """Compute the true density ratio p_target(x) / p_source(x)."""
    x = np.asarray(x, dtype=float)

    if scenario == "S0":
        return np.ones_like(x)

    # 常数倍不影响加权分位数，因此这里直接使用未归一化权重。
    elif scenario in {"S1", "S4"}:
        return np.exp(1.2 * x)

    elif scenario == "S2":
        return np.exp(2.5 * x)

    elif scenario == "S3":
        return np.select(
            [
                (x >= -2) & (x < 1),
                (x >= 1) & (x <= 2),
                (x > 2) & (x <= 3),
            ],
            [0.0, 2.0, np.inf],
            default=np.nan,
        )

    raise ValueError(f"Unknown scenario: {scenario}")


