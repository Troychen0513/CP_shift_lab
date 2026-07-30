"""Plotting utilities for experiment outputs."""

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from src.data import sample_target_x

def tilted_density(x:np.ndarray,beta:float) -> np.ndarray:
    """Return the normalized density proportional to exp(beta*x) on [-2, 2].
    计算 S1/S2 的理论密度曲线 """
    normalizer = np.exp(2*beta) - np.exp(-2*beta)
    
    return beta * np.exp(beta * x) / normalizer
    
   
# 判断：我们生成出来的 S0/S1/S2 的 X 分布，是否真的符合实验方案？ 
def make_data_sanity_plot(output_dir: Path, seed: int) -> Path:
    """Save S0/S1/S2 empirical and theoretical X densities."""
    output_dir.mkdir(exist_ok=True)
    
    rng = np.random.default_rng(seed)
    n = 50_000
    x_grid = np.linspace(-2,2,500)
    x_s0 = sample_target_x("S0", n, rng)
    x_s1 = sample_target_x("S1", n, rng)
    x_s2 = sample_target_x("S2", n, rng)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    ax.hist(x_s0, bins=80, density=True, alpha=0.35, label="S0 empirical")
    ax.hist(x_s1, bins=80, density=True, alpha=0.35, label="S1 empirical")
    ax.hist(x_s2, bins=80, density=True, alpha=0.35, label="S2 empirical")
    
    ax.plot(x_grid, np.full_like(x_grid, 0.25), label="S0 theory", linewidth=2)
    ax.plot(x_grid, tilted_density(x_grid, 1.2), label="S1 theory", linewidth=2)
    ax.plot(x_grid, tilted_density(x_grid, 2.5), label="S2 theory", linewidth=2)
    
    ax.set_title("T0 data sanity check: X density")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.25) # 设置透明度
    
    output_path = output_dir / "data_sanity.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    
    return output_path
    