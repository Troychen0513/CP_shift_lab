"""T2 workflow: adaptive Split CP on S0."""

import numpy as np

from src.conformal import conformal_order_statistic
from src.data import sample_source_xy, sample_target_xy
from src.metrics import (
    binned_coverage,
    coverage,
    mean_interval_length,
    mean_interval_score,
)
from src.models import PolyModel
from src.t1 import run_split_cp_once



# 自适应
def fit_adaptive_cp(config: dict, seed: int, scenario: str = "S0")-> dict:
    """Fit one adaptive Split CP run and keep data for metrics and plots."""
    rng = np.random.default_rng(seed)
    
    x_fit, y_fit = sample_source_xy(config["n_fit"], rng)
    x_cal, y_cal = sample_source_xy(config["n_cal"], rng)
    x_test, y_test = sample_target_xy(scenario, config["n_test"], rng)
    
    center_model = PolyModel(degree=config["model_degree"])
    center_model.fit(x_fit,y_fit)
    
    pred_fit = center_model.predict(x_fit)
    fit_residuals = np.abs(y_fit - pred_fit)   # 训练集上的绝对残差
    
    scale_model = PolyModel(degree=config["model_degree"])
    scale_model.fit(x_fit, np.log(fit_residuals + 1e-4))   # 预测误差尺度 sigma_hat(x)，这里预测的是 log(residual)，原因是残差必须为正。
    
    scale_cal = np.exp(scale_model.predict(x_cal))
    scale_test = np.exp(scale_model.predict(x_test))
    
    pred_cal = center_model.predict(x_cal)
    scores = np.abs(y_cal - pred_cal) / scale_cal
    q = conformal_order_statistic(scores, alpha=config["alpha"])
    
    pred_test = center_model.predict(x_test)
    lower = pred_test - q * scale_test
    upper = pred_test + q * scale_test
    
    return {
        "center_model": center_model,
        "scale_model": scale_model,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_test": x_test,
        "y_test": y_test,
        "scores": scores,
        "q": q,
        "scale_test": scale_test,
        "pred_test": pred_test,
        "lower": lower,
        "upper": upper,
    }
    
    
def run_adaptive_cp_once(config: dict, seed: int, scenario: str = "S0") -> dict:
    """Run one adaptive Split CP experiment and return its metrics."""
    result = fit_adaptive_cp(config, seed, scenario)

    x_test = result["x_test"]
    y_test = result["y_test"]
    lower = result["lower"]
    upper = result["upper"]

    metrics = {
        "method": "adaptive_cp",
        "scenario": scenario,
        "seed": seed,
        "q": result["q"],
        "coverage": coverage(y_test, lower, upper),
        "mean_length": mean_interval_length(lower, upper),
        "mean_interval_score": mean_interval_score(
            y_test,
            lower,
            upper,
            config["alpha"],
        ),
    }

    metrics.update(binned_coverage(x_test, y_test, lower, upper))
    return metrics


def run_t2_repeats(config: dict, n_repeats: int = 200) -> list[dict]:
    """Run repeated S0 experiments for Split CP and adaptive CP."""
    base_seed = int(config["seed"])
    rows = []

    for i in range(n_repeats):
        seed = base_seed + i

        rows.append(run_split_cp_once(config, seed, scenario="S0"))
        rows.append(run_adaptive_cp_once(config, seed, scenario="S0"))

    return rows


def summarize_t2(rows: list[dict]) -> list[dict]:
    """Summarize T2 repeated results by method."""
    summary = []
    methods = sorted({row["method"] for row in rows})
    bin_names = [f"bin_{i}_coverage" for i in range(1, 6)]

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]

        coverages = np.array([row["coverage"] for row in method_rows])
        lengths = np.array([row["mean_length"] for row in method_rows])
        scores = np.array([row["mean_interval_score"] for row in method_rows])

        bin_means = {
            f"{name}_mean": float(np.mean([row[name] for row in method_rows]))
            for name in bin_names
        }

        method_summary = {
            "method": method,
            "n_repeats": len(method_rows),
            "coverage_mean": float(np.mean(coverages)),
            "coverage_std": float(np.std(coverages, ddof=1)),
            "coverage_5pct": float(np.quantile(coverages, 0.05)),
            "coverage_95pct": float(np.quantile(coverages, 0.95)),
            "mean_length": float(np.mean(lengths)),
            "mean_interval_score": float(np.mean(scores)),
            **bin_means,
            "worst_bin_coverage_mean": min(bin_means.values()),
        }

        summary.append(method_summary)

    return summary