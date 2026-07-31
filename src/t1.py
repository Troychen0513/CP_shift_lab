"""T1 workflow: ordinary Split CP on S0."""

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


def run_split_cp_once(config: dict, seed: int, scenario: str = "S0") -> dict:
    """Run one ordinary Split CP experiment and return its metrics."""
    rng = np.random.default_rng(seed)

    x_fit, y_fit = sample_source_xy(config["n_fit"], rng)
    x_cal, y_cal = sample_source_xy(config["n_cal"], rng)
    x_test, y_test = sample_target_xy(scenario, config["n_test"], rng)

    model = PolyModel(degree=config["model_degree"])
    model.fit(x_fit, y_fit)

    pred_cal = model.predict(x_cal)
    scores = np.abs(y_cal - pred_cal)
    q = conformal_order_statistic(scores, alpha=config["alpha"])

    pred_test = model.predict(x_test)
    lower = pred_test - q
    upper = pred_test + q

    metrics = {
        "method": "split_cp",
        "scenario": scenario,
        "seed": seed,
        "q": q,
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


def run_m0_once(config: dict, seed: int, scenario: str = "S0") -> dict:
    """Run one Gaussian residual interval baseline."""
    rng = np.random.default_rng(seed)

    x_fit, y_fit = sample_source_xy(config["n_fit"], rng)
    x_cal, y_cal = sample_source_xy(config["n_cal"], rng)
    x_test, y_test = sample_target_xy(scenario, config["n_test"], rng)

    model = PolyModel(degree=config["model_degree"])
    model.fit(x_fit, y_fit)

    pred_cal = model.predict(x_cal)
    sigma_hat = np.std(y_cal - pred_cal, ddof=1)

    pred_test = model.predict(x_test)
    radius = 1.645 * sigma_hat
    lower = pred_test - radius
    upper = pred_test + radius

    metrics = {
        "method": "m0_gaussian",
        "scenario": scenario,
        "seed": seed,
        "q": radius,
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


def run_t1_repeats(config: dict, n_repeats: int = 200) -> list[dict]:
    """Run repeated S0 experiments for M0 and Split CP."""
    base_seed = int(config["seed"])
    rows = []

    for i in range(n_repeats):
        seed = base_seed + i
        rows.append(run_m0_once(config, seed, scenario="S0"))
        rows.append(run_split_cp_once(config, seed, scenario="S0"))

    return rows


def summarize_t1(rows: list[dict]) -> list[dict]:
    """Summarize T1 repeated results by method."""
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
