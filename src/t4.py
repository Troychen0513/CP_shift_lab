"""T4 workflow: oracle weighted conformal prediction under covariate shift."""

import numpy as np

from src.conformal import conformal_order_statistic
from src.data import sample_source_xy, sample_target_xy, true_density_ratio
from src.metrics import (
    binned_coverage,
    coverage,
    effective_sample_size,
    mean_interval_length,
    mean_interval_score,
)
from src.models import PolyModel
from src.t1 import run_split_cp_once


def weighted_thresholds(
    scores: np.ndarray,
    weights_cal: np.ndarray,
    weights_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return one WCP threshold for each test point."""
    scores = np.asarray(scores, dtype=float)
    weights_cal = np.asarray(weights_cal, dtype=float)
    weights_test = np.asarray(weights_test, dtype=float)

    if scores.shape != weights_cal.shape:
        raise ValueError("scores and weights_cal must have the same shape")
    if np.any(weights_cal < 0) or np.any(weights_test < 0):
        raise ValueError("weights must be nonnegative")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = weights_cal[order]
    cumulative_weight = np.cumsum(sorted_weights)
    total_cal_weight = float(cumulative_weight[-1])

    target_weight = (1 - alpha) * (total_cal_weight + weights_test)
    thresholds = np.full_like(weights_test, np.inf, dtype=float)

    finite_mask = target_weight <= total_cal_weight
    indexes = np.searchsorted(cumulative_weight, target_weight[finite_mask], side="left")
    thresholds[finite_mask] = sorted_scores[indexes]

    return thresholds


def fit_oracle_wcp(config: dict, seed: int, scenario: str = "S1") -> dict:
    """Fit Split CP on source data and apply oracle WCP on the target domain."""
    rng = np.random.default_rng(seed)

    x_fit, y_fit = sample_source_xy(config["n_fit"], rng)
    x_cal, y_cal = sample_source_xy(config["n_cal"], rng)
    x_test, y_test = sample_target_xy(scenario, config["n_test"], rng)

    model = PolyModel(degree=config["model_degree"])
    model.fit(x_fit, y_fit)

    pred_cal = model.predict(x_cal)
    scores = np.abs(y_cal - pred_cal)
    q_split = conformal_order_statistic(scores, alpha=config["alpha"])

    weights_cal = true_density_ratio(x_cal, scenario)
    weights_test = true_density_ratio(x_test, scenario)
    q_w = weighted_thresholds(scores, weights_cal, weights_test, config["alpha"])

    pred_test = model.predict(x_test)
    lower = pred_test - q_w
    upper = pred_test + q_w

    return {
        "model": model,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_test": x_test,
        "y_test": y_test,
        "scores": scores,
        "q_split": q_split,
        "q_w": q_w,
        "weights_cal": weights_cal,
        "weights_test": weights_test,
        "pred_test": pred_test,
        "lower": lower,
        "upper": upper,
    }


def run_oracle_wcp_once(config: dict, seed: int, scenario: str = "S1") -> dict:
    """Run one oracle WCP experiment and return its metrics."""
    result = fit_oracle_wcp(config, seed, scenario)

    x_test = result["x_test"]
    y_test = result["y_test"]
    lower = result["lower"]
    upper = result["upper"]
    weights_cal = result["weights_cal"]

    metrics = {
        "method": "oracle_wcp",
        "scenario": scenario,
        "seed": seed,
        "q": float(np.mean(result["q_w"])),
        "coverage": coverage(y_test, lower, upper),
        "mean_length": mean_interval_length(lower, upper),
        "mean_interval_score": mean_interval_score(
            y_test,
            lower,
            upper,
            config["alpha"],
        ),
        "ess": effective_sample_size(weights_cal),
        "max_weight": float(np.max(weights_cal)),
        "infinite_interval_rate": float(np.mean(np.isinf(result["q_w"]))),
    }

    metrics.update(binned_coverage(x_test, y_test, lower, upper))
    return metrics


def run_t4_repeats(config: dict, n_repeats: int = 200) -> list[dict]:
    """Run repeated S1 experiments for Split CP and oracle WCP."""
    base_seed = int(config["seed"])
    rows = []

    for i in range(n_repeats):
        seed = base_seed + i
        rows.append(run_split_cp_once(config, seed, scenario="S1"))
        rows.append(run_oracle_wcp_once(config, seed, scenario="S1"))

    return rows


def summarize_t4(rows: list[dict]) -> list[dict]:
    """Summarize T4 repeated results by method."""
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

        if method == "oracle_wcp":
            method_summary["ess_mean"] = float(np.mean([row["ess"] for row in method_rows]))
            method_summary["max_weight_mean"] = float(np.mean([row["max_weight"] for row in method_rows]))
            method_summary["infinite_interval_rate_mean"] = float(
                np.mean([row["infinite_interval_rate"] for row in method_rows])
            )

        summary.append(method_summary)

    return summary
