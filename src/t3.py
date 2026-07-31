"""T3 workflow: show ordinary Split CP under covariate shift."""

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

def eval_split_interval(x_test: np.ndarray,y_test: np.ndarray,model: PolyModel,q: float,alpha: float)->dict:
    """Evaluate one fixed Split CP interval on a test set."""
    
    pred_test = model.predict(x_test)
    lower = pred_test - q
    upper = pred_test + q
    
    result = {
        "coverage": coverage(y_test, lower, upper),
        "mean_length": mean_interval_length(lower, upper),
        "mean_interval_score": mean_interval_score(y_test, lower, upper, alpha),
        "lower": lower,
        "upper": upper,
        "pred_test": pred_test,
    }

    result.update(binned_coverage(x_test, y_test, lower, upper))
    return result

def fit_t3_shift_cp(config: dict, seed: int, target_scenario: str = "S1") -> dict:
    """Fit Split CP on source data, then test it on source and target data."""
    rng = np.random.default_rng(seed)
    
    x_fit, y_fit = sample_source_xy(config["n_fit"], rng)
    x_cal, y_cal = sample_source_xy(config["n_cal"], rng)
    
    x_source_test, y_source_test = sample_target_xy("S0", config["n_test"], rng)
    x_target_test, y_target_test = sample_target_xy(target_scenario, config["n_test"], rng)
    
    model = PolyModel(degree=config["model_degree"])
    model.fit(x_fit,y_fit)
    
    pred_cal = model.predict(x_cal)
    scores = np.abs(pred_cal-y_cal)
    q = conformal_order_statistic(scores, alpha=config["alpha"])
    
    source_eval = eval_split_interval(
        x_source_test,
        y_source_test,
        model,
        q,
        config["alpha"],
    )
    target_eval = eval_split_interval(
        x_target_test,
        y_target_test,
        model,
        q,
        config["alpha"],
    )

    return {
        "model": model,
        "q": q,
        "scores": scores,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_source_test": x_source_test,
        "y_source_test": y_source_test,
        "x_target_test": x_target_test,
        "y_target_test": y_target_test,
        "source_eval": source_eval,
        "target_eval": target_eval,
    }
    
    
def run_t3_repeats(config: dict, n_repeats: int = 200) -> list[dict]:
    """Run repeated T3 experiments on source and S1 target data."""
    base_seed = int(config["seed"])
    rows = []

    for i in range(n_repeats):
        seed = base_seed + i
        result = fit_t3_shift_cp(config, seed, target_scenario="S1")

        for domain, eval_result in [
            ("source", result["source_eval"]),
            ("target_s1", result["target_eval"]),
        ]:
            row = {
                "method": "split_cp",
                "domain": domain,
                "scenario": "S0" if domain == "source" else "S1",
                "seed": seed,
                "q": result["q"],
                "coverage": eval_result["coverage"],
                "mean_length": eval_result["mean_length"],
                "mean_interval_score": eval_result["mean_interval_score"],
            }

            for bin_id in range(1, 6):
                name = f"bin_{bin_id}_coverage"
                row[name] = eval_result[name]

            rows.append(row)

    return rows


def summarize_t3(rows: list[dict]) -> list[dict]:
    """Summarize T3 results by test domain."""
    summary = []
    domains = sorted({row["domain"] for row in rows})
    bin_names = [f"bin_{i}_coverage" for i in range(1, 6)]

    for domain in domains:
        domain_rows = [row for row in rows if row["domain"] == domain]

        coverages = np.array([row["coverage"] for row in domain_rows])
        lengths = np.array([row["mean_length"] for row in domain_rows])
        scores = np.array([row["mean_interval_score"] for row in domain_rows])

        bin_means = {
            f"{name}_mean": float(np.mean([row[name] for row in domain_rows]))
            for name in bin_names
        }

        summary.append(
            {
                "method": "split_cp",
                "domain": domain,
                "n_repeats": len(domain_rows),
                "coverage_mean": float(np.mean(coverages)),
                "coverage_std": float(np.std(coverages, ddof=1)),
                "coverage_5pct": float(np.quantile(coverages, 0.05)),
                "coverage_95pct": float(np.quantile(coverages, 0.95)),
                "mean_length": float(np.mean(lengths)),
                "mean_interval_score": float(np.mean(scores)),
                **bin_means,
                "worst_bin_coverage_mean": min(bin_means.values()),
            }
        )

    return summary