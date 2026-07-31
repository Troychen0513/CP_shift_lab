"""T5 workflow: estimate density ratios with a domain classifier."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.conformal import conformal_order_statistic
from src.data import sample_source_xy, sample_target_x, sample_target_xy, true_density_ratio
from src.metrics import (
    binned_coverage,
    coverage,
    effective_sample_size,
    mean_interval_length,
    mean_interval_score,
)
from src.models import PolyModel
from src.t1 import run_split_cp_once
from src.t4 import run_oracle_wcp_once, weighted_thresholds


def nan_mean(values: list[float]) -> float:
    """Return the mean after ignoring nan values."""
    values = np.asarray(values, dtype=float)

    if np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmean(values))


def make_domain_data(
    x_source: np.ndarray,
    x_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one-feature data and labels for source-vs-target classification."""
    x_source = np.asarray(x_source, dtype=float)
    x_target = np.asarray(x_target, dtype=float)

    x = np.concatenate([x_source, x_target]).reshape(-1, 1)
    y = np.concatenate(
        [
            np.zeros(len(x_source), dtype=int),
            np.ones(len(x_target), dtype=int),
        ]
    )

    return x, y


class DomainClassifier:
    """Estimate whether an x value looks like target-domain data."""

    def __init__(self, clip: float = 0.01) -> None:
        self.clip = clip
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)

    def fit(self, x_source: np.ndarray, x_target: np.ndarray) -> None:
        """Train the domain classifier using source X and unlabeled target X."""
        x_train, y_train = make_domain_data(x_source, x_target)
        self.model.fit(x_train, y_train)

    def predict_target_prob(self, x: np.ndarray) -> np.ndarray:
        """Predict P(target domain | x)."""
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]

    def density_ratio(self, x: np.ndarray, clipped: bool = True) -> np.ndarray:
        """Convert target probabilities into estimated density ratios."""
        prob = self.predict_target_prob(x)

        if clipped:
            prob = np.clip(prob, self.clip, 1 - self.clip)

        return prob / (1 - prob)

    def auc(self, x_source: np.ndarray, x_target: np.ndarray) -> float:
        """Measure how well the classifier separates source and target X."""
        x_eval, y_eval = make_domain_data(x_source, x_target)
        prob = self.model.predict_proba(x_eval)[:, 1]
        return float(roc_auc_score(y_eval, prob))


def fit_estimated_wcp(
    config: dict,
    seed: int,
    scenario: str = "S1",
    clip: float = 0.01,
) -> dict:
    """Fit Split CP and apply WCP with density ratios estimated from X only."""
    rng = np.random.default_rng(seed)

    x_fit, y_fit = sample_source_xy(config["n_fit"], rng)
    x_cal, y_cal = sample_source_xy(config["n_cal"], rng)
    x_target_u = sample_target_x(scenario, config["n_target_u"], rng)
    x_test, y_test = sample_target_xy(scenario, config["n_test"], rng)

    model = PolyModel(degree=config["model_degree"])
    model.fit(x_fit, y_fit)

    pred_cal = model.predict(x_cal)
    scores = np.abs(y_cal - pred_cal)
    q_split = conformal_order_statistic(scores, alpha=config["alpha"])

    domain_model = DomainClassifier(clip=clip)
    domain_model.fit(x_cal, x_target_u)

    weights_cal_hat = domain_model.density_ratio(x_cal, clipped=True)
    weights_test_hat = domain_model.density_ratio(x_test, clipped=True)
    q_hat = weighted_thresholds(scores, weights_cal_hat, weights_test_hat, config["alpha"])

    pred_test = model.predict(x_test)
    lower = pred_test - q_hat
    upper = pred_test + q_hat

    return {
        "model": model,
        "domain_model": domain_model,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_target_u": x_target_u,
        "x_test": x_test,
        "y_test": y_test,
        "scores": scores,
        "q_split": q_split,
        "q_hat": q_hat,
        "weights_cal_hat": weights_cal_hat,
        "weights_test_hat": weights_test_hat,
        "weights_cal_true": true_density_ratio(x_cal, scenario),
        "weights_test_true": true_density_ratio(x_test, scenario),
        "domain_auc": domain_model.auc(x_cal, x_target_u),
        "pred_test": pred_test,
        "lower": lower,
        "upper": upper,
    }


def run_estimated_wcp_once(
    config: dict,
    seed: int,
    scenario: str = "S1",
    clip: float = 0.01,
) -> dict:
    """Run one estimated WCP experiment and return its metrics."""
    result = fit_estimated_wcp(config, seed, scenario, clip)

    x_test = result["x_test"]
    y_test = result["y_test"]
    lower = result["lower"]
    upper = result["upper"]
    weights_cal_hat = result["weights_cal_hat"]

    metrics = {
        "method": "estimated_wcp",
        "scenario": scenario,
        "seed": seed,
        "q": float(np.mean(result["q_hat"])),
        "coverage": coverage(y_test, lower, upper),
        "mean_length": mean_interval_length(lower, upper),
        "mean_interval_score": mean_interval_score(
            y_test,
            lower,
            upper,
            config["alpha"],
        ),
        "ess": effective_sample_size(weights_cal_hat),
        "max_weight": float(np.max(weights_cal_hat)),
        "weight_q99": float(np.quantile(weights_cal_hat, 0.99)),
        "infinite_interval_rate": float(np.mean(np.isinf(result["q_hat"]))),
        "domain_auc": result["domain_auc"],
    }

    metrics.update(binned_coverage(x_test, y_test, lower, upper))
    return metrics


def add_t5_diagnostics(row: dict) -> dict:
    """Add missing T5 diagnostic fields so all methods share one table schema."""
    defaults = {
        "ess": np.nan,
        "max_weight": np.nan,
        "weight_q99": np.nan,
        "infinite_interval_rate": np.nan,
        "domain_auc": np.nan,
    }

    for key, value in defaults.items():
        row.setdefault(key, value)

    return row


def run_t5_repeats(config: dict, n_repeats: int = 200) -> list[dict]:
    """Run repeated S1 experiments for Split, oracle WCP, and estimated WCP."""
    base_seed = int(config["seed"])
    rows = []

    for i in range(n_repeats):
        seed = base_seed + i

        rows.append(add_t5_diagnostics(run_split_cp_once(config, seed, scenario="S1")))

        oracle_row = run_oracle_wcp_once(config, seed, scenario="S1")
        oracle_row.setdefault("weight_q99", np.nan)
        oracle_row.setdefault("domain_auc", np.nan)
        rows.append(add_t5_diagnostics(oracle_row))

        rows.append(add_t5_diagnostics(run_estimated_wcp_once(config, seed, scenario="S1")))

    return rows


def summarize_t5(rows: list[dict]) -> list[dict]:
    """Summarize T5 repeated results by method."""
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

        summary.append(
            {
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
                "ess_mean": nan_mean([row["ess"] for row in method_rows]),
                "max_weight_mean": nan_mean([row["max_weight"] for row in method_rows]),
                "weight_q99_mean": nan_mean([row["weight_q99"] for row in method_rows]),
                "infinite_interval_rate_mean": nan_mean(
                    [row["infinite_interval_rate"] for row in method_rows]
                ),
                "domain_auc_mean": nan_mean([row["domain_auc"] for row in method_rows]),
            }
        )

    return summary
