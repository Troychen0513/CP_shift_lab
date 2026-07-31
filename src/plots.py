"""Plotting utilities for experiment outputs."""

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from src.data import sample_target_x, f_true, sigma_true, true_density_ratio

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
    
    
    
def plot_t1_example(result: dict, output_dir: Path) -> Path:
    """Plot data, true curve, fitted curve, and Split CP interval."""
    output_dir.mkdir(exist_ok=True)

    model = result["model"]
    q = result["q"]
    x_fit = result["x_fit"]
    y_fit = result["y_fit"]

    x_grid = np.linspace(-2, 2, 500)
    true_y = f_true(x_grid)
    pred_y = model.predict(x_grid)
    lower = pred_y - q
    upper = pred_y + q

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.scatter(x_fit, y_fit, s=8, alpha=0.25, label="fit data")
    ax.plot(x_grid, true_y, linewidth=2, label="true f(x)")
    ax.plot(x_grid, pred_y, linewidth=2, label="predicted center")
    ax.fill_between(x_grid, lower, upper, alpha=0.25, label="M1 interval")

    ax.set_title("T1 Split CP example interval")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "t1_example_interval.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path
    


    
def plot_t1_residuals(result: dict, output_dir: Path) -> Path:
    """Plot calibration residuals and the Split CP threshold q."""
    output_dir.mkdir(exist_ok=True)

    scores = result["scores"]
    q = result["q"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.hist(scores, bins=40, alpha=0.75, edgecolor="white")
    ax.axvline(q, color="red", linewidth=2, label=f"q = {q:.3f}")

    ax.set_title("T1 calibration residuals")
    ax.set_xlabel("absolute residual")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "t1_residual_hist.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path



def plot_t1_binned_coverage(summary_rows: list[dict], output_dir: Path) -> Path:
    """Plot mean coverage in five X bins for each method."""
    output_dir.mkdir(exist_ok=True)

    bin_names = [f"bin_{i}_coverage_mean" for i in range(1, 6)]
    x = np.arange(1, 6)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for offset, row in zip([-width / 2, width / 2], summary_rows):
        values = [row[name] for name in bin_names]
        ax.bar(x + offset, values, width=width, label=row["method"])

    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="target 0.90")
    ax.set_title("T1 binned coverage by X")
    ax.set_xlabel("X bin")
    ax.set_ylabel("coverage")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    output_path = output_dir / "t1_binned_coverage.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path



def plot_t1_method_compare(summary_rows: list[dict], output_dir: Path) -> Path:
    """Compare coverage and interval length between M0 and Split CP."""
    output_dir.mkdir(exist_ok=True)

    methods = [row["method"] for row in summary_rows]
    coverage_values = [row["coverage_mean"] for row in summary_rows]
    length_values = [row["mean_length"] for row in summary_rows]

    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    axes[0].bar(x, coverage_values)
    axes[0].axhline(0.90, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Coverage")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, length_values)
    axes[1].set_title("Mean interval length")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].grid(axis="y", alpha=0.25)

    output_path = output_dir / "t1_method_compare.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t2_binned_coverage(summary_rows: list[dict], output_dir: Path) -> Path:
    """Plot five-bin coverage for Split CP and adaptive CP."""
    output_dir.mkdir(exist_ok=True)

    bin_names = [f"bin_{i}_coverage_mean" for i in range(1, 6)]
    x = np.arange(1, 6)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for offset, row in zip([-width / 2, width / 2], summary_rows):
        values = [row[name] for name in bin_names]
        ax.bar(x + offset, values, width=width, label=row["method"])

    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="target 0.90")
    ax.set_title("T2 binned coverage by X")
    ax.set_xlabel("X bin")
    ax.set_ylabel("coverage")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    output_path = output_dir / "t2_binned_coverage.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path

def plot_t2_length_by_x(adaptive_result: dict, output_dir: Path) -> Path:
    """Plot how adaptive CP interval length changes with X."""
    output_dir.mkdir(exist_ok=True)

    x_test = adaptive_result["x_test"]
    lower = adaptive_result["lower"]
    upper = adaptive_result["upper"]

    length = upper - lower
    order = np.argsort(x_test)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.scatter(x_test, length, s=8, alpha=0.20, label="test intervals")
    ax.plot(x_test[order], length[order], linewidth=2, label="sorted length")

    ax.set_title("T2 adaptive interval length by X")
    ax.set_xlabel("x")
    ax.set_ylabel("interval length")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "t2_length_by_x.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path

def plot_t2_scale_diagnostic(adaptive_result: dict, output_dir: Path) -> Path:
    """Compare learned scale with the true noise scale."""
    output_dir.mkdir(exist_ok=True)

    scale_model = adaptive_result["scale_model"]

    x_grid = np.linspace(-2, 2, 500)
    true_scale = sigma_true(x_grid)
    learned_scale = np.exp(scale_model.predict(x_grid))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(x_grid, true_scale, linewidth=2, label="true sigma(x)")
    ax.plot(x_grid, learned_scale, linewidth=2, label="learned scale")

    ax.set_title("T2 scale model diagnostic")
    ax.set_xlabel("x")
    ax.set_ylabel("scale")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "t2_scale_diagnostic.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def binned_curve(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers and bin means after sorting points by x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    bins = np.array_split(order, n_bins)

    centers = np.array([float(np.mean(x[idx])) for idx in bins], dtype=float)
    means = np.array([float(np.mean(y[idx])) for idx in bins], dtype=float)
    return centers, means


def plot_t3_density_comparison(result: dict, output_dir: Path) -> Path:
    """Plot source and target X densities for the T3 covariate shift case."""
    output_dir.mkdir(exist_ok=True)

    x_source = result["x_source_test"]
    x_target = result["x_target_test"]
    x_grid = np.linspace(-2, 2, 500)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharex=True, sharey=True)

    axes[0].hist(x_source, bins=60, density=True, alpha=0.75, color="C0", edgecolor="white")
    axes[0].plot(x_grid, np.full_like(x_grid, 0.25), color="black", linewidth=2)
    axes[0].set_title("Source S0")

    axes[1].hist(x_target, bins=60, density=True, alpha=0.75, color="C1", edgecolor="white")
    axes[1].plot(x_grid, tilted_density(x_grid, 1.2), color="black", linewidth=2)
    axes[1].set_title("Target S1")

    for ax in axes:
        ax.set_xlabel("x")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("density")
    fig.suptitle("T3 source and target X densities")
    fig.tight_layout()

    output_path = output_dir / "t3_x_density.png"
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t3_residual_by_x(result: dict, output_dir: Path) -> Path:
    """Plot absolute residuals versus x for source and target test sets."""
    output_dir.mkdir(exist_ok=True)

    model = result["model"]
    q = result["q"]

    x_source = result["x_source_test"]
    y_source = result["y_source_test"]
    x_target = result["x_target_test"]
    y_target = result["y_target_test"]

    residual_source = np.abs(y_source - model.predict(x_source))
    residual_target = np.abs(y_target - model.predict(x_target))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharex=True, sharey=True)

    for ax, x, residual, title, color in [
        (axes[0], x_source, residual_source, "Source S0", "C0"),
        (axes[1], x_target, residual_target, "Target S1", "C1"),
    ]:
        ax.scatter(x, residual, s=8, alpha=0.16, color=color, edgecolors="none")
        centers, means = binned_curve(x, residual, n_bins=20)
        ax.plot(centers, means, color=color, linewidth=2)
        ax.axhline(q, color="black", linestyle="--", linewidth=1.5, label="Split CP q")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("|y - mu_hat(x)|")
    axes[0].legend()

    fig.suptitle("T3 absolute residuals by x")
    fig.tight_layout()

    output_path = output_dir / "t3_residual_by_x.png"
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t3_coverage_compare(summary_rows: list[dict], output_dir: Path) -> Path:
    """Compare overall Split CP coverage on source and target domains."""
    output_dir.mkdir(exist_ok=True)

    domains = [row["domain"] for row in summary_rows]
    values = [row["coverage_mean"] for row in summary_rows]
    x = np.arange(len(domains))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    bars = ax.bar(x, values, color=["C0", "C1"])
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="target 0.90")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("T3 coverage comparison")
    ax.set_xlabel("domain")
    ax.set_ylabel("coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    output_path = output_dir / "t3_coverage_compare.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t3_binned_coverage(summary_rows: list[dict], output_dir: Path) -> Path:
    """Compare five-bin coverage between source and target domains."""
    output_dir.mkdir(exist_ok=True)

    bin_names = [f"bin_{i}_coverage_mean" for i in range(1, 6)]
    x = np.arange(1, 6)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for offset, row, color in zip([-width / 2, width / 2], summary_rows, ["C0", "C1"]):
        values = [row[name] for name in bin_names]
        ax.bar(x + offset, values, width=width, label=row["domain"], color=color)

    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="target 0.90")
    ax.set_title("T3 binned coverage by X")
    ax.set_xlabel("X bin")
    ax.set_ylabel("coverage")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    output_path = output_dir / "t3_binned_coverage.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t4_weight_curve(output_dir: Path, scenario: str = "S1") -> Path:
    """Plot the true density ratio used by oracle WCP."""
    output_dir.mkdir(exist_ok=True)

    x_grid = np.linspace(-2, 2, 500)
    weights = true_density_ratio(x_grid, scenario)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(x_grid, weights, linewidth=2)
    ax.set_title("T4 true density ratio")
    ax.set_xlabel("x")
    ax.set_ylabel("w(x)")
    ax.grid(alpha=0.25)

    output_path = output_dir / "t4_weight_curve.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t4_weighted_residuals(result: dict, output_dir: Path) -> Path:
    """Compare calibration residuals before and after oracle weighting."""
    output_dir.mkdir(exist_ok=True)

    scores = result["scores"]
    weights = result["weights_cal"]
    q_split = result["q_split"]
    finite_q_w = result["q_w"][np.isfinite(result["q_w"])]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.hist(scores, bins=40, density=True, alpha=0.45, label="unweighted")
    ax.hist(scores, bins=40, density=True, weights=weights, alpha=0.45, label="weighted")
    ax.axvline(q_split, color="C0", linestyle="--", linewidth=2, label="Split CP q")

    if len(finite_q_w) > 0:
        ax.axvline(
            float(np.median(finite_q_w)),
            color="C1",
            linestyle="--",
            linewidth=2,
            label="median WCP q",
        )

    ax.set_title("T4 calibration residual distribution")
    ax.set_xlabel("absolute residual")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "t4_weighted_residuals.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t4_threshold_by_x(result: dict, output_dir: Path) -> Path:
    """Plot how oracle WCP thresholds change with test x."""
    output_dir.mkdir(exist_ok=True)

    x_test = result["x_test"]
    q_w = result["q_w"]
    finite_mask = np.isfinite(q_w)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.scatter(x_test[finite_mask], q_w[finite_mask], s=8, alpha=0.20, label="test points")

    if np.any(finite_mask):
        centers, means = binned_curve(x_test[finite_mask], q_w[finite_mask], n_bins=20)
        ax.plot(centers, means, color="black", linewidth=2, label="binned mean")

    ax.axhline(result["q_split"], color="red", linestyle="--", linewidth=1.5, label="Split CP q")
    ax.set_title("T4 oracle WCP threshold by x")
    ax.set_xlabel("x")
    ax.set_ylabel("q_w(x)")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "t4_threshold_by_x.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t4_method_compare(summary_rows: list[dict], output_dir: Path) -> Path:
    """Compare coverage and interval length for Split CP and oracle WCP."""
    output_dir.mkdir(exist_ok=True)

    methods = [row["method"] for row in summary_rows]
    coverage_values = [row["coverage_mean"] for row in summary_rows]
    length_values = [row["mean_length"] for row in summary_rows]
    score_values = [row["mean_interval_score"] for row in summary_rows]

    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    axes[0].bar(x, coverage_values)
    axes[0].axhline(0.90, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Coverage")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(x, length_values)
    axes[1].set_title("Mean length")

    axes[2].bar(x, score_values)
    axes[2].set_title("Interval score")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    output_path = output_dir / "t4_method_compare.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t5_weight_curve(result: dict, output_dir: Path, scenario: str = "S1") -> Path:
    """Plot true and estimated density ratios across x."""
    output_dir.mkdir(exist_ok=True)

    domain_model = result["domain_model"]
    x_grid = np.linspace(-2, 2, 500)
    true_weights = true_density_ratio(x_grid, scenario)
    estimated_weights = domain_model.density_ratio(x_grid, clipped=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(x_grid, true_weights, linewidth=2, label="true w(x)")
    ax.plot(x_grid, estimated_weights, linewidth=2, label="estimated w(x)")
    ax.set_title("T5 true and estimated density ratios")
    ax.set_xlabel("x")
    ax.set_ylabel("weight")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "t5_weight_curve.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t5_domain_prob_hist(result: dict, output_dir: Path) -> Path:
    """Plot domain classifier probabilities for source and target X."""
    output_dir.mkdir(exist_ok=True)

    domain_model = result["domain_model"]
    prob_source = domain_model.predict_target_prob(result["x_cal"])
    prob_target = domain_model.predict_target_prob(result["x_target_u"])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.hist(prob_source, bins=40, alpha=0.55, density=True, label="source cal X")
    ax.hist(prob_target, bins=40, alpha=0.55, density=True, label="target unlabeled X")
    ax.set_title("T5 domain classifier probabilities")
    ax.set_xlabel("P(target domain | x)")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.25)

    output_path = output_dir / "t5_domain_prob_hist.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_t5_method_compare(summary_rows: list[dict], output_dir: Path) -> Path:
    """Compare Split CP, oracle WCP, and estimated WCP."""
    output_dir.mkdir(exist_ok=True)

    methods = [row["method"] for row in summary_rows]
    coverage_values = [row["coverage_mean"] for row in summary_rows]
    length_values = [row["mean_length"] for row in summary_rows]
    score_values = [row["mean_interval_score"] for row in summary_rows]

    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    axes[0].bar(x, coverage_values)
    axes[0].axhline(0.90, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Coverage")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(x, length_values)
    axes[1].set_title("Mean length")

    axes[2].bar(x, score_values)
    axes[2].set_title("Interval score")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    output_path = output_dir / "t5_method_compare.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path
