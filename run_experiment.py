"""Main entry point for the CP shift lab."""

from pathlib import Path

from src.io_utils import load_config, save_rows
from src.plots import (
    make_data_sanity_plot,
    plot_t1_binned_coverage,
    plot_t1_example,
    plot_t1_method_compare,
    plot_t1_residuals,
    plot_t2_binned_coverage,
    plot_t2_length_by_x,
    plot_t2_scale_diagnostic,
    plot_t3_binned_coverage,
    plot_t3_coverage_compare,
    plot_t3_density_comparison,
    plot_t3_residual_by_x,
    plot_t4_method_compare,
    plot_t4_threshold_by_x,
    plot_t4_weight_curve,
    plot_t4_weighted_residuals,
)
from src.t0 import run_pytest_and_write_log
from src.t1 import fit_split_cp, run_t1_repeats, summarize_t1
from src.t2 import fit_adaptive_cp, run_t2_repeats, summarize_t2
from src.t3 import fit_t3_shift_cp, run_t3_repeats, summarize_t3
from src.t4 import fit_oracle_wcp, run_t4_repeats, summarize_t4


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
OUTPUT_DIR = ROOT / "outputs"
T0_DIR = OUTPUT_DIR / "T0"
T1_DIR = OUTPUT_DIR / "T1"
T2_DIR = OUTPUT_DIR / "T2"
T3_DIR = OUTPUT_DIR / "T3"
T4_DIR = OUTPUT_DIR / "T4"


def main() -> None:
    """Generate T0-T4 outputs."""
    config = load_config(CONFIG_PATH)
    seed = int(config["seed"])

    t0_plot = make_data_sanity_plot(T0_DIR, seed)
    t0_log = run_pytest_and_write_log(ROOT, T0_DIR, seed)

    t1_rows = run_t1_repeats(config, n_repeats=200)
    t1_raw = save_rows(t1_rows, T1_DIR, "t1_raw_metrics.csv")

    t1_summary_rows = summarize_t1(t1_rows)
    t1_summary = save_rows(t1_summary_rows, T1_DIR, "t1_summary.csv")

    t2_rows = run_t2_repeats(config, n_repeats=200)
    t2_raw = save_rows(t2_rows, T2_DIR, "t2_raw_metrics.csv")

    t2_summary_rows = summarize_t2(t2_rows)
    t2_summary = save_rows(t2_summary_rows, T2_DIR, "t2_summary.csv")

    t3_rows = run_t3_repeats(config, n_repeats=200)
    t3_raw = save_rows(t3_rows, T3_DIR, "t3_raw_metrics.csv")

    t3_summary_rows = summarize_t3(t3_rows)
    t3_summary = save_rows(t3_summary_rows, T3_DIR, "t3_summary.csv")

    t4_rows = run_t4_repeats(config, n_repeats=200)
    t4_raw = save_rows(t4_rows, T4_DIR, "t4_raw_metrics.csv")

    t4_summary_rows = summarize_t4(t4_rows)
    t4_summary = save_rows(t4_summary_rows, T4_DIR, "t4_summary.csv")

    example = fit_split_cp(config, seed, scenario="S0")
    t1_example_plot = plot_t1_example(example, T1_DIR)
    t1_residual_plot = plot_t1_residuals(example, T1_DIR)
    t1_binned_plot = plot_t1_binned_coverage(t1_summary_rows, T1_DIR)
    t1_compare_plot = plot_t1_method_compare(t1_summary_rows, T1_DIR)

    adaptive_example = fit_adaptive_cp(config, seed, scenario="S0")
    t2_binned_plot = plot_t2_binned_coverage(t2_summary_rows, T2_DIR)
    t2_length_plot = plot_t2_length_by_x(adaptive_example, T2_DIR)
    t2_scale_plot = plot_t2_scale_diagnostic(adaptive_example, T2_DIR)

    t3_example = fit_t3_shift_cp(config, seed, target_scenario="S1")
    t3_density_plot = plot_t3_density_comparison(t3_example, T3_DIR)
    t3_residual_plot = plot_t3_residual_by_x(t3_example, T3_DIR)
    t3_coverage_plot = plot_t3_coverage_compare(t3_summary_rows, T3_DIR)
    t3_binned_plot = plot_t3_binned_coverage(t3_summary_rows, T3_DIR)

    t4_example = fit_oracle_wcp(config, seed, scenario="S1")
    t4_weight_plot = plot_t4_weight_curve(T4_DIR, scenario="S1")
    t4_residual_plot = plot_t4_weighted_residuals(t4_example, T4_DIR)
    t4_threshold_plot = plot_t4_threshold_by_x(t4_example, T4_DIR)
    t4_compare_plot = plot_t4_method_compare(t4_summary_rows, T4_DIR)

    print(f"saved: {t0_plot}")
    print(f"saved: {t0_log}")
    print(f"saved: {t1_raw}")
    print(f"saved: {t1_summary}")
    print(f"saved: {t2_raw}")
    print(f"saved: {t2_summary}")
    print(f"saved: {t3_raw}")
    print(f"saved: {t3_summary}")
    print(f"saved: {t4_raw}")
    print(f"saved: {t4_summary}")
    print(f"saved: {t1_example_plot}")
    print(f"saved: {t1_residual_plot}")
    print(f"saved: {t1_binned_plot}")
    print(f"saved: {t1_compare_plot}")
    print(f"saved: {t2_binned_plot}")
    print(f"saved: {t2_length_plot}")
    print(f"saved: {t2_scale_plot}")
    print(f"saved: {t3_density_plot}")
    print(f"saved: {t3_residual_plot}")
    print(f"saved: {t3_coverage_plot}")
    print(f"saved: {t3_binned_plot}")
    print(f"saved: {t4_weight_plot}")
    print(f"saved: {t4_residual_plot}")
    print(f"saved: {t4_threshold_plot}")
    print(f"saved: {t4_compare_plot}")


if __name__ == "__main__":
    main()
