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
)
from src.t0 import run_pytest_and_write_log
from src.t1 import run_t1_repeats, summarize_t1,fit_split_cp
from src.t2 import fit_adaptive_cp, run_t2_repeats, summarize_t2


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    """Generate T0-T2 outputs."""
    config = load_config(CONFIG_PATH)
    seed = int(config["seed"])

    t0_plot = make_data_sanity_plot(OUTPUT_DIR, seed)
    t0_log = run_pytest_and_write_log(ROOT, OUTPUT_DIR, seed)

    t1_rows = run_t1_repeats(config, n_repeats=200)
    t1_raw = save_rows(t1_rows, OUTPUT_DIR, "t1_raw_metrics.csv")

    t1_summary_rows = summarize_t1(t1_rows)
    t1_summary = save_rows(t1_summary_rows, OUTPUT_DIR, "t1_summary.csv")

    t2_rows = run_t2_repeats(config, n_repeats=200)
    t2_raw = save_rows(t2_rows, OUTPUT_DIR, "t2_raw_metrics.csv")

    t2_summary_rows = summarize_t2(t2_rows)
    t2_summary = save_rows(t2_summary_rows, OUTPUT_DIR, "t2_summary.csv")
    
    example = fit_split_cp(config, seed, scenario="S0")
    t1_example_plot = plot_t1_example(example, OUTPUT_DIR)
    t1_residual_plot = plot_t1_residuals(example, OUTPUT_DIR)
    t1_binned_plot = plot_t1_binned_coverage(t1_summary_rows, OUTPUT_DIR)
    t1_compare_plot = plot_t1_method_compare(t1_summary_rows, OUTPUT_DIR)
    
    adaptive_example = fit_adaptive_cp(config, seed, scenario="S0")
    t2_binned_plot = plot_t2_binned_coverage(t2_summary_rows, OUTPUT_DIR)
    t2_length_plot = plot_t2_length_by_x(adaptive_example, OUTPUT_DIR)
    t2_scale_plot = plot_t2_scale_diagnostic(adaptive_example, OUTPUT_DIR)

    print(f"saved: {t0_plot}")
    print(f"saved: {t0_log}")
    print(f"saved: {t1_raw}")
    print(f"saved: {t1_summary}")
    print(f"saved: {t2_raw}")
    print(f"saved: {t2_summary}")
    print(f"saved: {t1_example_plot}")
    print(f"saved: {t1_residual_plot}")
    print(f"saved: {t1_binned_plot}")
    print(f"saved: {t1_compare_plot}")
    print(f"saved: {t2_binned_plot}")
    print(f"saved: {t2_length_plot}")
    print(f"saved: {t2_scale_plot}")


if __name__ == "__main__":
    main()
