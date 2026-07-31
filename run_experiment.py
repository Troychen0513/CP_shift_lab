"""Main entry point for the CP shift lab."""

from pathlib import Path

from src.io_utils import load_config, save_rows
from src.plots import make_data_sanity_plot
from src.t0 import run_pytest_and_write_log
from src.t1 import run_t1_repeats, summarize_t1


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    """Generate T0 outputs and T1 repeated-experiment tables."""
    config = load_config(CONFIG_PATH)
    seed = int(config["seed"])

    t0_plot = make_data_sanity_plot(OUTPUT_DIR, seed)
    t0_log = run_pytest_and_write_log(ROOT, OUTPUT_DIR, seed)

    t1_rows = run_t1_repeats(config, n_repeats=200)
    t1_raw = save_rows(t1_rows, OUTPUT_DIR, "t1_raw_metrics.csv")

    t1_summary_rows = summarize_t1(t1_rows)
    t1_summary = save_rows(t1_summary_rows, OUTPUT_DIR, "t1_summary.csv")

    print(f"saved: {t0_plot}")
    print(f"saved: {t0_log}")
    print(f"saved: {t1_raw}")
    print(f"saved: {t1_summary}")


if __name__ == "__main__":
    main()
