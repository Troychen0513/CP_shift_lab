"""Main entry point for the CP shift lab."""

from pathlib import Path

from src.plots import make_data_sanity_plot
from src.experiment import load_config, run_pytest_and_write_log

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT/ "config.json"
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    """Generate required T0 outputs."""
    config = load_config(CONFIG_PATH)
    seed = int(config["seed"])
    plot_path = make_data_sanity_plot(OUTPUT_DIR, seed)
    log_path = run_pytest_and_write_log(ROOT, OUTPUT_DIR, seed)
    
    print(f"saved: {plot_path}")
    print(f"saved: {log_path}")



if __name__ == "__main__":
    main()