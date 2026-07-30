"""Experiment orchestration helpers."""

import json
import subprocess
import sys
from pathlib import Path

from src.conformal import conformal_order_statistic
from src.data import sample_source_xy, sample_target_xy
from src.metrics import coverage, mean_interval_length, mean_interval_score
from src.models import PolyModel



def load_config(config_path:Path) ->dict:
    """Load experiment settings from config.json."""
    with open(config_path,"r",encoding="utf-8") as f:
        return json.load(f)
    
    
def run_pytest_and_write_log(root: Path, output_dir: Path, seed: int) -> Path:
    """Run pytest and save stdout/stderr to unit_test_log.txt."""
    output_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        text=True,
        capture_output=True,
    )

    log_path = output_dir / "unit_test_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("T0 unit test log\n")
        f.write("================\n\n")
        f.write(f"seed: {seed}\n")
        f.write(f"command: {sys.executable} -m pytest -q\n")
        f.write(f"returncode: {result.returncode}\n\n")

        f.write("[stdout]\n")
        f.write(result.stdout)
        f.write("\n[stderr]\n")
        f.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"pytest failed; see {log_path}")

    return log_path
    
    
    
def run_split_cp_once():
    """Run one Split CP experiment and return its metrics."""
    