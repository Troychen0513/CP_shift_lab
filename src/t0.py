"""T0 workflow: data sanity output and unit-test log."""

import subprocess
import sys
from pathlib import Path


def run_pytest_and_write_log(root: Path, output_dir: Path, seed: int) -> Path:
    """Run pytest and save the test output as T0 evidence."""
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
