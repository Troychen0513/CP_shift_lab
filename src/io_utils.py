"""Small helpers for reading config files and saving tables."""

import csv
import json
from pathlib import Path


def load_config(config_path: Path) -> dict:
    """Read experiment settings from config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_single_metrics(metrics: dict, output_dir: Path, filename: str) -> Path:
    """Save one row of experiment metrics to a CSV file."""
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    return output_path


def save_rows(rows: list[dict], output_dir: Path, filename: str) -> Path:
    """Save many rows of experiment metrics to a CSV file."""
    if not rows:
        raise ValueError("rows must not be empty")

    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path
