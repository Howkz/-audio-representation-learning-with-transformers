from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a mapping at top level.")
    return cfg


def ensure_project_dirs(cfg: Dict[str, Any]) -> None:
    experiment = cfg["experiment"]
    paths = [
        Path(experiment["output_dir"]),
        Path(experiment["results_dir"]),
        Path(experiment["cache_dir"]),
        Path(experiment["processed_dir"]),
        Path(experiment["results_dir"]) / "benchmark_results",
        Path(experiment["results_dir"]) / "tables",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

