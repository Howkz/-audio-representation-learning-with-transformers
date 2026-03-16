from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_run_partial(
    partial_path: Path,
    run_id: str,
    payload: Dict[str, Any],
    model_name: str,
    architecture: str,
    adaptation: str,
) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    data = _safe_read_json(partial_path)
    if not data:
        data = {
            "model_name": model_name,
            "architecture": architecture,
            "adaptation_technique": adaptation,
            "runs": {},
        }
    data["runs"][str(run_id)] = payload
    with open(partial_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def aggregate_partial_to_final(partial_path: Path, final_path: Path) -> Dict[str, Any]:
    data = _safe_read_json(partial_path)
    runs = data.get("runs", {})
    aggregated = {
        "model_name": data.get("model_name"),
        "architecture": data.get("architecture"),
        "adaptation_technique": data.get("adaptation_technique"),
        "runs": runs,
        "metrics": {},
    }
    if not runs:
        with open(final_path, "w", encoding="utf-8") as handle:
            json.dump(aggregated, handle, indent=2)
        return aggregated

    sample_run = next(iter(runs.values()))
    metric_keys = [k for k, v in sample_run.items() if isinstance(v, (int, float))]
    for key in metric_keys:
        values = [float(run[key]) for run in runs.values() if isinstance(run.get(key), (int, float))]
        if values:
            aggregated["metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_path, "w", encoding="utf-8") as handle:
        json.dump(aggregated, handle, indent=2)
    return aggregated

