from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _fmt(metric: Dict[str, float]) -> str:
    if not metric:
        return "n/a"
    return f"{metric.get('mean', 0.0):.4f} +- {metric.get('std', 0.0):.4f}"


def write_final_table(final_json_path: Path, table_md_path: Path, title: str) -> None:
    with open(final_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    metrics = data.get("metrics", {})
    rows = [
        ("WER", _fmt(metrics.get("wer", {}))),
        ("Accuracy", _fmt(metrics.get("accuracy", {}))),
        ("Inference runtime (sec)", _fmt(metrics.get("inference_runtime_sec", {}))),
        ("Samples/sec", _fmt(metrics.get("inference_samples_per_sec", {}))),
        ("Peak GPU mem (MB)", _fmt(metrics.get("inference_peak_gpu_mem_mb", {}))),
    ]
    table_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| Metric | Mean +- Std |\n")
        handle.write("|---|---|\n")
        for key, value in rows:
            handle.write(f"| {key} | {value} |\n")


def write_dataset_breakdown_table(dataset_final_json_path: Path, table_md_path: Path, title: str) -> None:
    with open(dataset_final_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    metrics = data.get("metrics", {})
    table_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| Dataset | WER mean +- std | Accuracy mean +- std |\n")
        handle.write("|---|---|---|\n")
        for dataset_name, payload in metrics.items():
            handle.write(
                f"| {dataset_name} | {_fmt(payload.get('wer', {}))} | {_fmt(payload.get('accuracy', {}))} |\n"
            )

