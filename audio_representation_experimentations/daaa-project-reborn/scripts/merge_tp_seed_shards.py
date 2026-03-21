from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge TP 5-seed shard results from two VDI runs.")
    parser.add_argument("--suite-config", type=str, default="configs/final_tp/suite_tp_5seeds.yaml")
    parser.add_argument("--primary-root", type=str, required=True, help="Root containing experiments/<ID>/...")
    parser.add_argument("--peer-root", type=str, required=True, help="Second shard root containing experiments/<ID>/...")
    parser.add_argument("--output-root", type=str, default="results/experiments")
    parser.add_argument("--regenerate-tables", action="store_true")
    return parser.parse_args()


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _merge_run_payloads(payloads: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    payloads = [payload for payload in payloads if payload]
    if not payloads:
        return {}
    base = dict(payloads[0])
    merged_runs: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        for run_id, run_payload in payload.get("runs", {}).items():
            merged_runs[str(run_id)] = run_payload
    base["runs"] = merged_runs
    metric_keys = sorted(
        {
            key
            for run in merged_runs.values()
            for key, value in run.items()
            if isinstance(value, (int, float))
        }
    )
    metrics: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = [float(run[key]) for run in merged_runs.values() if isinstance(run.get(key), (int, float))]
        if values:
            metrics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    base["metrics"] = metrics
    return base


def _combine_mean_std(existing: Optional[Tuple[int, float, float]], n: int, mean: float, std: float) -> Tuple[int, float, float]:
    sum_ = float(n) * float(mean)
    sumsq = float(n) * (float(std) ** 2 + float(mean) ** 2)
    if existing is None:
        return n, sum_, sumsq
    prev_n, prev_sum, prev_sumsq = existing
    return prev_n + n, prev_sum + sum_, prev_sumsq + sumsq


def _merge_dataset_breakdown(payloads_with_counts: Iterable[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
    payloads_with_counts = [(payload, int(count)) for payload, count in payloads_with_counts if payload and int(count) > 0]
    if not payloads_with_counts:
        return {}
    accumulator: Dict[str, Dict[str, Tuple[int, float, float]]] = {}
    for payload, count in payloads_with_counts:
        for dataset_name, metric_block in payload.get("metrics", {}).items():
            dataset_acc = accumulator.setdefault(str(dataset_name), {})
            if not isinstance(metric_block, dict):
                continue
            for metric_name, metric_payload in metric_block.items():
                if not isinstance(metric_payload, dict):
                    continue
                mean = metric_payload.get("mean")
                std = metric_payload.get("std")
                if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
                    continue
                dataset_acc[metric_name] = _combine_mean_std(dataset_acc.get(metric_name), count, float(mean), float(std))

    merged: Dict[str, Any] = {"metrics": {}}
    for dataset_name, metrics in accumulator.items():
        merged["metrics"][dataset_name] = {}
        for metric_name, (count, sum_, sumsq) in metrics.items():
            mean = sum_ / max(1, count)
            variance = max(0.0, (sumsq / max(1, count)) - (mean ** 2))
            merged["metrics"][dataset_name][metric_name] = {"mean": float(mean), "std": float(np.sqrt(variance))}
    return merged


def _experiment_dir(root: Path, exp_id: str) -> Path:
    return root / exp_id / "benchmark_results"


def _merge_one_artifact(
    exp_id: str,
    *,
    primary_root: Path,
    peer_root: Path,
    output_root: Path,
    stem: str,
    has_dataset_breakdown: bool,
) -> None:
    final_name = f"{stem}_{exp_id}_final.json"
    primary_final = _read_json(_experiment_dir(primary_root, exp_id) / final_name)
    peer_final = _read_json(_experiment_dir(peer_root, exp_id) / final_name)
    merged_final = _merge_run_payloads([primary_final, peer_final])
    if merged_final:
        out_dir = _experiment_dir(output_root, exp_id)
        _write_json(out_dir / final_name, merged_final)

    if has_dataset_breakdown:
        dataset_name = final_name.replace(f"{stem}_", f"{stem}_by_dataset_")
        primary_dataset = _read_json(_experiment_dir(primary_root, exp_id) / dataset_name)
        peer_dataset = _read_json(_experiment_dir(peer_root, exp_id) / dataset_name)
        n_primary = len(primary_final.get("runs", {})) if primary_final else 0
        n_peer = len(peer_final.get("runs", {})) if peer_final else 0
        merged_dataset = _merge_dataset_breakdown(
            [(primary_dataset, n_primary), (peer_dataset, n_peer)]
        )
        if merged_dataset:
            out_dir = _experiment_dir(output_root, exp_id)
            _write_json(out_dir / dataset_name, merged_dataset)


def main() -> None:
    args = parse_args()
    suite_cfg = _load_yaml(_resolve_input_path(args.suite_config))
    primary_root = _resolve_input_path(args.primary_root)
    peer_root = _resolve_input_path(args.peer_root)
    output_root = _resolve_input_path(args.output_root)

    experiments = [exp for exp in suite_cfg.get("experiments", []) if bool(exp.get("enabled", True))]
    for exp in experiments:
        exp_id = str(exp["id"])
        _merge_one_artifact(
            exp_id,
            primary_root=primary_root,
            peer_root=peer_root,
            output_root=output_root,
            stem="train_audio_transformer",
            has_dataset_breakdown=False,
        )
        _merge_one_artifact(
            exp_id,
            primary_root=primary_root,
            peer_root=peer_root,
            output_root=output_root,
            stem="asr_benchmark",
            has_dataset_breakdown=True,
        )
        _merge_one_artifact(
            exp_id,
            primary_root=primary_root,
            peer_root=peer_root,
            output_root=output_root,
            stem="probe_benchmark",
            has_dataset_breakdown=True,
        )

    print(f"[MERGE] Merged shard results into {output_root}")
    if args.regenerate_tables:
        import subprocess
        import sys

        command = [
            sys.executable,
            "scripts/generate_tp_5seeds_tables.py",
            "--suite-config",
            str(_resolve_input_path(args.suite_config)),
        ]
        raise SystemExit(subprocess.run(command, cwd=str(PROJECT_ROOT), check=False).returncode)


if __name__ == "__main__":
    main()
