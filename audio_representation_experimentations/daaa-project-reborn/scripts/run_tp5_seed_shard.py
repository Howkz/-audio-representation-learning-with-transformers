from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TP 5-seed suite on a seed shard.")
    parser.add_argument("--suite-config", type=str, default="configs/final_tp/suite_tp_5seeds.yaml")
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--from-id", type=str, default=None)
    parser.add_argument("--to-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disk-guard-gb", type=float, default=1.5)
    parser.add_argument("--storage-interval-sec", type=int, default=30)
    parser.add_argument("--set", dest="cli_overrides", action="append", default=[])
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
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return payload


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    suite_path = _resolve_input_path(args.suite_config)
    suite_cfg = _load_yaml(suite_path)

    experiments = suite_cfg.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("Suite config must contain an 'experiments' list.")

    shard_seeds = [int(seed) for seed in args.seeds]
    for experiment in experiments:
        if isinstance(experiment, dict):
            experiment["seeds"] = list(shard_seeds)

    suite_meta = suite_cfg.setdefault("suite", {})
    if isinstance(suite_meta, dict):
        base_name = str(suite_meta.get("name", "tp_5seeds"))
        suite_meta["name"] = f"{base_name}_{args.label}"

    runtime_suite_path = PROJECT_ROOT / "results" / "suite" / "runtime_configs" / f"suite_tp_5seeds_{args.label}.yaml"
    _save_yaml(runtime_suite_path, suite_cfg)

    command: List[str] = [sys.executable, "scripts/run_experiment_suite.py", "--suite-config", str(runtime_suite_path)]
    if args.from_id:
        command.extend(["--from-id", args.from_id])
    if args.to_id:
        command.extend(["--to-id", args.to_id])
    if args.dry_run:
        command.append("--dry-run")
    if args.resume:
        command.append("--resume")
    if args.verbose:
        command.append("--verbose")
    command.extend(["--disk-guard-gb", str(args.disk_guard_gb)])
    command.extend(["--storage-interval-sec", str(args.storage_interval_sec)])
    for item in args.cli_overrides:
        command.extend(["--set", item])

    print(f"[TP5-SHARD] label={args.label} seeds={shard_seeds}")
    print(f"[TP5-SHARD] runtime suite config: {runtime_suite_path}")
    print(f"[TP5-SHARD] command: {' '.join(command)}")
    raise SystemExit(subprocess.run(command, cwd=str(PROJECT_ROOT), check=False).returncode)


if __name__ == "__main__":
    main()
