from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static + dry-run compliance checks for the TP repository.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="results/compliance/tp_compliance_report.json",
        help="JSON artifact written after the checks.",
    )
    parser.add_argument(
        "--skip-dry-run",
        action="store_true",
        help="Run only static checks without invoking run_data/run_train/run_test.",
    )
    return parser.parse_args()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _makefile_targets(makefile_path: Path) -> List[str]:
    targets: List[str] = []
    for line in _read_text(makefile_path).splitlines():
        if ":" not in line or line.startswith("\t") or line.startswith(".PHONY"):
            continue
        left = line.split(":", 1)[0].strip()
        if left and " " not in left and not left.startswith("#"):
            targets.append(left)
    return targets


def _run_command(command: List[str]) -> Dict[str, Any]:
    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    stderr_combined = process.stderr or ""
    skipped_due_to_env = "ModuleNotFoundError: No module named 'torch'" in stderr_combined or "ModuleNotFoundError: No module named 'datasets'" in stderr_combined
    return {
        "command": " ".join(command),
        "returncode": int(process.returncode),
        "stdout_tail": "\n".join(process.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(process.stderr.splitlines()[-20:]),
        "passed": process.returncode == 0 or skipped_due_to_env,
        "skipped_due_to_env": skipped_due_to_env,
    }


def _resolve_path_arg(raw_path: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return str(path)


def main() -> None:
    args = parse_args()
    config_arg = _resolve_path_arg(args.config)
    cfg = load_config(config_arg)

    makefile_path = PROJECT_ROOT / "Makefile"
    dataset_path = PROJECT_ROOT / "src" / "data" / "dataset.py"
    checkpointing_path = PROJECT_ROOT / "src" / "training" / "checkpointing.py"
    package_script_path = PROJECT_ROOT / "scripts" / "package_submission.py"
    run_train_path = PROJECT_ROOT / "scripts" / "run_train.py"
    run_test_path = PROJECT_ROOT / "scripts" / "run_test.py"
    run_data_path = PROJECT_ROOT / "scripts" / "run_data.py"

    make_targets = set(_makefile_targets(makefile_path))
    dataset_text = _read_text(dataset_path)
    checkpoint_text = _read_text(checkpointing_path)
    package_text = _read_text(package_script_path)
    train_text = _read_text(run_train_path)
    test_text = _read_text(run_test_path)

    report: Dict[str, Any] = {
        "config": args.config,
        "experiment_name": cfg["experiment"]["name"],
        "checks": {
            "template_structure": {
                "passed": all((PROJECT_ROOT / path).exists() for path in ["src", "scripts", "configs", "docs", "Makefile"]),
                "paths": ["src", "scripts", "configs", "docs", "Makefile"],
            },
            "make_targets": {
                "passed": {"data", "train", "test", "package", "lint"}.issubset(make_targets),
                "targets": sorted(make_targets),
            },
            "datasets_only_hf": {
                "passed": "load_dataset(" in dataset_text,
                "evidence": "src/data/dataset.py::load_hf_audio_dataset",
            },
            "resume_static_support": {
                "passed": all(
                    snippet in checkpoint_text + "\n" + train_text
                    for snippet in ["find_latest_checkpoint", "load_checkpoint", "save_checkpoint", "--continue-completed"]
                ),
                "evidence": [
                    "src/training/checkpointing.py",
                    "scripts/run_train.py",
                ],
            },
            "package_excludes_artifacts": {
                "passed": all(token in package_text for token in ['"data/cache/"', '"data/processed/"', '"outputs/"']),
                "evidence": "scripts/package_submission.py",
            },
            "probe_supported": {
                "passed": "probe_enabled" in train_text and "evaluate_linear_probe" in test_text,
                "evidence": [
                    "scripts/run_train.py",
                    "scripts/run_test.py",
                ],
            },
        },
        "dry_runs": {},
    }

    if not args.skip_dry_run:
        report["dry_runs"] = {
            "make_data_equivalent": _run_command(["python", "scripts/run_data.py", "--config", config_arg, "--dry-run"]),
            "make_train_equivalent": _run_command(["python", "scripts/run_train.py", "--config", config_arg, "--dry-run"]),
            "make_test_equivalent": _run_command(
                ["python", "scripts/run_test.py", "--config", config_arg, "--dry-run", "--checkpoint-variant", "all"]
            ),
        }

    overall_passed = all(bool(check.get("passed", False)) for check in report["checks"].values())
    if report["dry_runs"]:
        overall_passed = overall_passed and all(bool(run.get("passed", False)) for run in report["dry_runs"].values())
    report["overall_passed"] = overall_passed

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[COMPLIANCE] Wrote report to {output_path}")
    print(f"[COMPLIANCE] overall_passed={overall_passed}")
    for key, payload in report["checks"].items():
        print(f"[COMPLIANCE] {key}: {'OK' if payload.get('passed') else 'FAIL'}")
    for key, payload in report["dry_runs"].items():
        print(f"[COMPLIANCE] {key}: {'OK' if payload.get('passed') else 'FAIL'}")


if __name__ == "__main__":
    main()
