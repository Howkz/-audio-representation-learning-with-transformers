from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test checkpoint resume on a tiny finetune run.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/compliance/resume_test_report.json")
    return parser.parse_args()


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()


def _run(command: list[str]) -> Dict[str, Any]:
    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": int(process.returncode),
        "stdout_tail": "\n".join(process.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(process.stderr.splitlines()[-20:]),
        "passed": process.returncode == 0,
    }


def _prepare_resume_cfg(base_cfg: Dict[str, Any], *, exp_id: str, max_steps: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["id"] = exp_id
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_{exp_id}"
    cfg["experiment"]["output_dir"] = "outputs/compliance"
    cfg["experiment"]["results_dir"] = f"results/compliance/{exp_id}"
    cfg["experiment"]["processed_dir"] = f"data/processed/compliance/{exp_id}"
    cfg["experiment"]["checkpoint_every_steps"] = 1
    cfg["experiment"]["keep_last_checkpoints"] = 2
    cfg["experiment"]["seeds"] = [42]
    cfg["pretrain"]["mode"] = "none"
    cfg["training"]["pretrain"]["enabled"] = False
    cfg["training"]["finetune"]["epochs"] = 2
    cfg["training"]["finetune"]["batch_size"] = 1
    cfg["training"]["finetune"]["grad_accum_steps"] = 1
    cfg["training"]["finetune"]["max_steps"] = max_steps
    cfg["training"]["finetune"]["early_stopping_min_epochs"] = 99
    cfg["training"]["finetune"]["early_stopping_patience"] = 99
    cfg["training"]["finetune"]["warmup_steps"] = 0
    cfg["training"]["amp"] = False
    cfg["data"]["num_workers"] = 0
    cfg["data"]["streaming"] = False
    cfg["probe"] = {"enabled": False}
    cfg["diagnostics"] = {"enabled": False, "forensics_enabled": False}
    cfg["datasets"]["asr_train"]["max_samples"] = min(32, int(cfg["datasets"]["asr_train"].get("max_samples") or 32))
    cfg["datasets"]["asr_valid"]["max_samples"] = min(16, int(cfg["datasets"]["asr_valid"].get("max_samples") or 16))
    cfg["datasets"]["asr_tests"] = []
    return cfg


def main() -> None:
    args = parse_args()
    config_path = _resolve_input_path(args.config)
    base_cfg = load_config(str(config_path))
    exp_id = "TP_RESUME_SMOKE"
    runtime_dir = PROJECT_ROOT / "results" / "compliance" / "runtime_configs"
    stage1_cfg_path = runtime_dir / "resume_stage1.yaml"
    stage2_cfg_path = runtime_dir / "resume_stage2.yaml"

    stage1_cfg = _prepare_resume_cfg(base_cfg, exp_id=exp_id, max_steps=2)
    stage2_cfg = _prepare_resume_cfg(base_cfg, exp_id=exp_id, max_steps=4)
    _save_yaml(stage1_cfg_path, stage1_cfg)
    _save_yaml(stage2_cfg_path, stage2_cfg)

    stage1 = _run(["python", "scripts/run_train.py", "--config", str(stage1_cfg_path), "--continue-completed"])

    finetune_root = PROJECT_ROOT / "outputs" / "compliance" / "checkpoints" / exp_id / "finetune" / "seed_42"
    run_completed = finetune_root / "run_completed.txt"
    final_model = finetune_root / "ctc_final.pt"
    if run_completed.exists():
        run_completed.unlink()
    if final_model.exists():
        final_model.unlink()

    stage2 = _run(["python", "scripts/run_train.py", "--config", str(stage2_cfg_path), "--continue-completed"])

    final_json = PROJECT_ROOT / "results" / "compliance" / exp_id / "benchmark_results" / f"train_audio_transformer_{exp_id}_final.json"
    final_payload = {}
    if final_json.exists():
        with open(final_json, "r", encoding="utf-8") as handle:
            final_payload = json.load(handle)

    micro_steps = (
        final_payload.get("metrics", {})
        .get("train_micro_steps_completed", {})
        .get("mean")
    )
    report = {
        "stage1": stage1,
        "stage2": stage2,
        "final_json": str(final_json),
        "train_micro_steps_completed_mean": micro_steps,
        "passed": bool(stage1.get("passed")) and bool(stage2.get("passed")) and isinstance(micro_steps, (int, float)) and float(micro_steps) >= 4.0,
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[RESUME] Wrote report to {output_path}")
    print(f"[RESUME] passed={report['passed']}")


if __name__ == "__main__":
    main()
