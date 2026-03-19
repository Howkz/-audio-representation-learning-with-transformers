from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full E00->E11 experiment suite.")
    parser.add_argument("--suite-config", type=str, default="configs/suite_e00_e11.yaml")
    parser.add_argument("--from-id", type=str, default=None)
    parser.add_argument("--to-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disk-guard-gb", type=float, default=1.5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--storage-interval-sec", type=int, default=30)
    parser.add_argument(
        "--set",
        dest="cli_overrides",
        action="append",
        default=[],
        help="Apply deterministic config override using dot path, e.g. --set training.finetune.max_steps=120",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML at {path}: expected mapping at root.")
    return data


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _enabled_experiments(suite_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    experiments = suite_cfg.get("experiments", [])
    return [exp for exp in experiments if bool(exp.get("enabled", True))]


def _slice_experiments(experiments: List[Dict[str, Any]], from_id: Optional[str], to_id: Optional[str]) -> List[Dict[str, Any]]:
    if not experiments:
        return []
    ids = [str(e["id"]) for e in experiments]
    start = 0
    end = len(experiments) - 1
    if from_id:
        if from_id not in ids:
            raise ValueError(f"--from-id={from_id} not found in enabled experiments.")
        start = ids.index(from_id)
    if to_id:
        if to_id not in ids:
            raise ValueError(f"--to-id={to_id} not found in enabled experiments.")
        end = ids.index(to_id)
    if start > end:
        raise ValueError("Invalid range: from-id is after to-id.")
    return experiments[start : end + 1]


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024 ** 3)


def _format_hms(total_seconds: float) -> str:
    seconds = max(0, int(round(float(total_seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _storage_snapshot() -> Dict[str, float]:
    root = Path(".").resolve()
    usage = shutil.disk_usage(root)
    hf_home = Path(os.environ.get("HF_HOME", "")).expanduser()
    tmp_dir = Path(os.environ.get("TMPDIR", "")).expanduser()
    return {
        "free_gb": _bytes_to_gb(usage.free),
        "cache_gb": _bytes_to_gb(_dir_size_bytes(root / "data" / "cache")),
        "checkpoints_gb": _bytes_to_gb(_dir_size_bytes(root / "outputs" / "checkpoints")),
        "results_gb": _bytes_to_gb(_dir_size_bytes(root / "results" / "experiments")),
        "hf_cache_gb": _bytes_to_gb(_dir_size_bytes(hf_home)) if str(hf_home) else 0.0,
        "tmp_gb": _bytes_to_gb(_dir_size_bytes(tmp_dir)) if str(tmp_dir) else 0.0,
    }


def _print_storage(tag: str, elapsed_sec: Optional[float] = None) -> None:
    snap = _storage_snapshot()
    elapsed_msg = f" elapsed={_format_hms(elapsed_sec)}" if elapsed_sec is not None else ""
    print(
        f"[STORAGE] {tag}{elapsed_msg} | free={snap['free_gb']:.2f}GB "
        f"cache={snap['cache_gb']:.2f}GB checkpoints={snap['checkpoints_gb']:.2f}GB "
        f"results={snap['results_gb']:.2f}GB hf_cache={snap['hf_cache_gb']:.2f}GB "
        f"tmp={snap['tmp_gb']:.2f}GB"
    )


def _guard_disk(min_free_gb: float) -> None:
    free = _storage_snapshot()["free_gb"]
    if free < min_free_gb:
        raise RuntimeError(
            f"Disk guard triggered: free space {free:.2f}GB < required {min_free_gb:.2f}GB."
        )


def _monitor_storage(stop_event: threading.Event, interval_sec: int, started_at: float, phase_tag: str) -> None:
    while not stop_event.wait(timeout=max(5, interval_sec)):
        _print_storage(f"runtime:{phase_tag}", elapsed_sec=time.time() - started_at)


def _run_command(
    command: List[str],
    phase_tag: str,
    dry_run: bool,
    verbose: bool,
    storage_interval_sec: int,
    min_free_gb: float,
) -> None:
    _guard_disk(min_free_gb)
    print(f"[SUITE] Running {phase_tag}: {' '.join(command)}")
    _print_storage(f"before {phase_tag}")
    if dry_run:
        print(f"[SUITE] Dry-run: command skipped for {phase_tag}.")
        return

    started_at = time.time()
    stop_event = threading.Event()
    monitor_thread = None
    if verbose:
        monitor_thread = threading.Thread(
            target=_monitor_storage,
            args=(stop_event, storage_interval_sec, started_at, phase_tag),
            daemon=True,
        )
        monitor_thread.start()

    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    root_path = str(PROJECT_ROOT)
    if existing_pythonpath:
        parts = existing_pythonpath.split(os.pathsep)
        if root_path not in parts:
            child_env["PYTHONPATH"] = root_path + os.pathsep + existing_pythonpath
    else:
        child_env["PYTHONPATH"] = root_path

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=child_env,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{phase_tag}] {line.rstrip()}")
    return_code = process.wait()

    stop_event.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=1)

    if return_code != 0:
        raise RuntimeError(f"Command failed in phase {phase_tag} with exit code {return_code}.")
    _print_storage(f"after {phase_tag}", elapsed_sec=time.time() - started_at)


def _runtime_config_path(exp_id: str) -> Path:
    return Path("results") / "suite" / "runtime_configs" / f"{exp_id}.yaml"


def _parse_override_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _build_cli_override_mapping(items: List[str]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Expected dot.path=value.")
        key_path, raw_value = item.split("=", 1)
        keys = [k for k in key_path.strip().split(".") if k]
        if not keys:
            raise ValueError(f"Invalid --set key path in '{item}'.")
        cursor = mapping
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = _parse_override_value(raw_value.strip())
    return mapping


def _normalize_experiment_name(name: str, exp_id: str) -> str:
    parts = str(name).split("_")
    while parts and re.fullmatch(r"(E|SEL)\d{2}", parts[-1]):
        parts.pop()
    base = "_".join(parts) if parts else str(name)
    return f"{base}_{exp_id}"


def _apply_final_full_dataset(cfg: Dict[str, Any]) -> None:
    datasets = cfg.get("datasets", {})
    if isinstance(datasets.get("pretrain"), dict):
        datasets["pretrain"]["max_samples"] = None
    if isinstance(datasets.get("asr_train"), dict):
        datasets["asr_train"]["max_samples"] = None
    if isinstance(datasets.get("asr_valid"), dict):
        datasets["asr_valid"]["max_samples"] = None
    asr_tests = datasets.get("asr_tests", [])
    if isinstance(asr_tests, list):
        for spec in asr_tests:
            if isinstance(spec, dict):
                spec["max_samples"] = None


def _resolved_config(
    suite_cfg: Dict[str, Any],
    experiment: Dict[str, Any],
    source_cfg: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if source_cfg is not None:
        cfg = copy.deepcopy(source_cfg)
    else:
        base_config_path = Path(str(experiment["base_config"]))
        cfg = _load_yaml(base_config_path)
    cfg = _deep_merge(cfg, experiment.get("overrides", {}))
    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)
    if "seeds" in experiment:
        cfg["experiment"]["seeds"] = list(experiment["seeds"])

    exp_id = str(experiment["id"])
    cfg["experiment"]["id"] = exp_id
    cfg["experiment"]["name"] = _normalize_experiment_name(str(cfg["experiment"]["name"]), exp_id)
    cfg["experiment"]["results_dir"] = f"results/experiments/{exp_id}"
    cfg["experiment"]["output_dir"] = "outputs"
    cfg["experiment"]["cache_dir"] = "data/cache"
    cfg["experiment"]["processed_dir"] = "data/processed"
    if bool(experiment.get("final_full_dataset", False)):
        _apply_final_full_dataset(cfg)
    return cfg


def _test_final_path(exp_id: str) -> Path:
    return Path("results") / "experiments" / exp_id / "benchmark_results" / f"asr_benchmark_{exp_id}_final.json"


def _train_final_path(exp_id: str) -> Path:
    return Path("results") / "experiments" / exp_id / "benchmark_results" / f"train_audio_transformer_{exp_id}_final.json"


def _done_marker_path(exp_id: str) -> Path:
    return Path("results") / "experiments" / exp_id / "suite_done.json"


def _parse_metric_mean(path: Path, metric_name: str) -> Optional[float]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    metric = data.get("metrics", {}).get(metric_name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _screening_rows(suite_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp in _enabled_experiments(suite_cfg):
        if str(exp.get("phase")) != "screening":
            continue
        exp_id = str(exp["id"])
        final_path = _test_final_path(exp_id)
        wer = _parse_metric_mean(final_path, "wer")
        runtime = _parse_metric_mean(final_path, "inference_runtime_sec")
        mem = _parse_metric_mean(final_path, "inference_peak_gpu_mem_mb")
        if wer is None or runtime is None or mem is None:
            continue
        rows.append(
            {
                "id": exp_id,
                "title": str(exp.get("title", exp_id)),
                "wer": wer,
                "inference_runtime_sec": runtime,
                "inference_peak_gpu_mem_mb": mem,
            }
        )
    rows.sort(key=lambda r: (r["wer"], r["inference_runtime_sec"], r["inference_peak_gpu_mem_mb"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _selection_config(suite_cfg: Dict[str, Any]) -> Dict[str, Any]:
    selection = suite_cfg.get("suite", {}).get("selection", {})
    if not isinstance(selection, dict):
        return {}
    return selection


def _selection_manifest_path() -> Path:
    return Path("results") / "suite" / "selection_manifest.json"


def _selection_id(rank: int) -> str:
    return f"SEL{rank:02d}"


def _build_selection_manifest(
    suite_cfg: Dict[str, Any],
    allow_fallback_order: bool,
) -> Dict[str, Any]:
    selection_cfg = _selection_config(suite_cfg)
    top_k = int(selection_cfg.get("top_k_from_screening", 5))
    selection_seeds = list(selection_cfg.get("seeds", [42, 123]))
    screening_rows = _screening_rows(suite_cfg)

    if len(screening_rows) < top_k:
        if not allow_fallback_order:
            raise RuntimeError(
                f"Selection phase requires top-{top_k} screening runs, found {len(screening_rows)}."
            )
        screening_ids = [
            str(e["id"])
            for e in _enabled_experiments(suite_cfg)
            if str(e.get("phase")) == "screening"
        ]
        if len(screening_ids) < top_k:
            raise RuntimeError(
                f"Cannot build fallback selection manifest: need {top_k} screening experiments, found {len(screening_ids)}."
            )
        picked = screening_ids[:top_k]
        return {
            "top_k": top_k,
            "seeds": selection_seeds,
            "runs": [
                {
                    "selection_id": _selection_id(rank),
                    "source_screening_id": exp_id,
                    "screening_rank": rank,
                }
                for rank, exp_id in enumerate(picked, start=1)
            ],
        }

    picked_rows = screening_rows[:top_k]
    return {
        "top_k": top_k,
        "seeds": selection_seeds,
        "runs": [
            {
                "selection_id": _selection_id(rank),
                "source_screening_id": str(row["id"]),
                "screening_rank": rank,
            }
            for rank, row in enumerate(picked_rows, start=1)
        ],
    }


def _selection_rows_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in manifest.get("runs", []):
        selection_id = str(run["selection_id"])
        final_path = _test_final_path(selection_id)
        wer = _parse_metric_mean(final_path, "wer")
        runtime = _parse_metric_mean(final_path, "inference_runtime_sec")
        mem = _parse_metric_mean(final_path, "inference_peak_gpu_mem_mb")
        if wer is None or runtime is None or mem is None:
            continue
        rows.append(
            {
                "id": selection_id,
                "source_screening_id": str(run["source_screening_id"]),
                "wer": wer,
                "inference_runtime_sec": runtime,
                "inference_peak_gpu_mem_mb": mem,
            }
        )
    rows.sort(key=lambda r: (r["wer"], r["inference_runtime_sec"], r["inference_peak_gpu_mem_mb"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _cleanup_checkpoints_for_experiment(exp_id: str, dry_run: bool) -> None:
    path = Path("outputs") / "checkpoints" / exp_id
    if not path.exists():
        return
    print(f"[SUITE] Cleanup checkpoints for {exp_id}: {path}")
    if not dry_run:
        shutil.rmtree(path)


def _archive_experiment_artifacts(exp_id: str, dry_run: bool) -> None:
    src_root = Path("results") / "experiments" / exp_id
    archive_root = Path("results") / "suite" / "archive" / exp_id
    files_to_copy = [
        src_root / "benchmark_results" / f"train_audio_transformer_{exp_id}_final.json",
        src_root / "benchmark_results" / f"asr_benchmark_{exp_id}_final.json",
        src_root / "benchmark_results" / f"asr_benchmark_by_dataset_{exp_id}_final.json",
        src_root / "tables" / f"asr_overall_table_{exp_id}.md",
        src_root / "tables" / f"asr_dataset_breakdown_{exp_id}.md",
    ]
    existing_files = [path for path in files_to_copy if path.exists()]
    if not existing_files:
        print(f"[SUITE] Archive {exp_id}: no final artifacts found yet.")
        return

    print(f"[SUITE] Archive {exp_id}: {len(existing_files)} files -> {archive_root}")
    if dry_run:
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    for file_path in existing_files:
        shutil.copy2(file_path, archive_root / file_path.name)


def _resolve_ranked_experiment(
    suite_cfg: Dict[str, Any],
    experiment: Dict[str, Any],
    resolved_cfg_cache: Dict[str, Dict[str, Any]],
    ranking_rows: List[Dict[str, Any]],
    rank_key: str,
    fallback_ids: List[str],
    label: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
    allow_fallback_order: bool = False,
) -> Tuple[Dict[str, Any], str]:
    rank = int(experiment[rank_key])
    if len(ranking_rows) < rank:
        if not allow_fallback_order:
            raise RuntimeError(
                f"Cannot resolve {experiment['id']}: {label} ranking has only {len(ranking_rows)} completed runs."
            )
        if len(fallback_ids) < rank:
            raise RuntimeError(
                f"Cannot resolve {experiment['id']}: not enough fallback ids for rank {rank} ({label})."
            )
        source_id = str(fallback_ids[rank - 1])
    else:
        source_id = str(ranking_rows[rank - 1]["id"])

    if source_id in resolved_cfg_cache:
        source_cfg = resolved_cfg_cache[source_id]
    else:
        source_cfg_path = _runtime_config_path(source_id)
        if not source_cfg_path.exists():
            raise RuntimeError(f"Missing runtime config for source {source_id}: {source_cfg_path}")
        source_cfg = _load_yaml(source_cfg_path)
    cfg = _resolved_config(
        suite_cfg,
        experiment,
        source_cfg=source_cfg,
        cli_overrides=cli_overrides,
    )
    return cfg, source_id


def _run_data_once(
    python_exec: str,
    runtime_config_path: Path,
    dry_run: bool,
    verbose: bool,
    storage_interval_sec: int,
    min_free_gb: float,
) -> None:
    cmd = [python_exec, "scripts/run_data.py", "--config", str(runtime_config_path)]
    _run_command(
        command=cmd,
        phase_tag="DATA",
        dry_run=dry_run,
        verbose=verbose,
        storage_interval_sec=storage_interval_sec,
        min_free_gb=min_free_gb,
    )


def _run_train_test_for_experiment(
    python_exec: str,
    exp_id: str,
    runtime_config_path: Path,
    resume: bool,
    dry_run: bool,
    verbose: bool,
    storage_interval_sec: int,
    min_free_gb: float,
) -> None:
    train_cmd = [python_exec, "scripts/run_train.py", "--config", str(runtime_config_path)]
    if resume:
        train_cmd.append("--continue-completed")
    _run_command(
        command=train_cmd,
        phase_tag=f"TRAIN-{exp_id}",
        dry_run=dry_run,
        verbose=verbose,
        storage_interval_sec=storage_interval_sec,
        min_free_gb=min_free_gb,
    )

    test_cmd = [python_exec, "scripts/run_test.py", "--config", str(runtime_config_path)]
    _run_command(
        command=test_cmd,
        phase_tag=f"TEST-{exp_id}",
        dry_run=dry_run,
        verbose=verbose,
        storage_interval_sec=storage_interval_sec,
        min_free_gb=min_free_gb,
    )


def _write_done_marker(exp_id: str, source_id: Optional[str], runtime_config_path: Path, dry_run: bool) -> None:
    marker = _done_marker_path(exp_id)
    payload = {
        "id": exp_id,
        "completed_at_unix": int(time.time()),
        "runtime_config": str(runtime_config_path),
        "rank_source_experiment": source_id,
    }
    if dry_run:
        print(f"[SUITE] Dry-run marker (not written): {payload}")
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _generate_report_template(
    python_exec: str,
    suite_config_path: Path,
    report_output: Path,
    dry_run: bool,
    verbose: bool,
    storage_interval_sec: int,
    min_free_gb: float,
) -> None:
    cmd = [
        python_exec,
        "scripts/generate_report_template.py",
        "--suite-config",
        str(suite_config_path),
        "--output",
        str(report_output),
    ]
    _run_command(
        command=cmd,
        phase_tag="REPORT",
        dry_run=dry_run,
        verbose=verbose,
        storage_interval_sec=storage_interval_sec,
        min_free_gb=min_free_gb,
    )


def _extract_summary_row(exp_id: str, phase: str, title: str, source_id: Optional[str]) -> Optional[Dict[str, Any]]:
    final_path = _test_final_path(exp_id)
    wer = _parse_metric_mean(final_path, "wer")
    runtime = _parse_metric_mean(final_path, "inference_runtime_sec")
    mem = _parse_metric_mean(final_path, "inference_peak_gpu_mem_mb")
    if wer is None or runtime is None or mem is None:
        return None
    return {
        "id": exp_id,
        "phase": phase,
        "title": title,
        "wer": wer,
        "inference_runtime_sec": runtime,
        "inference_peak_gpu_mem_mb": mem,
        "source_id": source_id or "",
    }


def _execute_experiment(
    python_exec: str,
    suite_cfg: Dict[str, Any],
    exp: Dict[str, Any],
    cfg: Dict[str, Any],
    source_id: Optional[str],
    resume: bool,
    dry_run: bool,
    verbose: bool,
    storage_interval_sec: int,
    min_free_gb: float,
) -> Optional[Dict[str, Any]]:
    exp_id = str(exp["id"])
    done_marker = _done_marker_path(exp_id)
    if resume and done_marker.exists():
        print(f"[SUITE] Resume: skipping already completed experiment {exp_id}.")
        return None

    runtime_cfg_path = _runtime_config_path(exp_id)
    _save_yaml(runtime_cfg_path, cfg)
    print(f"[SUITE] Runtime config written: {runtime_cfg_path}")

    _run_train_test_for_experiment(
        python_exec=python_exec,
        exp_id=exp_id,
        runtime_config_path=runtime_cfg_path,
        resume=resume,
        dry_run=dry_run,
        verbose=verbose,
        storage_interval_sec=storage_interval_sec,
        min_free_gb=min_free_gb,
    )

    _write_done_marker(exp_id, source_id, runtime_cfg_path, dry_run=dry_run)
    _archive_experiment_artifacts(exp_id, dry_run=dry_run)
    _cleanup_checkpoints_for_experiment(exp_id, dry_run=dry_run)
    _print_storage(f"post-cleanup {exp_id}")

    return _extract_summary_row(
        exp_id=exp_id,
        phase=str(exp.get("phase", "")),
        title=str(exp.get("title", exp_id)),
        source_id=source_id,
    )


def main() -> None:
    args = parse_args()
    cli_override_mapping = _build_cli_override_mapping(args.cli_overrides)
    suite_config_path = Path(args.suite_config)
    suite_cfg = _load_yaml(suite_config_path)
    experiments = _enabled_experiments(suite_cfg)
    selected_experiments = _slice_experiments(experiments, args.from_id, args.to_id)
    if not selected_experiments:
        raise RuntimeError("No experiments selected.")

    python_exec = sys.executable
    Path("results/suite/runtime_configs").mkdir(parents=True, exist_ok=True)
    Path("results/suite").mkdir(parents=True, exist_ok=True)

    print(f"[SUITE] Name: {suite_cfg.get('suite', {}).get('name', 'N/A')}")
    print(f"[SUITE] Selected experiments: {[e['id'] for e in selected_experiments]}")
    print(f"[SUITE] Python executable: {python_exec}")
    _print_storage("suite start")

    first_exp = next(
        (
            exp
            for exp in selected_experiments
            if "auto_from_screening_rank" not in exp and "auto_from_selection_rank" not in exp
        ),
        selected_experiments[0],
    )
    first_cfg = _resolved_config(suite_cfg, first_exp, cli_overrides=cli_override_mapping)
    first_cfg_path = _runtime_config_path(str(first_exp["id"]))
    _save_yaml(first_cfg_path, first_cfg)
    _run_data_once(
        python_exec=python_exec,
        runtime_config_path=first_cfg_path,
        dry_run=args.dry_run,
        verbose=args.verbose,
        storage_interval_sec=args.storage_interval_sec,
        min_free_gb=args.disk_guard_gb,
    )

    resolved_cfg_cache: Dict[str, Dict[str, Any]] = {str(first_exp["id"]): first_cfg}
    summary_rows: List[Dict[str, Any]] = []
    deferred_selection_finals: List[Dict[str, Any]] = []

    for exp in selected_experiments:
        if "auto_from_selection_rank" in exp:
            deferred_selection_finals.append(exp)
            continue

        exp_id = str(exp["id"])

        source_id: Optional[str] = None
        if "auto_from_screening_rank" in exp:
            screening_rows = _screening_rows(suite_cfg)
            fallback_ids = [
                str(e["id"])
                for e in _enabled_experiments(suite_cfg)
                if str(e.get("phase")) == "screening"
            ]
            cfg, source_id = _resolve_ranked_experiment(
                suite_cfg,
                exp,
                resolved_cfg_cache,
                ranking_rows=screening_rows,
                rank_key="auto_from_screening_rank",
                fallback_ids=fallback_ids,
                label="screening",
                cli_overrides=cli_override_mapping,
                allow_fallback_order=args.dry_run,
            )
            print(f"[SUITE] {exp_id} resolved from screening rank source {source_id}.")
        else:
            cfg = _resolved_config(suite_cfg, exp, cli_overrides=cli_override_mapping)
        resolved_cfg_cache[exp_id] = cfg

        row = _execute_experiment(
            python_exec=python_exec,
            suite_cfg=suite_cfg,
            exp=exp,
            cfg=cfg,
            source_id=source_id,
            resume=args.resume,
            dry_run=args.dry_run,
            verbose=args.verbose,
            storage_interval_sec=args.storage_interval_sec,
            min_free_gb=args.disk_guard_gb,
        )
        if row is not None:
            summary_rows.append(row)

    # Write screening leaderboard.
    screening_rows = _screening_rows(suite_cfg)
    leaderboard_path = Path(str(suite_cfg["suite"]["leaderboard_output"]))
    _write_csv(
        path=leaderboard_path,
        rows=screening_rows,
        fieldnames=["rank", "id", "title", "wer", "inference_runtime_sec", "inference_peak_gpu_mem_mb"],
    )
    print(f"[SUITE] Screening leaderboard: {leaderboard_path}")

    if deferred_selection_finals:
        selection_cfg = _selection_config(suite_cfg)
        if not bool(selection_cfg.get("enabled", True)):
            raise RuntimeError("Selection phase is disabled but final experiments require auto_from_selection_rank.")

        selection_manifest = _build_selection_manifest(
            suite_cfg=suite_cfg,
            allow_fallback_order=args.dry_run,
        )
        selection_manifest_path = _selection_manifest_path()
        if args.dry_run:
            print(f"[SUITE] Dry-run selection manifest: {selection_manifest}")
        else:
            selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(selection_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(selection_manifest, handle, indent=2)
            print(f"[SUITE] Selection manifest written: {selection_manifest_path}")

        selection_seeds = list(selection_manifest.get("seeds", [42, 123]))
        for run in selection_manifest.get("runs", []):
            selection_id = str(run["selection_id"])
            source_screening_id = str(run["source_screening_id"])
            if source_screening_id in resolved_cfg_cache:
                source_cfg = resolved_cfg_cache[source_screening_id]
            else:
                source_cfg = _load_yaml(_runtime_config_path(source_screening_id))
            selection_exp = {
                "id": selection_id,
                "phase": "selection",
                "title": f"Selection from {source_screening_id}",
                "seeds": selection_seeds,
            }
            cfg = _resolved_config(
                suite_cfg=suite_cfg,
                experiment=selection_exp,
                source_cfg=source_cfg,
                cli_overrides=cli_override_mapping,
            )
            resolved_cfg_cache[selection_id] = cfg
            row = _execute_experiment(
                python_exec=python_exec,
                suite_cfg=suite_cfg,
                exp=selection_exp,
                cfg=cfg,
                source_id=source_screening_id,
                resume=args.resume,
                dry_run=args.dry_run,
                verbose=args.verbose,
                storage_interval_sec=args.storage_interval_sec,
                min_free_gb=args.disk_guard_gb,
            )
            if row is not None:
                summary_rows.append(row)

        selection_rows = _selection_rows_from_manifest(selection_manifest)
        selection_leaderboard_path = Path(
            str(selection_cfg.get("leaderboard_output", "results/suite/leaderboard_selection.csv"))
        )
        _write_csv(
            path=selection_leaderboard_path,
            rows=selection_rows,
            fieldnames=["rank", "id", "source_screening_id", "wer", "inference_runtime_sec", "inference_peak_gpu_mem_mb"],
        )
        print(f"[SUITE] Selection leaderboard: {selection_leaderboard_path}")

        selection_fallback_ids = [str(run["selection_id"]) for run in selection_manifest.get("runs", [])]
        for exp in deferred_selection_finals:
            exp_id = str(exp["id"])
            cfg, source_id = _resolve_ranked_experiment(
                suite_cfg=suite_cfg,
                experiment=exp,
                resolved_cfg_cache=resolved_cfg_cache,
                ranking_rows=selection_rows,
                rank_key="auto_from_selection_rank",
                fallback_ids=selection_fallback_ids,
                label="selection",
                cli_overrides=cli_override_mapping,
                allow_fallback_order=args.dry_run,
            )
            print(f"[SUITE] {exp_id} resolved from selection rank source {source_id}.")
            resolved_cfg_cache[exp_id] = cfg
            row = _execute_experiment(
                python_exec=python_exec,
                suite_cfg=suite_cfg,
                exp=exp,
                cfg=cfg,
                source_id=source_id,
                resume=args.resume,
                dry_run=args.dry_run,
                verbose=args.verbose,
                storage_interval_sec=args.storage_interval_sec,
                min_free_gb=args.disk_guard_gb,
            )
            if row is not None:
                summary_rows.append(row)

    # Write global suite summary for selected experiments.
    summary_path = Path(str(suite_cfg["suite"]["summary_output"]))
    _write_csv(
        path=summary_path,
        rows=summary_rows,
        fieldnames=["id", "phase", "title", "wer", "inference_runtime_sec", "inference_peak_gpu_mem_mb", "source_id"],
    )
    print(f"[SUITE] Summary CSV: {summary_path}")

    report_output = Path(str(suite_cfg["suite"]["report_output"]))
    _generate_report_template(
        python_exec=python_exec,
        suite_config_path=suite_config_path,
        report_output=report_output,
        dry_run=args.dry_run,
        verbose=args.verbose,
        storage_interval_sec=args.storage_interval_sec,
        min_free_gb=args.disk_guard_gb,
    )
    _print_storage("suite end")
    print("[SUITE] Completed.")


if __name__ == "__main__":
    main()
