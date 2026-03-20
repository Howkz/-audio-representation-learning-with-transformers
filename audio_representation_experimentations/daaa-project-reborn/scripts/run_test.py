from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure project root is importable when running `python scripts/run_test.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_project_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR checkpoints and aggregate 5-seed results.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate evaluation plan without running inference.")
    return parser.parse_args()


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _reduce_eval_batch(cfg: Dict[str, Any]) -> bool:
    current_batch = int(cfg["training"]["finetune"]["batch_size"])
    if current_batch <= 1:
        return False
    cfg["training"]["finetune"]["batch_size"] = max(1, current_batch // 2)
    return True


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _find_seed_checkpoint(cfg: Dict[str, Any], seed: int) -> Path:
    root = Path(cfg["experiment"]["output_dir"]) / "checkpoints"
    exp_id = cfg["experiment"].get("id")
    if exp_id:
        root = root / str(exp_id)
    root = root / "finetune" / f"seed_{seed}"
    best = root / "ctc_best.pt"
    final = root / "ctc_final.pt"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(f"No fine-tuned checkpoint found for seed {seed}: {root}")


def _aggregate_by_dataset(dataset_runs: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {"metrics": {}}
    for dataset_name, runs in dataset_runs.items():
        values: Dict[str, list] = {}
        for run_payload in runs.values():
            for key, value in run_payload.items():
                if isinstance(value, (int, float)):
                    values.setdefault(key, []).append(float(value))
        aggregated["metrics"][dataset_name] = {}
        for key, series in values.items():
            aggregated["metrics"][dataset_name][key] = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
            }
    return aggregated


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    if args.dry_run:
        print("[TEST] Dry-run enabled.")
        print(f"[TEST] Seeds: {cfg['experiment']['seeds']}")
        for spec in cfg["datasets"]["asr_tests"]:
            print(f"[TEST] Planned dataset: {spec['name']}:{spec['split']} max={spec.get('max_samples')}")
        return

    from src.data.dataset import (
        AudioFeatureDataset,
        apply_dataset_filters,
        build_audio_preprocess_config,
        load_hf_audio_dataset,
        resolve_transcript_key,
    )
    from src.data.text import CharCTCTokenizer
    from src.evaluation.reporting import write_dataset_breakdown_table, write_final_table
    from src.training.loops import evaluate_seed_on_dataset
    from src.training.results import aggregate_partial_to_final, write_json_artifact, write_run_partial

    def _load_audio_dataset(local_cfg: Dict[str, Any], spec: Dict[str, Any], local_audio_cfg):
        default_streaming = bool(local_cfg.get("data", {}).get("streaming", False))
        ds = load_hf_audio_dataset(
            dataset_name=spec["name"],
            dataset_config=spec.get("config"),
            split=spec["split"],
            cache_dir=local_cfg["experiment"]["cache_dir"],
            max_samples=spec.get("max_samples"),
            streaming=bool(spec.get("streaming", default_streaming)),
        )
        transcript_key = resolve_transcript_key(ds[0], spec.get("transcript_key")) if len(ds) > 0 else spec.get("transcript_key")
        ds = apply_dataset_filters(ds, transcript_key=transcript_key, spec=spec)
        return AudioFeatureDataset(ds, audio_cfg=local_audio_cfg, transcript_key=spec.get("transcript_key"))

    tokenizer = CharCTCTokenizer()

    tests = cfg["datasets"]["asr_tests"]
    test_datasets = {}
    for spec in tests:
        dataset_label = f"{spec['name']}:{spec['split']}"
        test_datasets[dataset_label] = _load_audio_dataset(
            cfg,
            spec,
            build_audio_preprocess_config(cfg, spec),
        )
        print(f"[TEST] Loaded {dataset_label} with {len(test_datasets[dataset_label])} samples.")

    exp_id = cfg["experiment"].get("id")
    filename_suffix = f"_{exp_id}" if exp_id else ""
    benchmark_dir = Path(cfg["experiment"]["results_dir"]) / "benchmark_results"
    partial_path = benchmark_dir / f"asr_benchmark{filename_suffix}_partial.json"
    final_path = benchmark_dir / f"asr_benchmark{filename_suffix}_final.json"
    dataset_breakdown_path = benchmark_dir / f"asr_benchmark_by_dataset{filename_suffix}_final.json"
    diagnostics_breakdown_path = benchmark_dir / f"asr_diagnostics_by_dataset{filename_suffix}_final.json"

    print(f"[TEST] Experiment id: {exp_id if exp_id else 'N/A'}")
    pretrain_enabled = bool(cfg["training"]["pretrain"].get("enabled", True))
    adaptation_label = "MAE pretrain -> CTC fine-tune" if pretrain_enabled else "CTC fine-tune only (no MAE)"

    dataset_runs: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k in test_datasets.keys()}

    for seed in cfg["experiment"]["seeds"]:
        eval_cfg = copy.deepcopy(cfg)
        ckpt_path = _find_seed_checkpoint(cfg, int(seed))
        per_dataset_metrics = []
        for dataset_label, dataset in test_datasets.items():
            for attempt in range(3):
                try:
                    metrics = evaluate_seed_on_dataset(
                        cfg=eval_cfg,
                        seed=int(seed),
                        dataset=dataset,
                        tokenizer=tokenizer,
                        checkpoint_path=ckpt_path,
                        dataset_label=dataset_label,
                        artifact_path=Path(cfg["experiment"]["results_dir"])
                        / "diagnostics"
                        / f"test_seed_{seed}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset_label)}.json",
                    )
                    break
                except RuntimeError as exc:
                    if not _is_oom_error(exc):
                        raise
                    if not _reduce_eval_batch(eval_cfg):
                        raise
                    _clear_cuda_cache()
                    print(
                        f"[TEST][OOM] seed={seed} dataset={dataset_label} retry {attempt + 1}/2 "
                        f"with eval_batch_size={eval_cfg['training']['finetune']['batch_size']}"
                    )
            dataset_runs[dataset_label][str(seed)] = metrics
            per_dataset_metrics.append(metrics)
            print(
                f"[TEST] seed={seed} dataset={dataset_label} "
                f"wer={metrics['wer']:.4f} accuracy={metrics['accuracy']:.4f} "
                f"runtime={metrics['inference_runtime_sec']:.2f}s "
                f"blank_ratio={metrics['blank_ratio']:.3f} empty_pred_ratio={metrics['empty_pred_ratio']:.3f}"
            )

        overall = {
            "seed": float(seed),
            "wer": float(np.mean([m["wer"] for m in per_dataset_metrics])),
            "accuracy": float(np.mean([m["accuracy"] for m in per_dataset_metrics])),
            "inference_runtime_sec": float(np.mean([m["inference_runtime_sec"] for m in per_dataset_metrics])),
            "inference_samples_per_sec": float(np.mean([m["inference_samples_per_sec"] for m in per_dataset_metrics])),
            "inference_peak_gpu_mem_mb": float(np.max([m["inference_peak_gpu_mem_mb"] for m in per_dataset_metrics])),
            "blank_ratio": float(np.mean([m["blank_ratio"] for m in per_dataset_metrics])),
            "empty_pred_ratio": float(np.mean([m["empty_pred_ratio"] for m in per_dataset_metrics])),
            "nonempty_pred_ratio": float(np.mean([m["nonempty_pred_ratio"] for m in per_dataset_metrics])),
            "invalid_length_ratio": float(np.mean([m["invalid_length_ratio"] for m in per_dataset_metrics])),
            "avg_out_length": float(np.mean([m["avg_out_length"] for m in per_dataset_metrics])),
            "avg_target_length": float(np.mean([m["avg_target_length"] for m in per_dataset_metrics])),
            "avg_length_margin": float(np.mean([m["avg_length_margin"] for m in per_dataset_metrics])),
        }
        write_run_partial(
            partial_path=partial_path,
            run_id=str(seed),
            payload=overall,
            model_name=cfg["experiment"]["name"],
            architecture="Audio Transformer Encoder + CTC",
            adaptation=adaptation_label,
        )

    aggregate_partial_to_final(partial_path=partial_path, final_path=final_path)
    dataset_breakdown = _aggregate_by_dataset(dataset_runs)
    write_json_artifact(dataset_breakdown_path, dataset_breakdown)
    write_json_artifact(
        diagnostics_breakdown_path,
        {
            "experiment_id": exp_id,
            "datasets": dataset_runs,
        },
    )

    tables_dir = Path(cfg["experiment"]["results_dir"]) / "tables"
    write_final_table(
        final_json_path=final_path,
        table_md_path=tables_dir / f"asr_overall_table{filename_suffix}.md",
        title="ASR Overall Benchmark (5 seeds)",
    )
    write_dataset_breakdown_table(
        dataset_final_json_path=dataset_breakdown_path,
        table_md_path=tables_dir / f"asr_dataset_breakdown{filename_suffix}.md",
        title="ASR Dataset Breakdown (5 seeds)",
    )
    print(
        f"[TEST] Aggregated files: {final_path}, {dataset_breakdown_path} "
        f"and {diagnostics_breakdown_path}"
    )


if __name__ == "__main__":
    main()
