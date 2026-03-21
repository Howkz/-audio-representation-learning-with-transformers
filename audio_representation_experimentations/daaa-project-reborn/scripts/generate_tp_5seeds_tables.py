from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final TP 5-seed summary tables.")
    parser.add_argument("--suite-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/suite/tp5")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(payload: Dict[str, Any], key: str) -> str:
    metrics = payload.get("metrics", {})
    metric = metrics.get(key, {})
    if not isinstance(metric, dict):
        return "n/a"
    return f"{metric.get('mean', 0.0):.4f} +- {metric.get('std', 0.0):.4f}"


def _metric_mean(payload: Dict[str, Any], key: str) -> Optional[float]:
    metric = payload.get("metrics", {}).get(key, {})
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _experiment_paths(exp_id: str) -> Dict[str, Path]:
    root = PROJECT_ROOT / "results" / "experiments" / exp_id / "benchmark_results"
    return {
        "train": root / f"train_audio_transformer_{exp_id}_final.json",
        "asr": root / f"asr_benchmark_{exp_id}_final.json",
        "probe": root / f"probe_benchmark_{exp_id}_final.json",
    }


def _write_markdown(path: Path, title: str, headers: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(row) + " |\n")


def main() -> None:
    args = parse_args()
    suite_cfg = _load_yaml(_resolve_input_path(args.suite_config))
    experiments = [exp for exp in suite_cfg.get("experiments", []) if bool(exp.get("enabled", True))]
    output_dir = PROJECT_ROOT / args.output_dir

    perf_rows: List[List[str]] = []
    frugality_rows: List[List[str]] = []
    index_by_label: Dict[tuple[str, str], Dict[str, Any]] = {}

    for exp in experiments:
        exp_id = str(exp["id"])
        paths = _experiment_paths(exp_id)
        train_payload = _read_json(paths["train"])
        asr_payload = _read_json(paths["asr"])
        probe_payload = _read_json(paths["probe"])

        benchmark = str(exp.get("benchmark", "unknown"))
        comparison_label = str(exp.get("comparison_label", exp_id))
        index_by_label[(benchmark, comparison_label)] = {
            "exp": exp,
            "train": train_payload,
            "asr": asr_payload,
            "probe": probe_payload,
        }

        if asr_payload:
            perf_rows.append(
                [
                    exp_id,
                    benchmark,
                    "ASR",
                    str(exp.get("input_type", "n/a")),
                    str(exp.get("pretraining", "n/a")),
                    "yes" if bool(exp.get("augmentation", False)) else "no",
                    str(exp.get("patch_time", "n/a")),
                    str(exp.get("distillation", "none")),
                    _metric(asr_payload, "wer"),
                    _metric(asr_payload, "accuracy"),
                ]
            )
            frugality_rows.append(
                [
                    exp_id,
                    benchmark,
                    "ASR",
                    _metric(train_payload, "train_runtime_sec"),
                    _metric(asr_payload, "inference_runtime_sec"),
                    _metric(train_payload, "train_peak_gpu_mem_mb"),
                    _metric(train_payload, "finetune_effective_batch_size"),
                    _metric(asr_payload, "model_trainable_params"),
                ]
            )

        if probe_payload:
            perf_rows.append(
                [
                    exp_id,
                    "fsc",
                    "Linear probe",
                    str(exp.get("input_type", "n/a")),
                    str(exp.get("pretraining", "n/a")),
                    "yes" if bool(exp.get("augmentation", False)) else "no",
                    str(exp.get("patch_time", "n/a")),
                    str(exp.get("distillation", "none")),
                    _metric(probe_payload, "accuracy"),
                    _metric(probe_payload, "macro_f1"),
                ]
            )
            frugality_rows.append(
                [
                    exp_id,
                    "fsc",
                    "Linear probe",
                    _metric(train_payload, "probe_train_runtime_sec"),
                    _metric(probe_payload, "inference_runtime_sec"),
                    _metric(train_payload, "probe_train_peak_gpu_mem_mb"),
                    _metric(train_payload, "probe_effective_batch_size"),
                    _metric(probe_payload, "model_trainable_params"),
                ]
            )

    ablation_rows: List[List[str]] = []
    pair_labels = [("R0", "R1"), ("R1", "R2"), ("R1", "R3"), ("R1", "R4"), ("R4", "A1")]
    for benchmark in sorted({str(exp.get("benchmark", "unknown")) for exp in experiments}):
        for left, right in pair_labels:
            left_payload = index_by_label.get((benchmark, left))
            right_payload = index_by_label.get((benchmark, right))
            if not left_payload or not right_payload:
                continue
            left_wer = _metric_mean(left_payload["asr"], "wer")
            right_wer = _metric_mean(right_payload["asr"], "wer")
            left_acc = _metric_mean(left_payload["probe"], "accuracy")
            right_acc = _metric_mean(right_payload["probe"], "accuracy")
            ablation_rows.append(
                [
                    benchmark,
                    f"{left} vs {right}",
                    f"{left_payload['exp']['id']} -> {right_payload['exp']['id']}",
                    "n/a" if left_wer is None or right_wer is None else f"{(right_wer - left_wer):+.4f}",
                    "n/a" if left_acc is None or right_acc is None else f"{(right_acc - left_acc):+.4f}",
                ]
            )

    _write_markdown(
        output_dir / "tp5_main_performance_table.md",
        "TP 5-Seeds Main Performance Table",
        ["Experiment", "Benchmark", "Task", "Input", "Pretraining", "Aug", "Patch time", "Distillation", "Metric A", "Metric B"],
        perf_rows,
    )
    _write_markdown(
        output_dir / "tp5_frugality_table.md",
        "TP 5-Seeds Frugality Table",
        ["Experiment", "Benchmark", "Task", "Train runtime", "Inference runtime", "Peak GPU mem", "Effective batch", "Trainable params"],
        frugality_rows,
    )
    _write_markdown(
        output_dir / "tp5_ablation_table.md",
        "TP 5-Seeds Ablation Table",
        ["Benchmark", "Comparison", "Experiments", "Delta ASR WER", "Delta FSC accuracy"],
        ablation_rows,
    )

    print(f"[TP5] Tables written to {output_dir}")


if __name__ == "__main__":
    main()
