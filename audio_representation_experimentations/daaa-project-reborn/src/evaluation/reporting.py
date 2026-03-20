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
        ("Blank ratio", _fmt(metrics.get("blank_ratio", {}))),
        ("Empty pred ratio", _fmt(metrics.get("empty_pred_ratio", {}))),
        ("Pred/ref char ratio", _fmt(metrics.get("pred_to_ref_char_ratio", {}))),
        ("Length deviation ratio", _fmt(metrics.get("length_deviation_ratio", {}))),
        ("Adjacent repeat ratio", _fmt(metrics.get("adjacent_repeat_ratio", {}))),
        ("Dominant char ratio", _fmt(metrics.get("dominant_char_ratio", {}))),
        ("Inference runtime (sec)", _fmt(metrics.get("inference_runtime_sec", {}))),
        ("Samples/sec", _fmt(metrics.get("inference_samples_per_sec", {}))),
        ("Peak GPU mem (MB)", _fmt(metrics.get("inference_peak_gpu_mem_mb", {}))),
        ("Model trainable params", _fmt(metrics.get("model_trainable_params", {}))),
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
        handle.write("| Dataset | WER mean +- std | Accuracy mean +- std | Empty pred ratio | Pred/ref ratio | Length deviation | Repeat ratio |\n")
        handle.write("|---|---|---|---|---|---|---|\n")
        for dataset_name, payload in metrics.items():
            handle.write(
                f"| {dataset_name} | {_fmt(payload.get('wer', {}))} | {_fmt(payload.get('accuracy', {}))} | "
                f"{_fmt(payload.get('empty_pred_ratio', {}))} | {_fmt(payload.get('pred_to_ref_char_ratio', {}))} | "
                f"{_fmt(payload.get('length_deviation_ratio', {}))} | {_fmt(payload.get('adjacent_repeat_ratio', {}))} |\n"
            )


def _fmt_scalar(metric: Dict[str, float], key: str) -> str:
    if not metric:
        return "n/a"
    payload = metric.get(key, {})
    if not isinstance(payload, dict):
        return "n/a"
    return _fmt(payload)


def _forensics_sections(data: Dict[str, Any]) -> list[tuple[str, Dict[str, Any]]]:
    sections: list[tuple[str, Dict[str, Any]]] = []
    train_probe = data.get("train_probe", {})
    validation = data.get("validation", {})
    test = data.get("test", {})
    if isinstance(train_probe, dict):
        sections.append(("train_probe", train_probe.get("metrics", {})))
    if isinstance(validation, dict):
        sections.append(("validation", validation.get("metrics", {})))
    if isinstance(test, dict):
        for variant in ("best", "final"):
            payload = test.get(variant, {})
            if isinstance(payload, dict):
                sections.append((f"test_{variant}", payload.get("overall", {}).get("metrics", {})))
    return sections


def write_forensics_overview_table(final_json_path: Path, table_md_path: Path, title: str) -> None:
    with open(final_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    table_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| Stage | WER | Accuracy | Blank ratio | Pred/ref ratio | Frame entropy | Raw dominant token | Collapsed dominant char |\n")
        handle.write("|---|---|---|---|---|---|---|---|\n")
        for label, metrics in _forensics_sections(data):
            handle.write(
                f"| {label} | {_fmt_scalar(metrics, 'wer')} | {_fmt_scalar(metrics, 'accuracy')} | "
                f"{_fmt_scalar(metrics, 'blank_ratio')} | {_fmt_scalar(metrics, 'pred_to_ref_char_ratio')} | "
                f"{_fmt_scalar(metrics, 'frame_entropy_mean')} | {_fmt_scalar(metrics, 'raw_dominant_token_ratio')} | "
                f"{_fmt_scalar(metrics, 'collapsed_dominant_char_ratio')} |\n"
            )


def write_forensics_alignment_table(final_json_path: Path, table_md_path: Path, title: str) -> None:
    with open(final_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    table_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| Stage | Student steps | Teacher steps | Teacher/student ratio | Interp ratio | Student FPS | Teacher FPS | Valid overlap | Stretch |\n")
        handle.write("|---|---|---|---|---|---|---|---|---|\n")
        for label, metrics in _forensics_sections(data):
            handle.write(
                f"| {label} | {_fmt_scalar(metrics, 'student_time_steps_mean')} | {_fmt_scalar(metrics, 'teacher_time_steps_mean')} | "
                f"{_fmt_scalar(metrics, 'teacher_to_student_time_ratio_mean')} | {_fmt_scalar(metrics, 'alignment_interpolation_ratio_mean')} | "
                f"{_fmt_scalar(metrics, 'student_frames_per_second_estimate')} | {_fmt_scalar(metrics, 'teacher_frames_per_second_estimate')} | "
                f"{_fmt_scalar(metrics, 'teacher_student_valid_frame_overlap_ratio')} | {_fmt_scalar(metrics, 'teacher_student_alignment_stretch_mean')} |\n"
            )


def write_forensics_teacher_student_table(final_json_path: Path, table_md_path: Path, title: str) -> None:
    with open(final_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    table_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| Stage | Teacher entropy | Teacher top1 conf | Argmax agree | Top-k@3 overlap | Feature cosine | Feature MSE | Norm ratio |\n")
        handle.write("|---|---|---|---|---|---|---|---|\n")
        for label, metrics in _forensics_sections(data):
            handle.write(
                f"| {label} | {_fmt_scalar(metrics, 'teacher_entropy_mean')} | {_fmt_scalar(metrics, 'teacher_top1_confidence_mean')} | "
                f"{_fmt_scalar(metrics, 'teacher_student_argmax_agreement')} | {_fmt_scalar(metrics, 'teacher_student_topk_overlap_at3')} | "
                f"{_fmt_scalar(metrics, 'teacher_student_feature_cosine_mean')} | {_fmt_scalar(metrics, 'teacher_student_feature_mse_mean')} | "
                f"{_fmt_scalar(metrics, 'teacher_student_feature_norm_ratio_mean')} |\n"
            )

