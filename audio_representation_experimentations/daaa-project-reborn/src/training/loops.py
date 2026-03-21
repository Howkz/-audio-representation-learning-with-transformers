from __future__ import annotations

import math
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler as LegacyGradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.data.collate import ctc_collate, pad_collate
from src.data.text import CharCTCTokenizer
from src.models.audio_transformer import (
    AudioMAEPretrain,
    AudioTransformerCTC,
    AudioTransformerEncoder,
    make_mae_mask,
)
from src.training.checkpointing import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.training.metrics import (
    collect_ctc_batch_diagnostics,
    compute_char_accuracy,
    compute_wer,
    finalize_ctc_diagnostics,
)
from src.training.results import write_json_artifact
from src.training.teachers import TeacherOutput, build_teacher
from src.training.utils import count_parameters, peak_memory_mb, reset_peak_memory


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_hms(total_seconds: float) -> str:
    seconds = max(0, int(round(float(total_seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _build_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return LegacyGradScaler(enabled=enabled)


def _use_tqdm() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _diagnostics_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = cfg.get("diagnostics", {})
    return diagnostics if isinstance(diagnostics, dict) else {}


def _forensics_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_diagnostics_cfg(cfg).get("forensics_enabled", False))


def _diagnostics_examples_limit(cfg: Dict[str, Any]) -> int:
    diagnostics = _diagnostics_cfg(cfg)
    return max(0, int(diagnostics.get("max_saved_examples", 8)))


def _diagnostics_frames_limit(cfg: Dict[str, Any]) -> int:
    diagnostics = _diagnostics_cfg(cfg)
    return max(1, int(diagnostics.get("max_saved_frames", 96)))


def _diagnostics_topk_tokens(cfg: Dict[str, Any]) -> int:
    diagnostics = _diagnostics_cfg(cfg)
    return max(1, int(diagnostics.get("topk_tokens", 5)))


def _train_probe_size(cfg: Dict[str, Any]) -> int:
    diagnostics = _diagnostics_cfg(cfg)
    return max(0, int(diagnostics.get("train_probe_size", 0)))


def _train_probe_seed(cfg: Dict[str, Any]) -> int:
    diagnostics = _diagnostics_cfg(cfg)
    return int(diagnostics.get("train_probe_seed", 1337))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "artifact"


def _diagnostics_artifact_path(
    cfg: Dict[str, Any],
    stage: str,
    seed: int,
    suffix: str,
) -> Path:
    diagnostics_dir = Path(cfg["experiment"]["results_dir"]) / "diagnostics"
    filename = f"{_safe_slug(stage)}_seed_{seed}_{_safe_slug(suffix)}.json"
    return diagnostics_dir / filename


def _merge_diagnostic_totals(target: Dict[str, float], update: Dict[str, float]) -> None:
    for key, value in update.items():
        if isinstance(value, dict):
            nested = target.get(key)
            if not isinstance(nested, dict):
                nested = {}
                target[key] = nested
            _merge_diagnostic_totals(nested, value)
        else:
            target[key] = float(target.get(key, 0.0)) + float(value)


def _maybe_warn_ctc(stage: str, metrics: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    diagnostics = _diagnostics_cfg(cfg)
    blank_threshold = float(diagnostics.get("warn_blank_ratio", 0.98))
    empty_threshold = float(diagnostics.get("warn_empty_pred_ratio", 0.80))
    invalid_threshold = float(diagnostics.get("warn_invalid_length_ratio", 0.0))

    if float(metrics.get("blank_ratio", 0.0)) >= blank_threshold:
        print(
            f"[{stage}][WARN] blank_ratio={float(metrics['blank_ratio']):.3f} "
            f">= {blank_threshold:.3f}. Les sorties sont probablement dominées par blank."
        )
    if float(metrics.get("empty_pred_ratio", 0.0)) >= empty_threshold:
        print(
            f"[{stage}][WARN] empty_pred_ratio={float(metrics['empty_pred_ratio']):.3f} "
            f">= {empty_threshold:.3f}. Les prédictions sont largement vides."
        )
    if float(metrics.get("invalid_length_ratio", 0.0)) > invalid_threshold:
        print(
            f"[{stage}][WARN] invalid_length_ratio={float(metrics['invalid_length_ratio']):.3f}. "
            "Une partie des séquences viole la contrainte CTC out_lengths >= target_lengths."
        )


def _pretrain_mode(cfg: Dict[str, Any]) -> str:
    pretrain_cfg = cfg.get("pretrain", {})
    if isinstance(pretrain_cfg, dict) and pretrain_cfg.get("mode") is not None:
        return str(pretrain_cfg.get("mode", "none")).lower()
    if bool(cfg["training"]["pretrain"].get("enabled", True)):
        return "mae"
    return "none"


def _pretrain_enabled(cfg: Dict[str, Any]) -> bool:
    return _pretrain_mode(cfg) == "mae"


def _distillation_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    distill_cfg = cfg.get("distillation", {})
    return distill_cfg if isinstance(distill_cfg, dict) else {}


def _distillation_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_distillation_cfg(cfg).get("enabled", False))


def _distillation_warmup_steps(cfg: Dict[str, Any]) -> int:
    return max(0, int(_distillation_cfg(cfg).get("warmup_steps", 0)))


def _distillation_warmup_start_factor(cfg: Dict[str, Any]) -> float:
    return float(max(0.0, min(1.0, float(_distillation_cfg(cfg).get("warmup_start_factor", 0.0)))))


def _effective_lambda_kd(cfg: Dict[str, Any], base_lambda_kd: float, optimizer_steps_completed: int) -> float:
    base_lambda_kd = float(base_lambda_kd)
    if base_lambda_kd <= 0.0:
        return 0.0
    warmup_steps = _distillation_warmup_steps(cfg)
    if warmup_steps <= 0:
        return base_lambda_kd
    start_factor = _distillation_warmup_start_factor(cfg)
    step_idx = max(0, int(optimizer_steps_completed))
    if step_idx >= warmup_steps:
        return base_lambda_kd
    if warmup_steps == 1:
        return base_lambda_kd
    progress = float(step_idx) / float(warmup_steps)
    factor = start_factor + progress * (1.0 - start_factor)
    return base_lambda_kd * float(max(0.0, min(1.0, factor)))


def _overemit_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    overemit_cfg = cfg.get("anti_overemit", {})
    return overemit_cfg if isinstance(overemit_cfg, dict) else {}


def _underemit_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    underemit_cfg = cfg.get("anti_underemit", {})
    return underemit_cfg if isinstance(underemit_cfg, dict) else {}


def _resolve_distill_projection_dim(
    cfg: Dict[str, Any],
    teacher: Optional[Any],
) -> Optional[int]:
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict) and model_cfg.get("distill_projection_dim") is not None:
        return int(model_cfg.get("distill_projection_dim"))
    if teacher is not None and getattr(teacher, "target_kind", "logits") == "hidden_states":
        hidden_dim = getattr(teacher, "hidden_dim", None)
        if hidden_dim is not None and int(hidden_dim) > 0:
            return int(hidden_dim)
    return None


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _adaptation_label(cfg: Dict[str, Any]) -> str:
    pretrain_mode = _pretrain_mode(cfg)
    if _distillation_enabled(cfg):
        teacher_cfg = cfg.get("teacher", {})
        source = str(teacher_cfg.get("source", "external"))
        family = str(teacher_cfg.get("family", source))
        if pretrain_mode == "mae":
            return f"MAE pretrain -> CTC fine-tune + distillation ({source}:{family})"
        return f"CTC fine-tune + distillation ({source}:{family})"
    if pretrain_mode == "mae":
        return "MAE pretrain -> CTC fine-tune"
    return "CTC fine-tune only (no MAE)"


def _align_teacher_values(
    values: torch.Tensor,
    source_lengths: torch.Tensor,
    target_steps: int,
    target_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3:
        raise ValueError("Teacher values must be [B, T, C].")

    if values.shape[1] == target_steps:
        aligned_values = values
    else:
        aligned_values = F.interpolate(
            values.transpose(1, 2),
            size=int(target_steps),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    source_mask = (
        torch.arange(values.shape[1], device=values.device, dtype=torch.long).unsqueeze(0)
        < source_lengths.to(values.device).unsqueeze(1)
    ).to(dtype=torch.float32)
    if source_mask.shape[1] == target_steps:
        aligned_mask = source_mask > 0.5
    else:
        aligned_mask = F.interpolate(
            source_mask.unsqueeze(1),
            size=int(target_steps),
            mode="nearest",
        ).squeeze(1) > 0.5
    student_mask = (
        torch.arange(target_steps, device=target_lengths.device, dtype=torch.long).unsqueeze(0)
        < target_lengths.unsqueeze(1)
    )
    return aligned_values, aligned_mask & student_mask.to(aligned_mask.device)


def _prepare_teacher_forensics(
    *,
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor,
    student_features: torch.Tensor,
    teacher_output: TeacherOutput,
) -> Dict[str, Any]:
    prepared: Dict[str, Any] = {
        "raw_lengths": teacher_output.lengths.detach(),
    }
    if teacher_output.target_kind == "hidden_states":
        aligned_features, feature_valid_mask = _align_teacher_values(
            values=teacher_output.values,
            source_lengths=teacher_output.lengths,
            target_steps=int(student_logits.shape[1]),
            target_lengths=student_lengths,
        )
        prepared["features"] = aligned_features.detach()
        prepared["feature_valid_mask"] = feature_valid_mask.detach()
    distribution_values = getattr(teacher_output, "distribution_values", None)
    if isinstance(distribution_values, torch.Tensor):
        aligned_distribution, distribution_valid_mask = _align_teacher_values(
            values=distribution_values,
            source_lengths=teacher_output.lengths,
            target_steps=int(student_logits.shape[1]),
            target_lengths=student_lengths,
        )
        prepared["distribution"] = aligned_distribution.detach()
        prepared["distribution_valid_mask"] = distribution_valid_mask.detach()
    return prepared


def _write_json_artifacts(
    primary_path: Optional[Path],
    payload: Dict[str, Any],
    extra_artifact_paths: Sequence[Path] = (),
) -> None:
    seen: set[str] = set()
    for path in [primary_path, *list(extra_artifact_paths)]:
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        write_json_artifact(path, payload)


def _make_train_probe_subset(
    dataset: torch.utils.data.Dataset,
    cfg: Dict[str, Any],
) -> Optional[Subset]:
    probe_size = min(len(dataset), _train_probe_size(cfg))
    if probe_size <= 0:
        return None
    generator = torch.Generator()
    generator.manual_seed(_train_probe_seed(cfg))
    indices = torch.randperm(len(dataset), generator=generator)[:probe_size].tolist()
    indices.sort()
    return Subset(dataset, indices)


def _compute_distillation_loss(
    *,
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor,
    student_features: torch.Tensor,
    teacher_output: TeacherOutput,
    temperature: float,
    informative_only: bool = False,
    min_nonblank_prob: float = 0.0,
    require_nonblank_argmax: bool = True,
) -> Tuple[torch.Tensor, float]:
    temperature = max(1e-4, float(temperature))
    aligned_values, valid_mask = _align_teacher_values(
        values=teacher_output.values,
        source_lengths=teacher_output.lengths,
        target_steps=int(student_logits.shape[1]),
        target_lengths=student_lengths,
    )
    if not valid_mask.any():
        return student_logits.new_zeros(()), 0.0

    if teacher_output.target_kind == "logits":
        teacher_probs = aligned_values / aligned_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        kd_mask = valid_mask
        if informative_only:
            blank_id = int(getattr(teacher_output, "blank_id", 0))
            blank_probs = teacher_probs[..., blank_id]
            informative_mask = (1.0 - blank_probs) >= max(0.0, float(min_nonblank_prob))
            if require_nonblank_argmax:
                informative_mask = informative_mask & (teacher_probs.argmax(dim=-1) != blank_id)
            kd_mask = kd_mask & informative_mask
            if not kd_mask.any():
                return student_logits.new_zeros(()), 0.0
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        loss = (token_kl * kd_mask.to(token_kl.dtype)).sum() / kd_mask.sum().clamp_min(1).to(token_kl.dtype)
        active_ratio = float(
            kd_mask.sum().item() / max(1.0, float(valid_mask.sum().item()))
        )
        return loss * (temperature ** 2), active_ratio

    if teacher_output.target_kind == "hidden_states":
        if aligned_values.shape[-1] != student_features.shape[-1]:
            raise ValueError(
                "Teacher hidden-state distillation requires matching hidden dims. "
                "Use a compatible student dim or add a projection layer."
            )
        mse = F.mse_loss(student_features, aligned_values, reduction="none").mean(dim=-1)
        active_ratio = float(valid_mask.sum().item() / max(1.0, float(valid_mask.sum().item())))
        return (
            (mse * valid_mask.to(mse.dtype)).sum() / valid_mask.sum().clamp_min(1).to(mse.dtype),
            active_ratio,
        )

    raise ValueError(f"Unsupported teacher target_kind '{teacher_output.target_kind}'.")


def _nonblank_density_stats(
    *,
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if student_logits.ndim != 3:
        raise ValueError("student_logits must be [B, T, C].")
    valid_mask = (
        torch.arange(student_logits.shape[1], device=student_lengths.device, dtype=torch.long).unsqueeze(0)
        < student_lengths.unsqueeze(1)
    )
    if not valid_mask.any():
        zeros = student_logits.new_zeros((student_logits.shape[0],))
        return zeros, zeros

    probs = torch.softmax(student_logits, dim=-1)
    nonblank_probs = 1.0 - probs[..., int(blank_id)]
    mean_nonblank = (
        (nonblank_probs * valid_mask.to(nonblank_probs.dtype)).sum(dim=1)
        / student_lengths.clamp_min(1).to(nonblank_probs.dtype)
    )
    target_density = (
        target_lengths.to(nonblank_probs.dtype)
        / student_lengths.clamp_min(1).to(nonblank_probs.dtype)
    ).clamp(min=0.0, max=1.0)
    return mean_nonblank, target_density


def _compute_overemit_loss(
    *,
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    density_scale: float,
    density_margin: float,
    power: float,
) -> Tuple[torch.Tensor, float, float]:
    mean_nonblank, target_density = _nonblank_density_stats(
        student_logits=student_logits,
        student_lengths=student_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id,
    )
    allowed_density = (
        target_density * max(0.0, float(density_scale)) + max(0.0, float(density_margin))
    ).clamp(max=1.0)
    excess_density = torch.relu(mean_nonblank - allowed_density)
    effective_power = max(1.0, float(power))
    if math.isclose(effective_power, 1.0, rel_tol=0.0, abs_tol=1e-8):
        penalty = excess_density
    else:
        penalty = excess_density.pow(effective_power)
    return penalty.mean(), float(mean_nonblank.mean().item()), float(target_density.mean().item())


def _compute_underemit_loss(
    *,
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    density_scale: float,
    density_margin: float,
    min_density: float,
    power: float,
) -> Tuple[torch.Tensor, float, float]:
    mean_nonblank, target_density = _nonblank_density_stats(
        student_logits=student_logits,
        student_lengths=student_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id,
    )
    required_density = (
        target_density * max(0.0, float(density_scale)) - max(0.0, float(density_margin))
    ).clamp(min=max(0.0, float(min_density)), max=1.0)
    deficit_density = torch.relu(required_density - mean_nonblank)
    effective_power = max(1.0, float(power))
    if math.isclose(effective_power, 1.0, rel_tol=0.0, abs_tol=1e-8):
        penalty = deficit_density
    else:
        penalty = deficit_density.pow(effective_power)
    return penalty.mean(), float(mean_nonblank.mean().item()), float(target_density.mean().item())


def _checkpoint_selection_score(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    selection_cfg = cfg.get("checkpoint_selection", {})
    if not isinstance(selection_cfg, dict):
        selection_cfg = {}
    empty_penalty = float(selection_cfg.get("empty_pred_penalty", 1.0))
    blank_penalty = float(selection_cfg.get("blank_ratio_penalty", 0.0))
    short_penalty = float(selection_cfg.get("short_pred_penalty", 0.0))
    long_penalty = float(selection_cfg.get("long_pred_penalty", 0.0))
    length_deviation_penalty = float(selection_cfg.get("length_deviation_penalty", 0.0))
    repeat_penalty = float(selection_cfg.get("repeat_ratio_penalty", 0.0))
    dominant_penalty = float(selection_cfg.get("dominant_char_penalty", 0.0))
    accuracy_bonus = float(selection_cfg.get("accuracy_bonus", 0.0))
    return (
        float(metrics["wer"])
        + empty_penalty * float(metrics.get("empty_pred_ratio", 0.0))
        + blank_penalty * float(metrics.get("blank_ratio", 0.0))
        + short_penalty * float(metrics.get("short_pred_ratio", 0.0))
        + long_penalty * float(metrics.get("long_pred_ratio", 0.0))
        + length_deviation_penalty * float(metrics.get("length_deviation_ratio", 0.0))
        + repeat_penalty * float(metrics.get("adjacent_repeat_ratio", 0.0))
        + dominant_penalty * float(metrics.get("dominant_char_ratio", 0.0))
        - accuracy_bonus * float(metrics.get("accuracy", 0.0))
    )


def _progress_speed_line(
    stage: str,
    seed: int,
    global_step: int,
    total_steps: int,
    start_ts: float,
    effective_batch_size: int,
) -> str:
    elapsed = max(1e-9, time.time() - start_ts)
    steps_per_sec = float(global_step / elapsed)
    samples_per_sec = float(steps_per_sec * max(1, effective_batch_size))
    if total_steps > 0 and steps_per_sec > 0.0:
        remaining = max(total_steps - global_step, 0)
        eta_sec = float(remaining / steps_per_sec)
        total_est_sec = float(elapsed + eta_sec)
        time_text = f"{_format_hms(elapsed)}/{_format_hms(total_est_sec)}"
        eta_text = _format_hms(eta_sec)
    else:
        time_text = f"{_format_hms(elapsed)}/N/A"
        eta_text = "N/A"
    return (
        f"[SPEED][{stage}] seed={seed} step={global_step}/{total_steps} "
        f"time={time_text} eta={eta_text} "
        f"steps/s={steps_per_sec:.2f} samples/s={samples_per_sec:.2f}"
    )


def _progress_postfix(
    global_step: int,
    total_steps: int,
    start_ts: float,
    effective_batch_size: int,
) -> str:
    elapsed = max(1e-9, time.time() - start_ts)
    steps_per_sec = float(global_step / elapsed) if global_step > 0 else 0.0
    samples_per_sec = float(steps_per_sec * max(1, effective_batch_size))
    if total_steps > 0 and steps_per_sec > 0.0:
        remaining = max(total_steps - global_step, 0)
        eta_sec = float(remaining / steps_per_sec)
        total_est_sec = float(elapsed + eta_sec)
        return (
            f"time={_format_hms(elapsed)}/{_format_hms(total_est_sec)} "
            f"eta={_format_hms(eta_sec)} "
            f"steps/s={steps_per_sec:.2f} samples/s={samples_per_sec:.2f}"
        )
    return (
        f"time={_format_hms(elapsed)}/N/A eta=N/A "
        f"steps/s={steps_per_sec:.2f} samples/s={samples_per_sec:.2f}"
    )


def _planned_training_schedule(
    dataset_size: int,
    batch_size: int,
    epochs: int,
    max_steps: int,
    stage: str,
    seed: int,
) -> Tuple[int, int, int]:
    steps_per_epoch = max(1, math.ceil(max(0, dataset_size) / max(1, batch_size)))
    requested_epochs = max(1, int(epochs))
    requested_max_steps = max(1, int(max_steps))
    effective_epochs = max(requested_epochs, math.ceil(requested_max_steps / steps_per_epoch))
    total_steps = min(requested_max_steps, effective_epochs * steps_per_epoch)
    if effective_epochs > requested_epochs:
        print(
            f"[{stage}][WARN] seed={seed} epochs={requested_epochs} insuffisant pour "
            f"atteindre max_steps={requested_max_steps} avec steps_per_epoch={steps_per_epoch}. "
            f"Extension automatique à effective_epochs={effective_epochs}."
        )
    return steps_per_epoch, total_steps, effective_epochs


def _optimizer_total_steps(total_micro_steps: int, grad_acc: int) -> int:
    return max(1, math.ceil(max(1, int(total_micro_steps)) / max(1, int(grad_acc))))


def _seeded_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        generator=generator,
        drop_last=False,
    )


def build_encoder(cfg: Dict[str, Any]) -> AudioTransformerEncoder:
    m = cfg["model"]
    audio = cfg["audio"]
    return AudioTransformerEncoder(
        n_mels=int(audio["n_mels"]),
        dim=int(m["dim"]),
        depth=int(m["depth"]),
        num_heads=int(m["num_heads"]),
        mlp_ratio=float(m["mlp_ratio"]),
        dropout=float(m["dropout"]),
        patch_size=int(m["patch_time"]),
        max_len=int(m["max_len"]),
        pos_embed=str(m["pos_embed"]),
        patch_strategy=str(m["patch_strategy"]),
        patch_freq=int(m["patch_freq"]),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_start_factor: float = 0.1,
    min_lr_ratio: float = 0.0,
):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    warmup_start_factor = float(max(1e-4, min(1.0, warmup_start_factor)))
    min_lr_ratio = float(max(0.0, min(1.0, min_lr_ratio)))

    def lr_lambda(step_idx: int) -> float:
        step_idx = max(0, int(step_idx))
        if warmup_steps > 0 and step_idx < warmup_steps:
            if warmup_steps == 1:
                return 1.0
            progress = float(step_idx + 1) / float(warmup_steps)
            return warmup_start_factor + progress * (1.0 - warmup_start_factor)

        cosine_steps = max(1, total_steps - warmup_steps)
        progress = float(step_idx - warmup_steps + 1) / float(cosine_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _checkpoint_root(cfg: Dict[str, Any], stage: str, seed: int) -> Path:
    exp = cfg["experiment"]
    exp_id = exp.get("id")
    base = Path(exp["output_dir"]) / "checkpoints"
    if exp_id:
        base = base / str(exp_id)
    return base / stage / f"seed_{seed}"


def run_pretrain_seed(
    cfg: Dict[str, Any],
    seed: int,
    dataset: torch.utils.data.Dataset,
    force_continue_completed: bool = False,
) -> Tuple[Path, Dict[str, float]]:
    device = _device()
    out_root = _checkpoint_root(cfg, stage="pretrain", seed=seed)
    out_root.mkdir(parents=True, exist_ok=True)
    run_completed = out_root / "run_completed.txt"
    if run_completed.exists() and not force_continue_completed:
        final_encoder = out_root / "encoder_final.pt"
        if not final_encoder.exists():
            raise FileNotFoundError(f"Missing encoder checkpoint: {final_encoder}")
        return final_encoder, {"status": 1.0}

    training_cfg = cfg["training"]["pretrain"]
    global_cfg = cfg["training"]
    checkpoint_every = int(cfg["experiment"]["checkpoint_every_steps"])
    keep_last = int(cfg["experiment"].get("keep_last_checkpoints", 2))
    amp_enabled = bool(global_cfg["amp"]) and torch.cuda.is_available()

    encoder = build_encoder(cfg)
    mae = AudioMAEPretrain(
        encoder=encoder,
        n_mels=int(cfg["audio"]["n_mels"]),
        dec_dim=int(cfg["model"]["mae_decoder_dim"]),
        dec_depth=int(cfg["model"]["mae_decoder_depth"]),
        dec_heads=int(cfg["model"]["mae_decoder_heads"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        mae.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    steps_per_epoch, total_steps, effective_epochs = _planned_training_schedule(
        dataset_size=len(dataset),
        batch_size=int(training_cfg["batch_size"]),
        epochs=int(training_cfg["epochs"]),
        max_steps=int(training_cfg["max_steps"]),
        stage="PRETRAIN",
        seed=seed,
    )
    optimizer_total_steps = _optimizer_total_steps(total_steps, grad_acc=int(training_cfg["grad_accum_steps"]))
    scheduler = _build_scheduler(
        optimizer,
        total_steps=optimizer_total_steps,
        warmup_steps=int(training_cfg.get("warmup_steps", 0)),
        warmup_start_factor=float(training_cfg.get("warmup_start_factor", 0.1)),
        min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.0)),
    )
    scaler = _build_grad_scaler(device=device, enabled=amp_enabled)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    latest = find_latest_checkpoint(out_root) if force_continue_completed else None
    if latest is not None:
        payload = load_checkpoint(latest, mae, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(payload["epoch"])
        start_step_in_epoch = int(payload["step_in_epoch"])
        global_step = int(payload["global_step"])

    num_workers = int(cfg["data"]["num_workers"])
    grad_acc = int(training_cfg["grad_accum_steps"])
    grad_clip = float(cfg["training"]["grad_clip_norm"])
    mask_ratio = float(cfg["model"]["mae_mask_ratio"])
    effective_batch_size = int(training_cfg["batch_size"]) * max(1, grad_acc)
    log_every_steps = int(cfg["training"].get("log_every_steps", 25))

    total_loss = 0.0
    total_count = 0
    start_ts = time.time()
    reset_peak_memory()
    pretrain_peak_gpu_mem_mb = 0.0
    accum_counter = 0
    print(
        f"[PRETRAIN] seed={seed} max_steps={int(training_cfg['max_steps'])} "
        f"micro_batch={int(training_cfg['batch_size'])} grad_acc={grad_acc} "
        f"effective_batch={effective_batch_size} "
        f"optimizer_steps={optimizer_total_steps} warmup_steps={int(training_cfg.get('warmup_steps', 0))}"
    )

    use_tqdm = _use_tqdm()
    progress = tqdm(
        total=total_steps,
        initial=min(global_step, total_steps),
        desc=f"pretrain seed={seed}",
        leave=False,
        mininterval=5.0,
        disable=not use_tqdm,
    )
    for epoch in range(start_epoch, effective_epochs):
        loader = _seeded_loader(
            dataset=dataset,
            batch_size=int(training_cfg["batch_size"]),
            num_workers=num_workers,
            collate_fn=pad_collate,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        )
        if use_tqdm:
            progress.set_description(f"pretrain seed={seed} epoch={epoch}")
        mae.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_step_in_epoch:
                continue
            x = batch["x_logmel"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            with torch.no_grad():
                patch_info = mae.encoder.patch_embedding.patchify(x, lengths=lengths)
                seq_len = int(patch_info["patches"].shape[1])
                valid_token_mask = None
                if patch_info.get("key_padding_mask") is not None:
                    valid_token_mask = ~patch_info["key_padding_mask"]
                mask = make_mae_mask(
                    batch_size=x.shape[0],
                    seq_len=seq_len,
                    mask_ratio=mask_ratio,
                    device=device,
                    valid_token_mask=valid_token_mask,
                )

            with _autocast_context(device=device, enabled=amp_enabled):
                loss = mae(x, mask, lengths=lengths)
                loss_scaled = loss / grad_acc
            scaler.scale(loss_scaled).backward()

            global_step += 1
            accum_counter += 1
            if use_tqdm:
                progress.update(1)
                progress.set_postfix_str(
                    _progress_postfix(
                        global_step=global_step,
                        total_steps=total_steps,
                        start_ts=start_ts,
                        effective_batch_size=effective_batch_size,
                    )
                )
            total_loss += float(loss.item())
            total_count += 1
            pretrain_peak_gpu_mem_mb = max(pretrain_peak_gpu_mem_mb, peak_memory_mb())

            if accum_counter >= grad_acc or global_step >= int(training_cfg["max_steps"]):
                scaler.unscale_(optimizer)
                clip_grad_norm_(mae.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum_counter = 0

            if global_step % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir=out_root,
                    model=mae,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    step_in_epoch=batch_idx + 1,
                    tag="step",
                    keep_last_checkpoints=keep_last,
                )

            if (not use_tqdm) and global_step % max(1, log_every_steps) == 0:
                print(
                    _progress_speed_line(
                        stage="PRETRAIN",
                        seed=seed,
                        global_step=global_step,
                        total_steps=total_steps,
                        start_ts=start_ts,
                        effective_batch_size=effective_batch_size,
                    )
                )

            if global_step >= int(training_cfg["max_steps"]):
                break

        start_step_in_epoch = 0
        save_checkpoint(
            checkpoint_dir=out_root,
            model=mae,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            step_in_epoch=0,
            tag="epoch",
            keep_last_checkpoints=keep_last,
        )
        if global_step >= int(training_cfg["max_steps"]):
            break
    if use_tqdm:
        progress.close()

    final_encoder = out_root / "encoder_final.pt"
    torch.save({"encoder_state_dict": mae.encoder.state_dict()}, final_encoder)
    with open(run_completed, "w", encoding="utf-8") as handle:
        handle.write("done\n")

    metrics = {
        "pretrain_loss": float(total_loss / max(1, total_count)),
        "pretrain_runtime_sec": float(time.time() - start_ts),
        "pretrain_peak_gpu_mem_mb": float(max(pretrain_peak_gpu_mem_mb, peak_memory_mb())),
        **{f"pretrain_{k}_params": float(v) for k, v in count_parameters(mae).items()},
    }
    return final_encoder, metrics


def _load_encoder_from_pretrain(encoder: AudioTransformerEncoder, pretrain_encoder_path: Path, device: torch.device) -> None:
    payload = torch.load(pretrain_encoder_path, map_location=device)
    state = payload["encoder_state_dict"]
    encoder.load_state_dict(state)


@torch.no_grad()
def _run_ctc_diagnostic_pass(
    *,
    model: AudioTransformerCTC,
    dataset: torch.utils.data.Dataset,
    tokenizer: CharCTCTokenizer,
    cfg: Dict[str, Any],
    seed: int,
    loader_epoch: int,
    stage_label: str,
    artifact_path: Optional[Path] = None,
    extra_artifact_paths: Sequence[Path] = (),
    teacher: Optional[Any] = None,
    dataset_label: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    payload_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = _device()
    num_workers = int(cfg["data"]["num_workers"])
    effective_batch_size = int(batch_size or cfg["training"]["finetune"]["batch_size"])
    loader = _seeded_loader(
        dataset=dataset,
        batch_size=effective_batch_size,
        num_workers=num_workers,
        collate_fn=partial(ctc_collate, tokenizer=tokenizer),
        seed=seed,
        epoch=loader_epoch,
        shuffle=False,
    )
    model.eval()
    predictions: List[str] = []
    references: List[str] = []
    diagnostic_totals: Dict[str, Any] = {}
    diagnostic_examples: List[Dict[str, Any]] = []
    max_examples = _diagnostics_examples_limit(cfg)
    max_frames = _diagnostics_frames_limit(cfg)
    topk_tokens = _diagnostics_topk_tokens(cfg)
    start_ts = time.time()
    reset_peak_memory()
    for batch in loader:
        x = batch["x_logmel"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)
        logits, out_lengths, student_features = model(x, lengths, return_features=True)
        teacher_forensics = None
        if teacher is not None:
            teacher_output = teacher.forward_teacher(
                x_logmel=x,
                lengths=lengths,
                waveforms=waveforms if getattr(teacher, "requires_waveform", False) else None,
                waveform_lengths=waveform_lengths if getattr(teacher, "requires_waveform", False) else None,
                temperature=float(_distillation_cfg(cfg).get("temperature", 1.0)),
            )
            teacher_forensics = _prepare_teacher_forensics(
                student_logits=logits,
                student_lengths=out_lengths,
                student_features=student_features,
                teacher_output=teacher_output,
            )
        batch_predictions, batch_totals, batch_examples = collect_ctc_batch_diagnostics(
            logits=logits,
            out_lengths=out_lengths,
            target_lengths=batch.get("target_lengths"),
            references=batch["transcripts"],
            tokenizer=tokenizer,
            max_examples=max(0, max_examples - len(diagnostic_examples)),
            max_frames=max_frames,
            topk_tokens=topk_tokens,
            sample_ids=batch.get("sample_ids"),
            source_datasets=batch.get("source_datasets"),
            source_splits=batch.get("source_splits"),
            waveform_lengths=batch.get("waveform_lengths"),
            sample_rate=int(cfg["audio"]["sample_rate"]),
            student_features=student_features,
            teacher_forensics=teacher_forensics,
        )
        predictions.extend(batch_predictions)
        references.extend(batch["transcripts"])
        _merge_diagnostic_totals(diagnostic_totals, batch_totals)
        diagnostic_examples.extend(batch_examples)
    runtime = max(1e-9, time.time() - start_ts)
    diagnostic_metrics = finalize_ctc_diagnostics(
        diagnostic_totals,
        tokenizer=tokenizer,
        expected_num_unique_samples=len(dataset),
    )
    result: Dict[str, Any] = {
        "wer": compute_wer(predictions, references),
        "accuracy": compute_char_accuracy(predictions, references),
        "eval_runtime_sec": float(runtime),
        "eval_samples_per_sec": float(len(references) / runtime),
        "eval_peak_gpu_mem_mb": peak_memory_mb(),
        **diagnostic_metrics,
    }
    if artifact_path is not None or extra_artifact_paths:
        payload: Dict[str, Any] = {
            "seed": int(seed),
            "stage": stage_label,
            "metrics": result,
            "examples": diagnostic_examples,
        }
        if dataset_label is not None:
            payload["dataset"] = dataset_label
        if checkpoint_path is not None:
            payload["checkpoint"] = str(checkpoint_path)
        if payload_metadata:
            payload.update(payload_metadata)
        _write_json_artifacts(artifact_path, payload, extra_artifact_paths=extra_artifact_paths)
    return result


@torch.no_grad()
def evaluate_ctc(
    model: AudioTransformerCTC,
    dataset: torch.utils.data.Dataset,
    tokenizer: CharCTCTokenizer,
    cfg: Dict[str, Any],
    seed: int,
    epoch: int,
    artifact_path: Optional[Path] = None,
    *,
    teacher: Optional[Any] = None,
    stage_label: str = "validation",
    extra_artifact_paths: Sequence[Path] = (),
) -> Dict[str, Any]:
    result = _run_ctc_diagnostic_pass(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        seed=seed,
        loader_epoch=epoch + 10_000,
        stage_label=stage_label,
        artifact_path=artifact_path,
        extra_artifact_paths=extra_artifact_paths,
        teacher=teacher,
        payload_metadata={"epoch": int(epoch)},
    )
    result["epoch"] = float(epoch)
    return result


def run_finetune_seed(
    cfg: Dict[str, Any],
    seed: int,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    pretrain_encoder_path: Optional[Path],
    tokenizer: CharCTCTokenizer,
    force_continue_completed: bool = False,
) -> Tuple[Path, Dict[str, float]]:
    device = _device()
    out_root = _checkpoint_root(cfg, stage="finetune", seed=seed)
    out_root.mkdir(parents=True, exist_ok=True)
    run_completed = out_root / "run_completed.txt"
    if run_completed.exists() and not force_continue_completed:
        final_model = out_root / "ctc_final.pt"
        if not final_model.exists():
            raise FileNotFoundError(f"Missing fine-tuned checkpoint: {final_model}")
        return final_model, {"status": 1.0}

    training_cfg = cfg["training"]["finetune"]
    global_cfg = cfg["training"]
    checkpoint_every = int(cfg["experiment"]["checkpoint_every_steps"])
    keep_last = int(cfg["experiment"].get("keep_last_checkpoints", 2))
    amp_enabled = bool(global_cfg["amp"]) and torch.cuda.is_available()

    teacher = build_teacher(cfg, tokenizer=tokenizer, device=device)
    distill_projection_dim = _resolve_distill_projection_dim(cfg, teacher)
    model_cfg_for_checkpoint = dict(cfg["model"])
    if distill_projection_dim is not None:
        model_cfg_for_checkpoint["distill_projection_dim"] = int(distill_projection_dim)
    encoder = build_encoder(cfg)
    if pretrain_encoder_path is not None:
        _load_encoder_from_pretrain(encoder, pretrain_encoder_path, device)
    model = AudioTransformerCTC(
        encoder=encoder,
        vocab_size=tokenizer.vocab_size,
        distill_projection_dim=distill_projection_dim,
    ).to(device)
    distill_cfg = _distillation_cfg(cfg)
    distill_enabled = teacher is not None
    lambda_ctc = float(distill_cfg.get("lambda_ctc", 1.0))
    lambda_kd = float(distill_cfg.get("lambda_kd", 0.0))
    kd_warmup_steps = _distillation_warmup_steps(cfg)
    kd_warmup_start_factor = _distillation_warmup_start_factor(cfg)
    temperature = float(distill_cfg.get("temperature", 1.0))
    kd_informative_only = bool(distill_cfg.get("informative_only", False))
    kd_min_nonblank_prob = float(distill_cfg.get("min_nonblank_prob", 0.0))
    kd_require_nonblank_argmax = bool(distill_cfg.get("require_nonblank_argmax", True))
    overemit_cfg = _overemit_cfg(cfg)
    overemit_enabled = bool(overemit_cfg.get("enabled", False))
    lambda_overemit = float(overemit_cfg.get("lambda", 0.0))
    overemit_density_scale = float(overemit_cfg.get("density_scale", 1.0))
    overemit_density_margin = float(overemit_cfg.get("density_margin", 0.0))
    overemit_power = float(overemit_cfg.get("power", 2.0))
    underemit_cfg = _underemit_cfg(cfg)
    underemit_enabled = bool(underemit_cfg.get("enabled", False))
    lambda_underemit = float(underemit_cfg.get("lambda", 0.0))
    underemit_density_scale = float(underemit_cfg.get("density_scale", 1.0))
    underemit_density_margin = float(underemit_cfg.get("density_margin", 0.0))
    underemit_min_density = float(underemit_cfg.get("min_density", 0.0))
    underemit_power = float(underemit_cfg.get("power", 2.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    steps_per_epoch, total_steps, effective_epochs = _planned_training_schedule(
        dataset_size=len(train_dataset),
        batch_size=int(training_cfg["batch_size"]),
        epochs=int(training_cfg["epochs"]),
        max_steps=int(training_cfg["max_steps"]),
        stage="FINETUNE",
        seed=seed,
    )
    optimizer_total_steps = _optimizer_total_steps(total_steps, grad_acc=int(training_cfg["grad_accum_steps"]))
    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    warmup_start_factor = float(training_cfg.get("warmup_start_factor", 0.1))
    min_lr_ratio = float(training_cfg.get("min_lr_ratio", 0.0))
    scheduler = _build_scheduler(
        optimizer,
        total_steps=optimizer_total_steps,
        warmup_steps=warmup_steps,
        warmup_start_factor=warmup_start_factor,
        min_lr_ratio=min_lr_ratio,
    )
    scaler = _build_grad_scaler(device=device, enabled=amp_enabled)
    ctc_loss_fn = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_wer: Optional[float] = None
    best_score: Optional[float] = None
    best_empty_pred_ratio: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs_without_improvement = 0
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 0))
    early_stopping_min_epochs = int(training_cfg.get("early_stopping_min_epochs", 0))
    early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
    early_stopped = False
    early_stop_epoch: Optional[int] = None
    latest = find_latest_checkpoint(out_root) if force_continue_completed else None
    if latest is not None:
        payload = load_checkpoint(latest, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(payload["epoch"])
        start_step_in_epoch = int(payload["step_in_epoch"])
        global_step = int(payload["global_step"])
        payload_extra = payload.get("extra") or {}
        if payload_extra.get("best_valid_wer") is not None:
            best_wer = float(payload_extra["best_valid_wer"])
        elif payload.get("best_metric") is not None:
            best_wer = float(payload["best_metric"])
        if payload_extra.get("best_selection_score") is not None:
            best_score = float(payload_extra["best_selection_score"])
        elif payload.get("best_metric") is not None:
            best_score = float(payload["best_metric"])
        if payload_extra.get("best_valid_empty_pred_ratio") is not None:
            best_empty_pred_ratio = float(payload_extra["best_valid_empty_pred_ratio"])
        if payload_extra.get("best_valid_epoch") is not None:
            best_epoch = int(payload_extra["best_valid_epoch"])
        if payload_extra.get("epochs_without_improvement") is not None:
            epochs_without_improvement = int(payload_extra["epochs_without_improvement"])

    num_workers = int(cfg["data"]["num_workers"])
    grad_acc = int(training_cfg["grad_accum_steps"])
    grad_clip = float(cfg["training"]["grad_clip_norm"])
    effective_batch_size = int(training_cfg["batch_size"]) * max(1, grad_acc)
    train_probe_dataset = _make_train_probe_subset(train_dataset, cfg) if _forensics_enabled(cfg) else None
    optimizer_steps_completed = 0
    log_every_steps = int(cfg["training"].get("log_every_steps", 25))
    train_start_ts = time.time()
    reset_peak_memory()
    running_loss = 0.0
    running_count = 0
    running_ctc_loss = 0.0
    running_kd_loss = 0.0
    running_kd_lambda_effective = 0.0
    running_kd_active_ratio = 0.0
    running_overemit_loss = 0.0
    running_underemit_loss = 0.0
    running_nonblank_density = 0.0
    running_target_density = 0.0
    train_sample_count = 0
    train_invalid_length_count = 0
    train_out_length_sum = 0.0
    train_target_length_sum = 0.0
    train_length_margin_sum = 0.0
    train_peak_gpu_mem_mb = 0.0
    eval_peak_gpu_mem_mb = 0.0
    emitted_invalid_length_warning = False
    accum_counter = 0
    print(
        f"[FINETUNE] seed={seed} max_steps={int(training_cfg['max_steps'])} "
        f"micro_batch={int(training_cfg['batch_size'])} grad_acc={grad_acc} "
        f"effective_batch={effective_batch_size} "
        f"distillation={distill_enabled} "
        f"lambda_ctc={lambda_ctc:.3f} lambda_kd={lambda_kd:.3f} "
        f"kd_warmup_steps={kd_warmup_steps} kd_warmup_start={kd_warmup_start_factor:.3f} "
        f"informative_only={kd_informative_only} min_nonblank_prob={kd_min_nonblank_prob:.3f} "
        f"overemit={overemit_enabled} lambda_overemit={lambda_overemit:.3f} "
        f"underemit={underemit_enabled} lambda_underemit={lambda_underemit:.3f} "
        f"optimizer_steps={optimizer_total_steps} warmup_steps={warmup_steps} "
        f"lr={float(training_cfg['learning_rate']):.6f} patience={early_stopping_patience}"
    )

    use_tqdm = _use_tqdm()
    progress = tqdm(
        total=total_steps,
        initial=min(global_step, total_steps),
        desc=f"finetune seed={seed}",
        leave=False,
        mininterval=5.0,
        disable=not use_tqdm,
    )
    for epoch in range(start_epoch, effective_epochs):
        loader = _seeded_loader(
            dataset=train_dataset,
            batch_size=int(training_cfg["batch_size"]),
            num_workers=num_workers,
            collate_fn=partial(ctc_collate, tokenizer=tokenizer),
            seed=seed,
            epoch=epoch,
            shuffle=True,
        )
        if use_tqdm:
            progress.set_description(f"finetune seed={seed} epoch={epoch}")
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_step_in_epoch:
                continue

            x = batch["x_logmel"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            waveform_lengths = batch["waveform_lengths"].to(device, non_blocking=True)

            with _autocast_context(device=device, enabled=amp_enabled):
                logits, out_lengths, student_features = model(x, lengths, return_features=True)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                ctc_loss = ctc_loss_fn(log_probs, targets, out_lengths, target_lengths)
                kd_loss = logits.new_zeros(())
                kd_active_ratio = 0.0
                overemit_loss = logits.new_zeros(())
                underemit_loss = logits.new_zeros(())
                mean_nonblank_density = 0.0
                mean_target_density = 0.0
                effective_lambda_kd = _effective_lambda_kd(cfg, lambda_kd, optimizer_steps_completed) if distill_enabled else 0.0
                if distill_enabled and effective_lambda_kd > 0.0:
                    teacher_output = teacher.forward_teacher(
                        x_logmel=x,
                        lengths=lengths,
                        waveforms=waveforms if getattr(teacher, "requires_waveform", False) else None,
                        waveform_lengths=waveform_lengths if getattr(teacher, "requires_waveform", False) else None,
                        temperature=temperature,
                    )
                    kd_loss, kd_active_ratio = _compute_distillation_loss(
                        student_logits=logits,
                        student_lengths=out_lengths,
                        student_features=student_features,
                        teacher_output=teacher_output,
                        temperature=temperature,
                        informative_only=kd_informative_only,
                        min_nonblank_prob=kd_min_nonblank_prob,
                        require_nonblank_argmax=kd_require_nonblank_argmax,
                    )
                if overemit_enabled and lambda_overemit > 0.0:
                    overemit_loss, mean_nonblank_density, mean_target_density = _compute_overemit_loss(
                        student_logits=logits,
                        student_lengths=out_lengths,
                        target_lengths=target_lengths,
                        blank_id=tokenizer.blank_id,
                        density_scale=overemit_density_scale,
                        density_margin=overemit_density_margin,
                        power=overemit_power,
                    )
                if underemit_enabled and lambda_underemit > 0.0:
                    underemit_loss, mean_nonblank_density, mean_target_density = _compute_underemit_loss(
                        student_logits=logits,
                        student_lengths=out_lengths,
                        target_lengths=target_lengths,
                        blank_id=tokenizer.blank_id,
                        density_scale=underemit_density_scale,
                        density_margin=underemit_density_margin,
                        min_density=underemit_min_density,
                        power=underemit_power,
                    )
                loss = (
                    lambda_ctc * ctc_loss
                    + effective_lambda_kd * kd_loss
                    + lambda_overemit * overemit_loss
                    + lambda_underemit * underemit_loss
                )
                loss_scaled = loss / grad_acc
            scaler.scale(loss_scaled).backward()

            global_step += 1
            accum_counter += 1
            if use_tqdm:
                progress.update(1)
                progress.set_postfix_str(
                    _progress_postfix(
                        global_step=global_step,
                        total_steps=total_steps,
                        start_ts=train_start_ts,
                        effective_batch_size=effective_batch_size,
                    )
                )
            running_loss += float(loss.item())
            running_ctc_loss += float(ctc_loss.item())
            running_kd_loss += float(kd_loss.item()) if isinstance(kd_loss, torch.Tensor) else float(kd_loss)
            running_kd_lambda_effective += float(effective_lambda_kd)
            running_kd_active_ratio += float(kd_active_ratio)
            running_overemit_loss += float(overemit_loss.item()) if isinstance(overemit_loss, torch.Tensor) else float(overemit_loss)
            running_underemit_loss += float(underemit_loss.item()) if isinstance(underemit_loss, torch.Tensor) else float(underemit_loss)
            running_nonblank_density += float(mean_nonblank_density)
            running_target_density += float(mean_target_density)
            running_count += 1
            batch_size = int(target_lengths.shape[0])
            invalid_length_mask = out_lengths < target_lengths
            train_sample_count += batch_size
            train_invalid_length_count += int(invalid_length_mask.sum().item())
            train_out_length_sum += float(out_lengths.sum().item())
            train_target_length_sum += float(target_lengths.sum().item())
            train_length_margin_sum += float((out_lengths - target_lengths).sum().item())
            train_peak_gpu_mem_mb = max(train_peak_gpu_mem_mb, peak_memory_mb())

            if invalid_length_mask.any() and not emitted_invalid_length_warning:
                emitted_invalid_length_warning = True
                min_margin = int((out_lengths - target_lengths).min().item())
                print(
                    f"[FINETUNE][WARN] seed={seed} global_step={global_step} "
                    f"{int(invalid_length_mask.sum().item())}/{batch_size} échantillons ont "
                    f"out_lengths < target_lengths (min_margin={min_margin})."
                )

            if accum_counter >= grad_acc or global_step >= int(training_cfg["max_steps"]):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum_counter = 0
                optimizer_steps_completed += 1

            if global_step % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir=out_root,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    step_in_epoch=batch_idx + 1,
                    best_metric=best_score,
                    tag="step",
                    keep_last_checkpoints=keep_last,
                )

            if (not use_tqdm) and global_step % max(1, log_every_steps) == 0:
                print(
                    _progress_speed_line(
                        stage="FINETUNE",
                        seed=seed,
                        global_step=global_step,
                        total_steps=total_steps,
                        start_ts=train_start_ts,
                        effective_batch_size=effective_batch_size,
                    )
                )

            if global_step >= int(training_cfg["max_steps"]):
                break

        start_step_in_epoch = 0
        eval_metrics = evaluate_ctc(
            model,
            valid_dataset,
            tokenizer,
            cfg,
            seed=seed,
            epoch=epoch,
            artifact_path=_diagnostics_artifact_path(cfg, "validation", seed, f"epoch_{epoch:03d}"),
        )
        eval_peak_gpu_mem_mb = max(eval_peak_gpu_mem_mb, float(eval_metrics["eval_peak_gpu_mem_mb"]))
        current_wer = float(eval_metrics["wer"])
        current_score = _checkpoint_selection_score(eval_metrics, cfg)
        print(
            f"[VALID] seed={seed} epoch={epoch} wer={current_wer:.4f} "
            f"accuracy={float(eval_metrics['accuracy']):.4f} "
            f"blank_ratio={float(eval_metrics['blank_ratio']):.3f} "
            f"empty_pred_ratio={float(eval_metrics['empty_pred_ratio']):.3f} "
            f"pred_to_ref_char_ratio={float(eval_metrics['pred_to_ref_char_ratio']):.3f} "
            f"invalid_length_ratio={float(eval_metrics['invalid_length_ratio']):.3f} "
            f"selection_score={current_score:.4f} lr={_current_lr(optimizer):.6f}"
        )
        _maybe_warn_ctc(stage="VALID", metrics=eval_metrics, cfg=cfg)
        improved = (
            best_score is None
            or current_score < (float(best_score) - early_stopping_min_delta)
            or (
                best_score is not None
                and math.isclose(current_score, best_score, rel_tol=0.0, abs_tol=max(1e-12, early_stopping_min_delta))
                and (best_wer is None or current_wer < best_wer)
            )
        )
        if improved:
            best_wer = current_wer
            best_score = current_score
            best_empty_pred_ratio = float(eval_metrics["empty_pred_ratio"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "blank_id": tokenizer.blank_id,
                    "student_config": {
                        "model": dict(model_cfg_for_checkpoint),
                        "audio": {"n_mels": int(cfg["audio"]["n_mels"])},
                    },
                    "selection_score": float(current_score),
                    "selection_components": {
                        "wer": float(current_wer),
                        "empty_pred_ratio": float(eval_metrics["empty_pred_ratio"]),
                        "blank_ratio": float(eval_metrics["blank_ratio"]),
                        "pred_to_ref_char_ratio": float(eval_metrics["pred_to_ref_char_ratio"]),
                        "short_pred_ratio": float(eval_metrics.get("short_pred_ratio", 0.0)),
                        "long_pred_ratio": float(eval_metrics.get("long_pred_ratio", 0.0)),
                        "length_deviation_ratio": float(eval_metrics.get("length_deviation_ratio", 0.0)),
                        "adjacent_repeat_ratio": float(eval_metrics.get("adjacent_repeat_ratio", 0.0)),
                        "dominant_char_ratio": float(eval_metrics.get("dominant_char_ratio", 0.0)),
                        "accuracy": float(eval_metrics["accuracy"]),
                    },
                },
                out_root / "ctc_best.pt",
            )
        else:
            if (epoch + 1) >= max(1, early_stopping_min_epochs):
                epochs_without_improvement += 1

        save_checkpoint(
            checkpoint_dir=out_root,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            step_in_epoch=0,
            best_metric=best_score,
            extra={
                "valid_wer": current_wer,
                "valid_accuracy": float(eval_metrics["accuracy"]),
                "valid_empty_pred_ratio": float(eval_metrics["empty_pred_ratio"]),
                "valid_selection_score": float(current_score),
                "best_valid_wer": None if best_wer is None else float(best_wer),
                "best_valid_empty_pred_ratio": (
                    None if best_empty_pred_ratio is None else float(best_empty_pred_ratio)
                ),
                "best_selection_score": None if best_score is None else float(best_score),
                "best_valid_epoch": None if best_epoch is None else int(best_epoch),
                "epochs_without_improvement": int(epochs_without_improvement),
            },
            tag="epoch",
            keep_last_checkpoints=keep_last,
        )
        if (
            early_stopping_patience > 0
            and (epoch + 1) >= max(1, early_stopping_min_epochs)
            and epochs_without_improvement >= early_stopping_patience
        ):
            early_stopped = True
            early_stop_epoch = int(epoch)
            print(
                f"[FINETUNE][EARLY-STOP] seed={seed} epoch={epoch} "
                f"best_epoch={best_epoch} best_score={float(best_score):.4f} "
                f"patience={early_stopping_patience}"
            )
            break
        if global_step >= int(training_cfg["max_steps"]):
            break
    if use_tqdm:
        progress.close()

    final_model = out_root / "ctc_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "blank_id": tokenizer.blank_id,
            "student_config": {
                "model": dict(model_cfg_for_checkpoint),
                "audio": {"n_mels": int(cfg["audio"]["n_mels"])},
            },
        },
        final_model,
    )
    with open(run_completed, "w", encoding="utf-8") as handle:
        handle.write("done\n")

    final_valid = evaluate_ctc(
        model,
        valid_dataset,
        tokenizer,
        cfg,
        seed=seed,
        epoch=99_999,
        artifact_path=_diagnostics_artifact_path(cfg, "validation", seed, "final"),
        teacher=teacher if _forensics_enabled(cfg) else None,
        extra_artifact_paths=(
            [
                Path(cfg["experiment"]["results_dir"]) / "diagnostics" / f"forensics_valid_seed_{seed}.json"
            ]
            if _forensics_enabled(cfg)
            else []
        ),
    )
    eval_peak_gpu_mem_mb = max(eval_peak_gpu_mem_mb, float(final_valid["eval_peak_gpu_mem_mb"]))
    _maybe_warn_ctc(stage="VALID-FINAL", metrics=final_valid, cfg=cfg)
    final_valid_score = _checkpoint_selection_score(final_valid, cfg)
    train_probe_metrics: Dict[str, Any] = {}
    if train_probe_dataset is not None:
        train_probe_metrics = evaluate_ctc(
            model,
            train_probe_dataset,
            tokenizer,
            cfg,
            seed=seed,
            epoch=99_998,
            artifact_path=Path(cfg["experiment"]["results_dir"]) / "diagnostics" / f"forensics_train_probe_seed_{seed}.json",
            teacher=teacher,
            stage_label="train_probe",
        )
    metrics = {
        "train_loss": float(running_loss / max(1, running_count)),
        "train_ctc_loss": float(running_ctc_loss / max(1, running_count)),
        "train_kd_loss": float(running_kd_loss / max(1, running_count)),
        "train_kd_lambda_effective": float(running_kd_lambda_effective / max(1, running_count)),
        "train_kd_active_ratio": float(running_kd_active_ratio / max(1, running_count)),
        "train_overemit_loss": float(running_overemit_loss / max(1, running_count)),
        "train_underemit_loss": float(running_underemit_loss / max(1, running_count)),
        "train_nonblank_density": float(running_nonblank_density / max(1, running_count)),
        "train_target_density": float(running_target_density / max(1, running_count)),
        "distillation_enabled": 1.0 if distill_enabled else 0.0,
        "distillation_lambda_kd_target": float(lambda_kd),
        "distillation_kd_warmup_steps": float(kd_warmup_steps),
        "distillation_kd_warmup_start_factor": float(kd_warmup_start_factor),
        "train_runtime_sec": float(time.time() - train_start_ts),
        "train_peak_gpu_mem_mb": float(train_peak_gpu_mem_mb),
        "train_invalid_length_ratio": float(train_invalid_length_count / max(1, train_sample_count)),
        "train_avg_out_length": float(train_out_length_sum / max(1, train_sample_count)),
        "train_avg_target_length": float(train_target_length_sum / max(1, train_sample_count)),
        "train_avg_length_margin": float(train_length_margin_sum / max(1, train_sample_count)),
        "train_micro_steps_completed": float(global_step),
        "train_optimizer_steps_completed": float(optimizer_steps_completed),
        "train_examples_seen": float(train_sample_count),
        "train_effective_epochs_completed": float(train_sample_count / max(1, len(train_dataset))),
        "train_dataset_size_after_filters": float(len(train_dataset)),
        "train_examples_seen_over_dataset_size": float(train_sample_count / max(1, len(train_dataset))),
        "train_max_steps_interpreted_as_micro_steps": 1.0,
        "train_grad_accum_steps": float(grad_acc),
        "train_micro_batch_size": float(training_cfg["batch_size"]),
        "train_effective_batch_size": float(effective_batch_size),
        "finetune_effective_batch_size": float(effective_batch_size),
        "finetune_optimizer_steps": float(optimizer_steps_completed),
        "finetune_optimizer_steps_planned": float(optimizer_total_steps),
        "finetune_warmup_steps": float(warmup_steps),
        "finetune_warmup_start_factor": float(warmup_start_factor),
        "finetune_min_lr_ratio": float(min_lr_ratio),
        "finetune_early_stopped": 1.0 if early_stopped else 0.0,
        "finetune_early_stop_epoch": float(-1 if early_stop_epoch is None else early_stop_epoch),
        "best_valid_epoch": float(-1 if best_epoch is None else best_epoch),
        "eval_peak_gpu_mem_mb": float(eval_peak_gpu_mem_mb),
        "valid_peak_gpu_mem_mb": float(eval_peak_gpu_mem_mb),
        "valid_wer": float(final_valid["wer"]),
        "valid_accuracy": float(final_valid["accuracy"]),
        "valid_runtime_sec": float(final_valid["eval_runtime_sec"]),
        "valid_samples_per_sec": float(final_valid["eval_samples_per_sec"]),
        "valid_blank_ratio": float(final_valid["blank_ratio"]),
        "valid_empty_pred_ratio": float(final_valid["empty_pred_ratio"]),
        "valid_nonempty_pred_ratio": float(final_valid["nonempty_pred_ratio"]),
        "valid_pred_to_ref_char_ratio": float(final_valid["pred_to_ref_char_ratio"]),
        "valid_invalid_length_ratio": float(final_valid["invalid_length_ratio"]),
        "valid_avg_out_length": float(final_valid["avg_out_length"]),
        "valid_avg_target_length": float(final_valid["avg_target_length"]),
        "valid_avg_length_margin": float(final_valid["avg_length_margin"]),
        "valid_exact_match_ratio": float(final_valid["exact_match_ratio"]),
        "valid_selection_score": float(final_valid_score),
        "best_valid_wer": float(best_wer if best_wer is not None else final_valid["wer"]),
        "best_valid_score": float(best_score if best_score is not None else final_valid_score),
        "best_valid_empty_pred_ratio": float(
            best_empty_pred_ratio if best_empty_pred_ratio is not None else final_valid["empty_pred_ratio"]
        ),
        "train_probe_size": float(0 if train_probe_dataset is None else len(train_probe_dataset)),
        "train_probe_unique_sample_ids_seen": float(train_probe_metrics.get("unique_sample_ids_seen", 0.0)),
        "train_probe_sample_coverage_ratio": float(train_probe_metrics.get("sample_coverage_ratio", 0.0)),
        "train_probe_sample_revisit_ratio": float(train_probe_metrics.get("sample_revisit_ratio", 0.0)),
        "train_probe_pred_to_ref_char_ratio": float(train_probe_metrics.get("pred_to_ref_char_ratio", 0.0)),
        "train_probe_blank_ratio": float(train_probe_metrics.get("blank_ratio", 0.0)),
        **{f"finetune_{k}_params": float(v) for k, v in count_parameters(model).items()},
    }
    return final_model, metrics


@torch.no_grad()
def evaluate_seed_on_dataset(
    cfg: Dict[str, Any],
    seed: int,
    dataset: torch.utils.data.Dataset,
    tokenizer: CharCTCTokenizer,
    checkpoint_path: Path,
    dataset_label: str,
    artifact_path: Optional[Path] = None,
    *,
    extra_artifact_paths: Sequence[Path] = (),
    stage_label: str = "test",
) -> Dict[str, Any]:
    device = _device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student_cfg = payload.get("student_config", {})
    model_cfg = dict(cfg["model"])
    audio_cfg = {"n_mels": int(cfg["audio"]["n_mels"])}
    if isinstance(student_cfg, dict):
        if isinstance(student_cfg.get("model"), dict):
            model_cfg.update(student_cfg["model"])
        if isinstance(student_cfg.get("audio"), dict) and student_cfg["audio"].get("n_mels") is not None:
            audio_cfg["n_mels"] = int(student_cfg["audio"]["n_mels"])
    eval_cfg = {
        **cfg,
        "model": model_cfg,
        "audio": {**cfg["audio"], **audio_cfg},
    }
    encoder = build_encoder(eval_cfg)
    model = AudioTransformerCTC(
        encoder=encoder,
        vocab_size=tokenizer.vocab_size,
        distill_projection_dim=model_cfg.get("distill_projection_dim"),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    param_counts = count_parameters(model)

    batch_size = int(cfg["training"]["finetune"]["batch_size"])
    teacher = build_teacher(eval_cfg, tokenizer=tokenizer, device=device) if _forensics_enabled(cfg) else None
    result = _run_ctc_diagnostic_pass(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        cfg=eval_cfg,
        seed=seed,
        loader_epoch=123456,
        stage_label=stage_label,
        artifact_path=artifact_path,
        extra_artifact_paths=extra_artifact_paths,
        teacher=teacher,
        dataset_label=dataset_label,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
    )
    result.update(
        {
        "seed": float(seed),
        "dataset": dataset_label,
        "eval_batch_size": float(batch_size),
        "inference_runtime_sec": float(result.get("eval_runtime_sec", 0.0)),
        "inference_samples_per_sec": float(result.get("eval_samples_per_sec", 0.0)),
        "inference_peak_gpu_mem_mb": float(result.get("eval_peak_gpu_mem_mb", 0.0)),
        "model_total_params": float(param_counts["total"]),
        "model_trainable_params": float(param_counts["trainable"]),
        }
    )
    if artifact_path is not None:
        result["diagnostics_artifact"] = str(artifact_path)
    return result
