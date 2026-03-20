from __future__ import annotations

import math
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler as LegacyGradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
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


def _diagnostics_examples_limit(cfg: Dict[str, Any]) -> int:
    diagnostics = _diagnostics_cfg(cfg)
    return max(0, int(diagnostics.get("max_saved_examples", 8)))


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


def _compute_distillation_loss(
    *,
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor,
    student_features: torch.Tensor,
    teacher_output: TeacherOutput,
    temperature: float,
) -> torch.Tensor:
    temperature = max(1e-4, float(temperature))
    aligned_values, valid_mask = _align_teacher_values(
        values=teacher_output.values,
        source_lengths=teacher_output.lengths,
        target_steps=int(student_logits.shape[1]),
        target_lengths=student_lengths,
    )
    if not valid_mask.any():
        return student_logits.new_zeros(())

    if teacher_output.target_kind == "logits":
        teacher_probs = aligned_values / aligned_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        loss = (token_kl * valid_mask.to(token_kl.dtype)).sum() / valid_mask.sum().clamp_min(1).to(token_kl.dtype)
        return loss * (temperature ** 2)

    if teacher_output.target_kind == "hidden_states":
        if aligned_values.shape[-1] != student_features.shape[-1]:
            raise ValueError(
                "Teacher hidden-state distillation requires matching hidden dims. "
                "Use a compatible student dim or add a projection layer."
            )
        mse = F.mse_loss(student_features, aligned_values, reduction="none").mean(dim=-1)
        return (mse * valid_mask.to(mse.dtype)).sum() / valid_mask.sum().clamp_min(1).to(mse.dtype)

    raise ValueError(f"Unsupported teacher target_kind '{teacher_output.target_kind}'.")


def _checkpoint_selection_score(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    selection_cfg = cfg.get("checkpoint_selection", {})
    if not isinstance(selection_cfg, dict):
        selection_cfg = {}
    empty_penalty = float(selection_cfg.get("empty_pred_penalty", 1.0))
    blank_penalty = float(selection_cfg.get("blank_ratio_penalty", 0.0))
    short_penalty = float(selection_cfg.get("short_pred_penalty", 0.0))
    accuracy_bonus = float(selection_cfg.get("accuracy_bonus", 0.0))
    return (
        float(metrics["wer"])
        + empty_penalty * float(metrics.get("empty_pred_ratio", 0.0))
        + blank_penalty * float(metrics.get("blank_ratio", 0.0))
        + short_penalty * float(metrics.get("short_pred_ratio", 0.0))
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


def _build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    total_steps = max(1, total_steps)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


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
    scheduler = _build_scheduler(optimizer, total_steps=total_steps)
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
    print(
        f"[PRETRAIN] seed={seed} max_steps={int(training_cfg['max_steps'])} "
        f"micro_batch={int(training_cfg['batch_size'])} grad_acc={grad_acc} "
        f"effective_batch={effective_batch_size}"
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

            if global_step % grad_acc == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(mae.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

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
def evaluate_ctc(
    model: AudioTransformerCTC,
    dataset: torch.utils.data.Dataset,
    tokenizer: CharCTCTokenizer,
    cfg: Dict[str, Any],
    seed: int,
    epoch: int,
    artifact_path: Optional[Path] = None,
) -> Dict[str, float]:
    device = _device()
    num_workers = int(cfg["data"]["num_workers"])
    batch_size = int(cfg["training"]["finetune"]["batch_size"])
    loader = _seeded_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(ctc_collate, tokenizer=tokenizer),
        seed=seed,
        epoch=epoch + 10_000,
        shuffle=False,
    )
    model.eval()
    predictions = []
    references = []
    diagnostic_totals: Dict[str, float] = {}
    diagnostic_examples = []
    max_examples = _diagnostics_examples_limit(cfg)
    start_ts = time.time()
    reset_peak_memory()
    for batch in loader:
        x = batch["x_logmel"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        logits, out_lengths = model(x, lengths)
        batch_predictions, batch_totals, batch_examples = collect_ctc_batch_diagnostics(
            logits=logits,
            out_lengths=out_lengths,
            target_lengths=batch["target_lengths"],
            references=batch["transcripts"],
            tokenizer=tokenizer,
            max_examples=max(0, max_examples - len(diagnostic_examples)),
        )
        predictions.extend(batch_predictions)
        references.extend(batch["transcripts"])
        _merge_diagnostic_totals(diagnostic_totals, batch_totals)
        diagnostic_examples.extend(batch_examples)
    runtime = max(1e-9, time.time() - start_ts)
    diagnostic_metrics = finalize_ctc_diagnostics(diagnostic_totals)
    result = {
        "wer": compute_wer(predictions, references),
        "accuracy": compute_char_accuracy(predictions, references),
        "eval_runtime_sec": float(runtime),
        "eval_samples_per_sec": float(len(references) / runtime),
        "eval_peak_gpu_mem_mb": peak_memory_mb(),
        **diagnostic_metrics,
    }
    if artifact_path is not None:
        write_json_artifact(
            artifact_path,
            {
                "seed": int(seed),
                "epoch": int(epoch),
                "stage": "validation",
                "metrics": result,
                "examples": diagnostic_examples,
            },
        )
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

    encoder = build_encoder(cfg)
    if pretrain_encoder_path is not None:
        _load_encoder_from_pretrain(encoder, pretrain_encoder_path, device)
    model = AudioTransformerCTC(encoder=encoder, vocab_size=tokenizer.vocab_size).to(device)
    teacher = build_teacher(cfg, tokenizer=tokenizer, device=device)
    distill_cfg = _distillation_cfg(cfg)
    distill_enabled = teacher is not None
    lambda_ctc = float(distill_cfg.get("lambda_ctc", 1.0))
    lambda_kd = float(distill_cfg.get("lambda_kd", 0.0))
    temperature = float(distill_cfg.get("temperature", 1.0))
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
    scheduler = _build_scheduler(optimizer, total_steps=total_steps)
    scaler = _build_grad_scaler(device=device, enabled=amp_enabled)
    ctc_loss_fn = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_wer: Optional[float] = None
    best_score: Optional[float] = None
    best_empty_pred_ratio: Optional[float] = None
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

    num_workers = int(cfg["data"]["num_workers"])
    grad_acc = int(training_cfg["grad_accum_steps"])
    grad_clip = float(cfg["training"]["grad_clip_norm"])
    effective_batch_size = int(training_cfg["batch_size"]) * max(1, grad_acc)
    log_every_steps = int(cfg["training"].get("log_every_steps", 25))
    train_start_ts = time.time()
    reset_peak_memory()
    running_loss = 0.0
    running_count = 0
    running_ctc_loss = 0.0
    running_kd_loss = 0.0
    train_sample_count = 0
    train_invalid_length_count = 0
    train_out_length_sum = 0.0
    train_target_length_sum = 0.0
    train_length_margin_sum = 0.0
    train_peak_gpu_mem_mb = 0.0
    eval_peak_gpu_mem_mb = 0.0
    emitted_invalid_length_warning = False
    print(
        f"[FINETUNE] seed={seed} max_steps={int(training_cfg['max_steps'])} "
        f"micro_batch={int(training_cfg['batch_size'])} grad_acc={grad_acc} "
        f"effective_batch={effective_batch_size} "
        f"distillation={distill_enabled}"
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
                if distill_enabled and lambda_kd > 0.0:
                    teacher_output = teacher.forward_teacher(
                        x_logmel=x,
                        lengths=lengths,
                        waveforms=waveforms if getattr(teacher, "requires_waveform", False) else None,
                        waveform_lengths=waveform_lengths if getattr(teacher, "requires_waveform", False) else None,
                        temperature=temperature,
                    )
                    kd_loss = _compute_distillation_loss(
                        student_logits=logits,
                        student_lengths=out_lengths,
                        student_features=student_features,
                        teacher_output=teacher_output,
                        temperature=temperature,
                    )
                loss = lambda_ctc * ctc_loss + lambda_kd * kd_loss
                loss_scaled = loss / grad_acc
            scaler.scale(loss_scaled).backward()

            global_step += 1
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

            if global_step % grad_acc == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

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
            f"selection_score={current_score:.4f}"
        )
        _maybe_warn_ctc(stage="VALID", metrics=eval_metrics, cfg=cfg)
        if (
            best_score is None
            or current_score < best_score
            or (
                math.isclose(current_score, best_score, rel_tol=0.0, abs_tol=1e-12)
                and (best_wer is None or current_wer < best_wer)
            )
        ):
            best_wer = current_wer
            best_score = current_score
            best_empty_pred_ratio = float(eval_metrics["empty_pred_ratio"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "blank_id": tokenizer.blank_id,
                    "student_config": {
                        "model": dict(cfg["model"]),
                        "audio": {"n_mels": int(cfg["audio"]["n_mels"])},
                    },
                    "selection_score": float(current_score),
                    "selection_components": {
                        "wer": float(current_wer),
                        "empty_pred_ratio": float(eval_metrics["empty_pred_ratio"]),
                        "blank_ratio": float(eval_metrics["blank_ratio"]),
                        "pred_to_ref_char_ratio": float(eval_metrics["pred_to_ref_char_ratio"]),
                        "accuracy": float(eval_metrics["accuracy"]),
                    },
                },
                out_root / "ctc_best.pt",
            )

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
            },
            tag="epoch",
            keep_last_checkpoints=keep_last,
        )
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
                "model": dict(cfg["model"]),
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
    )
    eval_peak_gpu_mem_mb = max(eval_peak_gpu_mem_mb, float(final_valid["eval_peak_gpu_mem_mb"]))
    _maybe_warn_ctc(stage="VALID-FINAL", metrics=final_valid, cfg=cfg)
    final_valid_score = _checkpoint_selection_score(final_valid, cfg)
    metrics = {
        "train_loss": float(running_loss / max(1, running_count)),
        "train_ctc_loss": float(running_ctc_loss / max(1, running_count)),
        "train_kd_loss": float(running_kd_loss / max(1, running_count)),
        "distillation_enabled": 1.0 if distill_enabled else 0.0,
        "train_runtime_sec": float(time.time() - train_start_ts),
        "train_peak_gpu_mem_mb": float(train_peak_gpu_mem_mb),
        "train_invalid_length_ratio": float(train_invalid_length_count / max(1, train_sample_count)),
        "train_avg_out_length": float(train_out_length_sum / max(1, train_sample_count)),
        "train_avg_target_length": float(train_target_length_sum / max(1, train_sample_count)),
        "train_avg_length_margin": float(train_length_margin_sum / max(1, train_sample_count)),
        "finetune_effective_batch_size": float(effective_batch_size),
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
) -> Dict[str, Any]:
    device = _device()
    encoder = build_encoder(cfg)
    model = AudioTransformerCTC(encoder=encoder, vocab_size=tokenizer.vocab_size).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    param_counts = count_parameters(model)

    batch_size = int(cfg["training"]["finetune"]["batch_size"])
    loader = _seeded_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=int(cfg["data"]["num_workers"]),
        collate_fn=partial(ctc_collate, tokenizer=tokenizer),
        seed=seed,
        epoch=123456,
        shuffle=False,
    )

    refs = []
    preds = []
    diagnostic_totals: Dict[str, float] = {}
    diagnostic_examples = []
    max_examples = _diagnostics_examples_limit(cfg)
    reset_peak_memory()
    start_ts = time.time()
    for batch in loader:
        x = batch["x_logmel"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        logits, out_lengths = model(x, lengths)
        batch_predictions, batch_totals, batch_examples = collect_ctc_batch_diagnostics(
            logits=logits,
            out_lengths=out_lengths,
            target_lengths=batch["target_lengths"],
            references=batch["transcripts"],
            tokenizer=tokenizer,
            max_examples=max(0, max_examples - len(diagnostic_examples)),
        )
        preds.extend(batch_predictions)
        refs.extend(batch["transcripts"])
        _merge_diagnostic_totals(diagnostic_totals, batch_totals)
        diagnostic_examples.extend(batch_examples)

    runtime = max(1e-9, time.time() - start_ts)
    result: Dict[str, Any] = {
        "seed": float(seed),
        "dataset": dataset_label,
        "wer": compute_wer(preds, refs),
        "accuracy": compute_char_accuracy(preds, refs),
        "eval_batch_size": float(batch_size),
        "inference_runtime_sec": float(runtime),
        "inference_samples_per_sec": float(len(refs) / runtime),
        "inference_peak_gpu_mem_mb": peak_memory_mb(),
        "model_total_params": float(param_counts["total"]),
        "model_trainable_params": float(param_counts["trainable"]),
        **finalize_ctc_diagnostics(diagnostic_totals),
    }
    if artifact_path is not None:
        write_json_artifact(
            artifact_path,
            {
                "seed": int(seed),
                "dataset": dataset_label,
                "checkpoint": str(checkpoint_path),
                "metrics": result,
                "examples": diagnostic_examples,
            },
        )
        result["diagnostics_artifact"] = str(artifact_path)
    return result
