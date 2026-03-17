from __future__ import annotations

import math
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
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
from src.training.metrics import compute_wer, greedy_decode_batch
from src.training.utils import count_parameters, peak_memory_mb, reset_peak_memory


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        eta_min = float((remaining / steps_per_sec) / 60.0)
        eta_text = f"{eta_min:.1f} min"
    else:
        eta_text = "N/A"
    return (
        f"[SPEED][{stage}] seed={seed} step={global_step}/{total_steps} "
        f"steps/s={steps_per_sec:.2f} samples/s={samples_per_sec:.2f} eta={eta_text}"
    )


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
    steps_per_epoch = max(1, math.ceil(len(dataset) / int(training_cfg["batch_size"])))
    total_steps = min(
        int(training_cfg["max_steps"]),
        int(training_cfg["epochs"]) * steps_per_epoch,
    )
    scheduler = _build_scheduler(optimizer, total_steps=total_steps)
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    latest = find_latest_checkpoint(out_root)
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

    for epoch in range(start_epoch, int(training_cfg["epochs"])):
        loader = _seeded_loader(
            dataset=dataset,
            batch_size=int(training_cfg["batch_size"]),
            num_workers=num_workers,
            collate_fn=pad_collate,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        )
        progress = tqdm(loader, desc=f"pretrain seed={seed} epoch={epoch}", leave=False)
        mae.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress):
            if epoch == start_epoch and batch_idx < start_step_in_epoch:
                continue
            x = batch["x_logmel"].to(device, non_blocking=True)
            with torch.no_grad():
                patch_info = mae.encoder.patch_embedding.patchify(x)
                seq_len = int(patch_info["patches"].shape[1])
                mask = make_mae_mask(
                    batch_size=x.shape[0],
                    seq_len=seq_len,
                    mask_ratio=mask_ratio,
                    device=device,
                )

            with autocast(enabled=amp_enabled):
                loss = mae(x, mask)
                loss_scaled = loss / grad_acc
            scaler.scale(loss_scaled).backward()

            global_step += 1
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

            if global_step % max(1, log_every_steps) == 0:
                print(
                    _progress_speed_line(
                        stage="PRETRAIN",
                        seed=seed,
                        global_step=global_step,
                        total_steps=int(training_cfg["max_steps"]),
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
    start_ts = time.time()
    reset_peak_memory()
    for batch in loader:
        x = batch["x_logmel"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        logits, _ = model(x, lengths)
        predictions.extend(greedy_decode_batch(logits, tokenizer))
        references.extend(batch["transcripts"])
    runtime = max(1e-9, time.time() - start_ts)
    result = {
        "wer": compute_wer(predictions, references),
        "eval_runtime_sec": float(runtime),
        "eval_samples_per_sec": float(len(references) / runtime),
        "eval_peak_gpu_mem_mb": peak_memory_mb(),
    }
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / int(training_cfg["batch_size"])))
    total_steps = min(
        int(training_cfg["max_steps"]),
        int(training_cfg["epochs"]) * steps_per_epoch,
    )
    scheduler = _build_scheduler(optimizer, total_steps=total_steps)
    scaler = GradScaler(enabled=amp_enabled)
    ctc_loss_fn = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_wer: Optional[float] = None
    latest = find_latest_checkpoint(out_root)
    if latest is not None:
        payload = load_checkpoint(latest, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(payload["epoch"])
        start_step_in_epoch = int(payload["step_in_epoch"])
        global_step = int(payload["global_step"])
        if payload.get("best_metric") is not None:
            best_wer = float(payload["best_metric"])

    num_workers = int(cfg["data"]["num_workers"])
    grad_acc = int(training_cfg["grad_accum_steps"])
    grad_clip = float(cfg["training"]["grad_clip_norm"])
    effective_batch_size = int(training_cfg["batch_size"]) * max(1, grad_acc)
    log_every_steps = int(cfg["training"].get("log_every_steps", 25))
    train_start_ts = time.time()
    reset_peak_memory()
    running_loss = 0.0
    running_count = 0
    train_peak_gpu_mem_mb = 0.0
    eval_peak_gpu_mem_mb = 0.0
    print(
        f"[FINETUNE] seed={seed} max_steps={int(training_cfg['max_steps'])} "
        f"micro_batch={int(training_cfg['batch_size'])} grad_acc={grad_acc} "
        f"effective_batch={effective_batch_size}"
    )

    for epoch in range(start_epoch, int(training_cfg["epochs"])):
        loader = _seeded_loader(
            dataset=train_dataset,
            batch_size=int(training_cfg["batch_size"]),
            num_workers=num_workers,
            collate_fn=partial(ctc_collate, tokenizer=tokenizer),
            seed=seed,
            epoch=epoch,
            shuffle=True,
        )
        progress = tqdm(loader, desc=f"finetune seed={seed} epoch={epoch}", leave=False)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress):
            if epoch == start_epoch and batch_idx < start_step_in_epoch:
                continue

            x = batch["x_logmel"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits, out_lengths = model(x, lengths)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = ctc_loss_fn(log_probs, targets, out_lengths, target_lengths)
                loss_scaled = loss / grad_acc
            scaler.scale(loss_scaled).backward()

            global_step += 1
            running_loss += float(loss.item())
            running_count += 1
            train_peak_gpu_mem_mb = max(train_peak_gpu_mem_mb, peak_memory_mb())

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
                    best_metric=best_wer,
                    tag="step",
                    keep_last_checkpoints=keep_last,
                )

            if global_step % max(1, log_every_steps) == 0:
                print(
                    _progress_speed_line(
                        stage="FINETUNE",
                        seed=seed,
                        global_step=global_step,
                        total_steps=int(training_cfg["max_steps"]),
                        start_ts=train_start_ts,
                        effective_batch_size=effective_batch_size,
                    )
                )

            if global_step >= int(training_cfg["max_steps"]):
                break

        start_step_in_epoch = 0
        eval_metrics = evaluate_ctc(model, valid_dataset, tokenizer, cfg, seed=seed, epoch=epoch)
        eval_peak_gpu_mem_mb = max(eval_peak_gpu_mem_mb, float(eval_metrics["eval_peak_gpu_mem_mb"]))
        current_wer = float(eval_metrics["wer"])
        if best_wer is None or current_wer < best_wer:
            best_wer = current_wer
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "blank_id": tokenizer.blank_id,
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
            best_metric=best_wer,
            extra={"valid_wer": current_wer},
            tag="epoch",
            keep_last_checkpoints=keep_last,
        )
        if global_step >= int(training_cfg["max_steps"]):
            break

    final_model = out_root / "ctc_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "blank_id": tokenizer.blank_id,
        },
        final_model,
    )
    with open(run_completed, "w", encoding="utf-8") as handle:
        handle.write("done\n")

    final_valid = evaluate_ctc(model, valid_dataset, tokenizer, cfg, seed=seed, epoch=99_999)
    eval_peak_gpu_mem_mb = max(eval_peak_gpu_mem_mb, float(final_valid["eval_peak_gpu_mem_mb"]))
    metrics = {
        "train_loss": float(running_loss / max(1, running_count)),
        "train_runtime_sec": float(time.time() - train_start_ts),
        "train_peak_gpu_mem_mb": float(train_peak_gpu_mem_mb),
        "eval_peak_gpu_mem_mb": float(eval_peak_gpu_mem_mb),
        "valid_peak_gpu_mem_mb": float(eval_peak_gpu_mem_mb),
        "valid_wer": float(final_valid["wer"]),
        "valid_runtime_sec": float(final_valid["eval_runtime_sec"]),
        "valid_samples_per_sec": float(final_valid["eval_samples_per_sec"]),
        "best_valid_wer": float(best_wer if best_wer is not None else final_valid["wer"]),
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
) -> Dict[str, float]:
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
    reset_peak_memory()
    start_ts = time.time()
    for batch in loader:
        x = batch["x_logmel"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        logits, _ = model(x, lengths)
        preds.extend(greedy_decode_batch(logits, tokenizer))
        refs.extend(batch["transcripts"])

    runtime = max(1e-9, time.time() - start_ts)
    return {
        "seed": float(seed),
        "dataset": dataset_label,
        "wer": compute_wer(preds, refs),
        "inference_runtime_sec": float(runtime),
        "inference_samples_per_sec": float(len(refs) / runtime),
        "inference_peak_gpu_mem_mb": peak_memory_mb(),
        "model_total_params": float(param_counts["total"]),
        "model_trainable_params": float(param_counts["trainable"]),
    }
