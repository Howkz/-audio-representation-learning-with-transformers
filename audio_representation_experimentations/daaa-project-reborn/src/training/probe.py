from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.data.collate import classification_collate
from src.models.audio_probe import AudioLinearProbe
from src.training.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from src.training.loops import (
    _autocast_context,
    _build_grad_scaler,
    _build_scheduler,
    _device,
    _optimizer_total_steps,
    _planned_training_schedule,
    _progress_postfix,
    _seeded_loader,
    build_encoder,
)
from src.training.utils import count_parameters, peak_memory_mb, reset_peak_memory


def _probe_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    probe_cfg = cfg.get("probe", {})
    return probe_cfg if isinstance(probe_cfg, dict) else {}


def probe_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_probe_cfg(cfg).get("enabled", False))


def _checkpoint_root(cfg: Dict[str, Any], seed: int) -> Path:
    exp = cfg["experiment"]
    exp_id = exp.get("id")
    base = Path(exp["output_dir"]) / "checkpoints"
    if exp_id:
        base = base / str(exp_id)
    return base / "probe" / f"seed_{seed}"


def _bundle_payload(
    model: AudioLinearProbe,
    cfg: Dict[str, Any],
    label_to_id: Dict[str, int],
    *,
    best_valid_accuracy: float,
    best_valid_macro_f1: float,
    epoch: int,
    global_step: int,
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model": dict(cfg["model"]),
        "audio": dict(cfg["audio"]),
        "probe": dict(_probe_cfg(cfg)),
        "label_to_id": dict(label_to_id),
        "best_valid_accuracy": float(best_valid_accuracy),
        "best_valid_macro_f1": float(best_valid_macro_f1),
        "epoch": int(epoch),
        "global_step": int(global_step),
    }


def _load_encoder_weights(encoder: torch.nn.Module, encoder_checkpoint_path: Optional[Path], device: torch.device) -> None:
    if encoder_checkpoint_path is None:
        return
    payload = torch.load(encoder_checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("encoder_state_dict")
    if isinstance(state_dict, dict):
        encoder.load_state_dict(state_dict)
        return

    model_state_dict = payload.get("model_state_dict")
    if isinstance(model_state_dict, dict):
        encoder_state = {
            key[len("encoder.") :]: value
            for key, value in model_state_dict.items()
            if isinstance(key, str) and key.startswith("encoder.")
        }
        if encoder_state:
            encoder.load_state_dict(encoder_state)
            return
    raise KeyError(f"Unable to extract encoder weights from {encoder_checkpoint_path}.")


def _extract_features(batch: Dict[str, Any]) -> torch.Tensor:
    features = batch.get("x_features")
    if isinstance(features, torch.Tensor):
        return features
    return batch["x_logmel"]


def _compute_accuracy(predictions: Sequence[int], references: Sequence[int]) -> float:
    total = max(1, len(references))
    correct = sum(int(p == r) for p, r in zip(predictions, references))
    return float(correct) / float(total)


def _compute_macro_f1(predictions: Sequence[int], references: Sequence[int], num_classes: int) -> float:
    if num_classes <= 0:
        return 0.0
    scores = []
    for cls in range(int(num_classes)):
        tp = sum(int(p == cls and r == cls) for p, r in zip(predictions, references))
        fp = sum(int(p == cls and r != cls) for p, r in zip(predictions, references))
        fn = sum(int(p != cls and r == cls) for p, r in zip(predictions, references))
        precision = float(tp) / float(max(1, tp + fp))
        recall = float(tp) / float(max(1, tp + fn))
        if precision + recall <= 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(sum(scores) / max(1, len(scores)))


@torch.inference_mode()
def _evaluate_linear_probe_model(
    *,
    model: AudioLinearProbe,
    num_classes: int,
    dataset: torch.utils.data.Dataset,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    device = _device()

    eval_batch_size = int(_probe_cfg(cfg).get("eval_batch_size", _probe_cfg(cfg).get("batch_size", 32)))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=classification_collate,
    )

    reset_peak_memory()
    start_ts = time.time()
    predictions: list[int] = []
    references: list[int] = []
    num_batches = 0
    num_samples = 0

    for batch in loader:
        x = _extract_features(batch).to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(x, lengths=lengths)
        preds = logits.argmax(dim=-1)
        predictions.extend(preds.cpu().tolist())
        references.extend(labels.cpu().tolist())
        num_batches += 1
        num_samples += int(labels.shape[0])

    runtime_sec = float(time.time() - start_ts)
    accuracy = _compute_accuracy(predictions, references)
    macro_f1 = _compute_macro_f1(predictions, references, num_classes=num_classes)
    params = count_parameters(model)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "eval_batch_size": float(max(1, eval_batch_size)),
        "inference_runtime_sec": runtime_sec,
        "inference_samples_per_sec": float(num_samples) / max(1e-9, runtime_sec),
        "inference_peak_gpu_mem_mb": float(peak_memory_mb()),
        "num_batches": float(num_batches),
        "num_samples": float(num_samples),
        "model_total_params": float(params["total"]),
        "model_trainable_params": float(params["trainable"]),
        "num_classes": float(num_classes),
    }


@torch.inference_mode()
def evaluate_linear_probe(
    *,
    cfg: Dict[str, Any],
    dataset: torch.utils.data.Dataset,
    checkpoint_path: Path,
    dataset_label: str,
) -> Dict[str, float]:
    device = _device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_to_id = payload.get("label_to_id", {})
    if not isinstance(label_to_id, dict) or not label_to_id:
        raise ValueError(f"Missing label_to_id in {checkpoint_path}.")

    encoder = build_encoder(cfg)
    model = AudioLinearProbe(
        encoder=encoder,
        num_classes=len(label_to_id),
        dropout=float(_probe_cfg(cfg).get("dropout", 0.1)),
        freeze_encoder=bool(_probe_cfg(cfg).get("freeze_encoder", True)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    metrics = _evaluate_linear_probe_model(
        model=model,
        num_classes=len(label_to_id),
        dataset=dataset,
        cfg=cfg,
    )
    metrics["dataset_label"] = dataset_label
    return metrics


def run_linear_probe_seed(
    *,
    cfg: Dict[str, Any],
    seed: int,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    label_to_id: Dict[str, int],
    encoder_checkpoint_path: Optional[Path],
    force_continue_completed: bool = False,
) -> Tuple[Path, Dict[str, float]]:
    if not probe_enabled(cfg):
        raise ValueError("Linear probe requested while probe.enabled=false.")

    device = _device()
    probe_cfg = _probe_cfg(cfg)
    out_root = _checkpoint_root(cfg, seed=seed)
    out_root.mkdir(parents=True, exist_ok=True)
    run_completed = out_root / "run_completed.txt"
    final_model = out_root / "linear_probe_final.pt"
    best_model = out_root / "linear_probe_best.pt"
    if run_completed.exists() and final_model.exists() and not force_continue_completed:
        return final_model, {"status": 1.0}

    model = AudioLinearProbe(
        encoder=build_encoder(cfg),
        num_classes=len(label_to_id),
        dropout=float(probe_cfg.get("dropout", 0.1)),
        freeze_encoder=bool(probe_cfg.get("freeze_encoder", True)),
    ).to(device)
    _load_encoder_weights(model.encoder, encoder_checkpoint_path=encoder_checkpoint_path, device=device)
    if bool(probe_cfg.get("freeze_encoder", True)):
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(probe_cfg["learning_rate"]),
        weight_decay=float(probe_cfg.get("weight_decay", 0.0)),
    )
    steps_per_epoch, total_steps, effective_epochs = _planned_training_schedule(
        dataset_size=len(train_dataset),
        batch_size=int(probe_cfg["batch_size"]),
        epochs=int(probe_cfg["epochs"]),
        max_steps=int(probe_cfg["max_steps"]),
        stage="PROBE",
        seed=seed,
    )
    optimizer_total_steps = _optimizer_total_steps(total_steps, grad_acc=int(probe_cfg.get("grad_accum_steps", 1)))
    scheduler = _build_scheduler(
        optimizer,
        total_steps=optimizer_total_steps,
        warmup_steps=int(probe_cfg.get("warmup_steps", 0)),
        warmup_start_factor=float(probe_cfg.get("warmup_start_factor", 0.1)),
        min_lr_ratio=float(probe_cfg.get("min_lr_ratio", 0.0)),
    )
    scaler = _build_grad_scaler(device=device, enabled=bool(cfg["training"]["amp"]) and torch.cuda.is_available())

    checkpoint_every = int(probe_cfg.get("checkpoint_every_steps", cfg["experiment"]["checkpoint_every_steps"]))
    keep_last = int(cfg["experiment"].get("keep_last_checkpoints", 2))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))

    start_epoch = 0
    global_step = 0
    step_in_epoch = 0
    optimizer_steps_completed = 0
    best_valid_accuracy = float("-inf")
    best_valid_macro_f1 = 0.0
    epochs_without_improvement = 0
    latest = find_latest_checkpoint(out_root) if force_continue_completed else None
    if latest is not None:
        payload = load_checkpoint(latest, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("global_step", 0))
        step_in_epoch = int(payload.get("step_in_epoch", 0))
        optimizer_steps_completed = int(payload.get("extra", {}).get("optimizer_steps_completed", 0))
        best_valid_accuracy = float(payload.get("extra", {}).get("best_valid_accuracy", best_valid_accuracy))
        best_valid_macro_f1 = float(payload.get("extra", {}).get("best_valid_macro_f1", best_valid_macro_f1))
        epochs_without_improvement = int(payload.get("extra", {}).get("epochs_without_improvement", 0))

    amp_enabled = bool(cfg["training"]["amp"]) and torch.cuda.is_available()
    grad_acc = max(1, int(probe_cfg.get("grad_accum_steps", 1)))
    train_peak_gpu_mem_mb = 0.0
    valid_peak_gpu_mem_mb = 0.0
    start_ts = time.time()
    last_train_loss = 0.0
    last_valid_metrics: Dict[str, float] = {}

    for epoch in range(start_epoch, effective_epochs):
        if global_step >= total_steps:
            break
        model.train()
        if bool(probe_cfg.get("freeze_encoder", True)):
            model.encoder.eval()
        loader = _seeded_loader(
            dataset=train_dataset,
            batch_size=int(probe_cfg["batch_size"]),
            num_workers=num_workers,
            collate_fn=classification_collate,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        )
        progress = None
        iterator = loader
        if hasattr(loader, "__len__"):
            try:
                from tqdm.auto import tqdm

                progress = tqdm(loader, total=len(loader), desc=f"probe seed={seed} epoch={epoch}", leave=False)
                iterator = progress
            except Exception:
                progress = None

        total_loss = 0.0
        total_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(iterator):
            if global_step >= total_steps:
                break
            x = _extract_features(batch).to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with _autocast_context(device=device, enabled=amp_enabled):
                logits = model(x, lengths=lengths)
                loss = F.cross_entropy(logits, labels)
                scaled_loss = loss / grad_acc
            scaler.scale(scaled_loss).backward()
            if ((batch_idx + 1) % grad_acc == 0) or (batch_idx + 1 == len(loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["training"].get("grad_clip_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps_completed += 1

            global_step += 1
            step_in_epoch = batch_idx + 1
            total_loss += float(loss.detach().item()) * float(labels.shape[0])
            total_count += int(labels.shape[0])
            train_peak_gpu_mem_mb = max(train_peak_gpu_mem_mb, peak_memory_mb())
            if progress is not None:
                progress.set_postfix_str(
                    _progress_postfix(
                        global_step=global_step,
                        total_steps=total_steps,
                        start_ts=start_ts,
                        effective_batch_size=int(probe_cfg["batch_size"]) * grad_acc,
                    )
                )

            if global_step % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir=out_root,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    step_in_epoch=step_in_epoch,
                    best_metric=best_valid_accuracy if best_valid_accuracy != float("-inf") else None,
                    extra={
                        "optimizer_steps_completed": optimizer_steps_completed,
                        "best_valid_accuracy": None if best_valid_accuracy == float("-inf") else best_valid_accuracy,
                        "best_valid_macro_f1": best_valid_macro_f1,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                    keep_last_checkpoints=keep_last,
                )

        last_train_loss = float(total_loss / max(1, total_count))
        reset_peak_memory()
        model.eval()
        valid_metrics = _evaluate_linear_probe_model(
            model=model,
            num_classes=len(label_to_id),
            dataset=valid_dataset,
            cfg=cfg,
        )
        model.train()
        if bool(probe_cfg.get("freeze_encoder", True)):
            model.encoder.eval()
        valid_peak_gpu_mem_mb = max(valid_peak_gpu_mem_mb, float(valid_metrics.get("inference_peak_gpu_mem_mb", 0.0)))
        last_valid_metrics = dict(valid_metrics)

        current_valid_accuracy = float(valid_metrics["accuracy"])
        if current_valid_accuracy > (best_valid_accuracy + float(probe_cfg.get("early_stopping_min_delta", 0.0))):
            best_valid_accuracy = current_valid_accuracy
            best_valid_macro_f1 = float(valid_metrics["macro_f1"])
            epochs_without_improvement = 0
            torch.save(
                _bundle_payload(
                    model,
                    cfg,
                    label_to_id,
                    best_valid_accuracy=best_valid_accuracy,
                    best_valid_macro_f1=best_valid_macro_f1,
                    epoch=epoch,
                    global_step=global_step,
                ),
                best_model,
            )
        else:
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
            best_metric=best_valid_accuracy if best_valid_accuracy != float("-inf") else None,
            extra={
                "optimizer_steps_completed": optimizer_steps_completed,
                "best_valid_accuracy": None if best_valid_accuracy == float("-inf") else best_valid_accuracy,
                "best_valid_macro_f1": best_valid_macro_f1,
                "epochs_without_improvement": epochs_without_improvement,
            },
            keep_last_checkpoints=keep_last,
        )

        if (epoch + 1) >= int(probe_cfg.get("early_stopping_min_epochs", 1)) and epochs_without_improvement >= int(
            probe_cfg.get("early_stopping_patience", 5)
        ):
            break

    torch.save(
        _bundle_payload(
            model,
            cfg,
            label_to_id,
            best_valid_accuracy=best_valid_accuracy if best_valid_accuracy != float("-inf") else 0.0,
            best_valid_macro_f1=best_valid_macro_f1,
            epoch=epoch if "epoch" in locals() else 0,
            global_step=global_step,
        ),
        final_model,
    )
    run_completed.write_text("done\n", encoding="utf-8")
    params = count_parameters(model)
    metrics = {
        "probe_train_loss": float(last_train_loss),
        "probe_valid_accuracy": float(last_valid_metrics.get("accuracy", 0.0)),
        "probe_valid_macro_f1": float(last_valid_metrics.get("macro_f1", 0.0)),
        "probe_train_runtime_sec": float(time.time() - start_ts),
        "probe_train_peak_gpu_mem_mb": float(train_peak_gpu_mem_mb),
        "probe_valid_peak_gpu_mem_mb": float(valid_peak_gpu_mem_mb),
        "probe_batch_size": float(probe_cfg["batch_size"]),
        "probe_grad_accum_steps": float(grad_acc),
        "probe_effective_batch_size": float(int(probe_cfg["batch_size"]) * grad_acc),
        "probe_num_classes": float(len(label_to_id)),
        "probe_best_valid_accuracy": 0.0 if best_valid_accuracy == float("-inf") else float(best_valid_accuracy),
        "probe_best_valid_macro_f1": float(best_valid_macro_f1),
        "probe_total_params": float(params["total"]),
        "probe_trainable_params": float(params["trainable"]),
        "probe_optimizer_steps": float(optimizer_steps_completed),
    }
    return final_model, metrics
