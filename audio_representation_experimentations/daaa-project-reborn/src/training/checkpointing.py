from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _coerce_rng_tensor(value: Any) -> torch.ByteTensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.uint8, device="cpu")
    return torch.tensor(value, dtype=torch.uint8)


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(_coerce_rng_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [_coerce_rng_tensor(item) for item in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def checkpoint_name(global_step: int, epoch: int, tag: str = "step") -> str:
    return f"checkpoint_{tag}_e{epoch:03d}_s{global_step:07d}.pt"


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    global_step: int,
    step_in_epoch: int,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    tag: str = "step",
    keep_last_checkpoints: Optional[int] = None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "step_in_epoch": int(step_in_epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": _rng_state(),
        "extra": extra or {},
    }
    ckpt_path = checkpoint_dir / checkpoint_name(global_step=global_step, epoch=epoch, tag=tag)
    torch.save(payload, ckpt_path)
    # Update stable pointer for quick resume.
    torch.save(payload, checkpoint_dir / "latest.pt")
    if keep_last_checkpoints is not None and keep_last_checkpoints > 0:
        candidates = sorted(
            checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        to_remove = max(0, len(candidates) - int(keep_last_checkpoints))
        for old_path in candidates[:to_remove]:
            if old_path.exists():
                old_path.unlink()
    return ckpt_path


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    # Training checkpoints are produced by this project and intentionally contain
    # optimizer/scheduler/scaler states plus Python/NumPy/Torch RNG states.
    # PyTorch 2.6 changed torch.load default `weights_only=True`, which rejects
    # these richer payloads unless we explicitly opt into full loading.
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if payload.get("rng_state") is not None:
        _restore_rng_state(payload["rng_state"])
    return payload
