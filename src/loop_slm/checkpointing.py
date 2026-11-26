"""Atomic, resumable training checkpoints.

Only load checkpoint files created by this project or another trusted source:
torch.load with weights_only=False can execute pickle payloads.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


FORMAT_VERSION = 1


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])

    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError as exc:
            raise RuntimeError("Checkpoint contains NumPy RNG state but NumPy is unavailable") from exc

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the checkpoint; exact RNG restoration is not possible"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    metadata: dict[str, Any] | None = None,
    sampler_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically save model, optimizer, optional training state, and RNG state.

    A single process must own writes to a given checkpoint path. Metadata and
    sampler state should contain serializable experiment information.
    """
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be a non-negative integer")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _capture_rng_state(),
        "metadata": metadata or {},
        "sampler_state": sampler_state,
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore a trusted checkpoint and return its metadata and progress state."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported or invalid checkpoint format")

    model.load_state_dict(payload["model"], strict=strict)
    optimizer.load_state_dict(payload["optimizer"])

    saved_scheduler = payload.get("scheduler")
    if scheduler is not None:
        if saved_scheduler is None:
            raise ValueError("A scheduler was supplied but the checkpoint has no scheduler state")
        scheduler.load_state_dict(saved_scheduler)
    elif saved_scheduler is not None:
        raise ValueError("Checkpoint contains scheduler state; supply the matching scheduler")

    saved_scaler = payload.get("scaler")
    if scaler is not None:
        if saved_scaler is None:
            raise ValueError("A scaler was supplied but the checkpoint has no scaler state")
        scaler.load_state_dict(saved_scaler)
    elif saved_scaler is not None:
        raise ValueError("Checkpoint contains scaler state; supply the matching scaler")

    _restore_rng_state(payload["rng"])
    return {
        "global_step": payload["global_step"],
        "metadata": payload.get("metadata", {}),
        "sampler_state": payload.get("sampler_state"),
    }
