"""
Local training computation — strictly isolated from all gRPC code.

This module contains everything that touches a GPU/CPU tensor:
    • Host-to-Device (H2D) model transfer
    • Local SGD / Adam training loop
    • VRAM peak measurement
    • Device-to-Host (D2H) gradient extraction

The gRPC client (`node/client.py`) calls ``run_local_training()`` as a black
box.  Swapping in Triton kernels or custom CUDA extensions only requires
modifying this file — the networking code remains untouched.

Telemetry captured here
-----------------------
    H2D_TRANSFER_START / _END   — time to move the global model to device
    LOCAL_TRAINING_START / _END — full local compute wall-time
    EPOCH_*                     — per-local-epoch loss
    VRAM_PEAK                   — peak VRAM during training (MiB)
    D2H_TRANSFER_START / _END   — time to move gradients back to CPU
"""

from __future__ import annotations

import io
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedsys.config import NodeConfig
from fedsys.logging_async import AsyncTelemetryLogger, TimedBlock


StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_local_training(
    model: nn.Module,
    global_state_bytes: bytes,
    dataloader: DataLoader,
    cfg: NodeConfig,
    logger: AsyncTelemetryLogger,
    epoch: int,
    num_samples: Optional[int] = None,
) -> tuple[bytes, int]:
    """
    Load the global model, train locally, and return the updated state dict.

    Parameters
    ----------
    model           : Freshly instantiated model (architecture must match).
    global_state_bytes : Serialized global state dict from the coordinator.
    dataloader      : Local training data loader.
    cfg             : Node configuration (device, lr, local_epochs, …).
    logger          : Async telemetry sink.
    epoch           : Current FL epoch index (for logging).
    num_samples     : Override for local sample count (defaults to len(dataset)).

    Returns
    -------
    (serialized_updated_state: bytes, num_local_samples: int)
    """
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu"
                          else "cpu")

    # ── H2D: load global model weights to device ─────────────────────────
    with TimedBlock(logger, "H2D_TRANSFER", node_id=cfg.node_id, epoch=epoch):
        state_dict = _load_state_dict(global_state_bytes, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.train()

    # Reset peak VRAM counter so we get a clean measurement for this round
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_samples = num_samples or (
        len(dataloader.dataset) if hasattr(dataloader, "dataset") else 0
    )

    # ── Local training loop ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    logger.log({
        "event": "LOCAL_TRAINING_START",
        "node_id": cfg.node_id,
        "epoch": epoch,
        "local_epochs": cfg.local_epochs,
        "num_samples": n_samples,
        "device": str(device),
    })

    t_train_start = time.perf_counter()

    for local_epoch in range(cfg.local_epochs):
        epoch_loss = _train_one_epoch(
            model, dataloader, optimizer, criterion, device
        )
        logger.log({
            "event": "LOCAL_EPOCH_END",
            "node_id": cfg.node_id,
            "fl_epoch": epoch,
            "local_epoch": local_epoch,
            "loss": round(epoch_loss, 6),
        })

    train_elapsed_ms = (time.perf_counter() - t_train_start) * 1_000
    vram_peak_mb = _peak_vram_mb(device)

    logger.log({
        "event": "LOCAL_TRAINING_END",
        "node_id": cfg.node_id,
        "epoch": epoch,
        "elapsed_ms": round(train_elapsed_ms, 4),
        "vram_peak_mb": round(vram_peak_mb, 2),
    })

    # ── D2H: move updated weights to CPU for serialization ───────────────
    with TimedBlock(logger, "D2H_TRANSFER", node_id=cfg.node_id, epoch=epoch):
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        buf = io.BytesIO()
        torch.save(cpu_state, buf)
        serialized = buf.getvalue()

    logger.log({
        "event": "GRADIENT_SERIALIZED",
        "node_id": cfg.node_id,
        "epoch": epoch,
        "bytes": len(serialized),
    })

    return serialized, n_samples


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one local epoch and return the mean batch loss."""
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)
        labels = batch["label"].float().to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(user_ids, item_ids)
        loss = criterion(logits.squeeze(-1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _load_state_dict(payload: bytes, device: torch.device) -> StateDict:
    """Deserialize a state dict and map tensors to the target device."""
    buf = io.BytesIO(payload)
    return torch.load(buf, map_location=device, weights_only=True)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type == "cuda":
        try:
            return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception:
            pass
    return 0.0
