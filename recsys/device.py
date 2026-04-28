from __future__ import annotations

import os

import torch


def resolve_runtime_device(requested_device: str) -> torch.device:
    device = torch.device(requested_device)
    if device.type != "cuda":
        return device

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() returned False.")
        torch.empty(1, device=device)
    except Exception as exc:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        raise RuntimeError(
            f"Failed to initialize CUDA for requested device '{requested_device}'. "
            "This is usually an environment problem on the cluster rather than a model bug: "
            "the job may not actually have a GPU allocation, the driver/runtime may be mismatched, "
            "or CUDA visibility changed after Python started. "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}. "
            "If you only want to validate the pipeline, rerun with `--device cpu`."
        ) from exc

    return device
