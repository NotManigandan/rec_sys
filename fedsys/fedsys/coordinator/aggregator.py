"""
Federated Averaging (FedAvg) aggregation.

McMahan et al., "Communication-Efficient Learning of Deep Networks from
Decentralized Data", AISTATS 2017.

This module is intentionally decoupled from both the gRPC layer and the
privacy layer:
  • It receives raw bytes (already reverse-transformed by the privacy middleware).
  • It operates purely on Python dicts of {str: torch.Tensor}.
  • It emits telemetry (VRAM peak, aggregation wall-time) via the async logger.

The aggregation step runs on CPU to avoid contention with training nodes on
the same machine.  The result is moved to the target device by the caller.
"""

from __future__ import annotations

import io
import time
from typing import Dict, List, Optional, Tuple

import torch

from fedsys.logging_async import AsyncTelemetryLogger


StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_state_dict(state_dict: StateDict) -> bytes:
    """
    Serialize a PyTorch state dict to bytes using torch.save.

    We save to CPU to keep the bytes device-agnostic.  The receiver is
    responsible for moving tensors to the correct device.
    """
    t0 = time.perf_counter()
    cpu_state = {k: v.cpu() for k, v in state_dict.items()}
    buf = io.BytesIO()
    torch.save(cpu_state, buf)
    payload = buf.getvalue()
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    return payload, elapsed_ms


def deserialize_state_dict(
    payload: bytes,
    device: str = "cpu",
) -> Tuple[StateDict, float]:
    """
    Deserialize bytes back into a state dict.

    Returns (state_dict, serde_ms).
    """
    t0 = time.perf_counter()
    buf = io.BytesIO(payload)
    state_dict = torch.load(buf, map_location=device, weights_only=True)
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    return state_dict, elapsed_ms


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

class FedAvgAggregator:
    """
    Weighted Federated Averaging.

    Each client's contribution is weighted proportionally to its number of
    local training samples so that larger datasets have more influence on
    the global model — matching the original FedAvg formulation.
    """

    def __init__(
        self,
        logger: AsyncTelemetryLogger,
        device: str = "cpu",
    ) -> None:
        self._logger = logger
        self._device = device

    def aggregate(
        self,
        updates: Dict[str, bytes],
        sample_counts: Dict[str, int],
        epoch: int,
    ) -> Tuple[StateDict, bytes]:
        """
        Aggregate a collection of serialized state-dict updates.

        Parameters
        ----------
        updates       : {node_id: serialized_bytes}
        sample_counts : {node_id: num_local_samples}
        epoch         : current FL epoch (for logging)

        Returns
        -------
        (aggregated_state_dict, serialized_bytes)
        """
        t_agg_start = time.perf_counter()
        vram_before = _peak_vram_mb()

        self._logger.log({
            "event": "AGGREGATION_START",
            "epoch": epoch,
            "num_updates": len(updates),
        })

        total_samples = sum(sample_counts.get(nid, 1) for nid in updates)

        aggregated: Optional[StateDict] = None
        serde_total_ms = 0.0

        for node_id, payload in updates.items():
            n_samples = sample_counts.get(node_id, 1)
            weight = n_samples / total_samples

            state_dict, serde_ms = deserialize_state_dict(payload, device=self._device)
            serde_total_ms += serde_ms

            self._logger.log({
                "event": "AGGREGATION_DESER",
                "epoch": epoch,
                "node_id": node_id,
                "weight": round(weight, 6),
                "serde_ms": round(serde_ms, 4),
                "payload_bytes": len(payload),
            })

            if aggregated is None:
                aggregated = {
                    k: v.float() * weight for k, v in state_dict.items()
                }
            else:
                for k, v in state_dict.items():
                    if k in aggregated:
                        aggregated[k] += v.float() * weight
                    else:
                        aggregated[k] = v.float() * weight

        if aggregated is None:
            raise ValueError("No updates to aggregate")

        vram_after = _peak_vram_mb()
        t_agg_end = time.perf_counter()
        elapsed_ms = (t_agg_end - t_agg_start) * 1_000

        serialized, ser_ms = serialize_state_dict(aggregated)

        self._logger.log({
            "event": "AGGREGATION_END",
            "epoch": epoch,
            "elapsed_ms": round(elapsed_ms, 4),
            "serde_total_ms": round(serde_total_ms + ser_ms, 4),
            "vram_peak_mb": round(vram_after, 2),
            "vram_delta_mb": round(vram_after - vram_before, 2),
            "output_bytes": len(serialized),
        })

        return aggregated, serialized


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _peak_vram_mb() -> float:
    """Return current peak VRAM allocated (MiB) on the default CUDA device."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0
