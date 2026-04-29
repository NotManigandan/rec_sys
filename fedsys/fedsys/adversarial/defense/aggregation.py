"""
fedsys.adversarial.defense.aggregation
========================================
Robust aggregation methods that replace or augment plain FedAvg.

Ported from ``recsys/federated/bpr.py`` (``aggregate_server_states``).

All methods operate on **raw serialized state-dict bytes** (the same payload
format used throughout the fedsys gRPC pipeline), just like
``FedAvgAggregator``.

Available methods (``DEFENSE_METHODS`` constant):
    "none"                 — plain weighted mean (identical to FedAvgAggregator)
    "clip_mean"            — L2 norm-clip deltas, then weighted mean
    "clip_trimmed_mean"    — L2 norm-clip + coordinate-wise trimmed mean
    "focus_clip_mean"      — focus-score weighting + norm-clip + mean
    "focus_clip_trimmed_mean" — focus-score + norm-clip + trimmed mean

Design notes
------------
* Deltas are computed as  Δ = local_state − global_state  and defense
  operations are applied on the delta space before re-composing the new
  global model.  This keeps the math equivalent to the recsys implementation
  even though fedsys transmits full state dicts (not only delta bytes).
* The "focus score" is derived from the item-embedding layer
  (``item_embedding.weight``), measuring how much each update concentrates
  gradient mass on a small set of items — a signal that distinguishes
  poisoned from benign updates.
* ``RobustAggregator`` wraps all methods behind a single ``aggregate()``
  interface that is drop-in compatible with ``FedAvgAggregator.aggregate()``.
"""

from __future__ import annotations

import io
import math
import time
from typing import Dict, List, Optional, Tuple

import torch

from fedsys.logging_async import AsyncTelemetryLogger
from fedsys.coordinator.aggregator import (
    deserialize_state_dict,
    serialize_state_dict,
)


StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Registry of available defense method names
# ---------------------------------------------------------------------------

DEFENSE_METHODS = (
    "none",
    "clip_mean",
    "clip_trimmed_mean",
    "focus_clip_mean",
    "focus_clip_trimmed_mean",
)


# ---------------------------------------------------------------------------
# Internal math helpers
# ---------------------------------------------------------------------------

def _delta(
    local: StateDict, global_state: StateDict
) -> StateDict:
    """Compute Δ = local − global for each parameter tensor."""
    return {
        k: (local[k].float() - global_state[k].float())
        for k in global_state
        if k in local
    }


def _l2_norm(delta: StateDict) -> float:
    """L2 norm of a flattened state-dict delta."""
    total = 0.0
    for v in delta.values():
        total += float(v.pow(2).sum())
    return math.sqrt(total)


def _scale_delta(delta: StateDict, factor: float) -> StateDict:
    """Scale all tensors in a delta by a scalar factor."""
    return {k: v * factor for k, v in delta.items()}


def _add_deltas(
    base: Optional[StateDict], other: StateDict, weight: float = 1.0
) -> StateDict:
    """Weighted accumulation: base += other * weight."""
    if base is None:
        return {k: v * weight for k, v in other.items()}
    for k, v in other.items():
        if k in base:
            base[k] = base[k] + v * weight
        else:
            base[k] = v * weight
    return base


def _trimmed_mean_tensor(
    tensors: List[torch.Tensor],
    trim_fraction: float = 0.10,
) -> torch.Tensor:
    """
    Coordinate-wise trimmed mean.

    For each scalar coordinate, sort the values across clients, drop the
    ``trim_fraction`` from each tail, and average the rest.

    Parameters
    ----------
    tensors       : List of tensors with the same shape.
    trim_fraction : Fraction to remove from each side; e.g. 0.10 removes
                    the highest and lowest 10 % of values per coordinate.
    """
    stacked = torch.stack([t.float() for t in tensors], dim=0)  # (n, ...)
    n = stacked.shape[0]
    k = max(1, int(n * trim_fraction))

    if 2 * k >= n:
        # Not enough clients to trim; fall back to mean
        return stacked.mean(dim=0)

    sorted_t, _ = stacked.sort(dim=0)
    trimmed = sorted_t[k : n - k]              # (n-2k, ...)
    return trimmed.mean(dim=0)


def _focus_score(
    delta: StateDict,
    global_item_emb: torch.Tensor,
    k_frac: float = 0.05,
) -> float:
    """
    Measure how much an update "focuses" gradient mass on a small subset of
    item embeddings — a signal of poisoning.

    A benign update spreads gradient roughly uniformly across all items
    (data sparsity permitting).  A poisoned update concentrates large deltas
    on just the target item and its genre neighbours.

    Algorithm (from recsys/federated/bpr.py):
    1. Compute row-wise L2 norm of the item-embedding delta  (shape: num_items)
    2. Sort descending
    3. score = sum of top-(k_frac * num_items) norms / total norm

    Higher score → more focused → more likely to be an attack.

    Parameters
    ----------
    delta            : Update delta from one client.
    global_item_emb  : The current global item embedding weight (for shape).
    k_frac           : Fraction of items to consider "concentrated".
    """
    key = "item_embedding.weight"
    if key not in delta:
        return 0.0

    emb_delta = delta[key].float()              # (num_items, dim)
    row_norms = emb_delta.norm(dim=1)           # (num_items,)
    total_norm = float(row_norms.sum()) + 1e-12

    k = max(1, int(len(row_norms) * k_frac))
    top_k_sum = float(row_norms.topk(k).values.sum())

    return top_k_sum / total_norm


# ---------------------------------------------------------------------------
# Per-round diagnostics (suppressed shard detection — ported from recsys)
# ---------------------------------------------------------------------------

def _detect_suppressed_shards(
    norms: Dict[str, float],
    focus_scores: Dict[str, float],
    clip_threshold: float,
    focus_threshold: float,
    logger: Optional[AsyncTelemetryLogger] = None,
    epoch: int = 0,
) -> List[str]:
    """
    Flag nodes whose update was heavily clipped (norm >> threshold) AND whose
    focus score exceeds the threshold — likely malicious nodes.

    Returns a list of suspicious node IDs.
    """
    suspicious = [
        nid for nid in norms
        if norms[nid] > 2 * clip_threshold
        and focus_scores.get(nid, 0.0) >= focus_threshold
    ]
    if suspicious and logger is not None:
        logger.log({
            "event": "DEFENSE_SUSPICIOUS_NODES",
            "epoch": epoch,
            "suspected": suspicious,
            "norms": {n: round(norms[n], 4) for n in suspicious},
            "focus_scores": {n: round(focus_scores.get(n, 0.0), 4) for n in suspicious},
        })
    return suspicious


# ---------------------------------------------------------------------------
# RobustAggregator
# ---------------------------------------------------------------------------

class RobustAggregator:
    """
    Drop-in replacement for ``FedAvgAggregator`` that supports robust
    aggregation methods.

    Usage
    -----
    ::

        agg = RobustAggregator(
            logger, method="focus_clip_mean",
            clip_threshold=5.0, trim_fraction=0.1,
        )
        new_state, serialized = agg.aggregate(
            updates, sample_counts, epoch,
            global_state=current_state_dict,
        )

    All parameters have sensible defaults and can be changed at runtime.

    Parameters
    ----------
    logger          : AsyncTelemetryLogger for emitting defense telemetry.
    method          : One of DEFENSE_METHODS.
    device          : Torch device for aggregation tensors.
    clip_threshold  : L2-norm threshold for norm clipping (Theta in literature).
    trim_fraction   : Coordinate-wise trim fraction (each tail).
    focus_k_frac    : k_frac parameter for focus score computation.
    focus_threshold : Focus score above which an update is considered
                      suspicious for detection logging.
    """

    def __init__(
        self,
        logger: AsyncTelemetryLogger,
        method: str = "clip_mean",
        device: str = "cpu",
        clip_threshold: float = 5.0,
        trim_fraction: float = 0.10,
        focus_k_frac: float = 0.05,
        focus_threshold: float = 0.50,
    ) -> None:
        if method not in DEFENSE_METHODS:
            raise ValueError(
                f"Unknown defense method {method!r}. "
                f"Choose from {DEFENSE_METHODS}."
            )
        self._logger           = logger
        self._method           = method
        self._device           = device
        self._clip_threshold   = clip_threshold
        self._trim_fraction    = trim_fraction
        self._focus_k_frac     = focus_k_frac
        self._focus_threshold  = focus_threshold

    @property
    def method(self) -> str:
        return self._method

    # ------------------------------------------------------------------
    # Public API (compatible with FedAvgAggregator.aggregate)
    # ------------------------------------------------------------------

    def aggregate(
        self,
        updates: Dict[str, bytes],
        sample_counts: Dict[str, int],
        epoch: int,
        global_state: Optional[StateDict] = None,
    ) -> Tuple[StateDict, bytes]:
        """
        Aggregate serialized local updates with robust defense.

        Parameters
        ----------
        updates       : {node_id: serialized_bytes_of_local_state_dict}
        sample_counts : {node_id: num_local_samples}
        epoch         : Current FL round (for telemetry).
        global_state  : Current global model state dict.  Required for all
                        methods except "none".  If None for a defense method
                        that requires it, falls back to "none".

        Returns
        -------
        (aggregated_state_dict, serialized_bytes)
        """
        t0 = time.perf_counter()
        self._logger.log({
            "event": "ROBUST_AGG_START",
            "method": self._method,
            "epoch": epoch,
            "num_updates": len(updates),
        })

        if self._method == "none" or global_state is None:
            result = self._plain_fedavg(updates, sample_counts, epoch)
        else:
            result = self._robust_aggregate(
                updates, sample_counts, epoch, global_state
            )

        elapsed = (time.perf_counter() - t0) * 1_000
        self._logger.log({
            "event": "ROBUST_AGG_END",
            "method": self._method,
            "epoch": epoch,
            "elapsed_ms": round(elapsed, 4),
        })
        return result

    # ------------------------------------------------------------------
    # Plain FedAvg (method == "none")
    # ------------------------------------------------------------------

    def _plain_fedavg(
        self,
        updates: Dict[str, bytes],
        sample_counts: Dict[str, int],
        epoch: int,
    ) -> Tuple[StateDict, bytes]:
        total_samples = sum(sample_counts.get(nid, 1) for nid in updates)
        aggregated: Optional[StateDict] = None

        for node_id, payload in updates.items():
            weight = sample_counts.get(node_id, 1) / total_samples
            state, _ = deserialize_state_dict(payload, device=self._device)
            if aggregated is None:
                aggregated = {k: v.float() * weight for k, v in state.items()}
            else:
                for k, v in state.items():
                    if k in aggregated:
                        aggregated[k] += v.float() * weight
                    else:
                        aggregated[k] = v.float() * weight

        if aggregated is None:
            raise ValueError("No updates to aggregate")

        serialized, _ = serialize_state_dict(aggregated)
        return aggregated, serialized

    # ------------------------------------------------------------------
    # Robust aggregation (all other methods)
    # ------------------------------------------------------------------

    def _robust_aggregate(
        self,
        updates: Dict[str, bytes],
        sample_counts: Dict[str, int],
        epoch: int,
        global_state: StateDict,
    ) -> Tuple[StateDict, bytes]:
        """
        1. Deserialize all local updates.
        2. Compute deltas vs. global state.
        3. (If focus-based) compute focus scores.
        4. Clip norms.
        5. (If trimmed) apply coordinate-wise trimmed mean.
        6. (Otherwise) apply weighted mean.
        7. Reconstruct new global state.
        """
        # ── Deserialize ────────────────────────────────────────────────
        local_states: Dict[str, StateDict] = {}
        for node_id, payload in updates.items():
            state, _ = deserialize_state_dict(payload, device=self._device)
            local_states[node_id] = state

        # ── Deltas ─────────────────────────────────────────────────────
        global_cpu = {k: v.to(self._device) for k, v in global_state.items()}
        deltas: Dict[str, StateDict] = {
            nid: _delta(s, global_cpu) for nid, s in local_states.items()
        }

        # ── Norms ──────────────────────────────────────────────────────
        norms = {nid: _l2_norm(d) for nid, d in deltas.items()}

        # ── Focus scores (only for focus methods) ──────────────────────
        focus_scores: Dict[str, float] = {}
        if self._method.startswith("focus"):
            global_item_emb = global_cpu.get(
                "item_embedding.weight",
                torch.zeros(1, 1)
            )
            for nid, d in deltas.items():
                focus_scores[nid] = _focus_score(
                    d, global_item_emb, k_frac=self._focus_k_frac
                )

            self._logger.log({
                "event": "DEFENSE_FOCUS_SCORES",
                "epoch": epoch,
                "scores": {n: round(v, 4) for n, v in focus_scores.items()},
            })

            _detect_suppressed_shards(
                norms, focus_scores,
                clip_threshold=self._clip_threshold,
                focus_threshold=self._focus_threshold,
                logger=self._logger,
                epoch=epoch,
            )

        # ── Compute weights ────────────────────────────────────────────
        total_samples = sum(sample_counts.get(nid, 1) for nid in updates)

        if self._method.startswith("focus"):
            # Down-weight updates with high focus scores
            raw_weights = {
                nid: (1.0 - focus_scores.get(nid, 0.0))
                     * sample_counts.get(nid, 1)
                for nid in updates
            }
            w_sum = sum(raw_weights.values()) or 1.0
            weights = {nid: w / w_sum for nid, w in raw_weights.items()}
        else:
            weights = {
                nid: sample_counts.get(nid, 1) / total_samples
                for nid in updates
            }

        self._logger.log({
            "event": "DEFENSE_WEIGHTS",
            "epoch": epoch,
            "weights": {n: round(v, 4) for n, v in weights.items()},
            "norms":   {n: round(v, 4) for n, v in norms.items()},
        })

        # ── Norm clipping ──────────────────────────────────────────────
        theta = self._clip_threshold
        clipped_deltas: Dict[str, StateDict] = {}
        for nid, d in deltas.items():
            norm = norms[nid]
            if norm > theta:
                scale = theta / (norm + 1e-12)
                clipped_deltas[nid] = _scale_delta(d, scale)
            else:
                clipped_deltas[nid] = d

        # ── Aggregation ────────────────────────────────────────────────
        if "trimmed" in self._method:
            aggregated_delta = self._trimmed_mean(clipped_deltas, weights)
        else:
            aggregated_delta = self._weighted_mean(clipped_deltas, weights)

        # ── Reconstruct ────────────────────────────────────────────────
        new_state = {
            k: global_cpu[k] + aggregated_delta.get(k, torch.zeros_like(global_cpu[k]))
            for k in global_cpu
        }

        serialized, _ = serialize_state_dict(new_state)
        return new_state, serialized

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _weighted_mean(
        self,
        deltas: Dict[str, StateDict],
        weights: Dict[str, float],
    ) -> StateDict:
        result: Optional[StateDict] = None
        for nid, d in deltas.items():
            result = _add_deltas(result, d, weight=weights.get(nid, 0.0))
        return result or {}

    def _trimmed_mean(
        self,
        deltas: Dict[str, StateDict],
        weights: Dict[str, float],
    ) -> StateDict:
        """
        Coordinate-wise trimmed mean on the delta tensors.

        First applies per-client weights (i.e. weighted trimmed mean) then
        trimming is done uniformly across clients to match recsys behavior.
        Clients with weight 0 are excluded from trimming.
        """
        if not deltas:
            return {}

        # Get all parameter keys from first delta
        keys = list(next(iter(deltas.values())).keys())
        nids = list(deltas.keys())

        result: StateDict = {}
        for key in keys:
            tensors = [deltas[nid][key] for nid in nids if key in deltas[nid]]
            if not tensors:
                continue
            result[key] = _trimmed_mean_tensor(tensors, self._trim_fraction)

        return result
