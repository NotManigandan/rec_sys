"""
Thread-safe in-memory registry of active training nodes.

The registry is the single source of truth for:
    • which nodes have registered this session
    • which nodes have submitted their update for the current epoch
    • synchronisation between the gRPC servicer threads and the aggregation
      loop running in its own thread

Concurrency model
-----------------
All public methods acquire ``self._lock`` (a reentrant lock).  The
``wait_for_updates`` method blocks on a ``threading.Condition`` that is
notified whenever a new gradient update arrives.  This allows the aggregation
thread to sleep efficiently rather than busy-polling.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NodeRecord:
    node_id: str
    address: str
    local_dataset_size: int
    partition: int
    registered_at: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)


class NodeRegistry:
    """
    Registry of active training nodes.

    Parameters
    ----------
    total_nodes : int
        Expected total number of nodes (N).  Used to assign partitions.
    min_nodes : int
        Minimum number of updates (K) required before FedAvg can proceed.
    round_timeout : float
        Seconds the registry will wait for K updates before timing out the
        round and dropping stragglers.
    """

    def __init__(
        self,
        total_nodes: int,
        min_nodes: int,
        round_timeout: float = 120.0,
    ) -> None:
        self._total_nodes = total_nodes
        self._min_nodes = min_nodes
        self._round_timeout = round_timeout

        self._lock = threading.RLock()
        self._update_condition = threading.Condition(self._lock)

        # node_id → NodeRecord
        self._nodes: Dict[str, NodeRecord] = {}
        # node_id → serialized gradient bytes for the current epoch
        self._pending_updates: Dict[str, bytes] = {}
        # node_id → number of local samples (for weighted FedAvg)
        self._pending_samples: Dict[str, int] = {}

        self._current_epoch: int = 0
        self._partition_counter: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        node_id: str,
        address: str,
        local_dataset_size: int,
        metadata: Dict[str, str],
    ) -> NodeRecord:
        """
        Register a node and assign it a data partition index.

        If a node re-registers (crash + restart) the existing record is
        refreshed in-place.

        Returns
        -------
        NodeRecord  — the (possibly updated) record for the node.
        """
        with self._lock:
            if node_id in self._nodes:
                rec = self._nodes[node_id]
                rec.address = address
                rec.local_dataset_size = local_dataset_size
                rec.metadata = metadata
                return rec

            partition = self._partition_counter % self._total_nodes
            self._partition_counter += 1
            rec = NodeRecord(
                node_id=node_id,
                address=address,
                local_dataset_size=local_dataset_size,
                partition=partition,
                metadata=metadata,
            )
            self._nodes[node_id] = rec
            return rec

    def active_node_ids(self) -> List[str]:
        with self._lock:
            return list(self._nodes.keys())

    def get_record(self, node_id: str) -> Optional[NodeRecord]:
        with self._lock:
            return self._nodes.get(node_id)

    @property
    def registered_count(self) -> int:
        with self._lock:
            return len(self._nodes)

    # ------------------------------------------------------------------
    # Update collection
    # ------------------------------------------------------------------

    def submit_update(
        self,
        node_id: str,
        epoch: int,
        payload_bytes: bytes,
        num_samples: int,
    ) -> bool:
        """
        Accept a gradient payload from a node for the current epoch.

        Stale updates (wrong epoch) are rejected.  Returns True iff the
        update was accepted.
        """
        with self._update_condition:
            if epoch != self._current_epoch:
                return False
            if node_id not in self._nodes:
                return False
            self._pending_updates[node_id] = payload_bytes
            self._pending_samples[node_id] = num_samples
            self._update_condition.notify_all()
            return True

    def wait_for_updates(self) -> Dict[str, bytes]:
        """
        Block until ≥ K updates arrive for the current epoch or timeout.

        Returns
        -------
        dict  — {node_id: payload_bytes} for all updates that arrived before
                the deadline.  May contain fewer than K entries if timeout
                fired (stragglers are implicitly dropped).
        """
        deadline = time.monotonic() + self._round_timeout

        with self._update_condition:
            while len(self._pending_updates) < self._min_nodes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._update_condition.wait(timeout=min(remaining, 5.0))

            updates = dict(self._pending_updates)
            samples = dict(self._pending_samples)

        return updates, samples

    def advance_epoch(self) -> int:
        """Clear pending updates and move to the next epoch."""
        with self._lock:
            self._pending_updates.clear()
            self._pending_samples.clear()
            self._current_epoch += 1
            return self._current_epoch

    @property
    def current_epoch(self) -> int:
        with self._lock:
            return self._current_epoch

    def pending_update_count(self) -> int:
        with self._lock:
            return len(self._pending_updates)
