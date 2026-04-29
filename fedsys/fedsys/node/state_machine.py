"""
Training node state machine.

Divides node execution into discrete, named operational states.  Each
state transition is logged to the telemetry sink so that precise
per-state durations can be extracted during post-hoc ML-systems profiling.

State transition diagram
------------------------

    IDLE
      │  (registration succeeds)
      ▼
    REGISTERING
      │  (assigned a partition)
      ▼
    WAITING_FOR_MODEL ◄─────────────────────────────────┐
      │  (FetchGlobalModel stream complete)               │
      ▼                                                   │
    RECEIVING_MODEL                                       │
      │  (tensors loaded to device)                       │
      ▼                                                   │
    LOCAL_TRAINING                                        │
      │  (local epochs done)                              │
      ▼                                                   │
    TRANSMITTING_UPDATE                                   │
      │  (SendLocalUpdate ACK received)                   │
      └──────────────────────────────────────────────────┘  (next round)

    (after all rounds)
      ▼
    DONE

    (on unrecoverable error)
      ▼
    ERROR
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable, Optional

from fedsys.logging_async import AsyncTelemetryLogger


class NodeState(Enum):
    IDLE               = auto()
    REGISTERING        = auto()
    WAITING_FOR_MODEL  = auto()
    RECEIVING_MODEL    = auto()
    LOCAL_TRAINING     = auto()
    TRANSMITTING_UPDATE = auto()
    DONE               = auto()
    ERROR              = auto()


# Valid transitions: current state → set of allowed next states
_TRANSITIONS: dict[NodeState, set[NodeState]] = {
    NodeState.IDLE:               {NodeState.REGISTERING},
    NodeState.REGISTERING:        {NodeState.WAITING_FOR_MODEL, NodeState.ERROR},
    NodeState.WAITING_FOR_MODEL:  {NodeState.RECEIVING_MODEL, NodeState.DONE, NodeState.ERROR},
    NodeState.RECEIVING_MODEL:    {NodeState.LOCAL_TRAINING, NodeState.ERROR},
    NodeState.LOCAL_TRAINING:     {NodeState.TRANSMITTING_UPDATE, NodeState.ERROR},
    NodeState.TRANSMITTING_UPDATE:{NodeState.WAITING_FOR_MODEL, NodeState.DONE, NodeState.ERROR},
    NodeState.DONE:               set(),
    NodeState.ERROR:              set(),
}


class StateMachine:
    """
    Lightweight finite-state machine for a training node.

    Enforces valid transitions and logs every transition event with
    nanosecond-resolution timestamps to the async telemetry logger.
    """

    def __init__(self, node_id: str, logger: AsyncTelemetryLogger) -> None:
        self._node_id = node_id
        self._logger = logger
        self._state = NodeState.IDLE
        self._entered_at: float = time.perf_counter()
        self._on_enter: dict[NodeState, Callable] = {}

        self._logger.log({
            "event": "STATE_ENTER",
            "state": NodeState.IDLE.name,
            "node_id": self._node_id,
        })

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> NodeState:
        return self._state

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def transition(self, new_state: NodeState, **metadata) -> None:
        """
        Attempt to move the machine to ``new_state``.

        Raises
        ------
        ValueError  if the transition is not permitted.
        """
        allowed = _TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition: {self._state.name} → {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )

        now = time.perf_counter()
        duration_ms = (now - self._entered_at) * 1_000

        self._logger.log({
            "event": "STATE_EXIT",
            "state": self._state.name,
            "duration_ms": round(duration_ms, 4),
            "node_id": self._node_id,
            **metadata,
        })

        self._state = new_state
        self._entered_at = now

        self._logger.log({
            "event": "STATE_ENTER",
            "state": new_state.name,
            "node_id": self._node_id,
            **metadata,
        })

        # Fire registered callback (if any)
        cb = self._on_enter.get(new_state)
        if cb is not None:
            cb()

    def on_enter(self, state: NodeState) -> Callable:
        """Decorator: register a callback invoked whenever state is entered."""
        def decorator(fn: Callable) -> Callable:
            self._on_enter[state] = fn
            return fn
        return decorator

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self._state in (NodeState.DONE, NodeState.ERROR)
