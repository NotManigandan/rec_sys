"""
Non-blocking telemetry sink.

All timing events, payload sizes, and hardware metrics are pushed onto a
thread-safe Queue.  A single daemon thread drains the queue and persists
records to two backends simultaneously:

    1. JSONL  — one JSON object per line; append-only; human-readable.
    2. SQLite — structured relational store; supports ad-hoc SQL analysis.

The Queue is bounded (maxsize=10_000) to exert back-pressure if the disk is
saturated, but the put() calls are non-blocking (block=False with a silent
drop on overflow) so that hot training / networking code is never stalled.

Usage
-----
    logger = AsyncTelemetryLogger(cfg)
    logger.start()                           # launch daemon thread
    logger.log({"event": "ROUND_START", "epoch": 3, ...})
    logger.stop()                            # graceful flush + join
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, Optional


# Sentinel object used to signal the drain thread to flush and exit.
_POISON = object()


class AsyncTelemetryLogger:
    """
    Thread-safe, non-blocking telemetry logger.

    Parameters
    ----------
    log_file : str
        Path to the JSONL output file.
    db_path : str
        Path to the SQLite database.
    maxsize : int
        Maximum number of records that can sit in the in-memory queue before
        new puts are silently dropped (prevents memory blow-up).
    flush_interval : float
        How often (in seconds) the drain thread commits the SQLite transaction
        even if the queue is not empty (batched writes).
    """

    def __init__(
        self,
        log_file: str,
        db_path: str,
        maxsize: int = 10_000,
        flush_interval: float = 1.0,
    ) -> None:
        self._log_file = log_file
        self._db_path = db_path
        self._flush_interval = flush_interval
        self._q: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0  # count of silently dropped records
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "AsyncTelemetryLogger":
        os.makedirs(os.path.dirname(self._log_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._thread = threading.Thread(
            target=self._drain_loop,
            name="telemetry-drain",
            daemon=True,
        )
        self._thread.start()
        return self

    def log(self, record: Dict[str, Any]) -> None:
        """Push a record onto the queue (non-blocking; drops on overflow)."""
        if "ts" not in record:
            record = {"ts": time.time(), **record}
        try:
            self._q.put_nowait(record)
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the drain thread, flush remaining records, then join."""
        self._q.put(_POISON)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain_loop(self) -> None:
        jsonl_fh = open(self._log_file, "a", buffering=1, encoding="utf-8")  # line-buffered
        conn = sqlite3.connect(self._db_path)
        self._init_db(conn)

        pending: list[Dict] = []
        last_flush = time.monotonic()

        def flush():
            nonlocal last_flush
            if pending:
                try:
                    conn.executemany(
                        "INSERT INTO events(ts, event, node_id, epoch, data) "
                        "VALUES (:ts,:event,:node_id,:epoch,:data)",
                        [
                            {
                                "ts": r.get("ts", 0.0),
                                "event": r.get("event", ""),
                                "node_id": r.get("node_id", ""),
                                "epoch": r.get("epoch", -1),
                                "data": json.dumps(r),
                            }
                            for r in pending
                        ],
                    )
                    conn.commit()
                except Exception as exc:
                    # SQLite errors must never propagate into the caller
                    print(f"[telemetry] SQLite flush error: {exc}")
                pending.clear()
            last_flush = time.monotonic()

        while True:
            try:
                item = self._q.get(timeout=self._flush_interval)
            except queue.Empty:
                flush()
                continue

            if item is _POISON:
                flush()
                break

            # Write to JSONL immediately (line-buffered → low latency)
            try:
                jsonl_fh.write(json.dumps(item) + "\n")
            except Exception:
                pass

            pending.append(item)
            self._q.task_done()

            now = time.monotonic()
            if now - last_flush >= self._flush_interval:
                flush()

        jsonl_fh.close()
        conn.close()

    @staticmethod
    def _init_db(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      REAL    NOT NULL,
                event   TEXT    NOT NULL,
                node_id TEXT,
                epoch   INTEGER,
                data    TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event ON events(event)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_epoch ON events(epoch)")
        conn.commit()


# ---------------------------------------------------------------------------
# Convenience timing context-manager
# ---------------------------------------------------------------------------

class TimedBlock:
    """
    Context manager that measures wall-clock duration and emits a telemetry
    record when the block exits.  Non-blocking: posts to the logger queue.

    Example
    -------
        with TimedBlock(logger, "H2D_TRANSFER", node_id="n0", epoch=1):
            model.load_state_dict(new_weights)
    """

    def __init__(
        self,
        logger: AsyncTelemetryLogger,
        event: str,
        **extra: Any,
    ) -> None:
        self._logger = logger
        self._event = event
        self._extra = extra
        self._t0: float = 0.0

    def __enter__(self) -> "TimedBlock":
        self._t0 = time.perf_counter()
        self._logger.log({"event": f"{self._event}_START", **self._extra})
        return self

    def __exit__(self, *_) -> None:
        elapsed_ms = (time.perf_counter() - self._t0) * 1_000
        self._logger.log(
            {
                "event": f"{self._event}_END",
                "elapsed_ms": round(elapsed_ms, 4),
                **self._extra,
            }
        )
