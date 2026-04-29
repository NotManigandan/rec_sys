# Log Analysis Guide

This document explains:

1. What each CSV output from `analysis/analyze_logs.py` means.
2. How to read raw telemetry JSON logs (`telemetry*.jsonl`) in detail.
3. How communication metrics are derived (bytes, calls, chunks, RTT, errors).

---

## 1) Inputs and outputs

## Input logs

The analysis script reads:

- `telemetry.jsonl` (coordinator-side telemetry)
- `telemetry_<node_id>.jsonl` (node-side telemetry)

from one run log directory (for example `logs/run_20260429_022913/`).

## Output CSV files

By default the script writes to:

- `analysis/out/analysis_<timestamp>/`

and generates:

- `node_summary.csv`
- `method_summary.csv`
- `coordinator_interactions.csv`
- `event_counts.csv`

---

## 2) `node_summary.csv`

One row per `node_id` (including `coordinator` if it has relevant events).

Columns:

- `node_id`
  - Logical emitter of events (`logtest-n0`, `logtest-n1`, `coordinator`, etc.).

- `rpc_calls`
  - Number of client RPC call starts counted from:
    - `CLIENT_RPC_SEND` (unary calls like `Register`, `FetchGlobalModel` request)
    - `CLIENT_STREAM_START` (stream upload call like `SendLocalUpdate`)

- `sent_bytes`
  - Total bytes sent by that node from client-side events:
    - `CLIENT_RPC_SEND.req_bytes`
    - `CLIENT_STREAM_END.total_sent_bytes`

- `recv_bytes`
  - Total bytes received by that node from client-side events:
    - `CLIENT_RPC_RECV.resp_bytes`
    - `STREAM_RECV_COMPLETE.total_bytes`

- `total_bytes`
  - `sent_bytes + recv_bytes`

- `stream_chunks_sent`
  - Count of stream chunks sent (primarily upload stream chunks).
  - Derived from `CLIENT_STREAM_END.chunks` and/or chunk fallback events.

- `stream_chunks_recv`
  - Count of stream chunks received (primarily model fetch stream chunks).
  - Derived from `STREAM_RECV_COMPLETE.total_chunks` and/or chunk fallback events.

- `avg_rtt_ms`
  - Mean RTT over RTT samples seen for that node.
  - RTT samples come from:
    - `CLIENT_RPC_RECV.rtt_ms`
    - `CLIENT_STREAM_END.rtt_ms`
    - `STREAM_RECV_COMPLETE.rtt_ms`

- `max_rtt_ms`
  - Maximum RTT sample for that node.

- `rounds_started`
  - Count of `FL_ROUND_START` events for that node.

- `rounds_done`
  - Sum of `NODE_DONE.rounds_completed` values.

- `errors`
  - Count of `NODE_ERROR` events.

Interpretation tips:

- If `errors > 0`, check `event_counts.csv` and node JSONL for the exact exception.
- Healthy runs typically show:
  - nonzero `rpc_calls`,
  - similar-order `sent_bytes` and `recv_bytes`,
  - `rounds_started == rounds_done` for the planned rounds.

---

## 3) `method_summary.csv`

One row per (`node_id`, RPC method).

`method` is normalized from gRPC full method names:

- `/federated.FederatedLearning/Register` -> `Register`
- `/federated.FederatedLearning/FetchGlobalModel` -> `FetchGlobalModel`
- `/federated.FederatedLearning/SendLocalUpdate` -> `SendLocalUpdate`

Columns:

- `node_id`
- `method`
- `rpc_calls`
  - Calls attributed to this method (client-side start events).
- `sent_bytes`
  - Bytes sent for this method.
- `recv_bytes`
  - Bytes received for this method.
- `total_bytes`
  - `sent_bytes + recv_bytes`
- `stream_chunks_sent`
  - Stream chunks sent for this method.
- `stream_chunks_recv`
  - Stream chunks received for this method.
- `avg_rtt_ms`
- `max_rtt_ms`

Typical patterns:

- `Register`: very small send/recv.
- `FetchGlobalModel`: tiny request send, large stream receive.
- `SendLocalUpdate`: large stream send, tiny ack response.

---

## 4) `coordinator_interactions.csv`

Coordinator-centric per-node communication summary.

Columns:

- `node_id`
  - Node as identified by coordinator-side events (`FETCH_MODEL_*`, `UPDATE_RECV_*`).

- `fetch_requests`
  - Number of times coordinator started model fetch stream for node.
  - From `FETCH_MODEL_START`.

- `fetch_total_bytes`
  - Sum of model payload bytes coordinator sent to node.
  - From `FETCH_MODEL_START.total_bytes`.

- `fetch_total_chunks`
  - Sum of stream chunks coordinator sent for fetch.
  - From `FETCH_MODEL_START.total_chunks`.

- `update_submissions`
  - Number of local update submissions coordinator started receiving.
  - From `UPDATE_RECV_START`.

- `update_expected_bytes`
  - Sum of bytes nodes declared for update payload (`ModelChunk.total_bytes` first chunk).
  - From `UPDATE_RECV_START.expected_bytes`.

- `update_actual_bytes`
  - Sum of bytes coordinator actually reconstructed from streamed chunks.
  - From `UPDATE_RECV_END.actual_bytes`.

- `update_total_chunks`
  - Total chunks expected over submitted updates.
  - From `UPDATE_RECV_START.total_chunks`.

Validation use:

- `update_expected_bytes == update_actual_bytes` is a strong integrity sanity check.

---

## 5) `event_counts.csv`

Raw event frequency table.

Columns:

- `node_id`
- `event`
- `count`

Use it to:

- quickly identify failures (`NODE_ERROR`),
- confirm stage progression (`STATE_ENTER`/`STATE_EXIT`),
- confirm transfers happened (`FETCH_MODEL_DONE`, `SEND_UPDATE_DONE`),
- verify round lifecycle (`FL_ROUND_START`, `NODE_DONE`).

---

## 6) JSON log format (raw telemetry)

Each line in `telemetry*.jsonl` is one JSON object.

Common fields:

- `ts`: Unix timestamp (seconds, float)
- `event`: event name (string)
- `node_id`: logical emitter or owner
- `method`: gRPC method path (for RPC events)
- `epoch`: round index (where applicable)

Extra fields depend on event type.

Example:

```json
{"ts": 1777443097.763103, "event": "CLIENT_RPC_SEND", "req_bytes": 8, "method": "/federated.FederatedLearning/FetchGlobalModel", "node_id": "ana-n0"}
```

---

## 7) Event families and meanings

## A) Node lifecycle / state machine

- `STATE_ENTER`, `STATE_EXIT`
  - Track node FSM transitions and time spent per state.
- `FL_ROUND_START`
  - Round started on node.
- `NODE_DONE`
  - Node completed configured rounds.
- `NODE_ERROR`
  - Node raised exception; inspect `error` text.

## B) Node control flow

- `REGISTER_SEND`, `REGISTER_OK`
- `FETCH_MODEL_REQ`, `FETCH_MODEL_DONE`
- `SEND_UPDATE_START`, `SEND_UPDATE_DONE`

These are high-level node milestones around RPC calls and training flow.

## C) gRPC client interceptor events (node side)

- `CLIENT_RPC_SEND`
  - Unary call request bytes sent.
- `CLIENT_RPC_RECV`
  - Unary call response bytes received + RTT.
- `CLIENT_STREAM_START`
  - Stream-unary call initiated.
- `STREAM_CHUNK_SENT`
  - One outbound chunk in client stream.
- `CLIENT_STREAM_END`
  - Stream send completed (total sent bytes/chunks + RTT).
- `STREAM_CHUNK_RECV`
  - One inbound chunk in server stream.
- `STREAM_RECV_COMPLETE`
  - Stream receive finished (total bytes/chunks + RTT).

## D) gRPC server interceptor events (coordinator side)

- `RPC_RECV`
  - Request entered coordinator RPC handler.
- `RPC_HANDLER_END`
  - Handler completed with elapsed time and payload info.
- `STREAM_CHUNK_RECV`
  - Coordinator received one chunk in client-streaming RPC.

## E) Coordinator application-level transfer events

- `FETCH_MODEL_START`, `FETCH_MODEL_END`
  - Coordinator-side model broadcast stream metrics.
- `UPDATE_RECV_START`, `UPDATE_RECV_END`
  - Coordinator-side update stream reassembly metrics.
- `NODE_REGISTERED`
  - Node registration accepted.

## F) Training / hardware timing events

- `H2D_TRANSFER_START`, `H2D_TRANSFER_END`
  - Host-to-device transfer timing on node.
- Other `*_START` / `*_END` timed blocks may appear similarly.

---

## 8) Why bytes can look asymmetric

Common scenarios:

- In healthy rounds:
  - large coordinator->node bytes during fetch,
  - large node->coordinator bytes during update upload,
  - usually same order, not always exact.

- In failed runs (node crashes before upload):
  - large receive bytes (download happened),
  - tiny send bytes (only control calls happened),
  - `NODE_ERROR` present.

---

## 9) Multi-run handling

The analyzer reads all `telemetry*.jsonl` under one `--log-dir`.

So for clean per-run analysis:

- keep each run in its own log directory (recommended),
- or pass explicit `--log-dir` per run when analyzing.

---

## 10) Quick workflow

1. Run coordinator + nodes (same `--log-dir` for one experiment).
2. Run:
   - `python analysis/analyze_logs.py --log-dir <run_log_dir>`
3. Read:
   - `node_summary.csv` for high-level health
   - `method_summary.csv` for RPC-level byte split
   - `coordinator_interactions.csv` for stream integrity checks
   - `event_counts.csv` for fast anomaly detection

