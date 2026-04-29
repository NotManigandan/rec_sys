# FedSys — gRPC Federated Learning Framework

A scalable, synchronous Federated Learning framework built for ML Systems research. Designed for deep profiling of the compute/communication boundary, VRAM pressure, and serialization overhead at scale (up to 300 M parameter models).

---

## Architecture Overview

```
 ┌─────────────────────────────────┐       gRPC streaming
 │         COORDINATOR             │◄──────────────────────┐
 │                                 │                        │
 │  NodeRegistry (thread-safe)     │  SendLocalUpdate       │
 │  FedAvgAggregator               │  (client→server stream)│
 │  FederatedLearningServicer      │                        │
 │  TelemetryServerInterceptor     │──────────────────────►│
 │  AsyncTelemetryLogger           │  FetchGlobalModel      │
 └─────────────────────────────────┘  (server→client stream)│
                                                            │
 ┌──────────────────────────────────────────────────────────┘
 │             TRAINING NODE (×N)
 │
 │  StateMachine  →  IDLE → REGISTERING → WAITING_FOR_MODEL
 │                         → RECEIVING_MODEL → LOCAL_TRAINING
 │                         → TRANSMITTING_UPDATE → (repeat)
 │
 │  FederatedNode          — networking + orchestration
 │  trainer.py             — isolated compute (H2D, backprop, D2H)
 │  TelemetryClientInterceptor
 │  AsyncTelemetryLogger
 └──────────────────────────────────────────────────────────
```

### Layer separation

| Layer | Files | Responsibility |
|---|---|---|
| **Protocol** | `proto/federated.proto`, `fedsys/generated/` | gRPC wire format |
| **Networking** | `fedsys/node/client.py`, `fedsys/coordinator/server.py` | streaming, chunking, registration |
| **Computation** | `fedsys/node/trainer.py`, `fedsys/coordinator/aggregator.py` | gradients, FedAvg — zero gRPC imports |
| **Privacy** | `fedsys/privacy.py` | byte-level middleware (placeholder) |
| **Observability** | `fedsys/logging_async.py`, `fedsys/interceptors.py` | non-blocking telemetry |

---

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA (optional; CPU is supported for testing)

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1 — Generate the proto stubs (once)

```bash
python scripts/generate_proto.py
```

### 2 — Start the coordinator

```bash
# N=3 nodes, aggregate when K=2 have reported
python scripts/run_coordinator.py --total-nodes 3 --min-nodes 2 --num-rounds 10
```

### 3 — Start each training node (separate terminals)

```bash
# Synthetic data (no download required)
python scripts/run_node.py --node-id node-0 --synthetic
python scripts/run_node.py --node-id node-1 --synthetic
python scripts/run_node.py --node-id node-2 --synthetic
```

### Real Amazon data

```bash
python scripts/run_node.py --node-id node-0 --partition 0 --num-partitions 3
python scripts/run_node.py --node-id node-1 --partition 1 --num-partitions 3
python scripts/run_node.py --node-id node-2 --partition 2 --num-partitions 3
```

The dataset (~1 GB compressed) downloads automatically to `~/.cache/fedsys/amazon/` on first use.

---

## Model

Neural Collaborative Filtering (`fedsys/models/recommendation.py`) targeting ~300 M parameters:

| Component | Size |
|---|---|
| User embedding: 1 M × 128 | 128.0 M |
| Item embedding: 500 K × 128 | 64.0 M |
| MLP backbone (256→4096→8192→8192→4096→2048→1024→1) | ~108.0 M |
| **Total** | **~300 M** |

The architecture is configurable via `ModelConfig`. The forward pass is plain PyTorch so individual layers can be replaced with Triton kernels without touching the FL framework.

---

## Telemetry

Every node and the coordinator writes two concurrent telemetry files:

| File | Format | Content |
|---|---|---|
| `logs/telemetry.jsonl` | JSONL | One JSON object per event |
| `logs/telemetry.db` | SQLite | Same data; queryable with SQL |

### Key events captured

| Event | Description |
|---|---|
| `H2D_TRANSFER_START/END` | Host-to-Device time (global model → GPU) |
| `LOCAL_TRAINING_START/END` | Full local compute wall-time |
| `LOCAL_EPOCH_END` | Per-local-epoch loss |
| `VRAM_PEAK` | Peak VRAM during training (MiB) |
| `D2H_TRANSFER_START/END` | Device-to-Host (gradients → CPU) |
| `FETCH_MODEL_START/END` | Server-side stream send time |
| `SEND_UPDATE_START/END` | Client-side stream send time |
| `CLIENT_RPC_RECV` | RTT per RPC call |
| `AGGREGATION_START/END` | FedAvg wall-time + VRAM delta |
| `STATE_ENTER/EXIT` | State machine transitions with duration |
| `STREAM_CHUNK_SENT/RECV` | Per-chunk byte sizes (via interceptors) |

All writes are non-blocking — a daemon thread drains a bounded `queue.Queue` and flushes to disk out-of-band so logging never inflates measured latencies.

### Example SQL queries

```sql
-- Compute vs communication ratio per node per round
SELECT node_id, epoch,
       MAX(CASE WHEN event='LOCAL_TRAINING_END'   THEN json_extract(data,'$.elapsed_ms') END) AS compute_ms,
       MAX(CASE WHEN event='SEND_UPDATE_DONE'     THEN json_extract(data,'$.net_ms') END)     AS send_ms,
       MAX(CASE WHEN event='FETCH_MODEL_DONE'     THEN json_extract(data,'$.net_ms') END)     AS recv_ms
FROM events GROUP BY node_id, epoch;

-- Peak VRAM per training round
SELECT epoch, node_id, json_extract(data,'$.vram_peak_mb') AS vram_mb
FROM events WHERE event='LOCAL_TRAINING_END' ORDER BY epoch;

-- H2D transfer cost per round
SELECT node_id, epoch, json_extract(data,'$.elapsed_ms') AS h2d_ms
FROM events WHERE event='H2D_TRANSFER_END';
```

---

## gRPC Design

### Message size

Both the channel and server override gRPC's default 4 MB message cap:

```python
options = [
    ("grpc.max_send_message_length", -1),   # unlimited
    ("grpc.max_receive_message_length", -1),
]
```

Large payloads are additionally chunked (default 4 MB/chunk) and streamed:

- `FetchGlobalModel(EpochRequest) returns (stream ModelChunk)` — coordinator streams the global model to each node
- `SendLocalUpdate(stream ModelChunk) returns (Ack)` — node streams gradient updates to coordinator

### Privacy middleware

`fedsys/privacy.py` exposes two functions called at every transmission boundary:

```python
def apply_privacy_transform(payload: bytes, *, context=None) -> bytes: ...
def reverse_privacy_transform(payload: bytes, *, context=None) -> bytes: ...
```

Both are currently identity functions. Replace with Differential Privacy noise injection, gradient clipping, or homomorphic encryption without touching any gRPC or training code.

### Straggler handling

The coordinator's `NodeRegistry.wait_for_updates()` blocks for at most `round_timeout_seconds`. Nodes that miss the deadline are silently dropped from the current aggregation round; their updates for future rounds are still accepted.

---

## Project Structure

```
fedsys/
├── proto/
│   └── federated.proto          # Service definition
├── fedsys/
│   ├── config.py                # CoordinatorConfig, NodeConfig, ModelConfig
│   ├── privacy.py               # Privacy middleware placeholder
│   ├── logging_async.py         # Non-blocking JSONL + SQLite telemetry
│   ├── interceptors.py          # gRPC client & server interceptors
│   ├── generated/               # Auto-generated protobuf stubs
│   ├── coordinator/
│   │   ├── registry.py          # Thread-safe node registry
│   │   ├── aggregator.py        # FedAvg with per-layer telemetry
│   │   └── server.py            # gRPC servicer + aggregation loop
│   ├── node/
│   │   ├── state_machine.py     # Finite-state machine with logged transitions
│   │   ├── trainer.py           # Isolated compute (H2D, backprop, D2H, VRAM)
│   │   └── client.py            # gRPC client + streaming logic
│   ├── data/
│   │   └── amazon_dataset.py    # Amazon 2023 Video Games + synthetic fallback
│   └── models/
│       └── recommendation.py    # NCF ~300 M param model
└── scripts/
    ├── generate_proto.py        # Compile proto -> Python stubs
    ├── run_coordinator.py       # Coordinator entry point
    └── run_node.py              # Node entry point
```
