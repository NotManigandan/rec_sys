# Federated Learning Flow (2 Nodes + Coordinator)

This document explains the runtime flow of your synchronous FL system with:

- 1 coordinator
- 2 training nodes (`node-0`, `node-1`)
- K-out-of-N aggregation (`min_nodes` out of `total_nodes`)

---

## Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant N0 as Node-0 (FederatedNode)
    participant N1 as Node-1 (FederatedNode)
    participant GRPC as Coordinator gRPC Servicer
    participant REG as NodeRegistry
    participant AGG as Aggregation Thread (run_aggregation_loop)
    participant FAVG as FedAvgAggregator

    Note over AGG: Background thread started in serve()<br/>run_aggregation_loop(...)

    par Node startup
        N0->>GRPC: Register(NodeInfo) RPC
        GRPC->>REG: register(node_id, address, local_dataset_size, metadata)
        REG-->>GRPC: NodeRecord(partition=0)
        GRPC-->>N0: RegistrationResponse(assigned_partition=0)
    and
        N1->>GRPC: Register(NodeInfo) RPC
        GRPC->>REG: register(...)
        REG-->>GRPC: NodeRecord(partition=1)
        GRPC-->>N1: RegistrationResponse(assigned_partition=1)
    end

    loop For each round epoch = 0..num_rounds-1
        par Node-0 round work
            N0->>GRPC: FetchGlobalModel(EpochRequest{epoch})
            GRPC->>GRPC: FetchGlobalModel()<br/>wait while _broadcast_epoch < requested_epoch
            GRPC->>GRPC: apply_privacy_transform(raw_payload)
            GRPC-->>N0: stream ModelChunk (server-side streaming)
            N0->>N0: _fetch_global_model()<br/>reassemble chunks + reverse_privacy_transform()

            N0->>N0: _train_locally() -> run_local_training()
            N0->>N0: _send_update()<br/>apply_privacy_transform(updated_bytes)
            N0-->>GRPC: SendLocalUpdate(stream ModelChunk) (client-side streaming)
            GRPC->>GRPC: SendLocalUpdate()<br/>reassemble + reverse_privacy_transform()
            GRPC->>REG: submit_update(node_id, epoch, payload_bytes, num_samples)
            REG-->>GRPC: accepted=True/False
            GRPC-->>N0: Ack(success, epoch)
        and Node-1 round work
            N1->>GRPC: FetchGlobalModel(EpochRequest{epoch})
            GRPC->>GRPC: FetchGlobalModel() wait gate
            GRPC-->>N1: stream ModelChunk
            N1->>N1: _fetch_global_model() + reverse_privacy_transform()
            N1->>N1: _train_locally() -> run_local_training()
            N1-->>GRPC: SendLocalUpdate(stream ModelChunk)
            GRPC->>REG: submit_update(...)
            GRPC-->>N1: Ack(...)
        and Coordinator aggregation thread
            AGG->>REG: wait_for_updates()
            Note over REG: blocks until >= min_nodes (K)<br/>or round timeout
            REG-->>AGG: updates, sample_counts
            AGG->>FAVG: aggregate(updates, sample_counts, epoch)
            FAVG-->>AGG: new_state_dict
            AGG->>GRPC: update_global_model(new_state_dict, epoch+1)
            AGG->>REG: advance_epoch()
        end
    end

    Note over AGG: After final round:<br/>save_checkpoint(... is_final=True)<br/>optional test evaluate()
```

---

## Function Call Map

### Node Side (`fedsys/node/client.py`)

- `FederatedNode.run()`
  - `self._register()`
  - loop for each round:
    - `self._fetch_global_model(epoch)`
      - RPC call: `FetchGlobalModel`
      - reassembles chunks
      - `reverse_privacy_transform(...)`
    - `self._train_locally(global_bytes, epoch)`
      - calls `run_local_training(...)` in `fedsys/node/trainer.py`
    - `self._send_update(updated_bytes, n_samples, epoch)`
      - `apply_privacy_transform(...)`
      - chunk generator yields `ModelChunk`
      - RPC call: `SendLocalUpdate`
      - receives `Ack`

### Coordinator RPC Side (`fedsys/coordinator/server.py`)

- `Register(...)`
  - calls `registry.register(...)`
- `FetchGlobalModel(...)` (server-streaming)
  - waits until requested epoch model is ready
  - `apply_privacy_transform(...)`
  - streams model chunks
- `SendLocalUpdate(...)` (client-streaming)
  - receives and reassembles chunks
  - `reverse_privacy_transform(...)`
  - extracts metadata (`epoch`, `num_samples`, `node_id`)
  - calls `registry.submit_update(...)`
  - returns `Ack`

### Coordinator Aggregation Thread (`run_aggregation_loop`)

- `registry.wait_for_updates()` (blocks until K updates or timeout)
- `aggregator.aggregate(updates, sample_counts, epoch)` (FedAvg)
- optional validation evaluation (`evaluate(...)`)
- `save_checkpoint(...)` (`model_epoch_N`, `model_best`, `model_final`)
- `servicer.update_global_model(new_state, epoch+1)`
- `registry.advance_epoch()`
- after all rounds: optional test evaluation (`evaluate(..., split="test")`)

---

## Threading Model (Important)

Your coordinator process runs multiple threads with different responsibilities:

- gRPC worker thread pool:
  - executes RPC handlers (`Register`, `FetchGlobalModel`, `SendLocalUpdate`)
- Aggregation loop thread:
  - executes `run_aggregation_loop(...)` and performs FedAvg orchestration
- Logger thread:
  - async telemetry flushes to JSONL / SQLite

`NodeRegistry` bridges gRPC threads and aggregation thread using a lock + condition:

- `submit_update(...)` stores one node's payload and notifies waiters
- `wait_for_updates(...)` sleeps efficiently until enough updates arrive

---

## Round Semantics

- `num_rounds` = number of global aggregation cycles.
- `local_epochs` = local passes each participating node performs per round.
- Effective per-node local training load (if node participates every round):

`total_local_epochs_per_node = num_rounds * local_epochs`

This is not identical to centralized "global epochs" because each node trains on only its partition.

