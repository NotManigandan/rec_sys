"""
Training node gRPC client.

Drives the node's state machine through a full federated learning session:

    IDLE → REGISTERING → [WAITING_FOR_MODEL → RECEIVING_MODEL →
                           LOCAL_TRAINING → TRANSMITTING_UPDATE] × num_rounds
         → DONE

Networking and computation are cleanly separated:
    • All gRPC calls live in this file.
    • All tensor operations live in node/trainer.py.
    • The state machine in node/state_machine.py enforces valid transitions.

Streaming protocol
------------------
FetchGlobalModel : iterate the server-side stream; reassemble chunk bytes.
SendLocalUpdate  : split serialized state dict into chunks; stream to server.

Privacy middleware is applied here at the boundary (symmetric with the server).
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Iterator, Optional

import grpc

from fedsys.config import ModelConfig, NodeConfig, ensure_log_dir
from fedsys.interceptors import TelemetryClientInterceptor
from fedsys.logging_async import AsyncTelemetryLogger, TimedBlock
from fedsys.models.recommendation import build_model
from fedsys.node.state_machine import NodeState, StateMachine
from fedsys.node.trainer import run_local_training
from fedsys.privacy import apply_privacy_transform, reverse_privacy_transform

from fedsys.generated import federated_pb2, federated_pb2_grpc


class FederatedNode:
    """
    A single training node that participates in federated learning.

    Parameters
    ----------
    cfg       : Node configuration (address, device, hyper-params, …).
    model_cfg : Model architecture config (must match coordinator).
    dataloader: Pre-built PyTorch DataLoader for this node's data partition.
    num_rounds: Number of FL rounds to participate in.
    """

    def __init__(
        self,
        cfg: NodeConfig,
        model_cfg: ModelConfig,
        dataloader,
        num_rounds: int = 10,
    ) -> None:
        self._cfg = cfg
        self._model_cfg = model_cfg
        self._dataloader = dataloader
        self._num_rounds = num_rounds

        ensure_log_dir(cfg)
        self._logger = AsyncTelemetryLogger(cfg.log_file, cfg.db_path).start()
        self._sm = StateMachine(cfg.node_id, self._logger)
        self._model: Optional[NCFRecommender] = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full FL session synchronously."""
        try:
            self._connect_and_register()
            for fl_round in range(self._num_rounds):
                self._logger.log({
                    "event": "FL_ROUND_START",
                    "node_id": self._cfg.node_id,
                    "epoch": fl_round,
                })
                global_bytes = self._fetch_global_model(fl_round)
                updated_bytes, n_samples = self._train_locally(global_bytes, fl_round)
                self._send_update(updated_bytes, n_samples, fl_round)

            self._sm.transition(NodeState.DONE)
            self._logger.log({
                "event": "NODE_DONE",
                "node_id": self._cfg.node_id,
                "rounds_completed": self._num_rounds,
            })
        except Exception as exc:
            self._sm.transition(NodeState.ERROR, error=str(exc))
            self._logger.log({
                "event": "NODE_ERROR",
                "node_id": self._cfg.node_id,
                "error": str(exc),
            })
            raise
        finally:
            self._logger.stop()

    # ------------------------------------------------------------------
    # Step 1: Registration
    # ------------------------------------------------------------------

    def _connect_and_register(self) -> None:
        self._sm.transition(NodeState.REGISTERING)

        interceptor = TelemetryClientInterceptor(self._logger, self._cfg.node_id)
        channel = grpc.intercept_channel(
            grpc.insecure_channel(
                self._cfg.coordinator_address,
                options=self._cfg.grpc_options,
            ),
            interceptor,
        )
        self._stub = federated_pb2_grpc.FederatedLearningStub(channel)
        self._channel = channel

        node_info = federated_pb2.NodeInfo(
            node_id=self._cfg.node_id,
            address="",
            local_dataset_size=len(self._dataloader.dataset)
            if hasattr(self._dataloader, "dataset") else 0,
            metadata={"device": self._cfg.device},
        )

        self._logger.log({
            "event": "REGISTER_SEND",
            "node_id": self._cfg.node_id,
            "coordinator": self._cfg.coordinator_address,
        })

        response = self._stub.Register(node_info)
        self._cfg.data_partition = response.assigned_partition
        self._cfg.num_partitions = response.total_partitions

        self._logger.log({
            "event": "REGISTER_OK",
            "node_id": self._cfg.node_id,
            "partition": response.assigned_partition,
            "message": response.message,
        })

        # Instantiate model once after registration (architecture is fixed)
        self._model = build_model(self._model_cfg)
        self._sm.transition(NodeState.WAITING_FOR_MODEL)

    # ------------------------------------------------------------------
    # Step 2: Fetch global model
    # ------------------------------------------------------------------

    def _fetch_global_model(self, epoch: int) -> bytes:
        self._sm.transition(NodeState.RECEIVING_MODEL)

        request = federated_pb2.EpochRequest(
            node_id=self._cfg.node_id,
            epoch=epoch,
        )

        self._logger.log({
            "event": "FETCH_MODEL_REQ",
            "node_id": self._cfg.node_id,
            "epoch": epoch,
        })

        t_net_start = time.perf_counter()
        chunks: list[bytes] = []
        total_chunks = 0
        expected_bytes = 0

        for chunk in self._stub.FetchGlobalModel(request):
            if not chunks:
                total_chunks = chunk.total_chunks
                expected_bytes = chunk.total_bytes
            chunks.append(chunk.payload)

        net_ms = (time.perf_counter() - t_net_start) * 1_000
        raw_payload = b"".join(chunks)

        # Reverse privacy transform after reassembly
        t_transform = time.perf_counter()
        payload = reverse_privacy_transform(
            raw_payload, context=f"coordinator->{self._cfg.node_id}"
        )
        transform_ms = (time.perf_counter() - t_transform) * 1_000

        self._logger.log({
            "event": "FETCH_MODEL_DONE",
            "node_id": self._cfg.node_id,
            "epoch": epoch,
            "total_bytes": len(payload),
            "total_chunks": total_chunks,
            "net_ms": round(net_ms, 4),
            "privacy_transform_ms": round(transform_ms, 4),
        })

        self._sm.transition(NodeState.LOCAL_TRAINING)
        return payload

    # ------------------------------------------------------------------
    # Step 3: Local training
    # ------------------------------------------------------------------

    def _train_locally(self, global_bytes: bytes, epoch: int) -> tuple[bytes, int]:
        updated_bytes, n_samples = run_local_training(
            model=self._model,
            global_state_bytes=global_bytes,
            dataloader=self._dataloader,
            cfg=self._cfg,
            logger=self._logger,
            epoch=epoch,
        )
        self._sm.transition(NodeState.TRANSMITTING_UPDATE)
        return updated_bytes, n_samples

    # ------------------------------------------------------------------
    # Step 4: Send local update (client-side streaming)
    # ------------------------------------------------------------------

    def _send_update(self, updated_bytes: bytes, n_samples: int, epoch: int) -> None:
        # Apply privacy transform before chunking
        t_transform = time.perf_counter()
        payload = apply_privacy_transform(
            updated_bytes, context=f"{self._cfg.node_id}->coordinator"
        )
        transform_ms = (time.perf_counter() - t_transform) * 1_000

        chunk_size = self._cfg.chunk_size_bytes
        total_bytes = len(payload)
        total_chunks = math.ceil(total_bytes / chunk_size)
        epoch_id = str(uuid.uuid4())

        self._logger.log({
            "event": "SEND_UPDATE_START",
            "node_id": self._cfg.node_id,
            "epoch": epoch,
            "total_bytes": total_bytes,
            "total_chunks": total_chunks,
            "privacy_transform_ms": round(transform_ms, 4),
        })

        t_net_start = time.perf_counter()
        ack = self._stub.SendLocalUpdate(
            self._chunk_generator(payload, total_chunks, chunk_size, epoch_id, epoch, n_samples)
        )
        net_ms = (time.perf_counter() - t_net_start) * 1_000

        self._logger.log({
            "event": "SEND_UPDATE_DONE",
            "node_id": self._cfg.node_id,
            "epoch": epoch,
            "ack_success": ack.success,
            "net_ms": round(net_ms, 4),
        })

        if not ack.success:
            raise RuntimeError(f"Server rejected update at epoch {epoch}: {ack.message}")

        # Transition back to waiting for the next round
        self._sm.transition(NodeState.WAITING_FOR_MODEL)

    def _chunk_generator(
        self,
        payload: bytes,
        total_chunks: int,
        chunk_size: int,
        epoch_id: str,
        epoch: int,
        n_samples: int,
    ) -> Iterator[federated_pb2.ModelChunk]:
        total_bytes = len(payload)
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_bytes)
            yield federated_pb2.ModelChunk(
                payload=payload[start:end],
                chunk_index=i,
                total_chunks=total_chunks,
                epoch_id=epoch_id,
                total_bytes=total_bytes,
                metadata={
                    "node_id": self._cfg.node_id,
                    "epoch": str(epoch),
                    "num_samples": str(n_samples),
                },
            )
