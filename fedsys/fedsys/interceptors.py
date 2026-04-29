"""
gRPC interceptors for automatic telemetry capture.

Server-side  : TelemetryServerInterceptor
    Wraps every incoming RPC handler (unary-unary, unary-stream,
    stream-unary, stream-stream) to record:
        • request / response byte sizes
        • serialization / deserialization overhead
        • server-side handler latency

Client-side  : TelemetryClientInterceptor
    Implements all four grpc.*ClientInterceptor interfaces to capture:
        • outbound payload size
        • round-trip time (RTT) — time between first byte sent and last
          byte of response received; isolates network from compute.
        • per-call metadata

All metrics are pushed to an AsyncTelemetryLogger (non-blocking).  The
interceptors never stall the RPC; errors inside the logging path are caught
and silently discarded.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import grpc

from fedsys.logging_async import AsyncTelemetryLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _byte_size(obj: Any) -> int:
    """Return serialized byte size of a protobuf message (0 if unknown)."""
    try:
        return obj.ByteSize()
    except Exception:
        return 0


def _now_ms() -> float:
    return time.perf_counter() * 1_000


# ---------------------------------------------------------------------------
# Iterator wrappers — measure streaming payload sizes without buffering
# ---------------------------------------------------------------------------

class _MeasuredRequestIterator:
    """Wraps a client-side streaming request iterator; tallies bytes sent."""

    def __init__(self, iterator, logger: AsyncTelemetryLogger, meta: dict):
        self._it = iterator
        self._logger = logger
        self._meta = meta
        self._total_bytes = 0
        self._chunks = 0

    def __iter__(self):
        return self

    def __next__(self):
        msg = next(self._it)
        size = _byte_size(msg)
        self._total_bytes += size
        self._chunks += 1
        self._logger.log({
            "event": "STREAM_CHUNK_SENT",
            "chunk_index": self._chunks,
            "chunk_bytes": size,
            **self._meta,
        })
        return msg


class _MeasuredResponseIterator:
    """Wraps a server-side streaming response iterator; tallies bytes received."""

    def __init__(self, iterator, logger: AsyncTelemetryLogger, meta: dict, t_start: float):
        self._it = iterator
        self._logger = logger
        self._meta = meta
        self._t_start = t_start
        self._total_bytes = 0
        self._chunks = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            msg = next(self._it)
        except StopIteration:
            elapsed = _now_ms() - self._t_start
            self._logger.log({
                "event": "STREAM_RECV_COMPLETE",
                "total_bytes": self._total_bytes,
                "total_chunks": self._chunks,
                "rtt_ms": round(elapsed, 4),
                **self._meta,
            })
            raise
        size = _byte_size(msg)
        self._total_bytes += size
        self._chunks += 1
        self._logger.log({
            "event": "STREAM_CHUNK_RECV",
            "chunk_index": self._chunks,
            "chunk_bytes": size,
            **self._meta,
        })
        return msg


# ---------------------------------------------------------------------------
# Server-side interceptor
# ---------------------------------------------------------------------------

class TelemetryServerInterceptor(grpc.ServerInterceptor):
    """
    Records per-RPC telemetry on the coordinator (server) side.

    Captured events per call:
        RPC_RECV        — request received (byte size + handler name)
        RPC_HANDLER_END — handler finished (elapsed ms)
        STREAM_*        — per-chunk events for streaming RPCs
    """

    def __init__(self, logger: AsyncTelemetryLogger, node_id: str = "coordinator") -> None:
        self._logger = logger
        self._node_id = node_id

    # ---- handler wrappers -------------------------------------------------

    def _wrap_unary_unary(self, fn: Callable, method: str) -> Callable:
        logger = self._logger
        node_id = self._node_id

        def wrapper(request, context):
            req_bytes = _byte_size(request)
            t0 = _now_ms()
            logger.log({
                "event": "RPC_RECV",
                "method": method,
                "req_bytes": req_bytes,
                "node_id": node_id,
            })
            resp = fn(request, context)
            elapsed = _now_ms() - t0
            logger.log({
                "event": "RPC_HANDLER_END",
                "method": method,
                "resp_bytes": _byte_size(resp),
                "elapsed_ms": round(elapsed, 4),
                "node_id": node_id,
            })
            return resp

        return wrapper

    def _wrap_unary_stream(self, fn: Callable, method: str) -> Callable:
        logger = self._logger
        node_id = self._node_id

        def wrapper(request, context):
            req_bytes = _byte_size(request)
            t0 = _now_ms()
            logger.log({
                "event": "RPC_RECV",
                "method": method,
                "req_bytes": req_bytes,
                "node_id": node_id,
            })
            total_sent = 0
            chunk_count = 0
            for chunk in fn(request, context):
                size = _byte_size(chunk)
                total_sent += size
                chunk_count += 1
                yield chunk
            elapsed = _now_ms() - t0
            logger.log({
                "event": "RPC_HANDLER_END",
                "method": method,
                "total_sent_bytes": total_sent,
                "chunks": chunk_count,
                "elapsed_ms": round(elapsed, 4),
                "node_id": node_id,
            })

        return wrapper

    def _wrap_stream_unary(self, fn: Callable, method: str) -> Callable:
        logger = self._logger
        node_id = self._node_id

        def wrapper(request_iterator, context):
            total_recv = 0
            chunk_count = 0
            t0 = _now_ms()

            def measured_iter():
                nonlocal total_recv, chunk_count
                for chunk in request_iterator:
                    size = _byte_size(chunk)
                    total_recv += size
                    chunk_count += 1
                    logger.log({
                        "event": "STREAM_CHUNK_RECV",
                        "method": method,
                        "chunk_index": chunk_count,
                        "chunk_bytes": size,
                        "node_id": node_id,
                    })
                    yield chunk

            resp = fn(measured_iter(), context)
            elapsed = _now_ms() - t0
            logger.log({
                "event": "RPC_HANDLER_END",
                "method": method,
                "total_recv_bytes": total_recv,
                "chunks": chunk_count,
                "resp_bytes": _byte_size(resp),
                "elapsed_ms": round(elapsed, 4),
                "node_id": node_id,
            })
            return resp

        return wrapper

    # ---- grpc.ServerInterceptor interface ---------------------------------

    def intercept_service(
        self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        if handler is None:
            return handler

        method = handler_call_details.method

        if handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                self._wrap_unary_unary(handler.unary_unary, method),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                self._wrap_unary_stream(handler.unary_stream, method),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                self._wrap_stream_unary(handler.stream_unary, method),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        # stream-stream: pass through (not used in this proto)
        return handler


# ---------------------------------------------------------------------------
# Client-side interceptors
# ---------------------------------------------------------------------------

class TelemetryClientInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):
    """
    Measures outbound payload size and round-trip time for every gRPC call
    made by the training node.

    RTT is defined as the interval between the first byte sent (just before
    invoking the continuation) and the last byte of the final response
    received.  This isolates network + serialization from the node's
    compute loop.
    """

    def __init__(self, logger: AsyncTelemetryLogger, node_id: str) -> None:
        self._logger = logger
        self._node_id = node_id

    def _meta(self, details) -> dict:
        return {
            "method": details.method,
            "node_id": self._node_id,
        }

    # ---- unary-unary -------------------------------------------------------

    def intercept_unary_unary(self, continuation, client_call_details, request):
        meta = self._meta(client_call_details)
        req_bytes = _byte_size(request)
        t0 = _now_ms()
        self._logger.log({"event": "CLIENT_RPC_SEND", "req_bytes": req_bytes, **meta})

        resp = continuation(client_call_details, request)

        try:
            resp_bytes = _byte_size(resp.result())
        except Exception:
            resp_bytes = 0
        rtt = _now_ms() - t0
        self._logger.log({
            "event": "CLIENT_RPC_RECV",
            "resp_bytes": resp_bytes,
            "rtt_ms": round(rtt, 4),
            **meta,
        })
        return resp

    # ---- unary-stream (FetchGlobalModel) -----------------------------------

    def intercept_unary_stream(self, continuation, client_call_details, request):
        meta = self._meta(client_call_details)
        req_bytes = _byte_size(request)
        t0 = _now_ms()
        self._logger.log({"event": "CLIENT_RPC_SEND", "req_bytes": req_bytes, **meta})

        resp_iter = continuation(client_call_details, request)
        return _MeasuredResponseIterator(resp_iter, self._logger, meta, t0)

    # ---- stream-unary (SendLocalUpdate) ------------------------------------

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        meta = self._meta(client_call_details)
        t0 = _now_ms()
        self._logger.log({"event": "CLIENT_STREAM_START", **meta})

        measured = _MeasuredRequestIterator(request_iterator, self._logger, meta)
        resp = continuation(client_call_details, measured)

        try:
            rtt = _now_ms() - t0
            self._logger.log({
                "event": "CLIENT_STREAM_END",
                "total_sent_bytes": measured._total_bytes,
                "chunks": measured._chunks,
                "rtt_ms": round(rtt, 4),
                **meta,
            })
        except Exception:
            pass
        return resp

    # ---- stream-stream (not used; pass-through) ----------------------------

    def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
        return continuation(client_call_details, request_iterator)
