from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _short_method(method: str) -> str:
    if not method:
        return "unknown"
    if "/" in method:
        return method.rsplit("/", 1)[-1]
    return method


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                obj["_source_file"] = str(path)
                obj["_line_no"] = line_no
                yield obj


def _safe_num(v: Any, cast=float, default=0.0):
    try:
        return cast(v)
    except Exception:
        return default


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def analyze(log_dir: Path, output_dir: Path) -> Dict[str, Path]:
    jsonl_files = sorted(log_dir.glob("telemetry*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No telemetry JSONL files found in: {log_dir}")

    event_counts: Counter[Tuple[str, str]] = Counter()
    method_stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "rpc_calls": 0,
            "sent_bytes": 0,
            "recv_bytes": 0,
            "stream_chunks_sent": 0,
            "stream_chunks_recv": 0,
            "rtt_samples": [],
        }
    )
    node_totals: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "rpc_calls": 0,
            "sent_bytes": 0,
            "recv_bytes": 0,
            "stream_chunks_sent": 0,
            "stream_chunks_recv": 0,
            "rtt_samples": [],
            "rounds_started": 0,
            "rounds_done": 0,
            "errors": 0,
        }
    )

    # Coordinator-side interaction counters (how coordinator sees each node)
    coord_interactions: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "fetch_requests": 0,
            "fetch_total_bytes": 0,
            "fetch_total_chunks": 0,
            "update_submissions": 0,
            "update_expected_bytes": 0,
            "update_actual_bytes": 0,
            "update_total_chunks": 0,
        }
    )

    for jf in jsonl_files:
        for rec in _read_jsonl(jf):
            event = str(rec.get("event", ""))
            node_id = str(rec.get("node_id", "unknown"))
            method = _short_method(str(rec.get("method", "")))
            event_counts[(node_id, event)] += 1

            # Generic node lifecycle signals
            if event == "FL_ROUND_START":
                node_totals[node_id]["rounds_started"] += 1
            elif event in ("NODE_DONE",):
                node_totals[node_id]["rounds_done"] += int(rec.get("rounds_completed", 0) or 0)
            elif event in ("NODE_ERROR",):
                node_totals[node_id]["errors"] += 1

            # Coordinator-side communication events
            if event == "FETCH_MODEL_START":
                coord_interactions[node_id]["fetch_requests"] += 1
                coord_interactions[node_id]["fetch_total_bytes"] += _safe_num(rec.get("total_bytes"), int, 0)
                coord_interactions[node_id]["fetch_total_chunks"] += _safe_num(rec.get("total_chunks"), int, 0)
            elif event == "UPDATE_RECV_START":
                coord_interactions[node_id]["update_submissions"] += 1
                coord_interactions[node_id]["update_expected_bytes"] += _safe_num(rec.get("expected_bytes"), int, 0)
                coord_interactions[node_id]["update_total_chunks"] += _safe_num(rec.get("total_chunks"), int, 0)
            elif event == "UPDATE_RECV_END":
                coord_interactions[node_id]["update_actual_bytes"] += _safe_num(rec.get("actual_bytes"), int, 0)

            # Client unary/unary and unary/stream request send
            if event == "CLIENT_RPC_SEND":
                key = (node_id, method)
                method_stats[key]["rpc_calls"] += 1
                b = _safe_num(rec.get("req_bytes"), int, 0)
                method_stats[key]["sent_bytes"] += b
                node_totals[node_id]["rpc_calls"] += 1
                node_totals[node_id]["sent_bytes"] += b
                continue

            # Client unary/unary response receive
            if event == "CLIENT_RPC_RECV":
                key = (node_id, method)
                b = _safe_num(rec.get("resp_bytes"), int, 0)
                rtt = _safe_num(rec.get("rtt_ms"), float, 0.0)
                method_stats[key]["recv_bytes"] += b
                method_stats[key]["rtt_samples"].append(rtt)
                node_totals[node_id]["recv_bytes"] += b
                node_totals[node_id]["rtt_samples"].append(rtt)
                continue

            # Client stream-unary call start/end (SendLocalUpdate)
            if event == "CLIENT_STREAM_START":
                key = (node_id, method)
                method_stats[key]["rpc_calls"] += 1
                node_totals[node_id]["rpc_calls"] += 1
                continue

            if event == "CLIENT_STREAM_END":
                key = (node_id, method)
                b = _safe_num(rec.get("total_sent_bytes"), int, 0)
                c = _safe_num(rec.get("chunks"), int, 0)
                rtt = _safe_num(rec.get("rtt_ms"), float, 0.0)
                method_stats[key]["sent_bytes"] += b
                method_stats[key]["stream_chunks_sent"] += c
                method_stats[key]["rtt_samples"].append(rtt)
                node_totals[node_id]["sent_bytes"] += b
                node_totals[node_id]["stream_chunks_sent"] += c
                node_totals[node_id]["rtt_samples"].append(rtt)
                continue

            # Unary-stream receive complete (FetchGlobalModel response stream)
            if event == "STREAM_RECV_COMPLETE":
                key = (node_id, method)
                b = _safe_num(rec.get("total_bytes"), int, 0)
                c = _safe_num(rec.get("total_chunks"), int, 0)
                rtt = _safe_num(rec.get("rtt_ms"), float, 0.0)
                method_stats[key]["recv_bytes"] += b
                method_stats[key]["stream_chunks_recv"] += c
                method_stats[key]["rtt_samples"].append(rtt)
                node_totals[node_id]["recv_bytes"] += b
                node_totals[node_id]["stream_chunks_recv"] += c
                node_totals[node_id]["rtt_samples"].append(rtt)
                continue

            # Optional chunk-level fallback counters
            if event == "STREAM_CHUNK_SENT":
                key = (node_id, method)
                method_stats[key]["stream_chunks_sent"] += 1
                node_totals[node_id]["stream_chunks_sent"] += 1
                continue
            if event == "STREAM_CHUNK_RECV":
                key = (node_id, method)
                method_stats[key]["stream_chunks_recv"] += 1
                node_totals[node_id]["stream_chunks_recv"] += 1
                continue

    # Build output tables
    method_rows: List[Dict[str, Any]] = []
    for (node_id, method), st in sorted(method_stats.items()):
        rtts = st["rtt_samples"]
        method_rows.append(
            {
                "node_id": node_id,
                "method": method,
                "rpc_calls": st["rpc_calls"],
                "sent_bytes": st["sent_bytes"],
                "recv_bytes": st["recv_bytes"],
                "total_bytes": st["sent_bytes"] + st["recv_bytes"],
                "stream_chunks_sent": st["stream_chunks_sent"],
                "stream_chunks_recv": st["stream_chunks_recv"],
                "avg_rtt_ms": round(sum(rtts) / len(rtts), 4) if rtts else 0.0,
                "max_rtt_ms": round(max(rtts), 4) if rtts else 0.0,
            }
        )

    node_rows: List[Dict[str, Any]] = []
    for node_id, st in sorted(node_totals.items()):
        rtts = st["rtt_samples"]
        node_rows.append(
            {
                "node_id": node_id,
                "rpc_calls": st["rpc_calls"],
                "sent_bytes": st["sent_bytes"],
                "recv_bytes": st["recv_bytes"],
                "total_bytes": st["sent_bytes"] + st["recv_bytes"],
                "stream_chunks_sent": st["stream_chunks_sent"],
                "stream_chunks_recv": st["stream_chunks_recv"],
                "avg_rtt_ms": round(sum(rtts) / len(rtts), 4) if rtts else 0.0,
                "max_rtt_ms": round(max(rtts), 4) if rtts else 0.0,
                "rounds_started": st["rounds_started"],
                "rounds_done": st["rounds_done"],
                "errors": st["errors"],
            }
        )

    coord_rows: List[Dict[str, Any]] = []
    for node_id, st in sorted(coord_interactions.items()):
        coord_rows.append({"node_id": node_id, **st})

    event_rows: List[Dict[str, Any]] = []
    for (node_id, event), count in sorted(event_counts.items()):
        event_rows.append({"node_id": node_id, "event": event, "count": count})

    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "node_summary": output_dir / "node_summary.csv",
        "method_summary": output_dir / "method_summary.csv",
        "coordinator_interactions": output_dir / "coordinator_interactions.csv",
        "event_counts": output_dir / "event_counts.csv",
    }

    _write_csv(
        out_paths["node_summary"],
        node_rows,
        [
            "node_id",
            "rpc_calls",
            "sent_bytes",
            "recv_bytes",
            "total_bytes",
            "stream_chunks_sent",
            "stream_chunks_recv",
            "avg_rtt_ms",
            "max_rtt_ms",
            "rounds_started",
            "rounds_done",
            "errors",
        ],
    )
    _write_csv(
        out_paths["method_summary"],
        method_rows,
        [
            "node_id",
            "method",
            "rpc_calls",
            "sent_bytes",
            "recv_bytes",
            "total_bytes",
            "stream_chunks_sent",
            "stream_chunks_recv",
            "avg_rtt_ms",
            "max_rtt_ms",
        ],
    )
    _write_csv(
        out_paths["coordinator_interactions"],
        coord_rows,
        [
            "node_id",
            "fetch_requests",
            "fetch_total_bytes",
            "fetch_total_chunks",
            "update_submissions",
            "update_expected_bytes",
            "update_actual_bytes",
            "update_total_chunks",
        ],
    )
    _write_csv(out_paths["event_counts"], event_rows, ["node_id", "event", "count"])
    return out_paths


def parse_args() -> argparse.Namespace:
    default_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out_dir = f"analysis/out/analysis_{default_ts}"
    p = argparse.ArgumentParser(
        description="Analyze fedsys telemetry logs and export CSV summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--log-dir",
        default="logs",
        help="Directory containing telemetry*.jsonl files.",
    )
    p.add_argument(
        "--out-dir",
        default=default_out_dir,
        help="Directory to write CSV outputs. Defaults to analysis/out/analysis_<timestamp>.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_paths = analyze(Path(args.log_dir), Path(args.out_dir))
    print("[analysis] Wrote CSV files:")
    for name, path in out_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
