"""
Integration test: TwoLayerModel + synthetic CSV data + val/test evaluation.

Runs a full coordinator + 2 nodes for 3 rounds, then prints:
  - Per-round validation metrics (loss, accuracy, AUC-ROC)
  - Final test-set metrics
  - Which checkpoint was saved as model_best.pt

Usage
-----
    python scripts/_test_simple.py
"""

import json
import os
import sys
import subprocess
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("logs", exist_ok=True)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
META_PATH = os.path.join(DATA_DIR, "meta.json")
VAL_CSV   = os.path.join(DATA_DIR, "val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

# ── Step 1: generate data (including val/test splits) ────────────────────────
if not os.path.exists(VAL_CSV) or not os.path.exists(TEST_CSV):
    print("[test] val/test CSVs not found — regenerating data ...")
    subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py",
         "--num-users", "1000", "--num-items", "500",
         "--num-samples", "10000", "--num-partitions", "2",
         "--val-fraction", "0.10", "--test-fraction", "0.10"],
        check=True,
    )

with open(META_PATH) as fh:
    meta = json.load(fh)

NUM_USERS = meta["num_users"]
NUM_ITEMS = meta["num_items"]

# ── Step 2: coordinator + model config ───────────────────────────────────────
from fedsys.config import CoordinatorConfig, ModelConfig, NodeConfig
from fedsys.coordinator.server import serve
from fedsys.data.synthetic_dataset import build_synthetic_dataloader
from fedsys.node.client import FederatedNode

model_cfg = ModelConfig(
    model_type="simple",
    num_users=NUM_USERS,
    num_items=NUM_ITEMS,
    embedding_dim=16,
    mlp_hidden=[64],
)

coord_cfg = CoordinatorConfig(
    host="127.0.0.1", port=50060,
    total_nodes=2, min_nodes=2, num_rounds=3,
    round_timeout_seconds=30,
    checkpoint_dir="checkpoints",
    val_data_path=VAL_CSV,
    test_data_path=TEST_CSV,
    log_dir="logs",
    db_path="logs/coord_test.db",
    log_file="logs/coord_test.jsonl",
)

threading.Thread(target=serve, args=(coord_cfg, model_cfg), daemon=True).start()
time.sleep(0.8)

# ── Step 3: training nodes ───────────────────────────────────────────────────
def make_node(node_id: str, partition: int) -> FederatedNode:
    cfg = NodeConfig(
        node_id=node_id,
        coordinator_host="127.0.0.1",
        coordinator_port=50060,
        device="cpu",
        local_epochs=2,
        batch_size=64,
        log_dir="logs",
        db_path=f"logs/{node_id}.db",
        log_file=f"logs/{node_id}.jsonl",
    )
    dl = build_synthetic_dataloader(
        data_dir=DATA_DIR, partition_index=partition, batch_size=64,
    )
    return FederatedNode(cfg, model_cfg, dl, num_rounds=3)


errors = []

def run(node: FederatedNode) -> None:
    try:
        node.run()
    except Exception as exc:
        errors.append(str(exc))
        import traceback; traceback.print_exc()


t0 = threading.Thread(target=run, args=(make_node("node-0", 0),))
t1 = threading.Thread(target=run, args=(make_node("node-1", 1),))
t0.start(); t1.start()
t0.join(timeout=120); t1.join(timeout=120)

# ── Step 4: report ───────────────────────────────────────────────────────────
print()
if errors:
    print(f"FAILED ({len(errors)} error(s)):")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)

print("PASSED: 2 nodes x 3 rounds | TwoLayerModel | synthetic CSV data")

# Node telemetry summary
for fname in ["logs/node-0.jsonl", "logs/node-1.jsonl"]:
    if not os.path.exists(fname):
        continue
    evts = [json.loads(l) for l in open(fname)]
    train = [e for e in evts if e["event"] == "LOCAL_TRAINING_END"]
    h2d   = [e for e in evts if e["event"] == "H2D_TRANSFER_END"]
    send  = [e for e in evts if e["event"] == "SEND_UPDATE_DONE"]
    avg = lambda lst, k: sum(e[k] for e in lst) / max(len(lst), 1)
    print(
        f"  {fname}: {len(evts)} events | "
        f"avg_train={avg(train,'elapsed_ms'):.1f} ms | "
        f"avg_h2d={avg(h2d,'elapsed_ms'):.1f} ms | "
        f"avg_net={avg(send,'net_ms'):.1f} ms"
    )

# Checkpoint summary
print()
ckpt_dir = "checkpoints"
for fname in sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []:
    path = os.path.join(ckpt_dir, fname)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {ckpt_dir}/{fname}  ({size_kb:.1f} KB)")
