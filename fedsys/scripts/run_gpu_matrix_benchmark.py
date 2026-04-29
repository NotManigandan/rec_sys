from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in current environment.")

print(
    "GPU:",
    torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
)

from fedsys.adversarial.attack.target import choose_target_genre, select_target_item
from fedsys.data.movielens_dataset import load_movielens_dataset

ml = load_movielens_dataset("data/", "ml-1m", show_progress=False)
target_genre = choose_target_genre(ml, min_segment_users=20)
target_item = select_target_item(ml, target_genre, min_popularity=1)
print("Target:", target_genre, target_item)

outdir = ROOT / "logs" / "gpu_full"
outdir.mkdir(parents=True, exist_ok=True)

MODELS = ["simple", "bpr", "neural_cf", "two_tower"]
BASE_PORT = 50500


def start_proc(args: list[str], out: Path, err: Path) -> subprocess.Popen:
    out.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        args,
        stdout=open(out, "w", encoding="utf-8"),
        stderr=open(err, "w", encoding="utf-8"),
    )


def parse_metrics(coord_out: Path) -> dict:
    text = coord_out.read_text(encoding="utf-8", errors="ignore").splitlines()
    m = {
        "val_hit10": None,
        "val_ndcg10": None,
        "adv_target_hit10": None,
        "adv_target_ndcg10": None,
        "test_hit10": None,
        "test_ndcg10": None,
        "test_target_hit10": None,
        "test_target_ndcg10": None,
    }
    for line in text:
        r = re.search(r"Epoch\s+\d+\s+hit@10=(\d+\.\d+)\s+ndcg@10=(\d+\.\d+)", line)
        if r:
            m["val_hit10"] = float(r.group(1))
            m["val_ndcg10"] = float(r.group(2))
        r = re.search(
            r"Adv epoch\s+\d+\s+target_hit@10=(\d+\.\d+)\s+target_ndcg@10=(\d+\.\d+)",
            line,
        )
        if r:
            m["adv_target_hit10"] = float(r.group(1))
            m["adv_target_ndcg10"] = float(r.group(2))
        r = re.search(r"\s+hit@10:\s*(\d+\.\d+)", line)
        if r:
            m["test_hit10"] = float(r.group(1))
        r = re.search(r"\s+ndcg@10:\s*(\d+\.\d+)", line)
        if r:
            m["test_ndcg10"] = float(r.group(1))
        r = re.search(r"\s+target_hit@10:\s*(\d+\.\d+)", line)
        if r:
            m["test_target_hit10"] = float(r.group(1))
        r = re.search(r"\s+target_ndcg@10:\s*(\d+\.\d+)", line)
        if r:
            m["test_target_ndcg10"] = float(r.group(1))
    return m


def run_case(model: str, mode: str, port: int) -> dict:
    prefix = outdir / f"{mode}_{model}"
    c_out = Path(str(prefix) + "_coord.out.log")
    c_err = Path(str(prefix) + "_coord.err.log")
    n0_out = Path(str(prefix) + "_n0.out.log")
    n0_err = Path(str(prefix) + "_n0.err.log")
    n1_out = Path(str(prefix) + "_n1.out.log")
    n1_err = Path(str(prefix) + "_n1.err.log")

    rounds = "1"
    epochs = "1"

    coord_args = [
        sys.executable,
        "scripts/run_coordinator.py",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--total-nodes",
        "2",
        "--min-nodes",
        "2",
        "--num-rounds",
        rounds,
        "--round-timeout",
        "180",
        "--model-type",
        model,
        "--ml-data-root",
        "data/",
        "--ml-variant",
        "ml-1m",
        "--defense",
        "none",
        "--adv-target-item",
        str(target_item),
        "--adv-target-genre",
        target_genre,
        "--checkpoint-dir",
        f"checkpoints/gpu_full_{mode}_{model}",
    ]
    if mode == "adversarial":
        coord_args += ["--attack-max-synth", "400"]

    t0 = time.time()
    coord = start_proc(coord_args, c_out, c_err)
    time.sleep(3)

    node_common = [
        "--coordinator",
        f"127.0.0.1:{port}",
        "--movielens",
        "data/",
        "--ml-variant",
        "ml-1m",
        "--num-partitions",
        "2",
        "--model-type",
        model,
        "--device",
        "cuda:0",
        "--local-epochs",
        epochs,
        "--batch-size",
        "512",
        "--num-rounds",
        rounds,
    ]

    n0_args = [
        sys.executable,
        "scripts/run_node.py",
        "--node-id",
        f"{mode}-{model}-n0",
        "--partition",
        "0",
    ] + node_common

    if mode == "adversarial":
        # Clean node passes --attack-max-synth to reserve the same embedding rows
        # as the coordinator, without actually poisoning (no --attack flag).
        n0_args = [
            sys.executable,
            "scripts/run_node.py",
            "--node-id",
            f"{mode}-{model}-n0",
            "--partition",
            "0",
        ] + node_common + [
            "--attack-max-synth", "400",
        ]
        n1_args = [
            sys.executable,
            "scripts/run_node.py",
            "--node-id",
            f"{mode}-{model}-n1",
            "--partition",
            "1",
        ] + node_common + [
            "--attack",
            "--attack-target-item",
            str(target_item),
            "--attack-target-genre",
            target_genre,
            "--attack-budget",
            "0.30",
            "--attack-max-synth",
            "400",
            "--attack-num-filler",
            "30",
            "--attack-num-neutral",
            "20",
            "--attack-target-weight",
            "1.0",
        ]
    else:
        n1_args = [
            sys.executable,
            "scripts/run_node.py",
            "--node-id",
            f"{mode}-{model}-n1",
            "--partition",
            "1",
        ] + node_common

    n0 = start_proc(n0_args, n0_out, n0_err)
    n1 = start_proc(n1_args, n1_out, n1_err)

    rc_n0 = n0.wait()
    rc_n1 = n1.wait()
    rc_c = coord.wait()
    elapsed = time.time() - t0

    errs = []
    for p in [c_err, n0_err, n1_err]:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if txt:
            errs.append({"file": str(p), "head": "\n".join(txt.splitlines()[:12])})

    metrics = parse_metrics(c_out)
    ok = rc_n0 == 0 and rc_n1 == 0 and rc_c == 0 and len(errs) == 0
    return {
        "model": model,
        "mode": mode,
        "runtime_sec": round(elapsed, 2),
        "port": port,
        "ok": ok,
        "exit_codes": {"coord": rc_c, "n0": rc_n0, "n1": rc_n1},
        "metrics": metrics,
        "errors": errs,
    }


selected_modes = ["normal", "adversarial"]
if len(sys.argv) > 1 and sys.argv[1] in {"normal", "adversarial"}:
    selected_modes = [sys.argv[1]]

results = []
port = BASE_PORT if selected_modes[0] == "normal" else BASE_PORT + 4
for mode in selected_modes:
    for model in MODELS:
        print(f"RUN {mode} {model} on port {port}")
        res = run_case(model, mode, port)
        results.append(res)
        print(" ->", "OK" if res["ok"] else "FAIL", "runtime", res["runtime_sec"], "s")
        port += 1

report = {"target_genre": target_genre, "target_item": target_item, "results": results}
(outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nSUMMARY")
for r in results:
    m = r["metrics"]
    print(
        f"{r['mode']:11s} {r['model']:10s} ok={r['ok']} rt={r['runtime_sec']:7.2f}s "
        f"test_hit10={m['test_hit10']} test_ndcg10={m['test_ndcg10']} "
        f"test_target_hit10={m['test_target_hit10']}"
    )

print("\nDIFF (adversarial - normal)")
by_model = {m: {} for m in MODELS}
for r in results:
    by_model[r["model"]][r["mode"]] = r

for mname in MODELS:
    n = by_model[mname].get("normal")
    a = by_model[mname].get("adversarial")
    if not n or not a:
        continue
    nm = n["metrics"]
    am = a["metrics"]

    def d(k: str):
        if nm.get(k) is None or am.get(k) is None:
            return None
        return round(am[k] - nm[k], 6)

    print(
        mname,
        "d_test_hit10=",
        d("test_hit10"),
        "d_test_ndcg10=",
        d("test_ndcg10"),
        "d_target_hit10=",
        d("test_target_hit10"),
        "d_target_ndcg10=",
        d("test_target_ndcg10"),
        "rt_normal=",
        n["runtime_sec"],
        "rt_adv=",
        a["runtime_sec"],
    )
