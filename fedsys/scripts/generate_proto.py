"""
Proto compilation script.

Generates fedsys/generated/federated_pb2.py and
fedsys/generated/federated_pb2_grpc.py from proto/federated.proto.

Run once before starting the coordinator or any node:

    python scripts/generate_proto.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_FILE = REPO_ROOT / "proto" / "federated.proto"
OUT_DIR    = REPO_ROOT / "fedsys" / "generated"


def main() -> None:
    if not PROTO_FILE.exists():
        print(f"[generate_proto] ERROR: proto file not found at {PROTO_FILE}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").write_text("# auto-generated — do not edit\n")

    cmd = [
        sys.executable,
        "-m", "grpc_tools.protoc",
        f"--proto_path={PROTO_FILE.parent}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        str(PROTO_FILE.name),
    ]

    print(f"[generate_proto] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print("[generate_proto] FAILED:\n", result.stderr)
        sys.exit(result.returncode)

    # Fix the relative import that grpc_tools generates for Python packages
    grpc_file = OUT_DIR / "federated_pb2_grpc.py"
    if grpc_file.exists():
        content = grpc_file.read_text()
        content = content.replace(
            "import federated_pb2 as federated__pb2",
            "from fedsys.generated import federated_pb2 as federated__pb2",
        )
        grpc_file.write_text(content)

    print(f"[generate_proto] OK  ->  {OUT_DIR}")


if __name__ == "__main__":
    main()
