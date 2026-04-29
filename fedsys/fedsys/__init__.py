"""
FedSys — gRPC Federated Learning Framework.

Package layout
--------------
fedsys/
├── config.py           — shared configuration dataclasses
├── privacy.py          — privacy middleware placeholder
├── logging_async.py    — non-blocking JSONL + SQLite telemetry sink
├── interceptors.py     — gRPC client & server interceptors
├── coordinator/        — server, FedAvg aggregator, node registry
├── node/               — state machine, trainer, gRPC client
├── data/               — Amazon 2023 dataset + synthetic fallback
├── models/             — NCF recommendation model (~300 M params)
└── generated/          — auto-generated protobuf stubs (run scripts/generate_proto.py)
"""
