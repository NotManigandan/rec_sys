import pytest
import torch

from recsys import device as device_module


def test_resolve_runtime_device_cpu() -> None:
    resolved = device_module.resolve_runtime_device("cpu")
    assert resolved == torch.device("cpu")


def test_resolve_runtime_device_reports_cuda_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)

    def fail_empty(*args, **kwargs):
        raise RuntimeError("CUDA unknown error")

    monkeypatch.setattr(device_module.torch, "empty", fail_empty)

    with pytest.raises(RuntimeError, match="Failed to initialize CUDA") as exc_info:
        device_module.resolve_runtime_device("cuda")

    assert "CUDA_VISIBLE_DEVICES='0'" in str(exc_info.value)
