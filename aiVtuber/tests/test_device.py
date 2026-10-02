from __future__ import annotations

import types

import pytest

from vtuber.device import select_device
from vtuber.errors import ConfigError


@pytest.fixture
def fake_torch(monkeypatch):
    """假的 torch：可設定 CUDA / MPS 是否可用，測試不需要安裝 torch。"""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    return torch


def test_auto_prefers_cuda_then_mps_then_cpu(fake_torch):
    assert select_device("auto") == "cpu"
    fake_torch.backends.mps.is_available = lambda: True
    assert select_device("auto") == "mps"
    fake_torch.cuda.is_available = lambda: True
    assert select_device("auto") == "cuda"


def test_explicit_device_that_is_missing_is_an_error(fake_torch):
    with pytest.raises(ConfigError, match="'cuda' is not available"):
        select_device("cuda")


def test_explicit_cpu_is_always_allowed(fake_torch):
    assert select_device("cpu") == "cpu"
