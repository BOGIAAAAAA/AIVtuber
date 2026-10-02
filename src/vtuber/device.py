"""推論裝置選擇（需要 torch，只在用到模型時呼叫）。"""

from __future__ import annotations

from vtuber.errors import ConfigError


def select_device(preference: str = "auto") -> str:
    """回傳 torch 裝置名稱：auto 依序選 cuda > mps > cpu；指定的裝置不可用時丟出 ConfigError。"""
    import torch

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if preference == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available.get(preference, False):
        raise ConfigError(f"device '{preference}' is not available on this machine")
    return preference
