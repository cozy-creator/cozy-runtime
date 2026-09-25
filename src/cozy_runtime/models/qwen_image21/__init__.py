"""Qwen Image 2.1 models; optional ML imports are loaded only on explicit use."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import QwenImage21Model

__all__ = ["QwenImage21Model"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import model

    value = getattr(model, name)
    globals()[name] = value
    return value
