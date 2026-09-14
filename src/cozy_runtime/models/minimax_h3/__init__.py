"""MiniMax H3 builtins; install ``cozy-runtime[minimax-h3]`` to execute them.

The namespace is lightweight. Accessing a model loads the optional Diffusers
implementation; ordinary Runtime and custom Model imports remain independent.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import H3Model, H3TurboBase, H3TurboLoRA

__all__ = ["H3Model", "H3TurboBase", "H3TurboLoRA"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import model

    value = getattr(model, name)
    globals()[name] = value
    return value
