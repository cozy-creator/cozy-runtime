"""Position-wise feed-forward modules run in token pieces when their intermediate would not fit
(memory v2 §3 rung 3: cuttable work is cut before any weight leaves).

Each token's row of a feed-forward module depends on that token alone, so the pieces join into
exactly the whole module's output, and its widest intermediate (H3's [T, 28672] projection and
[T, 14336] SwiGLU at 15 s, 6.4 and 3.2 GB) never exists whole. While the module fits, it runs as
one whole call: one O(1) check, no copy and no allocation of its own.

Found by class, never by guessing a module's structure: diffusers' `FeedForward` and its
subclasses, and MLPs with gate, up and down projections (transformers' Llama/Qwen family). A
module these miss declares its cut through `author.split`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cozy_runtime.author._errors import MEMORY_EXHAUSTED, is_device_oom
from cozy_runtime.author._split import Cut, split


def widest(module: Any) -> int:
    """The widest intermediate one token takes in `module`, in elements (0: not a position-wise
    feed-forward module this recognises)."""
    diffusers = any(
        kind.__name__ == "FeedForward" and kind.__module__.startswith("diffusers.")
        for kind in type(module).__mro__
    )
    gated = all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj"))
    if not (diffusers or gated):
        return 0
    widths = [
        width
        for child in module.modules()
        if isinstance(width := getattr(child, "out_features", None), int)
    ]
    return max(widths, default=0)


def pieces(
    forward: Callable[..., Any], width: int, fits: Callable[[int], bool]
) -> Callable[..., Any]:
    """`forward` whole while `fits` says its intermediate (twice `width` per token: the
    projection and what its activation makes of it) has room, else in token pieces
    (`author.split`); a whole call that runs out of memory anyway runs again in pieces."""

    def run(*args: Any, **kwargs: Any) -> Any:
        if len(args) != 1 or kwargs or not hasattr(args[0], "element_size"):
            return forward(*args, **kwargs)
        x = args[0]
        if x.dim() < 2:
            return forward(x)
        need = x.numel() // x.shape[-1] * 2 * width * x.element_size()
        if fits(need):
            try:
                return forward(x)
            except Exception as exc:
                if not is_device_oom(exc) and getattr(exc, "code", None) != MEMORY_EXHAUSTED:
                    raise
                exc.__traceback__ = None  # the failed call's frames free their tensors
        return split(forward, x, Cut(x.dim() - 2), per_element=need / x.numel())

    return run
