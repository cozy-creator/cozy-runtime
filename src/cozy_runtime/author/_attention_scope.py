"""Explicit model-owned layout for approximate attention calls."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AttentionLayout:
    """One live document and its protected prefix, supplied by model inference.

    Dense paths are component-relative attention module prefixes. They protect
    refiners or sensitive layers without making a generic backend infer model
    semantics from tensor values. Step indices are zero-based. This describes
    inference only; it does not authorize a backend or select a kernel.
    """

    live_tokens: int
    protected_prefix: int
    step: int
    dense_until_step: int = 0
    dense_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("live_tokens", "protected_prefix", "step", "dense_until_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.live_tokens or self.protected_prefix > self.live_tokens:
            raise ValueError("attention layout requires a live document containing its prefix")
        if not isinstance(self.dense_paths, tuple) or any(
            not isinstance(path, str) or not path or path.startswith(".") or path.endswith(".")
            for path in self.dense_paths
        ):
            raise ValueError("dense_paths must be a tuple of nonempty module prefixes")


_ACTIVE_LAYOUT: ContextVar[AttentionLayout | None] = ContextVar(
    "cozy_attention_layout", default=None
)


@contextmanager
def attention_scope(layout: AttentionLayout) -> Iterator[None]:
    """Apply layout only within this inference call; restore it even on failure."""
    if not isinstance(layout, AttentionLayout):
        raise TypeError("attention_scope requires an AttentionLayout")
    token = _ACTIVE_LAYOUT.set(layout)
    try:
        yield
    finally:
        _ACTIVE_LAYOUT.reset(token)


#: What the runtime does with an early length statement; the executor's attention installs it.
_EXPECTED: list[Callable[[int, Any], None]] = []


def expect_attention(live_tokens: int, module: Any) -> None:
    """State, before inference reaches it, the live document length `module`'s attention sites
    will see, so a kernel specialized on that length can start preparing now (h3a-087).

    Advisory only: the layout each forward supplies stays the authority, and a wrong length
    costs a wasted background compile, never an output."""
    if type(live_tokens) is int and live_tokens > 0:
        for listener in _EXPECTED:
            listener(live_tokens, module)
