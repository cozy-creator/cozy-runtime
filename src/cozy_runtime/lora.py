"""A LoRA a model owns and arms per request (a turbo or lightning overlay), and the one rule every
LoRA update follows, a package's or the Runtime's own PEFT layers'.

`accumulate` adds `scale * (x @ down.T) @ up.T` into the base output in place: at the activation's
precision (never a copy of `x` in the factors' dtype), with no `[rows, out]` temporary. A caller
whose rows are split across GPUs (Ulysses) passes `chunk`: fixed-row GEMMs, so a row's bits never
depend on how many rows a GPU holds. `Hook` is a linear's permanent forward hook for an overlay:
the Runtime bakes the armed factors into the linear's resident bytes when it can (`_cozy_baked`),
and the hook adds nothing then."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch.nn import functional as F


def accumulate(
    x: torch.Tensor,
    out: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    scale: float,
    *,
    chunk: int | None = None,
) -> None:
    """`out += scale * (x @ down.T) @ up.T`, in place: one GEMM pair, or with `chunk` one per
    `chunk` rows, a short tail padded to it (through a temporary)."""
    if not isinstance(x, torch.Tensor):
        raise ValueError("a LoRA update requires the original tensor operand")
    if not out.is_contiguous():
        raise ValueError("a LoRA update accumulates into a contiguous base output only")
    flat, inputs = out.view(-1, out.shape[-1]), x.reshape(-1, x.shape[-1])
    # Each factor read once (`.t()` reads a weight varena holds unbound; `.to` alone defers it)
    down, up = down.to(inputs.dtype).t(), up.to(flat.dtype).t()
    if chunk is None:
        flat.addmm_(inputs.mm(down).to(flat.dtype), up, alpha=scale)
        return
    for start in range(0, inputs.shape[0], chunk):
        rows = inputs[start : start + chunk].contiguous()  # row-major whatever x's layout
        count = rows.shape[0]
        if count == chunk:
            flat[start : start + count].addmm_(rows.mm(down).to(flat.dtype), up, alpha=scale)
            continue
        pad = chunk - count
        low = F.pad(rows, (0, 0, 0, pad)).mm(down).to(flat.dtype)
        flat[start:].copy_(F.pad(flat[start:], (0, 0, 0, pad)).addmm_(low, up, alpha=scale)[:count])


class Factors(Protocol):
    """One linear's armed factors, a module of the overlay's own loaded model: its weights are bound
    while the overlay's component is held, else each is read on demand. Its forward `(x, out)` adds
    the update (`accumulate`)."""

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> None: ...

    def factors(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(down [r, in], up [out, r], scale)."""
        ...


class Hook:
    """A linear's permanent forward hook for an overlay: `armed()` is this linear's factors while
    an overlay is armed, else None (a base forward)."""

    __slots__ = ("_cozy_baked", "armed")

    def __init__(self, armed: Callable[[], Factors | None]) -> None:
        self.armed, self._cozy_baked = armed, False

    def __call__(self, module: Any, args: tuple[Any, ...], out: torch.Tensor) -> None:
        factors = self.armed()
        if factors is not None and not self._cozy_baked:
            factors(args[0], out)

    def lora_factors(self) -> tuple[object, torch.Tensor, torch.Tensor, float] | None:
        """The Runtime's bake reads these (`weights.LoRASource`)."""
        factors = self.armed()
        return None if factors is None else (factors, *factors.factors())
