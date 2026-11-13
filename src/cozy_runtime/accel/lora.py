"""A LoRA a model owns and arms per request (a turbo or lightning overlay), and the one rule every
LoRA update follows, a package's or the Runtime's own PEFT layers'.

`accumulate` adds `scale * (x @ down.T) @ up.T` into the base output in place: at the activation's
precision (never a copy of `x` in the factors' dtype), with no `[rows, out]` temporary. A caller
whose rows are split across GPUs (Ulysses) passes `chunk`: fixed-row GEMMs, so a row's bits never
depend on how many rows a GPU holds. `Hook` is a linear's permanent forward hook for an overlay:
the Runtime bakes the armed factors into the linear's resident bytes when it can (`_cozy_baked`),
and the hook adds nothing then. `Sites` puts one on every linear of a component and arms an
overlay's factors (`Pair`, by module path) for a block of calls: an adapter's LoRA, applied only
while the adapter steers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import Protocol

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
    try:  # a column range of a packed sibling output too: its rows are a strided 2-D view
        flat = out.view(-1, out.shape[-1])
    except RuntimeError as exc:
        raise ValueError("a LoRA update accumulates into a base output whose rows flatten") from exc
    inputs = x.reshape(-1, x.shape[-1])
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

    def __call__(
        self, module: torch.nn.Module, args: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        factors = self.armed()
        if self._cozy_baked or factors is None:
            return  # Bound weights already contain the delta; the fill owns factor reads.
        factors(args[0], out)

    def lora_factors(self) -> tuple[Factors, torch.Tensor, torch.Tensor, float] | None:
        """The Runtime's bake reads these (`weights.LoRASource`)."""
        factors = self.armed()
        return None if factors is None else (factors, *factors.factors())


class Pair(torch.nn.Module):
    """One linear's factors as most LoRA files store them, `lora_A` [r, in] and `lora_B` [out, r],
    applied as `scale * B @ A`: the `Factors` an overlay arms on `Sites`."""

    def __init__(self, inputs: int, outputs: int, rank: int, scale: float = 1.0) -> None:
        super().__init__()
        self.lora_A = torch.nn.Linear(inputs, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, outputs, bias=False)
        self.scale = scale

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> None:
        accumulate(x, out, self.lora_A.weight, self.lora_B.weight, self.scale)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self.lora_A.weight, self.lora_B.weight, self.scale


class Sites:
    """Where a component takes an overlay its model owns: a `Hook` on each of its linears, put on
    as the model is built. `applied` arms factors by module path for a block of calls. Unarmed,
    the hooks add nothing and the component's calls may replay recorded graphs; armed, its calls
    run as written, with the factors baked into the pages filled for them, else added per call."""

    def __init__(self, root: torch.nn.Module) -> None:
        self.armed: Mapping[str, Factors] | None = None
        linears = [(p, m) for p, m in root.named_modules() if isinstance(m, torch.nn.Linear)]
        self.paths = frozenset(path for path, _ in linears)
        for path, module in linears:
            module.register_forward_hook(_Site(self, path))

    @contextmanager
    def applied(self, factors: Mapping[str, Factors]) -> Iterator[None]:
        """`factors`, each by its linear's path under the root, armed while the block runs."""
        if self.armed is not None:
            raise ValueError("another overlay is armed on these sites")
        if unknown := sorted(set(factors) - self.paths):
            raise ValueError(f"the overlay names {len(unknown)} linears not here: {unknown[0]}")
        self.armed = factors
        try:
            yield
        finally:
            self.armed = None


class _Site(Hook):
    __slots__ = ("sites",)

    def __init__(self, sites: Sites, path: str) -> None:
        super().__init__(partial(_armed, sites, path))
        self.sites = sites


def _armed(sites: Sites, path: str) -> Factors | None:
    return None if sites.armed is None else sites.armed.get(path)
