"""The package contract's seal on torch (CONTRACT §9): what a package may not reach raises, with
the contract's API named. Applied once per executor, before its package is imported; the Runtime's
own code never calls these names (it reads varena: `vt.stats`, `vt.room`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

from cozy_runtime.author._errors import ContractViolation

#: Room and allocator readers: memory is varena's, and nothing sizes work from these.
_ROOM = (
    "mem_get_info",
    "memory_stats",
    "memory_allocated",
    "memory_reserved",
    "max_memory_allocated",
    "max_memory_reserved",
    "reset_peak_memory_stats",
    "set_per_process_memory_fraction",
)
_USE = (
    "memory is the Runtime's: an op too big for the card is `author.split`, a layout a loop keeps "
    "is `author.windows`, state kept across steps is `author.retained`"
)


#: The sealed names' originals: for varena's own gate cells, which read the driver on purpose.
_ORIGINAL: dict[str, Callable[..., Any]] = {}


def unsealed(torch: Any, name: str) -> Callable[..., Any]:
    """`torch.cuda.<name>` as torch has it, sealed or not. Never for sizing work."""
    return _ORIGINAL.get(name) or getattr(torch.cuda, name)


def _raiser(name: str) -> Callable[..., NoReturn]:
    def refused(*_: Any, **__: Any) -> NoReturn:
        raise ContractViolation(f"torch.cuda.{name} is not in the package contract: {_USE}")

    refused.__name__ = refused.__qualname__ = name
    return refused


def seal(torch: Any) -> None:
    """Replace the room readers on `torch.cuda` (and `torch.cuda.memory`) with raisers."""
    for name in _ROOM:
        if hasattr(torch.cuda, name) and name not in _ORIGINAL:
            _ORIGINAL[name] = getattr(torch.cuda, name)
    for module in (torch.cuda, torch.cuda.memory):
        for name in _ROOM:
            if hasattr(module, name):
                setattr(module, name, _raiser(name))
