"""An attempt's device allocation failures reach the memory manager, never a library's fallback.

cuDNN tries its heuristic engines in order. When one's workspace cannot be allocated it runs the
next, and PyTorch keeps that engine in the calling thread's plan cache for every later call of
the same shape (`Conv_v8.cpp`, `try_configs`). The executor's call threads outlive attempts, so a
conv first run with too little free memory changed the numerics of every later attempt that
thread served: after a crashed H3 attempt, its audio decode differed until the executor was
replaced. On an attempt's thread the allocator's failure is raised instead. varena's catch, which
the attempt's request opens (`cuts.caught`), makes room and runs the op again with the engine the
heuristics prefer, so a cached engine never depends on the free memory of its first call.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

_LOCAL = threading.local()
_INSTALLED: list[bool] = []


def install(torch: ModuleType) -> None:
    """Observe the process's device allocation failures, once."""
    if _INSTALLED:
        return
    out_of_memory = torch.OutOfMemoryError

    def observed(device: int, alloc: int, _limit: int, free: int) -> None:
        if getattr(_LOCAL, "depth", 0):
            raise out_of_memory(
                f"CUDA out of memory. Tried to allocate {alloc} B on device {device} ({free} B "
                "free); raised to the memory manager instead of a library's fallback"
            )

    torch._C._cuda_attach_out_of_memory_observer(observed)
    _INSTALLED.append(True)


@contextmanager
def surfaced() -> Iterator[None]:
    """This thread's allocation failures raise until the scope ends (an attempt's request)."""
    _LOCAL.depth = getattr(_LOCAL, "depth", 0) + 1
    try:
        yield
    finally:
        _LOCAL.depth -= 1
