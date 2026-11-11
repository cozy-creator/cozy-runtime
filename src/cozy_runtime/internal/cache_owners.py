"""Lifecycle-only weak cache owners; native Varena chooses requests and byte amounts."""
from __future__ import annotations

import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Protocol

from tensorfs.varena import _runtime as varena

if TYPE_CHECKING:
    import torch


class _Lock(Protocol):
    def acquire(self, blocking: bool = ..., timeout: float = ...) -> bool: ...
    def release(self) -> None: ...


@contextmanager
def try_locks(locks: Sequence[_Lock | None]) -> Iterator[bool]:
    """Claim actual owner/domain locks without waiting on the allocating caller."""
    acquired = []
    seen = set()
    try:
        for lock in locks:
            if lock is None or id(lock) in seen:
                continue
            seen.add(id(lock))
            if not lock.acquire(blocking=False):
                yield False
                return
            acquired.append(lock)
        yield True
    finally:
        for lock in reversed(acquired):
            lock.release()


class Registration:
    def __init__(self, device: int, target: object, method: str) -> None:
        reference = weakref.ref(target)

        def shrink(_requested_bytes: int) -> bool:
            import torch

            # Native has acquired the GIL and capture-only exclusion before upgrading
            # this weak owner. Every temporary owning reference dies in that scope too.
            with torch.cuda.device(device):
                found = reference()
                return False if found is None else bool(getattr(found, method)())

        self.callback = shrink  # native keeps only a weak callable reference
        self.native = varena.CacheOwner(device, shrink)
        self.stopping = False

    def check(self) -> None:
        self.native.check()

    def touch(self) -> None:
        if not self.stopping:
            self.native.touch()

    def suspend(self) -> None:
        if not self.stopping:
            self.native.suspend()

    def stop(self) -> None:
        if not self.stopping:
            self.stopping = True
            self.native.defer_close()

    def close(self) -> None:
        self.stop()
        # Worker/capture refusal retains the weak callable; admission is already closed.
        # The native worker drains it later without an allocator waiting for Python.
        with suppress(varena.CacheOwnerDeferredError):
            self.native.close()


def register(device: torch.device, target: object, method: str) -> Registration | None:
    if device.type != "cuda":
        return None
    return Registration(device.index or 0, target, method)
