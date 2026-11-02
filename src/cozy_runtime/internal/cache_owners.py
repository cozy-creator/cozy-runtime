"""Lifecycle-only weak cache owners; native Varena chooses requests and byte amounts."""
from __future__ import annotations

import weakref
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from tensorfs import varena


def supported() -> bool:
    owner = getattr(varena, "CacheOwner", None)
    return owner is not None and getattr(owner, "capability", lambda: 0)() == 1


@contextmanager
def try_locks(locks: list[Any]) -> Iterator[bool]:
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
    def __init__(self, device: int, target: Any, method: str) -> None:
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


def register(device: Any, target: Any, method: str) -> Registration | None:
    if device.type != "cuda" or not supported():
        return None  # supported SDK skew retains the existing synchronous owner lifecycle
    return Registration(device.index or 0, target, method)
