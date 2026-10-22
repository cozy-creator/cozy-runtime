"""An owned capture stream and opaque library workspace shared by serialized regions.

Torch's ordinary Stream factory draws from a finite pool. A domain uses an external stream
it created itself, so unrelated eager work cannot borrow its library workspace accidentally.
Idle domains are reused between component generations, like Torch's process context caches;
their charged library allocations never belong to one disposable region graph.
"""

from __future__ import annotations

import ctypes
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class Domain:
    def __init__(self, torch: Any, memory: Any) -> None:
        self.torch, self.memory = torch, memory
        self.device = torch.cuda.current_device()
        self.lock = threading.RLock()
        self.stream: Any = None
        self.pool: Any = None
        self.done: Any = None
        self.last_stream: Any = None
        self.fenced = True
        self.owned = True
        self.generation = 0
        self.local = threading.local()

    def initialize(self) -> None:
        """Driver/library preparation runs under the caller's admitted discovery budget."""
        if self.stream is not None:
            return
        pointer = ctypes.c_void_p()
        status = self.torch.cuda.cudart().cudaStreamCreate(ctypes.addressof(pointer))
        code = int(getattr(status, "value", status))
        if code:
            error = self.torch.AcceleratorError(f"cudaStreamCreate failed with CUDA error {code}")
            error.error_code = code
            raise error
        try:
            stream = self.torch.cuda.ExternalStream(pointer.value, device=self.device)
            pool = self.memory.new_pool() if self.memory is not None else self.torch.cuda.MemPool()
            done = self.torch.cuda.Event()
            done.record(stream)
        except BaseException:
            self.torch.cuda.cudart().cudaStreamDestroy(pointer.value)
            raise
        self.stream, self.pool, self.done = stream, pool, done

    @contextmanager
    def using(self) -> Iterator[None]:
        """Serialize every use of shared workspace, including output copies across streams."""
        with self.lock:
            current = self.torch.cuda.current_stream(self.device)
            depth = getattr(self.local, "depth", 0)
            if not depth and self.done is not None:
                current.wait_event(self.done)
                self.last_stream, self.fenced = current, False
            self.local.depth = depth + 1
            try:
                yield
            finally:
                self.local.depth -= 1
                if not depth and self.done is not None:
                    self.last_stream, self.fenced = current, False
                    self.done.record(current)
                    self.fenced = True

    def prime_blas(self) -> None:
        """Initialize this thread's BLAS handle/workspace in the domain's persistent pool."""
        context = (
            self.memory.allocating(self.pool)
            if self.memory is not None
            else self.torch.cuda.use_mem_pool(self.pool)
        )
        with context, self.torch.cuda.stream(self.stream):
            self.torch.cuda.current_blas_handle()

    def finish(self) -> None:
        with self.lock:
            if not self.fenced:
                self.last_stream.synchronize()
                self.fenced = True
            elif self.done is not None:
                self.done.synchronize()
            self.owned = False

    def nbytes(self) -> int:
        return (
            0
            if self.pool is None
            else sum(segment["total_size"] for segment in self.pool.snapshot(include_traces=False))
        )


_LOCK = threading.RLock()
_OWNERS: weakref.WeakKeyDictionary[Any, Domain] = weakref.WeakKeyDictionary()
# The stream and the library's cached pointers have process-context lifetime. Reuse an idle
# domain rather than creating new streams/workspaces for every successive model generation.
_DOMAINS: list[Domain] = []


def acquire(torch: Any, root: Any, memory: Any = None) -> Domain:
    with _LOCK:
        found = _OWNERS.get(root)
        if found is None:
            device = torch.cuda.current_device()
            found = next(
                (d for d in _DOMAINS if not d.owned and d.device == device and d.memory is memory),
                None,
            )
            if found is None:
                found = Domain(torch, memory)
                _DOMAINS.append(found)
            found.owned = True
            found.generation += 1
            _OWNERS[root] = found
            weakref.finalize(root, _retire, found, found.generation)
        return found


def release(root: Any) -> None:
    with _LOCK:
        found = _OWNERS.pop(root, None)
    if found is not None:
        _retire(found, found.generation)


def existing(root: Any) -> Domain | None:
    """Observe an existing owner without creating a domain or touching CUDA."""
    with _LOCK:
        return _OWNERS.get(root)


def _retire(found: Domain, generation: int) -> None:
    with _LOCK:
        if found.generation == generation and found.owned:
            found.finish()
