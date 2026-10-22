"""Component-owned library and graph scratch, serialized across its graphs.

Torch 2.14 resets cuBLAS workspaces on a graph's capture stream. Region graphs therefore
begin/end on a control stream and fork onto a separate, uniquely owned library stream. A
small owner graph on the library stream is reset only at final domain teardown; it is never
replayed. Its reset frees that stream's cached library pointers before the pool/streams die.

A component's graphs also share one capture pool. `using` serializes every region call of the
component on the device (a host lock and a completion event across streams), and each replay
copies its inputs in and its outputs out inside that interval, so no two graphs' scratch is
ever live at once: the pool holds one region's working set plus every graph's static buffers.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class Unavailable(RuntimeError):
    """This runtime cannot honor the graph domain's workspace lifetime contract."""


def unsupported(torch: Any) -> str:
    return (
        ""
        if torch.__version__.split(".")[:2] == ["2", "14"]
        else ("library-workspace graph lifetime is qualified only on Torch 2.14")
    )


class Domain:
    def __init__(self, torch: Any, memory: Any) -> None:
        self.torch, self.memory = torch, memory
        self.device = torch.cuda.current_device()
        self.lock = threading.RLock()
        self.stream: Any = None
        self.control: Any = None
        self.pool: Any = None
        self.owner: Any = None
        self.fork: Any = None
        self.join: Any = None
        self.done: Any = None
        self.last_stream: Any = None
        self.fenced = True
        self.closed = False
        self.config: Any = None
        self.graphs: weakref.WeakSet[Any] = weakref.WeakSet()
        self.local = threading.local()
        #: the graphs' shared capture pool, its accounting cache, and the recorders using it
        self.captures: Any = None
        self.cache: Any = None
        self.recorders: weakref.WeakSet[Any] = weakref.WeakSet()
        #: each graph output's address -> (its recording, its index): segment boundaries
        self.lent: dict[int, tuple[weakref.ref[Any], int]] = {}

    def _stream(self) -> Any:
        pointer = ctypes.c_void_p()
        status = self.torch.cuda.cudart().cudaStreamCreate(ctypes.addressof(pointer))
        code = int(getattr(status, "value", status))
        if code:
            error = self.torch.AcceleratorError(f"cudaStreamCreate failed with CUDA error {code}")
            error.error_code = code
            raise error
        try:
            return self.torch.cuda.ExternalStream(pointer.value, device=self.device)
        except BaseException:
            self.torch.cuda.cudart().cudaStreamDestroy(pointer.value)
            raise

    def configuration(self) -> tuple[Any, ...]:
        cuda = self.torch.backends.cuda
        return (
            cuda.cublas_workspace_size(),
            cuda.cublaslt_workspace_size(),
            cuda.preferred_blas_library(),
        )

    def refusal(self) -> str:
        if self.closed:
            return "graph domain is closed"
        if self.config is not None and self.configuration() != self.config:
            return "library workspace configuration changed while graphs retain its pointers"
        return unsupported(self.torch)

    def initialize(self) -> None:
        """Preparation, including the never-replayed owner, uses the admitted budget."""
        if why := self.refusal():
            raise Unavailable(why)
        if self.stream is not None:
            return
        try:
            self.stream = self._stream()
            self.control = self._stream()
            self.pool = (
                self.memory.new_pool() if self.memory is not None else self.torch.cuda.MemPool()
            )
            self.fork, self.join, self.done = (self.torch.cuda.Event() for _ in range(3))
            self.fork.record(self.control)
            self.join.record(self.stream)
            self.done.record(self.stream)
            self.owner = self.torch.cuda.CUDAGraph()
            # No library work or allocator routing inside this owner graph. Recording an
            # event makes its capture stream explicit; reset later clears only this stream.
            with self.torch.cuda.stream(self.stream):
                self.owner.capture_begin(capture_error_mode="thread_local")
                self.join.record(self.stream)
                self.owner.capture_end()
            self.config = self.configuration()
        except BaseException:
            self.finish()
            raise

    @contextmanager
    def using(self) -> Iterator[None]:
        """Serialize shared workspace and order producers/output copies across streams."""
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
        """Prime the actual capture thread's handle into persistent domain ownership."""
        if why := self.refusal():
            raise Unavailable(why)
        context = (
            self.memory.allocating(self.pool)
            if self.memory is not None
            else self.torch.cuda.use_mem_pool(self.pool)
        )
        with context, self.torch.cuda.stream(self.stream):
            self.torch.cuda.current_blas_handle()

    def finish(self) -> None:
        """Drain all uses, invalidate executables, then free only this domain's resources."""
        # At interpreter shutdown, pinned-host tensor destructors may still need these
        # streams. The CUDA process owns final reclamation; ordinary unload calls us early.
        if sys.is_finalizing():
            return
        with self.lock:
            if self.closed:
                return
            if not self.fenced and self.last_stream is not None:
                self.last_stream.synchronize()
            if self.done is not None:
                self.done.synchronize()
            for stream in (self.stream, self.control):
                if stream is not None:
                    stream.synchronize()
            for graph in self.graphs:
                graph.reset()
            self.graphs.clear()
            if self.owner is not None:
                self.owner.reset()
                self.owner = None
            self.drop_captures()
            self.pool = None  # MemPool destruction releases this pool, never a global flush.
            for stream in (self.stream, self.control):
                if stream is not None:
                    self.torch.cuda.cudart().cudaStreamDestroy(stream.cuda_stream)
            self.stream = self.control = self.fork = self.join = self.done = None
            self.closed = self.fenced = True

    def capture_pool(self) -> Any:
        """The pool every graph of this component records into (made on first use)."""
        if self.captures is None:
            self.captures = (
                self.memory.new_pool() if self.memory is not None else self.torch.cuda.MemPool()
            )
        return self.captures

    def drop_captures(self) -> None:
        """After every graph recorded into it is reset and its static buffers dropped."""
        self.captures = None  # MemPool destruction releases this pool, never a global flush.
        self.lent.clear()
        if self.cache is not None:
            self.cache.close()
            self.cache = None

    def nbytes(self) -> int:
        return (
            0
            if self.pool is None
            else sum(s["total_size"] for s in self.pool.snapshot(include_traces=False))
        )

    def capture_bytes(self) -> int:
        return (
            0
            if self.captures is None
            else sum(s["total_size"] for s in self.captures.snapshot(include_traces=False))
        )


_LOCK = threading.RLock()
_OWNERS: weakref.WeakKeyDictionary[Any, Domain] = weakref.WeakKeyDictionary()


def acquire(torch: Any, root: Any, memory: Any = None) -> Domain:
    with _LOCK:
        found = _OWNERS.get(root)
        if found is None:
            found = Domain(torch, memory)
            _OWNERS[root] = found
            finalizer = weakref.finalize(root, found.finish)
            finalizer.atexit = False
        return found


def release(root: Any) -> None:
    with _LOCK:
        found = _OWNERS.pop(root, None)
    if found is not None:
        found.finish()


def existing(root: Any) -> Domain | None:
    """Observe an existing owner without creating a domain or touching CUDA."""
    with _LOCK:
        return _OWNERS.get(root)
