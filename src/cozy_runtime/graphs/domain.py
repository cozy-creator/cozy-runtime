"""Component-owned library and graph scratch, serialized across its graphs.

Torch 2.14 resets cuBLAS workspaces on a graph's capture stream. Region graphs therefore
begin/end on a control stream and fork onto a separate, uniquely owned library stream. A
small owner graph on the library stream is reset only at final domain teardown; it is never
replayed. Its reset frees that stream's cached library pointers before the pool/streams die.

A placed model's components run one after another on a device, so all their graphs share the
device's one domain and its capture-pool token (`Pool`). `using` serializes graph use on the
device (a host lock and
a completion event across streams), and each replay copies its inputs in and its outputs out
inside that interval, so no two graphs' scratch is ever live at once: the pool holds one
region's working set plus every graph's static buffers.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any

from cozy_runtime.author._context import device as placed_device
from cozy_runtime.internal import accel


class Unavailable(RuntimeError):
    """This runtime cannot honor the graph domain's workspace lifetime contract."""


def unsupported(torch: Any) -> str:
    return (
        ""
        if torch.__version__.split(".")[:2] == ["2", "14"]
        else ("library-workspace graph lifetime is qualified only on Torch 2.14")
    )


@dataclass(eq=False, slots=True)
class Pool:
    """A stock Torch graph-pool token. Graphs and their static tensors own its allocations.

    Runtime retires every graph and waits for its readers before dropping those tensors.
    The native weight manager does not own this activation/scratch pool.
    """

    handle: tuple[int, int]


class Domain:
    def __init__(self, torch: Any, memory: Any) -> None:
        self.torch, self.memory = torch, memory
        self.device = placed_device().index or 0
        self.lock = threading.RLock()
        self.stream: Any = None
        self.control: Any = None
        self.primed = False
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

    def recovery_blocked(self, *, replay: bool = False) -> AbstractContextManager[Any]:
        """Block reclaim; actual capture also excludes native producers."""
        from tensorfs import varena

        # Older native cohorts retain their conservative capture guard.
        guard = getattr(varena, "replay", varena.capture) if replay else varena.capture
        return guard(self.device)

    def initialize(self) -> None:
        """Preparation, including the never-replayed owner, uses the admitted budget."""
        if why := self.refusal():
            raise Unavailable(why)
        if self.stream is not None:
            return
        from cozy_runtime.internal import kernel_ready

        # Refuse before allocating the owner graph/streams; the same device's background
        # kernel probes must not invalidate this small lifetime-owner capture either.
        with kernel_ready.recording():
            try:
                self.stream = self._stream()
                self.control = self._stream()
                self.fork, self.join, self.done = (self.torch.cuda.Event() for _ in range(3))
                self.fork.record(self.control)
                self.join.record(self.stream)
                self.done.record(self.stream)
                self.owner = self.torch.cuda.CUDAGraph()
                # No library work or allocation inside this owner graph. Recording an event makes
                # its capture stream explicit; reset later clears only this stream.
                with self.recovery_blocked(), self.torch.cuda.stream(self.stream):
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
        """The library stream's cuBLAS and cuDNN handles and workspaces, which torch keeps per
        stream for the process's life: made once, outside any request, so they live in the
        process and never depend on a request's temporary allocations."""
        if why := self.refusal():
            raise Unavailable(why)
        if not self.primed:
            with self.torch.cuda.stream(self.stream):
                accel.first_launches(self.torch, "cuda")
            self.primed = True

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
            for stream in (self.stream, self.control):
                if stream is not None:
                    self.torch.cuda.cudart().cudaStreamDestroy(stream.cuda_stream)
            self.stream = self.control = self.fork = self.join = self.done = None
            self.closed = self.fenced = True


_LOCK = threading.RLock()
_OWNERS: weakref.WeakKeyDictionary[Any, Domain] = weakref.WeakKeyDictionary()
#: per device memory, the one domain its placed models' graphs share
_DEVICES: weakref.WeakKeyDictionary[Any, Domain] = weakref.WeakKeyDictionary()


def device(torch: Any, memory: Any) -> Domain:
    """The device's domain: every placed component's graphs, serialized."""
    with _LOCK:
        found = _DEVICES.get(memory)
        if found is None:
            found = _DEVICES[memory] = Domain(torch, memory)
            finalizer: Any = weakref.finalize(memory, found.finish)
            finalizer.atexit = False
        return found


def acquire(torch: Any, root: Any, memory: Any = None) -> Domain:
    with _LOCK:
        found = _OWNERS.get(root)
        if found is None:
            found = Domain(torch, memory)
            _OWNERS[root] = found
            finalizer: Any = weakref.finalize(root, found.finish)
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
