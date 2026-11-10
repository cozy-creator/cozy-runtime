"""Component-owned library and graph scratch, serialized across its graphs.

Torch 2.15 resets cuBLAS workspaces on a graph's capture stream. Region graphs therefore
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

import sys
import threading
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import tensorfs.varena.torch as vt
from tensorfs.varena import _runtime as varena

from cozy_runtime.author._context import device as placed_device
from cozy_runtime.internal import accel, kernel_ready

if TYPE_CHECKING:
    import torch

    from cozy_runtime.internal.residency import Memory
else:
    try:
        import torch
    except ImportError:  # pragma: no cover - the base package is torch-free
        torch = None


class Unavailable(RuntimeError):
    """This runtime cannot honor the graph domain's workspace lifetime contract."""


def unsupported() -> str:
    return (
        ""
        if torch.__version__.split(".")[:2] in (["2", "14"], ["2", "15"])
        else ("library-workspace graph lifetime is qualified only on Torch 2.14 and 2.15")
    )


@dataclass(eq=False, slots=True)
class Pool:
    """A Varena-backed graph-pool token; graphs, tensors and completion fences own backing.

    Runtime retires every graph and joins its readers before dropping those tensors.
    """

    handle: tuple[int, int]


# A failed CUDA destruction does not prove whether its handle remains valid.
# Keep its events/library/stream owners until the executor exits; never probe it again.
_UNPROVEN: list[Domain] = []


class Domain:
    def __init__(self, memory: Memory | None) -> None:
        self.memory = memory
        self.device = placed_device().index or 0
        self.lock = threading.RLock()
        self.stream: torch.cuda.Stream | None = None
        self.control: torch.cuda.Stream | None = None
        self.primed = False
        self.owner: torch.cuda.CUDAGraph | None = None
        self.fork: torch.cuda.Event | None = None
        self.join: torch.cuda.Event | None = None
        self.done: torch.cuda.Event | None = None
        self.last_stream: torch.cuda.Stream | None = None
        self.fenced = True
        self.closed = False
        self.closing = False
        self._destruction_error: BaseException | None = None
        self.config: _Configuration | None = None
        self.graphs: weakref.WeakSet[torch.cuda.CUDAGraph] = weakref.WeakSet()
        self.local = threading.local()

    def configuration(self) -> _Configuration:
        cuda = torch.backends.cuda
        return (
            cuda.cublas_workspace_size(),
            cuda.cublaslt_workspace_size(),
            cuda.preferred_blas_library(),
        )

    def refusal(self) -> str:
        if self.closed:
            return "graph domain is closed"
        if self.closing:
            return "graph domain is closing"
        if self.config is not None and self.configuration() != self.config:
            return "library workspace configuration changed while graphs retain its pointers"
        return unsupported()

    def capturing(
        self,
    ) -> tuple[torch.cuda.Stream, torch.cuda.Stream, torch.cuda.Event, torch.cuda.Event]:
        """Its control and library streams and fork/join events, for a capture."""
        # `initialize` made them, and every capture follows it.
        return cast(
            "tuple[torch.cuda.Stream, torch.cuda.Stream, torch.cuda.Event, torch.cuda.Event]",
            (self.control, self.stream, self.fork, self.join),
        )

    def recovery_blocked(self, *, replay: bool = False) -> AbstractContextManager[object]:
        """Block reclaim; actual capture also excludes native producers."""
        guard = varena.replay if replay else varena.capture
        return guard(self.device)

    def initialize(self) -> None:
        """Preparation, including the never-replayed owner, uses the admitted budget."""
        if why := self.refusal():
            raise Unavailable(why)
        if self.stream is not None:
            return
        # Refuse before allocating the owner graph/streams; the same device's background
        # kernel probes must not invalidate this small lifetime-owner capture either.
        with kernel_ready.recording():
            try:
                self.stream = stream = accel.owned_stream(torch, self.device)
                self.control = control = accel.owned_stream(torch, self.device)
                fork, join, done = (
                    torch.cuda.Event()  # type: ignore[no-untyped-call]  # unannotated in torch
                    for _ in range(3)
                )
                self.fork, self.join, self.done = fork, join, done
                fork.record(control)
                join.record(stream)
                done.record(stream)
                self.owner = owner = torch.cuda.CUDAGraph()
                # No library work or allocation inside this owner graph. Recording an event makes
                # its capture stream explicit; reset later clears only this stream.
                with vt.graph(owner, stream=stream, reserve_bytes=0):
                    join.record(stream)
                self.config = self.configuration()
            except BaseException:
                self.finish()
                raise

    @contextmanager
    def using(self) -> Iterator[None]:
        """Serialize shared workspace and order producers/output copies across streams."""
        with self.lock:
            if self.closed or self.closing:
                raise Unavailable("graph domain is closing or closed")
            current = torch.cuda.current_stream(self.device)
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
            with torch.cuda.stream(self.stream):
                accel.first_launches(torch, "cuda")
            self.primed = True

    def finish(self) -> None:
        """Drain uses, retire every live stream, then destroy with explicit progress.

        No submission or consumer declaration may follow closing. A fallible retirement
        leaves all handles alive; a successful destroy clears its field immediately. An
        unknown destruction outcome is terminal and retains all remaining custody.
        """
        # At interpreter shutdown, pinned-host tensor destructors may still need these
        # streams. The CUDA process owns final reclamation; ordinary unload calls us early.
        if sys.is_finalizing():
            return
        with self.lock:
            if self._destruction_error is not None:
                self._unproven_destruction(self._destruction_error)
            if self.closed:
                return
            self.closing = True
            if not self.fenced and self.last_stream is not None:
                self.last_stream.synchronize()
            if self.done is not None:
                self.done.synchronize()
            for stream in (self.stream, self.control):
                if stream is not None:
                    stream.synchronize()
            self.last_stream = None
            self.fenced = True
            for graph in self.graphs:
                graph.reset()
            self.graphs.clear()
            if self.owner is not None:
                self.owner.reset()
                self.owner = None
            for stream in (self.stream, self.control):
                if stream is not None:
                    # Graph reset may enqueue allocator retirement events.
                    stream.synchronize()
                    vt.retire_stream(stream, device=self.device)
            # Retire all live handles before destroying either: a refusal for control
            # must not leave a destroyed library-stream handle for a later finish().
            for name in ("stream", "control"):
                stream = getattr(self, name)
                if stream is not None:
                    try:
                        accel.destroy_stream(torch, stream)
                        setattr(self, name, None)
                    except BaseException as error:
                        self._unproven_destruction(error)
            self.fork = self.join = self.done = None
            self.closed = self.fenced = True

    def _unproven_destruction(self, cause: BaseException) -> None:
        if self._destruction_error is None:
            self._destruction_error = cause
            _UNPROVEN.append(self)
        error = torch.AcceleratorError(
            f"graph stream destruction is unproven; retire executor: {cause}"
        )
        raise error from cause


_LOCK = threading.RLock()
_OWNERS: weakref.WeakKeyDictionary[torch.nn.Module, Domain] = weakref.WeakKeyDictionary()
#: per device memory, the one domain its placed models' graphs share
_DEVICES: weakref.WeakKeyDictionary[Memory, Domain] = weakref.WeakKeyDictionary()
type _Configuration = tuple[int, int, torch._C._BlasBackend]


def device(memory: Memory) -> Domain:
    """The device's domain: every placed component's graphs, serialized."""
    with _LOCK:
        found = _DEVICES.get(memory)
        if found is None:
            found = _DEVICES[memory] = Domain(memory)
            weakref.finalize(memory, found.finish).atexit = False
        return found


def acquire(root: torch.nn.Module, memory: Memory | None = None) -> Domain:
    with _LOCK:
        found = _OWNERS.get(root)
        if found is None:
            found = Domain(memory)
            _OWNERS[root] = found
            weakref.finalize(root, found.finish).atexit = False
        return found


def release(root: torch.nn.Module) -> None:
    with _LOCK:
        found = _OWNERS.pop(root, None)
    if found is not None:
        found.finish()


def existing(root: torch.nn.Module) -> Domain | None:
    """Observe an existing owner without creating a domain or touching CUDA."""
    with _LOCK:
        return _OWNERS.get(root)
