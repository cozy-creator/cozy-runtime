"""cr-005 — the streaming fill plane: CAS bytes into the contract's exact destinations.

This is the real body behind cr-002b's `Backend` seam. The endpoint declares WHICH
components a method touches; it never learns HOW their bytes got there (§1.1/§3.2), so
everything in this module is runtime-owned and invisible to author code.

The path, once, in order:

    verified CAS read -> pinned staging ring -> async H2D on a runtime stream
      -> event fence -> atomic commit

Five properties are load-bearing rather than incidental:

**One canonical logical tensor to one canonical destination.** The keys walked here are
exactly the keys `contract.check_agreement` proved bijective between the harness's deep
walk and the fill plane's own census — a destination absent from the contract has no path
to bytes, and a contract key that never arrives refuses the whole generation.

**The freeze point.** A generation is observable as READY only after every H2D completion
event has fired. Until then the transaction is `staging`, and any failure takes it to
`rolled_back` with device memory released and verified CAS untouched. There is no state in
which a partially filled component answers a request.

**Leases outlive the copy.** A pinned slot is reusable only after the CUDA event recorded
on the stream that consumed it. Returning a slot with a live copy is a typed refusal, not
a race — `lease_live`.

**Read order is the STORE's, write order is the CONTRACT's.** The DestinationSet says which
keys; TensorFS decides how to touch the disk for them. Since tfs-007 that split falls exactly
on the boundary: this plane hands over the code-topology traversal and the store plans and
reads it, while every delivered batch names the destination its bytes belong to.

**Verification is the store's job, and since tfs-007 it is done BY the store.** The interim
Python CAS reader is gone (decisions #264): `Checkpoint` opens a real TensorFS store through
the tfs-007 facade, `acquire()` takes a verified read lease whose GC-safe hold registers
first, and `ReadLease.stream` drives a native reader pool into pinned slots this plane owns.
Nothing here knows CAS layout, caches a file descriptor, or hashes an object. The ORDERING
that buys is the point: the snapshot is verified when the lease is taken, so a corrupt
artifact refuses before a single destination is reserved, where the interim reader could
only find it mid-stream with gigabytes already live.
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cozy_runtime.author._loader import Census, TensorLike, TensorSpec, census

# ------------------------------------------------------------------------- refusals


class FillRefusal(Exception):
    """A typed fill refusal. Every one of these fires BEFORE any byte reaches a
    destination, or aborts the generation that was going to hold it."""

    def __init__(self, message: str, *, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(fields)


#: The complete refusal vocabulary of this plane. Named as a frozen set so a new code
#: cannot appear without touching this line.
REFUSALS = frozenset(
    {
        "missing_object",  # the CAS has no object file for a declared segment
        "digest_mismatch",  # object bytes do not hash to the object id
        "length_mismatch",  # segment/destination/object lengths disagree
        "dtype_mismatch",  # stored dtype is not the destination's (decode is cr-006)
        "budget_overrun",  # the ledger cannot grant this class
        "destination_absent",  # a fill for a key the contract does not name
        "incomplete_fill",  # commit with a DestinationSet key never enqueued
        "lease_live",  # a staging slot reused before its completion event
        "not_ready",  # the generation was read before the fence fired
        "derived_unmaterialized",  # a non-persistent buffer the fill plane cannot supply
    }
)


def _refuse(code: str, message: str, fields: Sequence[str] = ()) -> FillRefusal:
    if code not in REFUSALS:
        raise AssertionError(f"{code} is not a declared fill refusal")
    return FillRefusal(message, code=code, fields=fields)


# --------------------------------------------------------------------------- ledger

#: The byte classes this plane owns. `vram` is the device destinations; `pinned` is the
#: page-locked ring (REGISTERED, kernel-unreclaimable); `staging` is the host bytes
#: actually in flight inside that ring (CHARGED). §3.2 keeps allocated and registered as
#: two distinct numbers on purpose, so the ledger carries both.
CLASSES = ("vram", "pinned", "staging")


@dataclass(slots=True)
class ResourceLedger:
    """One unified ledger over the classes this plane moves bytes through.

    Grants are ATOMIC over classes (§3.2: no incremental acquisition, so no hold-and-wait
    cycle can form) and every counter declares how it was measured: `charged` is this
    plane's own arithmetic, and the driver reconciles it against the allocator's view.
    """

    budgets: dict[str, int] = field(default_factory=dict)
    charged: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    peak: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    grants: int = 0

    def grant(self, request: Mapping[str, int], why: str) -> None:
        """Charge every class or none. A class over budget refuses naming need and have."""
        for cls, nbytes in request.items():
            if cls not in CLASSES:
                raise AssertionError(f"{cls} is not a ledger class")
            budget = self.budgets.get(cls)
            if budget is None:
                continue
            free = budget - self.charged[cls]
            if nbytes > free:
                raise _refuse(
                    "budget_overrun",
                    f"{why}: {cls} needs {nbytes} B, {free} B free of a {budget} B budget "
                    f"(short by {nbytes - free} B) — a fill is granted its whole envelope "
                    "before it moves a byte, never incrementally under pressure",
                    [cls],
                )
        for cls, nbytes in request.items():
            self.charged[cls] += nbytes
            self.peak[cls] = max(self.peak[cls], self.charged[cls])
        self.grants += 1

    def release(self, request: Mapping[str, int]) -> None:
        for cls, nbytes in request.items():
            self.charged[cls] = max(0, self.charged[cls] - nbytes)

    def rows(self) -> dict[str, dict[str, int]]:
        return {
            cls: {
                "charged": self.charged[cls],
                "peak": self.peak[cls],
                "budget": self.budgets.get(cls, -1),
            }
            for cls in CLASSES
        }


# ------------------------------------------------------------------- the TensorFS reader


def tensorfs_module() -> Any:
    """The tfs-007 facade, or a typed refusal naming what is missing.

    TensorFS is a compiled extension and the base runtime package stays installable without
    it, exactly as it stays installable without torch. An absent reader is a REFUSAL that
    names the install, never an `ImportError` surfacing out of a fill.
    """
    try:
        import tensorfs
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise _refuse(
            "reader_unavailable",
            "the TensorFS reader (tfs-007's `tensorfs` facade) is not installed in this "
            "environment; the fill plane has no other way to read a verified object and "
            "will not invent one",
        ) from exc
    return tensorfs


#: How a TensorFS refusal is spoken in THIS plane's vocabulary. Closed and explicit: a code
#: this table does not name is re-raised UNCHANGED, because a plane that renames a refusal
#: it does not understand is worse than one that admits it did not.
STORE_REFUSALS: dict[str, str] = {
    "OBJECT_CORRUPT": "digest_mismatch",
    "OBJECT_ID_MISMATCH": "digest_mismatch",
    "MALFORMED_DIGEST": "digest_mismatch",
    "OBJECT_ABSENT": "missing_object",
    "LEASE_NOT_COVERED": "missing_object",
    "SHORT_READ": "length_mismatch",
    "RANGE_BOUNDS": "length_mismatch",
    "BUFFER_SIZE": "length_mismatch",
    "LENGTH_MISMATCH": "length_mismatch",
    "BYTE_LENGTH_MISMATCH": "length_mismatch",
    "DTYPE_MISMATCH": "dtype_mismatch",
    "DTYPE_UNKNOWN": "dtype_mismatch",
    "MISSING_TENSOR": "destination_absent",
    "TRAVERSAL_INCOMPLETE": "incomplete_fill",
    "SLOT_STARVED": "lease_live",
    "LEASE_REVOKED": "lease_revoked",
    "NONCANONICAL_ENCODING": "checkpoint_unreadable",
    "MALFORMED_JSON": "checkpoint_unreadable",
    "UNKNOWN_FORMAT": "checkpoint_unreadable",
    "MISSING_FIELD": "checkpoint_unreadable",
}


@contextlib.contextmanager
def store_refusals(what: str) -> Iterator[None]:
    """Speak a TensorFS refusal in this plane's vocabulary, or let it through untranslated."""
    tensorfs = tensorfs_module()
    try:
        yield
    except tensorfs.errors.Refusal as exc:
        code = STORE_REFUSALS.get(getattr(exc, "code", ""))
        if code is None:
            raise
        raise _refuse(
            code,
            f"{what}: TensorFS refused {exc.code} - {getattr(exc, 'detail', exc)}",
        ) from exc


@dataclass(frozen=True, slots=True)
class PlanRow:
    """One checkpoint header row: what the store holds for one logical tensor.

    `what` is TensorFS's own name for the destination (`<component>/<key>#<part>`), carried
    verbatim so a batch item can be routed back to its tensor without this plane re-deriving
    a naming rule the store already owns.
    """

    key: str
    """The CONTRACT's key: component-qualified, e.g. `unet.conv_in.weight`."""
    name: str
    """The HEADER's key for the same tensor, unqualified, e.g. `conv_in.weight`."""
    component: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    what: str
    """TensorFS's own destination name, `<component>/<name>#<part>`."""


class Checkpoint:
    """One snapshot of a TensorFS store, opened through the tfs-007 facade.

    This class is the ONLY place in the runtime that speaks to TensorFS, and it is
    deliberately thin. Object verification, GC-safe holds, plan ordering, dtype names, the
    reader pool and every refusal are the store's; what this plane adds is the projection
    from a header into the contract's rows and the H2D transaction on top of the stream.

    It replaces cr-005's interim `CasStore` + `read_plan` outright (decisions #264). The
    interim reader knew CAS layout, cached a file descriptor per (thread, object) and
    verified per object in Python; none of that survives, and none of it needs to.
    """

    def __init__(self, root: str | Path, snapshot_id: str) -> None:
        tensorfs = tensorfs_module()
        self.tensorfs = tensorfs
        self.root = str(root)
        self.snapshot_id = snapshot_id
        with store_refusals(f"opening snapshot {snapshot_id[:23]}"):
            self.store = tensorfs.Store.open(self.root)
            snapshot = self.store.snapshot(snapshot_id)
            self.header_bytes: bytes = snapshot["header"]
            self.header: dict[str, Any] = tensorfs.parse_header(self.header_bytes)

    def rows(self, component: str) -> list[PlanRow]:
        """The header's table for one component, in the header's own key order."""
        table = self.header["components"].get(component)
        if table is None:
            raise _refuse(
                "destination_absent",
                f"this checkpoint has no component {component!r} (it has "
                f"{', '.join(sorted(self.header['components'])) or 'none'})",
            )
        out: list[PlanRow] = []
        for key, entry in table.items():
            logical = entry["logical"]
            shape = tuple(int(n) for n in logical["shape"])
            dtype = str(logical["dtype"])
            width = self.tensorfs.DTYPES.get(dtype)
            if width is None:
                raise _refuse(
                    "dtype_mismatch",
                    f"{key}: the header declares dtype {dtype!r}, which this build of "
                    "TensorFS does not know",
                    [key],
                )
            if dtype not in TORCH_DTYPES:
                # The store knows this dtype and this runtime has no torch spelling for it.
                # Saying so is the point of keeping the map CHECKED rather than trusted: the
                # alternative is passing the raw name through and comparing it against a
                # torch dtype it can never equal, which refuses too — with a message about
                # the wrong thing.
                raise _refuse(
                    "dtype_mismatch",
                    f"{key}: TensorFS supplies dtype {dtype!r} and this runtime has no "
                    "torch binding for it; the map lives in the runtime because torch is "
                    "not TensorFS's business, and a missing row is a refusal, not a guess",
                    [key],
                )
            nbytes = width
            for extent in shape:
                nbytes *= extent
            out.append(
                PlanRow(
                    # The contract's key is component-qualified (`unet.conv_in.weight`);
                    # the header's is not, and TensorFS's `what` is a third spelling.
                    # Carrying all three is what keeps the plane from re-deriving any of
                    # them — a naming rule guessed in two places is a wrong copy waiting.
                    key=f"{component}.{key}",
                    name=key,
                    component=component,
                    dtype=dtype,
                    shape=shape,
                    nbytes=nbytes,
                    what=f"{component}/{key}#value",
                )
            )
        return out

    def read_plan(self, traversal: Sequence[tuple[str, str]], window_bytes: int) -> Any:
        """The store's own read plan over the traversal THIS PLANE asks for.

        cr-005's rule survives the swap and is now the store's to keep: read order is the
        STORE's, write order is the CONTRACT's. The runtime hands over the code-topology
        traversal and TensorFS decides how to touch the disk for it.
        """
        with store_refusals("planning the read"):
            return self.tensorfs.plan(self.header_bytes, list(traversal), window=window_bytes)

    def acquire(self) -> Any:
        """A verified read lease over the whole snapshot. The GC-safe hold registers FIRST."""
        with store_refusals(f"acquiring a lease on {self.snapshot_id[:23]}"):
            return self.store.acquire_snapshot(self.snapshot_id)


def topology_of(rows: Sequence[PlanRow]) -> dict[str, TensorSpec]:
    """The checkpoint topology table the loader matches the code against."""
    return {
        row.key: TensorSpec(row.shape, _dtype_name(row.dtype), "plain", row.nbytes) for row in rows
    }


class StagingRing:
    """A fixed pinned ring, never a residence tier (§3.2): bytes pass through, warm bytes
    are never page-locked, and the ring never grows under pressure.

    Since tfs-007 the ring is this plane's CONTRIBUTION to the store's reader pool rather
    than a loop it drives: the runtime allocates the pinned slots and hands their views to
    `ReadLease.stream`, and TensorFS fills them from its own native pool — which is what
    lets reading continue while this process holds the GIL. What stays here is the half that
    is the runtime's by boundary (pinned staging is the runtime's, boundaries.md) and the
    discipline that actually matters: a slot goes back to the ring only when the H2D
    completion event for its bytes has FIRED, never on the return of the callback.

    Size BOTH consumers. Readers and copies in flight share the same slots, so holding
    `slots - 1` copies leaves the pool one slot to fill and reads at queue depth 1 — tfs-007
    measured exactly that (0.836 GB/s at 4 slots / 2 in flight, 1.63 GB/s at 16 / 8).
    """

    def __init__(
        self,
        torch: Any,
        device: Any,
        *,
        window_bytes: int,
        slots: int,
        ledger: ResourceLedger,
        inflight: int,
        fence: bool = True,
    ) -> None:
        if slots < 2:
            raise _refuse(
                "lease_live",
                f"a {slots}-slot ring cannot double-buffer: the single slot is refilled "
                "while its copy is in flight, so there is no lease to hold",
            )
        if inflight >= slots:
            raise _refuse(
                "lease_live",
                f"{inflight} copies in flight against {slots} slots leaves the reader pool "
                "nothing to fill: a slot whose H2D has not completed is NOT free, so the "
                "ring must hold strictly more slots than in-flight copies",
            )
        self.torch = torch
        self.device = device
        self.window_bytes = int(window_bytes)
        self.ledger = ledger
        self.fence = fence
        self.keep = int(inflight)
        self.charge = {"pinned": self.window_bytes * slots}
        ledger.grant(self.charge, f"pinned staging ring ({slots} x {window_bytes} B)")
        self.stream = torch.cuda.Stream(device=device)
        self.buffers: list[Any] = []
        self.views: list[memoryview] = []
        for _ in range(slots):
            # device="cpu" is NOT redundant: construction runs under an ambient meta
            # device, and a meta staging buffer has data_ptr() == 0 — the ring would read
            # into address zero. Every allocation this plane makes names its device.
            buffer = torch.empty(
                self.window_bytes, dtype=torch.uint8, pin_memory=True, device="cpu"
            )
            block = (ctypes.c_ubyte * buffer.numel()).from_address(buffer.data_ptr())
            self.buffers.append(buffer)
            self.views.append(memoryview(block).cast("B"))
        self.inflight: collections.deque[tuple[Any, Any]] = collections.deque()
        self.waits = 0
        self.charged = 0

    def hold(self, event: Any, batch: Any) -> None:
        """This batch's slot is now DMA-live; it is charged until its event fires."""
        self.inflight.append((event, batch))
        self.charged += 1
        self.ledger.grant({"staging": self.window_bytes}, "staging slot")

    def drain(self, keep: int) -> None:
        """Return slots whose H2D has actually completed. QUERY FIRST, BLOCK LAST.

        The callback runs on the store's delivery thread, so blocking here stops the ring
        from handing out the next filled slot even when readers have work ready. Blocking
        only when the caller is genuinely out of slots is worth ~1.7x (tfs-007, measured).
        """
        while self.inflight and self.inflight[0][0].query():
            self._retire()
        while len(self.inflight) > keep:
            self.inflight[0][0].synchronize()
            self.waits += 1
            self._retire()

    def _retire(self) -> None:
        _, batch = self.inflight.popleft()
        batch.release()
        self.charged -= 1
        self.ledger.release({"staging": self.window_bytes})

    def track(self, tensor: Any) -> None:
        """The destination is written on OUR stream; the allocator must know."""
        tensor.record_stream(self.stream)

    def finish(self) -> None:
        self.drain(0)
        self.stream.synchronize()
        self.torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def close(self) -> None:
        """Give every byte back, including the slots an aborted stream never handed in —
        an interrupted fill that leaks its own staging charge is an interrupted fill that
        the next grant prices wrong."""
        self.ledger.release({"staging": self.window_bytes * max(self.charged, 0)})
        self.inflight.clear()
        self.buffers.clear()
        self.views.clear()
        self.ledger.release(self.charge)


# -------------------------------------------------------------------------- backend


State = Literal["open", "staging", "ready", "rolled_back"]


class StreamingFillBackend:
    """The `author.Backend` implementation that moves real bytes (§1.1's ONE fill path).

    `fill` VALIDATES and enqueues; `commit` moves, fences and freezes. Splitting them is
    what makes the transaction real: the whole component's destinations are known before
    a single byte lands, so a refusal has nothing to undo and a failure has one thing.
    """

    def __init__(
        self,
        checkpoint: Checkpoint,
        rows: Sequence[PlanRow],
        *,
        device: str = "cuda",
        window_bytes: int = 16 << 20,
        slots: int = 16,
        readers: int = 16,
        inflight: int = 8,
        ledger: ResourceLedger | None = None,
        fence: bool = True,
        on_position: Callable[[int], None] | None = None,
    ) -> None:
        from cozy_runtime.internal.derive import torch_module

        self.torch = torch_module()
        self.checkpoint = checkpoint
        self.plan = {row.key: row for row in rows}
        self.order = [row.key for row in rows]
        self.device = self.torch.device(device)
        self.device_str = str(self.torch.empty(0, device=self.device).device)
        # The window is the store's PLANNING unit and the slot is this plane's staging unit;
        # tfs-007 measured 4 MiB windows into 16 MiB slots as the shape that works.
        self.window_bytes = int(window_bytes)
        self.slot_bytes = max(int(window_bytes) * 4, 16 << 20)
        self.slots = int(slots)
        self.readers = int(readers)
        self.inflight = int(inflight)
        self.ledger = ledger or ResourceLedger()
        self.fence = fence
        #: cr-011's liveness feed: the MONOTONE integral-MiB position of the logical stream,
        #: reported per batch. It is the only thing the wedge detector reads, and it is a
        #: position rather than a heartbeat — a stalled transfer stops moving it, while a
        #: slow one keeps moving it however slowly.
        self.on_position = on_position
        self.lease: Any = None
        self.state: State = "open"
        self.enqueued: dict[str, Any] = {}
        self.expected: dict[str, tuple[str, ...]] = {}
        self.components: dict[str, Any] = {}
        self.vram_charge: dict[str, int] = {}
        self.stats: dict[str, Any] = {"windows": 0, "waits": 0}

    # -- reservation -------------------------------------------------------------

    def expect(self, sets: Mapping[str, Sequence[str]]) -> None:
        """The contract's COMPLETE DestinationSets, and the verified read lease.

        Commit refuses on any key that never arrives, which is what makes a narrow component
        declaration authoritative. The LEASE is acquired here rather than at commit, and the
        ordering is the point: TensorFS verifies every object in the snapshot when the hold
        registers, so a corrupt or truncated artifact refuses BEFORE this plane reserves a
        single byte of device memory. The interim reader could only discover that mid-stream,
        with gigabytes of destinations already live.
        """
        self.expected = {name: tuple(keys) for name, keys in sets.items()}
        self.lease = self.checkpoint.acquire()
        self.stats["lease_bytes"] = int(self.lease.bytes)
        self.stats["lease_objects"] = len(self.lease.objects)

    def materialize(self, obj: object, walked: Census) -> Mapping[str, TensorLike]:
        """Reserve ordinary CUDA destinations for the constructed component roots.

        Baseline residency is ordinary CUDA allocation and that is sufficient (§3.2); a
        promoted Varena adapter would replace this method and nothing else. The census is
        re-taken afterwards and must agree key-for-key: a materialization that changes the
        topology is not a placement, it is a different construction.
        """
        from cozy_runtime.author._loader import _members
        from cozy_runtime.internal.derive import placing

        roots = dict(_members(obj))
        started = time.perf_counter_ns()
        with placing():
            for name in walked.components:
                member = roots.get(name)
                if member is None or not hasattr(member, "to_empty"):
                    continue
                stray = sorted(_non_persistent(member))
                if stray:
                    raise _refuse(
                        "derived_unmaterialized",
                        f"component {name!r} carries {len(stray)} non-persistent buffer(s) "
                        f"({', '.join(stray[:4])}): no checkpoint supplies them and this "
                        "plane moves stored bytes only. A derived table is a pure function "
                        "of config and its re-execution on the target device is cr-008b's "
                        "staging pass — serving uninitialized memory under a derived name "
                        "is the alternative, and it is not one",
                        stray[:6],
                    )
                want = sum(t.numel() * t.element_size() for _, t in _tensors(member))
                self.ledger.grant({"vram": want}, f"component {name!r} destinations")
                self.vram_charge[name] = want
                try:
                    member.to_empty(device=self.device, recurse=True)
                except BaseException:
                    # RESERVATION IS A TRANSACTION TOO. `to_empty` on a 4.78 GiB component
                    # can run out of device memory halfway, and the half it moved is held by
                    # the module — not by `components` (set below) and not by `enqueued`
                    # (the fill has not run). Nothing else would ever free it, so the worker
                    # would sit on gigabytes it can never use until the process died, while
                    # reporting a rollback that returned them. Measured: two concurrent
                    # residents on one 8 GiB card left 2,345,812,992 B held after the
                    # refusal. Same lesson as cr-005's finding 3, a different loop: release
                    # is a POSITIVE ACT.
                    self._free_storages(t for _, t in _tensors(member))
                    member.to_empty(device=self.torch.device("meta"), recurse=True)
                    self.ledger.release({"vram": self.vram_charge.pop(name, 0)})
                    raise
                self.components[name] = member
            # Inside the placement window too: censusing a REAL module detaches every
            # tensor, and a detach on cuda is exactly the "real allocation during
            # construction" the derive guard exists to catch in author code.
            fresh = census(obj)
        self.stats["materialize_ms"] = (time.perf_counter_ns() - started) // 1_000_000
        if [d.key for d in fresh.destinations] != [d.key for d in walked.destinations]:
            raise _refuse(
                "destination_absent",
                "the census after materialization is not the census before it — "
                "placement moves bytes, never topology",
            )
        return fresh.live

    # -- the Backend protocol ----------------------------------------------------

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None:
        """Validate one destination against the stored row and enqueue it. No I/O here:
        every refusal in the matrix must fire before the first byte moves."""
        row = self.plan.get(key)
        if row is None:
            raise _refuse(
                "destination_absent",
                f"{key!r} is not a destination this checkpoint supplies — the fill plane "
                "walks the contract's keys and nothing else, so an unknown destination has "
                "no path to bytes",
                [key],
            )
        tensor = destination
        nbytes = int(tensor.numel() * tensor.element_size())  # type: ignore[attr-defined]
        if nbytes != row.nbytes:
            raise _refuse(
                "length_mismatch",
                f"{key}: the destination holds {nbytes} B and the checkpoint supplies "
                f"{row.nbytes} B — one canonical logical tensor to one canonical "
                "destination, with no reshaping in between",
                [key],
            )
        if _dtype_name(tensor.dtype) != _dtype_name(row.dtype):
            raise _refuse(
                "dtype_mismatch",
                f"{key}: stored as {row.dtype}, the destination is "
                f"{_dtype_name(tensor.dtype)} — this plane executes the ENCODING CONTRACT "
                "only; a dtype change is a decode and decoders are cr-006's",
                [key],
            )
        if str(getattr(tensor, "device", "")) != self.device_str:
            raise _refuse(
                "destination_absent",
                f"{key}: destination is on {tensor.device!s}, the fill device is "  # type: ignore[attr-defined]
                f"{self.device_str}",
                [key],
            )
        if spec.nbytes and spec.nbytes != row.nbytes:
            raise _refuse(
                "length_mismatch",
                f"{key}: the code topology says {spec.nbytes} B, the checkpoint header "
                f"{row.nbytes} B",
                [key],
            )
        self.enqueued[key] = tensor
        self.state = "staging"

    def commit(self) -> None:
        """Move every enqueued destination, fence the copies, then freeze.

        `ready` is set after the last completion event and not one statement earlier: the
        freeze point is what makes "no partially filled generation is ever observable"
        a property of the code rather than of the caller's discipline.
        """
        if self.state == "open":
            self.state = "ready"
            return
        missing = [
            key for name, keys in self.expected.items() for key in keys if key not in self.enqueued
        ]
        if missing:
            raise _refuse(
                "incomplete_fill",
                f"{len(missing)} destination(s) of a declared component never reached the "
                f"fill plane ({', '.join(missing[:4])}) — a DestinationSet is COMPLETE or "
                "the component does not commit",
                missing[:6],
            )
        from cozy_runtime.internal.derive import placing

        started = time.perf_counter_ns()
        with placing():
            ring = StagingRing(
                self.torch,
                self.device,
                window_bytes=self.slot_bytes,
                slots=self.slots,
                ledger=self.ledger,
                inflight=self.inflight,
                fence=self.fence,
            )
            try:
                for tensor in self.enqueued.values():
                    ring.track(tensor)
                windows = self._stream(ring)
                ring.finish()
                self.torch.cuda.synchronize()
                self.stats["windows"] = windows
                self.stats["waits"] = ring.waits
            finally:
                ring.close()
                # The lease is released AFTER the fence, explicitly. A hold that outlives
                # the process is the store's to reap; a hold dropped before the copies land
                # is a read against bytes the GC may already consider free.
                self.release_lease()
        self.stats["stream_ms"] = (time.perf_counter_ns() - started) // 1_000_000
        # The tfs-018-comparable leg: reserve device destinations, read, copy, fence. It
        # deliberately EXCLUDES the meta construction, which those arms have no analogue
        # for — comparing a number that includes it would flatter neither side honestly.
        self.stats["fill_ms"] = self.stats["stream_ms"] + self.stats.get("materialize_ms", 0)
        self.state = "ready"

    def release_lease(self) -> None:
        """Give the hold back. Idempotent: a rollback and a commit may both reach here."""
        lease = self.lease
        self.lease = None
        if lease is not None and lease.live:
            lease.release()

    def rollback(self, keys: Sequence[str]) -> None:
        """Abort the new generation; verified CAS is untouched by construction (this plane
        never writes to the store).

        The release is a POSITIVE ACT, not a dropped reference. Half-filled destinations
        are freed by resizing their storage to zero, because dropping references frees
        nothing while anything still holds one — and after a failed fill the thing holding
        one is the EXCEPTION: a Python traceback retains every frame it passed through,
        including the fill loop's live destination map. Measured here (an interrupted 4.78
        GiB fill "released" 0 bytes until the storages were freed explicitly), and it is
        the same hazard the moment a supervisor journals a fill failure.
        """
        from cozy_runtime.internal.derive import placing

        self.release_lease()
        with placing():
            self._free_storages(self.enqueued.values())
            self.enqueued.clear()
            for name, member in list(self.components.items()):
                self._free_storages(t for _, t in _tensors(member))
                member.to_empty(device=self.torch.device("meta"), recurse=True)
                self.ledger.release({"vram": self.vram_charge.pop(name, 0)})
        self.components.clear()
        self.torch.cuda.empty_cache()
        self.state = "rolled_back"

    def _free_storages(self, tensors: Any) -> int:
        """Free the backing allocation of every distinct storage, once. Views share it, so
        a set of storage identities is the unit that can be freed exactly once."""
        seen: set[int] = set()
        freed = 0
        for tensor in tensors:
            try:
                storage = tensor.untyped_storage()
            except Exception:
                continue
            token = storage.data_ptr()
            if token in seen or token == 0:
                continue
            seen.add(token)
            freed += storage.nbytes()
            storage.resize_(0)
        return freed

    # -- the unload/release path -------------------------------------------------

    def release(self, component: str) -> int:
        """Component release RETURNS VRAM. Release makes bytes evictable and this plane's
        baseline backend frees them outright (§3.2: device eviction FREES, never copies to
        host). Returns the allocator's own delta, not this ledger's arithmetic."""
        member = self.components.pop(component, None)
        if member is None:
            raise _refuse(
                "destination_absent",
                f"{component!r} is not a component this generation holds",
                [component],
            )
        from cozy_runtime.internal.derive import placing

        before = int(self.torch.cuda.memory_allocated(self.device))
        with placing():
            self._free_storages(t for _, t in _tensors(member))
            member.to_empty(device=self.torch.device("meta"), recurse=True)
        for key in [k for k in self.enqueued if k.split(".")[0] == component]:
            del self.enqueued[key]
        self.torch.cuda.empty_cache()
        self.ledger.release({"vram": self.vram_charge.pop(component, 0)})
        return before - int(self.torch.cuda.memory_allocated(self.device))

    def require_ready(self, what: str) -> None:
        if self.state != "ready":
            raise _refuse(
                "not_ready",
                f"{what} before the fill fence: the generation is {self.state!r}. Content "
                "validity is the committed tuple (construction generation, backing epoch, "
                "exact content binding, COMPLETED fill)",
            )

    # -- the stream ---------------------------------------------------------------

    def _stream(self, ring: StagingRing) -> int:
        """The store streams; this plane splices and fences.

        cr-005's rule survives the reader swap and is now split across the boundary exactly
        where it belongs: READ ORDER IS THE STORE'S — TensorFS plans the disk touches for
        the traversal it is given — and WRITE ORDER IS THE CONTRACT'S, because the traversal
        handed over IS the contract's DestinationSet order and every batch item carries the
        destination it belongs to.

        Queue depth is a MEASURED requirement, not a preference: tfs-018 found 3.5x between
        depth 1 and depth 4 on this store, tfs-005 confirmed it, and tfs-007 confirmed it a
        third time from a third language. What tfs-007 added is that COPIES IN FLIGHT share
        the ring with readers, so the caller sizes both or reads at the depth the leftovers
        allow.
        """
        rows = [self.plan[key] for key in self.order if key in self.enqueued]
        if not rows:
            return 0
        traversal = [(row.component, row.name) for row in rows]
        plan = self.checkpoint.read_plan(traversal, self.window_bytes)
        expect = sum(row.nbytes for row in rows)
        if int(plan.bytes) != expect:
            raise _refuse(
                "length_mismatch",
                f"the store's read plan carries {plan.bytes} B for a traversal this "
                f"contract prices at {expect} B",
            )

        flat = {row.what: self.enqueued[row.key].view(-1).view(self.torch.uint8) for row in rows}
        base: dict[str, int] = {}
        for item in plan.items():
            base.setdefault(str(item["what"]), int(item["dest_off"]))
        missing = sorted(set(flat) - set(base))
        if missing:
            raise _refuse(
                "incomplete_fill",
                f"{len(missing)} enqueued destination(s) are absent from the store's read "
                f"plan ({', '.join(missing[:4])})",
                missing[:6],
            )

        batches = [0]
        cursor = [0]
        torch = self.torch

        def on_batch(batch: Any) -> None:
            """Splice one filled slot across the destinations it covers.

            A batch item carries `off` — where its bytes sit IN THE SLOT — and the store
            delivers batches in plan order, so the cursor is what turns a slot offset back
            into a position in the logical stream. `base` then turns that into an offset
            inside one destination tensor. Every arithmetic step here comes from a number
            the store handed over; none of it is re-derived from a naming rule.
            """
            start = cursor[0]
            with torch.cuda.stream(ring.stream):
                for item in batch.items():
                    what = str(item["what"])
                    source = int(item["off"])
                    length = int(item["len"])
                    target = start + source - base[what]
                    flat[what][target : target + length].copy_(
                        ring.buffers[batch.slot][source : source + length], non_blocking=True
                    )
                event = torch.cuda.Event()
                event.record(ring.stream)
            cursor[0] = start + int(batch.nbytes)
            ring.hold(event, batch)
            batches[0] += 1
            if self.on_position is not None:
                self.on_position(cursor[0] >> 20)
            # A slot whose H2D has not completed is NOT free. Query first, block last.
            ring.drain(ring.keep)

        started = time.perf_counter_ns()
        with store_refusals("streaming the component"):
            stats = self.lease.stream(plan, ring.views, on_batch, readers=self.readers)
        self.stats["read_ns"] = time.perf_counter_ns() - started
        self.stats["store"] = dict(stats)
        return batches[0]


# ------------------------------------------------------------------------- helpers


#: TensorFS dtype name -> torch dtype name. This is the ONE thing the store cannot own: it
#: is a fact about torch, and TensorFS has no business knowing torch exists. What the store
#: DOES own is the set of names and their widths, so this map is CHECKED against
#: `tensorfs.DTYPES` rather than trusted — a name the store knows and this map does not is a
#: refusal, not a silent passthrough that would land the wrong number of bytes.
TORCH_DTYPES: dict[str, str] = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "f8_e4m3fn": "float8_e4m3fn",
    "f8_e5m2": "float8_e5m2",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "u8": "uint8",
    "bool": "bool",
}


def _dtype_name(dtype: object) -> str:
    """The one dtype spelling. The store writes TensorFS names (`f16`), torch prints its
    own (`torch.float16`), and a fill that guesses which is which is a silent wrong copy."""
    name = str(getattr(dtype, "name", dtype)).removeprefix("torch.")
    return TORCH_DTYPES.get(name, name)


def _tensors(module: Any) -> Iterator[tuple[str, Any]]:
    yield from module.named_parameters(remove_duplicate=False)
    yield from module.named_buffers(remove_duplicate=False)


def _non_persistent(root: Any) -> set[str]:
    names: set[str] = set()
    for prefix, sub in root.named_modules():
        for local in getattr(sub, "_non_persistent_buffers_set", ()):
            names.add(f"{prefix}.{local}" if prefix else local)
    return names
