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

**Read order is the STORE's, write order is the CONTRACT's.** The DestinationSet says
which keys; the checkpoint header says in what order their bytes sit on disk. Ordering is
movement, and movement is the runtime's — so the ring streams in header order and splices
into whichever destination each window covers.

**Verification is the store's job, done here on loan.** Until tfs-005/007 land, this module
reads CAS object files directly and verifies sha256 == object id on every object, through
the store's own verification record (tfs-002's `tensorfs.verification/1`: a record whose
(dev, ino, length, mtime) still describes the live inode asserts that object's digest
without a rehash). A missing or stale record REHASHES. That is the interim seam, and the
whole of it is `CasStore` — tfs-007 replaces that class with the real read lease.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
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


# ------------------------------------------------------------------------ CAS source


@dataclass(frozen=True, slots=True)
class Segment:
    """One contiguous run of a logical tensor's bytes. `oid` is None for inline bytes
    (the header carries tensors too small to be worth an object)."""

    oid: str | None
    length: int
    inline: bytes | None = None


@dataclass(frozen=True, slots=True)
class PlanRow:
    """One checkpoint header row: what the store holds for one logical tensor."""

    key: str
    component: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    segments: tuple[Segment, ...]


Verify = Literal["record", "rehash"]


class CasStore:
    """Direct verified reads over a tensorfs CAS tree — the tfs-007 interim seam.

    Thread-safe by construction: verification is decided once per object under a lock, and
    each reader thread keeps its own file handle (a shared fd would serialize the pool).
    """

    def __init__(self, root: str | Path, *, verify: Verify = "record") -> None:
        self.root = Path(root)
        self.verify = verify
        self._ok: set[str] = set()
        self._gates: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._local = threading.local()
        self.opens = 0
        self.verifies = 0
        self.rehashes = 0
        self.verify_ns = 0
        self.bytes = 0

    # -- layout ------------------------------------------------------------------

    def object_path(self, oid: str) -> Path:
        return self.root / "objects" / "sha256" / oid[:2] / oid[2:4] / oid

    def record_path(self, oid: str) -> Path:
        return self.root / "verified" / oid[:2] / oid[2:4] / oid

    # -- verification ------------------------------------------------------------

    def _rehash(self, path: Path, oid: str) -> int:
        digest = hashlib.sha256()
        length = 0
        with open(path, "rb", buffering=0) as handle:
            while chunk := handle.read(1 << 22):
                digest.update(chunk)
                length += len(chunk)
        if digest.hexdigest() != oid:
            raise _refuse(
                "digest_mismatch",
                f"CAS object {oid[:16]}… hashes to {digest.hexdigest()[:16]}… — content "
                "address is the only identity a stored byte has; a mismatch is a corrupt "
                "or substituted object and never a fill input",
                [oid],
            )
        return length

    def ensure_verified(self, oid: str, expect: int) -> None:
        """sha256 == object id, by record hit where the store's own record still describes
        the live inode, and by rehash otherwise.

        SINGLE-FLIGHT per object (§3.2's registry rule, at the byte tier): several reader
        threads hitting one object must verify it ONCE. Without the per-object gate a
        1,679-object corpus verifies a few dozen objects twice, which is invisible on a
        record hit and doubles the work on a rehash.
        """
        with self._lock:
            if oid in self._ok:
                return
            gate = self._gates.setdefault(oid, threading.Lock())
        with gate:
            if oid in self._ok:
                return
            self._verify_one(oid, expect)

    def _verify_one(self, oid: str, expect: int) -> None:
        started = time.perf_counter_ns()
        path = self.object_path(oid)
        try:
            stat = path.stat()
        except FileNotFoundError:
            raise _refuse(
                "missing_object",
                f"CAS object {oid[:16]}… is not in {self.root} — a declared segment with "
                "no object is a broken checkpoint, not a cache miss to fetch around",
                [oid],
            ) from None
        rehashed = False
        if self.verify == "rehash":
            length = self._rehash(path, oid)
            rehashed = True
        else:
            record = self._record(oid)
            if record is None or not self._fresh(record, stat, oid):
                length = self._rehash(path, oid)
                rehashed = True
            else:
                length = int(record["length"])
        if length != stat.st_size:
            raise _refuse(
                "length_mismatch",
                f"CAS object {oid[:16]}… is {stat.st_size} B, verified as {length} B",
                [oid],
            )
        if expect > length:
            raise _refuse(
                "length_mismatch",
                f"a segment claims {expect} B of CAS object {oid[:16]}…, which holds "
                f"{length} B — the header and the store disagree about the bytes",
                [oid],
            )
        with self._lock:
            self._ok.add(oid)
            self.verifies += 1
            self.rehashes += int(rehashed)
            self.verify_ns += time.perf_counter_ns() - started

    def _record(self, oid: str) -> dict[str, Any] | None:
        try:
            body = self.record_path(oid).read_bytes()
        except FileNotFoundError:
            return None
        record: dict[str, Any] = json.loads(body)
        if record.get("format") != "tensorfs.verification/1":
            return None
        if record.get("sha256") != oid:
            raise _refuse(
                "digest_mismatch",
                f"the verification record at {self.record_path(oid)} asserts "
                f"{str(record.get('sha256'))[:16]}… for object {oid[:16]}… — a record that "
                "names a different digest than the object it files under is not evidence",
                [oid],
            )
        return record

    @staticmethod
    def _fresh(record: Mapping[str, Any], stat: os.stat_result, oid: str) -> bool:
        """Does this record still describe the live inode? The record asserts sha256 for
        exactly (dev, ino, length, mtime); any drift and it asserts nothing."""
        return (
            int(record.get("dev", -1)) == stat.st_dev
            and int(record.get("ino", -1)) == stat.st_ino
            and int(record.get("length", -1)) == stat.st_size
            and int(record.get("mtime_s", -1)) == stat.st_mtime_ns // 1_000_000_000
            and int(record.get("mtime_ns", -1)) == stat.st_mtime_ns % 1_000_000_000
        )

    # -- reads -------------------------------------------------------------------

    def _handle(self, oid: str) -> int:
        cache: dict[str, int] | None = getattr(self._local, "handles", None)
        if cache is None:
            cache = {}
            self._local.handles = cache
        fd = cache.get(oid)
        if fd is None:
            fd = os.open(self.object_path(oid), os.O_RDONLY)
            cache[oid] = fd
            with self._lock:
                self.opens += 1
        return fd

    def read_into(self, oid: str, offset: int, length: int, view: memoryview) -> None:
        """`length` VERIFIED bytes of object `oid` from `offset`, into caller memory."""
        self.ensure_verified(oid, offset + length)
        fd = self._handle(oid)
        at = 0
        while at < length:
            got = os.preadv(fd, [view[at:length]], offset + at)
            if got <= 0:
                raise _refuse(
                    "length_mismatch",
                    f"short read on CAS object {oid[:16]}…: {at} of {length} B",
                    [oid],
                )
            at += got
        with self._lock:
            self.bytes += length

    def close(self) -> None:
        cache: dict[str, int] | None = getattr(self._local, "handles", None)
        for fd in (cache or {}).values():
            os.close(fd)
        self._local.handles = {}

    def stats(self) -> dict[str, int | str]:
        return {
            "verify": self.verify,
            "opens": self.opens,
            "verifies": self.verifies,
            "rehashes": self.rehashes,
            "verify_ms": self.verify_ns // 1_000_000,
            "read_bytes": self.bytes,
        }


# --------------------------------------------------------------------------- ring


@dataclass(slots=True)
class Slot:
    """One pinned staging buffer and the lease that keeps it alive: `event` is the exact
    H2D completion the next user must wait on."""

    tensor: Any
    view: memoryview
    ptr: int
    event: Any = None
    dirty: bool = False
    """Holds bytes a copy may still be reading. Cleared by the EVENT, never by the return."""
    charged: bool = False
    """Its staging bytes are charged to the ledger right now."""


class StagingRing:
    """A fixed pinned ring, never a residence tier (§3.2): bytes pass through, warm bytes
    are never page-locked, and the ring never grows under pressure.

    `fence` exists to be turned OFF exactly once, by the red arm that proves the event
    fence is load-bearing rather than decorative.
    """

    def __init__(
        self,
        torch: Any,
        device: Any,
        *,
        window_bytes: int,
        slots: int,
        ledger: ResourceLedger,
        fence: bool = True,
    ) -> None:
        if slots < 2:
            raise _refuse(
                "lease_live",
                f"a {slots}-slot ring cannot double-buffer: the single slot is refilled "
                "while its copy is in flight, so there is no lease to hold",
            )
        self.torch = torch
        self.device = device
        self.window_bytes = int(window_bytes)
        self.ledger = ledger
        self.fence = fence
        self.charge = {"pinned": self.window_bytes * slots}
        ledger.grant(self.charge, f"pinned staging ring ({slots} x {window_bytes} B)")
        self.stream = torch.cuda.Stream(device=device)
        self.slots: list[Slot] = []
        for _ in range(slots):
            # device="cpu" is NOT redundant: construction runs under an ambient meta
            # device, and a meta staging buffer has data_ptr() == 0 — the ring would read
            # into address zero. Every allocation this plane makes names its device.
            buffer = torch.empty(
                self.window_bytes, dtype=torch.uint8, pin_memory=True, device="cpu"
            )
            block = (ctypes.c_ubyte * buffer.numel()).from_address(buffer.data_ptr())
            self.slots.append(
                Slot(buffer, memoryview(block).cast("B"), buffer.data_ptr())
            )
        self.waits = 0

    def take(self, slot: Slot) -> None:
        """Claim a slot for a new window. A dirty slot with no recorded event is a lease
        that was dropped while its bytes were still DMA-live."""
        if slot.dirty and slot.event is None:
            raise _refuse(
                "lease_live",
                "a staging slot was returned before its H2D completion event was "
                "recorded — the ring would overwrite bytes the copy engine is still "
                "reading. A lease is released by the EVENT, never by the return",
            )
        if slot.event is not None:
            self.waits += 1
            slot.event.synchronize()
            slot.event = None
        slot.dirty = True
        slot.charged = True
        self.ledger.grant({"staging": self.window_bytes}, "staging slot")

    def copy_out(self, slot: Slot, src_off: int, dst: Any, dst_off: int, count: int) -> None:
        with self.torch.cuda.stream(self.stream):
            dst[dst_off : dst_off + count].copy_(
                slot.tensor[src_off : src_off + count], non_blocking=True
            )

    def give_back(self, slot: Slot) -> None:
        if self.fence:
            event = self.torch.cuda.Event()
            event.record(self.stream)
            slot.event = event
        slot.charged = False
        self.ledger.release({"staging": self.window_bytes})

    def track(self, tensor: Any) -> None:
        """The destination is written on OUR stream; the allocator must know."""
        tensor.record_stream(self.stream)

    def finish(self) -> None:
        self.stream.synchronize()
        self.torch.cuda.current_stream(self.device).wait_stream(self.stream)
        for slot in self.slots:
            slot.event = None
            slot.dirty = False
            slot.charged = False

    def close(self) -> None:
        """Give every byte back, including the slots an aborted stream never handed in —
        an interrupted fill that leaks its own staging charge is an interrupted fill that
        the next grant prices wrong."""
        outstanding = sum(1 for slot in self.slots if slot.charged)
        self.ledger.release({"staging": self.window_bytes * outstanding})
        self.slots.clear()
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
        plan: Sequence[PlanRow],
        store: CasStore,
        *,
        device: str = "cuda",
        window_bytes: int = 16 << 20,
        slots: int = 4,
        readers: int = 4,
        ledger: ResourceLedger | None = None,
        fence: bool = True,
        interrupt_at: float | None = None,
    ) -> None:
        from cozy_runtime.internal.derive import torch_module

        self.torch = torch_module()
        self.plan = {row.key: row for row in plan}
        self.order = [row.key for row in plan]
        self.store = store
        self.device = self.torch.device(device)
        self.device_str = str(self.torch.empty(0, device=self.device).device)
        self.window_bytes = int(window_bytes)
        self.slots = int(slots)
        self.readers = int(readers)
        self.ledger = ledger or ResourceLedger()
        self.fence = fence
        self.interrupt_at = interrupt_at
        self.state: State = "open"
        self.enqueued: dict[str, Any] = {}
        self.expected: dict[str, tuple[str, ...]] = {}
        self.components: dict[str, Any] = {}
        self.vram_charge: dict[str, int] = {}
        self.stats: dict[str, Any] = {"windows": 0, "waits": 0}

    # -- reservation -------------------------------------------------------------

    def expect(self, sets: Mapping[str, Sequence[str]]) -> None:
        """The contract's COMPLETE DestinationSets. Commit refuses on any key that never
        arrives, which is what makes a narrow component declaration authoritative."""
        self.expected = {name: tuple(keys) for name, keys in sets.items()}

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
                member.to_empty(device=self.device, recurse=True)
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
            key
            for name, keys in self.expected.items()
            for key in keys
            if key not in self.enqueued
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
                window_bytes=self.window_bytes,
                slots=self.slots,
                ledger=self.ledger,
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
                self.store.close()
        self.stats["stream_ms"] = (time.perf_counter_ns() - started) // 1_000_000
        # The tfs-018-comparable leg: reserve device destinations, read, copy, fence. It
        # deliberately EXCLUDES the meta construction, which those arms have no analogue
        # for — comparing a number that includes it would flatter neither side honestly.
        self.stats["fill_ms"] = self.stats["stream_ms"] + self.stats.get("materialize_ms", 0)
        self.state = "ready"

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

    def _segments(self) -> tuple[list[tuple[int, int, Segment]], list[tuple[int, int, Any]]]:
        """The virtual byte stream, in STORE order over the enqueued keys, and the
        destination placements that stream covers."""
        segments: list[tuple[int, int, Segment]] = []
        placements: list[tuple[int, int, Any]] = []
        cursor = 0
        for key in self.order:
            tensor = self.enqueued.get(key)
            if tensor is None:
                continue
            row = self.plan[key]
            start = cursor
            for segment in row.segments:
                segments.append((cursor, cursor + segment.length, segment))
                cursor += segment.length
            if cursor - start != row.nbytes:
                raise _refuse(
                    "length_mismatch",
                    f"{key}: segments carry {cursor - start} B for a {row.nbytes} B tensor",
                    [key],
                )
            placements.append(
                (start, cursor, tensor.view(-1).view(self.torch.uint8))
            )
        return segments, placements

    def _stream(self, ring: StagingRing) -> int:
        """Reader pool over disjoint windows; the copy is issued by the consumer thread on
        one runtime stream, and the slot's lease ends at that copy's event.

        Queue depth is a MEASURED requirement, not a preference: tfs-018 showed a
        queue-depth-1 ring loses cold against safetensors on this very store, and wins by
        1.30x at depth 4. A serial reader is the wrong shape for a 1,679-object corpus.
        """
        import bisect
        import queue

        segments, placements = self._segments()
        if not segments:
            return 0
        total = segments[-1][1]
        starts = [p[0] for p in placements]
        seg_starts = [s[0] for s in segments]
        windows = [
            (at, min(self.window_bytes, total - at))
            for at in range(0, total, self.window_bytes)
        ]
        free: queue.Queue[Slot] = queue.Queue()
        for slot in ring.slots:
            free.put(slot)
        done: queue.Queue[tuple[int, int, Slot] | BaseException] = queue.Queue()
        pending = iter(windows)
        guard = threading.Lock()
        stop = threading.Event()
        cut = (
            int(len(windows) * self.interrupt_at) if self.interrupt_at is not None else -1
        )

        def read_window(at: int, count: int, slot: Slot) -> None:
            index = bisect.bisect_right(seg_starts, at) - 1
            written = 0
            while written < count:
                start, end, segment = segments[index]
                skip = at + written - start
                take = min(end - start - skip, count - written)
                if segment.oid is None:
                    body = segment.inline or b""
                    slot.view[written : written + take] = body[skip : skip + take]
                else:
                    self.store.read_into(
                        segment.oid, skip, take, slot.view[written : written + take]
                    )
                written += take
                index += 1

        timing = {"read_ns": 0, "slot_wait_ns": 0, "copy_ns": 0, "done_wait_ns": 0}

        def worker() -> None:
            while not stop.is_set():
                with guard:
                    window = next(pending, None)
                if window is None:
                    return
                waited = time.perf_counter_ns()
                slot = free.get()
                try:
                    ring.take(slot)
                    started = time.perf_counter_ns()
                    timing["slot_wait_ns"] += started - waited
                    read_window(window[0], window[1], slot)
                    timing["read_ns"] += time.perf_counter_ns() - started
                except BaseException as exc:
                    stop.set()
                    done.put(exc)
                    return
                done.put((window[0], window[1], slot))

        pool = [
            threading.Thread(target=worker, name=f"fill-reader-{n}", daemon=True)
            for n in range(self.readers)
        ]
        for thread in pool:
            thread.start()
        try:
            for n in range(len(windows)):
                waited = time.perf_counter_ns()
                item = done.get()
                started = time.perf_counter_ns()
                timing["done_wait_ns"] += started - waited
                if isinstance(item, BaseException):
                    raise item
                at, count, slot = item
                if n == cut:
                    raise _refuse(
                        "incomplete_fill",
                        f"planted interruption after {n} of {len(windows)} windows — the "
                        "arm that proves a killed fill leaves nothing Ready",
                    )
                index = bisect.bisect_right(starts, at) - 1
                while index < len(placements) and placements[index][0] < at + count:
                    start, end, flat = placements[index]
                    if end > at:
                        low, high = max(start, at), min(end, at + count)
                        ring.copy_out(slot, low - at, flat, low - start, high - low)
                    index += 1
                ring.give_back(slot)
                free.put(slot)
                timing["copy_ns"] += time.perf_counter_ns() - started
        finally:
            stop.set()
            for slot in ring.slots:
                free.put(slot)
            for thread in pool:
                thread.join(timeout=5)
        self.stats.update({k: v // 1_000_000 for k, v in timing.items()})
        return len(windows)


# ------------------------------------------------------------------------- helpers


def _dtype_name(dtype: object) -> str:
    """The one dtype spelling. The store writes tensorfs names (`f16`), torch prints its
    own (`torch.float16`), and a fill that guesses which is which is a silent wrong copy."""
    name = str(getattr(dtype, "name", dtype)).removeprefix("torch.")
    return {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
        "i64": "int64",
        "i32": "int32",
        "i16": "int16",
        "i8": "int8",
        "u8": "uint8",
    }.get(name, name)


def _tensors(module: Any) -> Iterator[tuple[str, Any]]:
    yield from module.named_parameters(remove_duplicate=False)
    yield from module.named_buffers(remove_duplicate=False)


def _non_persistent(root: Any) -> set[str]:
    names: set[str] = set()
    for prefix, sub in root.named_modules():
        for local in getattr(sub, "_non_persistent_buffers_set", ()):
            names.add(f"{prefix}.{local}" if prefix else local)
    return names


def read_plan(path: str | Path, component: str | None = None) -> list[PlanRow]:
    """Parse a tensorfs read plan (component, key, dtype, shape, nbytes, segments).

    This is the header table in the store's own order, which is the order the fill plane
    reads in. tfs-007 replaces this parser with the real `Header` reader; the ROW SHAPE is
    the seam and is deliberately identical.
    """
    import base64

    rows: list[PlanRow] = []
    for line in Path(path).read_text().strip().split("\n"):
        comp, key, dtype, shape, nbytes, where = line.split("\t")
        if component is not None:
            comp = component
        segments: list[Segment] = []
        if where.startswith("INLINE:"):
            body = base64.b64decode(where[7:])
            segments.append(Segment(None, len(body), body))
        else:
            for piece in where.split(","):
                oid, length = piece.split(":")
                segments.append(Segment(oid, int(length)))
        rows.append(
            PlanRow(
                key=f"{comp}.{key}",
                component=comp,
                dtype=dtype,
                shape=tuple(int(d) for d in shape.split(",") if d),
                nbytes=int(nbytes),
                segments=tuple(segments),
            )
        )
    return rows


def topology(plan: Sequence[PlanRow]) -> dict[str, TensorSpec]:
    """The checkpoint topology table the loader matches the code against."""
    return {
        row.key: TensorSpec(row.shape, _dtype_name(row.dtype), "plain", row.nbytes)
        for row in plan
    }
