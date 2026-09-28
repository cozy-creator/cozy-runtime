"""cr-005 — the streaming fill plane: CAS bytes into the contract's exact destinations.

This is the real body behind cr-002b's `Backend` seam. The package declares WHICH
components a method touches; it never learns HOW their bytes got there (§1.1/§3.2), so
everything in this module is runtime-owned and invisible to author code.

The path, once, in order:

    verified CAS read -> pinned staging ring -> async H2D on a runtime stream
      -> event fence -> atomic commit

Five properties are load-bearing rather than incidental:

**One canonical logical tensor to one canonical destination.** The keys walked here are
exactly the keys the derive harness proved bijective between its deep walk and the fill
plane's own census — a destination absent from the census has no path to bytes, and a
censused key that never arrives refuses the whole generation.

**The freeze point.** A generation is observable as READY only after every H2D completion
event has fired. Until then the transaction is `staging`, and any failure takes it to
`poisoned` with verified CAS untouched. There is no state in which a partially filled
component answers a request.

**The executor process is the transaction** (decisions #613, cr-025). This plane keeps no
ledger and unwinds nothing on a failure path: a fill that fails after a byte moved poisons
the generation, the worker replaces the executor process (cr-024), and external reclaim
frees the card — se-002 banked the 90.8 GiB refill at ~3 s warm, which is cheaper and more
honest than any incremental unwind. The ONE piece of arithmetic that survives is ADMISSION:
`price_envelope`/`complete_device_envelope` say whether the plan fits measured capacity
before the first allocation, and the executor refuses on it. Nothing here charges, releases,
clamps or reconciles a counter after that.

**Leases outlive the copy.** A pinned slot is reusable only after the CUDA event recorded
on the stream that consumed it. Returning a slot with a live copy is a typed refusal, not
a race — `lease_live`.

**Read order is the CONSTRUCTION's, byte location is the STORE's.** The construction census
gives the complete ordered DestinationSet to TensorFS; the header is lookup data that says
where those tensors' bytes live, never a second traversal authority. TensorFS validates and
plans the supplied traversal, while every delivered batch names the destination its bytes
belong to.

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
import hashlib
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, Unpack, cast, get_args

import tensorfs

from cozy_runtime.author._loader import (
    Census,
    TensorLike,
    TensorSpec,
    _members,
    canonical_dtype,
    census,
)
from cozy_runtime.internal import accel, canonical
from cozy_runtime.internal.derive import placing, torch_module
from cozy_runtime.internal.encoding import (
    TORCH_DTYPES,
    Capabilities,
    DeviceFacts,
    Encoded,
    LeafProvider,
    Provider,
    RolePart,
    Selection,
    aliases,
    device_line,
    implementation_digest,
    launch_providers,
    measure_device,
    measure_runtime,
)
from cozy_runtime.internal.forensics import FillForensics, distinct_storage_bytes
from cozy_runtime.internal.paging import BlockLayout, PageUnit, partition
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.probe import QualificationResult, observed_capability_records, qualify
from cozy_runtime.internal.resolution import (
    PlanRefusal,
    ResolvedModelPlan,
    TensorResolution,
    script_plan,
)

# Every `Any` below is a torch module, tensor, event or device: torch is absent from the
# check venv.

if TYPE_CHECKING:
    from tensorfs import Batch, ReadPlan, Tensor
    from tensorfs import ReadLease as ReadLease
    from tensorfs import Store as Store

# ------------------------------------------------------------------------- refusals


class FillRefusal(Exception):
    """A typed fill refusal. Every one of these fires BEFORE any byte reaches a
    destination, or aborts the generation that was going to hold it."""

    def __init__(self, message: str, *, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(fields)


#: The complete refusal vocabulary of this plane, and EXACTLY the producible codes. Named
#: as a frozen set so a new code cannot appear without touching this line — and a code
#: leaves with its last producer, because an entry nothing can raise makes the registry lie
#: in the same way an unregistered producer does, just in the other direction (#504).
REFUSALS = frozenset(
    {
        "missing_object",  # the CAS has no object file for a declared segment
        "digest_mismatch",  # object bytes do not hash to the object id
        "length_mismatch",  # segment/destination/object lengths disagree
        "dtype_mismatch",  # stored dtype is not the destination's (decode is cr-006)
        "destination_absent",  # a fill for a key the contract does not name
        "incomplete_fill",  # commit with a DestinationSet key never enqueued
        "lease_live",  # a staging slot reused before its completion event
        "derived_unmaterialized",  # a non-persistent buffer the fill plane cannot supply
        # cr-006's three, spoken in the SAME vocabulary because the worker, the
        # worker records and the CLI all read one refusal table. `encoding.py` owns their
        # meaning; this line is what keeps a new code from appearing unnoticed.
        "unknown_encoding",  # the header cites a spec digest no provider claims
        "encoding_unqualified",  # a claimed encoding, unqualified on THIS device
        "role_mismatch",  # the stored roles are not the roles the decoder accounts for
        # The store's own lifecycle and header refusals, spoken here because
        # `STORE_REFUSALS` already routes TensorFS codes to them. They were being spoken
        # WITHOUT being declared, so `_refuse`'s guard turned every one of them into an
        # AssertionError abort — the guard catching exactly what it exists to catch.
        "lease_revoked",  # the read lease covering these bytes was revoked mid-fill
        "checkpoint_unreadable",  # the snapshot's header/document bytes do not parse
        "store_unusable",  # a caller-prepared Store root that TensorFS refuses to open
        # The stored carrier does not match the encoding spec it CITES: a role set that is
        # not the spec's exact set, or a role whose shape is not what the spec's relation
        # derives. Distinct from `role_mismatch` on purpose — that one says THIS RUNTIME's
        # decoder does not account for the roles (a runtime gap, READER side), and this one
        # says the stored bytes are not the encoding they claim (DATA side, rebuild the
        # artifact). Collapsing them would print the wrong remedy, which is the exact cost
        # `worker/refusal.py` exists to avoid (#503b). It becomes producible with mxfp8: a
        # block scale is the first role whose geometry is DERIVED (`ceil_div(cols, 32)`)
        # rather than fixed, so "the right roles at the wrong shape" is a state that can
        # exist for the first time.
        "carrier_geometry",
        # #549.4 and #549.5, spoken here for the same reason cr-006's other codes are:
        # one refusal table. `spec_unreviewed` says the registry knows this exact spec
        # digest and no provider in this build was ever read against it (remedy: review a
        # provider, or stop citing the spec) — the alias-keyed launch table could not
        # produce it, because it handed the digest a sibling's decoder. `geometry_unsupported`
        # says every implementation refuses this tensor's own SHAPE (remedy: the artifact's
        # geometry, or a provider with declared padding semantics); the device and the bytes
        # are both fine, which is why it is not `encoding_unqualified`.
        "spec_unreviewed",
        "geometry_unsupported",
        # RETIRED with #549.3: `objective_unranked`. It fired when two providers qualified
        # and the deployment's objective named no per-record axis — which was `latency`, the
        # DEFAULT, so the default path refused as soon as a second implementation existed.
        # The axis was never missing; it lived one layer up, in the measured envelopes of
        # whole plans, and the chooser that held them never ran. `resolution.py` makes the
        # joint decision now and CONFESSES uncalibrated instead of refusing.
        # The `encoded_gemm` route's own two. Distinct on purpose, because they print
        # different remedies: `leaf_unconsented` says the PACKAGE must declare that its
        # code survives a replaced linear op (an author edit), while `leaf_schema` says
        # the module tree cannot take a leaf where this destination is, or did not end up
        # holding what the substitution declared (a runtime/artifact disagreement).
        "leaf_unconsented",
        "leaf_schema",
        # cr-025's ONE-SELECTION-AUTHORITY law. `plan_divergence`: the resolved plan this
        # generation was priced and identified under names a tensor, an implementation, a
        # route, a geometry or a dtype that is not what this checkpoint holds or this build
        # and device can execute. The plane never re-selects to paper over it — the
        # planner-A/executor-B split is exactly what the digest in generation identity
        # forbids. `plan_unresolved`: no executable plan resolves for these bytes here,
        # spoken in this vocabulary by the honesty surface (`servability`).
        "plan_divergence",
        "plan_unresolved",
    }
)


class _Holder:
    """A minimal census subject: the census walks `components`, so one component's restage
    walks exactly one component. It exists so `stage` reuses the SAME census the whole
    construction used instead of a second traversal rule."""

    __slots__ = ("components",)

    def __init__(self, components: Mapping[str, Any]) -> None:
        self.components = dict(components)


def _refuse(code: str, message: str, fields: Sequence[str] = ()) -> FillRefusal:
    if code not in REFUSALS:
        raise AssertionError(f"{code} is not a declared fill refusal")
    return FillRefusal(message, code=code, fields=fields)


# ------------------------------------------------------------------- admission arithmetic


@dataclass(slots=True)
class ComponentMemory:
    """One component's construction, settled residency, and overlapping fill storage."""

    destinations: int = 0
    native_destinations: int = 0
    native: int = 0
    native_resident: int = 0
    native_scratch: int = 0
    decode: int = 0
    resident: int | None = None

    @property
    def settled(self) -> int:
        if self.resident is not None:
            return self.resident
        return self.destinations - self.native_destinations + self.native_resident

    @property
    def base(self) -> int:
        # Both the initial float tree and its final encoded tree must fit the ceiling.
        return max(self.destinations, self.settled)

    @property
    def peak(self) -> int:
        # _stream installs native leaves at component boundaries. Until then the entire
        # component's encoded payload coexists with its original logical destinations.
        return self.destinations + self.native + self.native_scratch + self.decode

    @property
    def overhead(self) -> int:
        return max(self.peak - self.base, 0)

    def admission(self, headroom: int) -> int:
        # Fill scratch and forward scratch are different phases. Already-live input
        # activations are accounted by the caller's measured allocator free space.
        return max(self.peak, self.settled + headroom)


class MoveRow(TypedDict):
    component: str
    action: Literal["park", "evict"]
    bytes: int


class LeavesRow(TypedDict):
    component: str
    action: Literal["leaves"]
    count: int
    bytes: int
    was: int


#: The legs that partition a stage's `ms` (cr-103), plus the destinations it put back.
type StageLeg = Literal[
    "reserve_ms",
    "lease_ms",
    "enqueue_ms",
    "commit_ms",
    "ring_ms",
    "copy_ms",
    "fence_ms",
    "destinations",
]
STAGE_LEGS: tuple[StageLeg, ...] = get_args(StageLeg.__value__)


class StageRow(TypedDict):
    component: str
    action: Literal["stage"]
    bytes: int
    ms: int
    stream_ms: int
    reserve_ms: int
    lease_ms: int
    enqueue_ms: int
    commit_ms: int
    copy_ms: int
    ring_ms: int
    fence_ms: int
    objects: int
    destinations: int


type StageEvent = MoveRow | LeavesRow | StageRow


class DecodeStats(TypedDict):
    tensors: int
    bytes: int
    flushes: int
    ms: int
    #: Destinations that became a MODULE instead of a decoded tensor, and the stored bytes
    #: those modules now hold RESIDENT: the opposite of `bytes`, which passed through.
    leaves: int
    resident_bytes: int


class Traversal(TypedDict):
    component: str
    order: str
    tensors: int
    digest: str
    first: str
    last: str


class FillStats(TypedDict, total=False):
    windows: int
    waits: int
    lease_bytes: int
    lease_objects: int
    lease_acquisitions: int
    leases: int
    materialize_ms: int
    ring_ms: int
    copy_ms: int
    fence_ms: int
    stream_ms: int
    fill_ms: int
    decode: DecodeStats
    traversals: list[Traversal]
    read_ns: int
    store: dict[str, object]


class PoisonReport(TypedDict, total=False):
    keys_in_flight: int
    enqueued: int
    resident: list[str]
    parked: list[str]
    allocator_bytes: int


class ComponentEnvelope(TypedDict):
    destinations: int
    native_destinations: int
    native: int
    native_resident: int
    native_scratch: int
    decode_total: int
    decode_largest: int
    decode_envelope: int
    rows: int


class DeviceEnvelope(TypedDict):
    components: dict[str, ComponentEnvelope]
    destination_bytes: int
    native_bytes: int
    decode_peak_bytes: int
    base_bytes: int
    fill_overhead_bytes: int
    required_bytes: int
    basis: str
    # Admission's resident view, which the executor adds before refusing a shortfall.
    resident_destination_bytes: NotRequired[int]
    local_resident_ceiling_bytes: NotRequired[int]
    resident_budget_bytes: NotRequired[int]
    authorized_device_limit_bytes: NotRequired[int]


class CompleteEnvelope(DeviceEnvelope):
    construction_destination_bytes: int
    derived_destination_bytes: int


def price_envelope(
    rows: Sequence[tuple[str, int, int, str, int, int]], decode_scratch_bytes: int
) -> DeviceEnvelope:
    """Price component-boundary fill peaks, retaining logical and settled bytes separately."""
    per: dict[str, ComponentEnvelope] = {}
    for component, logical, stored, route, native_resident, native_scratch in rows:
        cell = per.setdefault(
            component,
            ComponentEnvelope(
                destinations=0,
                native_destinations=0,
                native=0,
                native_resident=0,
                native_scratch=0,
                decode_total=0,
                decode_largest=0,
                decode_envelope=0,
                rows=0,
            ),
        )
        cell["rows"] += 1
        cell["destinations"] += logical
        if route == "encoded_gemm":
            cell["native"] += stored
            cell["native_destinations"] += logical
            cell["native_resident"] += native_resident
            cell["native_scratch"] += native_scratch
        elif route == "decoded_float":
            cell["decode_total"] += stored
            cell["decode_largest"] = max(cell["decode_largest"], stored)
    for cell in per.values():
        cell["decode_envelope"] = (
            min(cell["decode_total"], decode_scratch_bytes + cell["decode_largest"])
            if cell["decode_total"]
            else 0
        )
    memory = [_component_memory(cell) for cell in per.values()]
    destinations = sum(cell.destinations for cell in memory)
    base = sum(cell.base for cell in memory)
    overhead = max((cell.overhead for cell in memory), default=0)
    return {
        "components": per,
        "destination_bytes": destinations,
        "native_bytes": sum(cell.native for cell in memory),
        "decode_peak_bytes": max((cell.decode for cell in memory), default=0),
        "base_bytes": base,
        "fill_overhead_bytes": overhead,
        "required_bytes": base + overhead,
        "basis": "component construction/settled maximum + widest component fill overhead; "
        "header floor excluding derived buffers",
    }


def _component_memory(cell: ComponentEnvelope) -> ComponentMemory:
    return ComponentMemory(
        destinations=cell["destinations"],
        native_destinations=cell["native_destinations"],
        native=cell["native"],
        native_resident=cell["native_resident"],
        native_scratch=cell["native_scratch"],
        decode=cell["decode_envelope"],
    )


def complete_device_envelope(
    header: DeviceEnvelope, construction_destination_bytes: int
) -> CompleteEnvelope:
    """Include the MCC's derived buffers without charging parked native components."""
    derived = max(construction_destination_bytes - header["destination_bytes"], 0)
    base = header["base_bytes"] + derived
    return {
        **header,
        "construction_destination_bytes": construction_destination_bytes,
        "derived_destination_bytes": derived,
        "base_bytes": base,
        "required_bytes": base + header["fill_overhead_bytes"],
    }


# ------------------------------------------------------------------- the TensorFS reader


def tensorfs_module() -> ModuleType:
    """TensorFS for `internal/worker` modules, which do not import it themselves (cr-067)."""
    return tensorfs


def capabilities() -> frozenset[str]:
    """The features this process's TensorFS build names (`tensorfs.CAPABILITIES`, e.g.
    "ensure/1", "deliver/1"); an older build names none. Detected by name, never by version."""
    return frozenset(getattr(tensorfs, "CAPABILITIES", ()))


_stores: dict[tuple[int, str], Store] = {}
_stores_lock = threading.Lock()


def store(root: str | Path) -> Store:
    """The process's ONE TensorFS Store handle for `root`.

    TensorFS keeps its trust index and catalog connection alive only while a handle to the
    root lives, so a fresh `Store.open` per call re-validated and re-loaded the catalog every
    time. Keyed by pid: a forked child never shares its parent's SQLite connection.
    """
    key = (os.getpid(), os.path.abspath(root))
    with _stores_lock:
        held = _stores.get(key)
        if held is None:
            held = _stores[key] = tensorfs.Store.open(key[1])
        return held


def ensure_store(root: str | Path) -> Store:
    """`store(root)`, first creating the Store when its catalog is absent."""
    path = os.path.abspath(root)
    if not os.path.isfile(os.path.join(path, "tensorfs.sqlite")):
        created: Store = tensorfs.Store.ensure(path)
        with _stores_lock:
            _stores[(os.getpid(), path)] = created
        return created
    return store(path)


def open_store(root: Path) -> Store:
    """The process's Store handle for one caller-prepared root, refusals in fill vocabulary."""

    try:
        return store(root)
    except Exception as exc:
        raise _refuse(
            "store_unusable",
            f"{root}: not a usable TensorFS store — {type(exc).__name__}: {exc}; "
            "have the Store owner ensure it before launching Runtime",
        ) from exc


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
    # The header re-check against its own encoding closure (`Checkpoint.__init__`), which is
    # law 18 made mechanical: a RecordOwner's authority is not worker safety, so the worker
    # asks TensorFS whether every tensor's stored roles ARE the roles its cited spec
    # declares. Both codes were escaping this plane UNTRANSLATED — a raw
    # `tensorfs.errors.Refusal` crossing a boundary whose whole contract is that the
    # worker records and the CLI read ONE refusal table. Same class as #496a, one
    # layer up: the check fired correctly and the answer had nowhere to go.
    "ROLE_SET_MISMATCH": "carrier_geometry",
    "SHAPE_MISMATCH": "carrier_geometry",
}

#: The table's codomain is CHECKED against the vocabulary at import, not at the moment a
#: store refusal happens to fire. A translation to an undeclared code is a defect in this
#: file, and finding it on the failure path means finding it during an outage.
_undeclared = sorted(set(STORE_REFUSALS.values()) - REFUSALS)
if _undeclared:
    raise AssertionError(f"STORE_REFUSALS translates to undeclared codes: {_undeclared}")


@contextlib.contextmanager
def store_refusals(what: str) -> Iterator[None]:
    """Speak a TensorFS refusal in this plane's vocabulary, or let it through untranslated."""
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


#: The staging geometry tfs-007 measured: 4 MiB store windows into 16 MiB pinned slots,
#: sixteen of them, sixteen readers, eight copies in flight. The executor's fill and the
#: worker's page warm (h3a-018) read the store with the same shape.
WINDOW_BYTES = 4 << 20
RING_SLOTS = 16
READERS = 16
INFLIGHT = 8


def slot_bytes(window_bytes: int) -> int:
    """This plane's staging unit over the store's planning unit (tfs-007)."""
    return max(int(window_bytes) * 4, 16 << 20)


def block_read_bytes() -> int:
    """Bytes THIS process has actually pulled from the block layer, or -1 (cr-100).

    The one cheap way to tell a warm page-cache read from a cold one: page faults on the
    mapped CAS objects are charged here, cache hits are not. Observation only, and it is
    allowed to be unavailable -- every non-Linux host reports -1 rather than a zero that
    would read as "warm".
    """
    try:
        with open("/proc/self/io", "rb") as rows:
            for row in rows:
                if row.startswith(b"read_bytes:"):
                    return int(row.split(b":")[1])
    except (OSError, ValueError):
        return -1
    return -1


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
    """The LOGICAL dtype: what the destination is and what a decode targets."""
    shape: tuple[int, ...]
    nbytes: int
    """The LOGICAL byte count. For an encoded tensor this is NOT what is stored."""
    encoded: Encoded
    """The per-tensor encoding assignment, from the header's own citation (cr-006)."""

    def role_what(self, role: str) -> str:
        """TensorFS's own name for one stored role, `<component>/<name>#<role>`."""
        return f"{self.component}/{self.name}#{role}"

    @property
    def stored_nbytes(self) -> int:
        return self.encoded.stored_nbytes


class Checkpoint:
    """One model manifest in a TensorFS store, opened through the tfs-007 facade.

    This class is the ONLY place in the runtime that speaks to TensorFS, and it is
    deliberately thin. Object verification, GC-safe holds, plan ordering, dtype names, the
    reader pool and every refusal are the store's; what this plane adds is the projection
    from a header into the contract's rows and the H2D transaction on top of the stream.

    It replaces cr-005's interim `CasStore` + `read_plan` outright (decisions #264). The
    interim reader knew CAS layout, cached a file descriptor per (thread, object) and
    verified per object in Python; none of that survives, and none of it needs to.
    """

    def __init__(self, root: str | Path, manifest_id: str) -> None:
        self.root = str(root)
        self.manifest_id = manifest_id
        with store_refusals(f"opening manifest {manifest_id[:23]}"):
            self.store = store(self.root)
            header = self.store.manifest(manifest_id)["header"]
            if header is None:
                raise _refuse("checkpoint_unreadable", f"{manifest_id}: no cozytensors header")
            self.header_bytes = header
            # Encoding definitions are nested directly in cozytensors/1. TensorFS parses
            # and validates the header once and exposes each definition with the digest
            # that tensor rows cite; there is no second closure-object lookup.
            self.header = tensorfs.parse_header(self.header_bytes)
            self.specs = tuple(row["id"] for row in self.header["encodings"])
        self._aliases = aliases()

    def has(self, component: str) -> bool:
        """Does the header carry this component? A constructed component it does not carry
        is the fit's `component_missing`, so nothing here refuses it first."""
        return component in self.header["components"]

    def rows(self, component: str) -> list[PlanRow]:
        """The header's lookup rows for one component.

        Their iteration order has no execution authority. Plan resolution surveys each row
        before construction; the later construction census supplies the traversal that
        TensorFS validates and executes.
        """
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
            shape = tuple(logical["shape"])
            dtype = logical["logical_dtype"]
            width = tensorfs.DTYPES.get(dtype)
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
                    encoded=self._encoded(f"{component}.{key}", entry),
                )
            )
        return out

    def _encoded(self, key: str, entry: Tensor) -> Encoded:
        """The tensor's encoding assignment and stored roles, verbatim from the header.

        `plain/1` is a real entry here, not an absent form: the selection path has one
        branch and the verbatim provider is what it selects for unencoded bytes.
        """
        parts: list[RolePart] = []
        for role, part in entry["parts"].items():
            shape, dtype = tuple(part["shape"]), part["dtype"]
            width = tensorfs.DTYPES.get(dtype)
            if width is None or dtype not in TORCH_DTYPES:
                # BOTH directions, for the same reason the logical row checks both: a
                # carrier dtype TensorFS knows and this runtime has no torch spelling for
                # would otherwise reach the scratch allocation as a KeyError, which is an
                # implementation detail escaping where a refusal belongs.
                why = (
                    "this build of TensorFS does not know"
                    if width is None
                    else "TensorFS knows and this runtime has no torch binding for"
                )
                raise _refuse(
                    "dtype_mismatch",
                    f"{key}: role {role!r} is stored as {dtype!r}, which {why}",
                    [key],
                )
            nbytes = width
            for extent in shape:
                nbytes *= extent
            parts.append(RolePart(role, dtype, shape, nbytes))
        digest = entry["encoding"]
        return Encoded(digest, self._aliases.get(digest, "unregistered"), tuple(parts))

    def read_plan(
        self,
        traversal: Sequence[tuple[str, str]],
        window_bytes: int,
        components: Sequence[str],
    ) -> ReadPlan:
        """The store's own read plan over the traversal THIS PLANE asks for.

        cr-005's rule survives the swap and is now the store's to keep: read order is the
        STORE's, write order is the CONTRACT's. The runtime hands over TensorRequirements
        traversal and TensorFS decides how to touch the disk for it.

        `components` DECLARES what this plan is for (#570b), and the fill plane always knows
        it: this loop is already grouped BY COMPONENT, so the scope is the group's own name.
        Completeness is then checked against those components' tensors rather than the whole
        header — which is what makes an N-ary artifact servable at all. Under the old
        whole-snapshot rule H3's Fl2VA class named 917 of 3445 tensors and was refused for
        the 2528 belonging to a sibling transformer its contract forbids it to touch.
        """
        with store_refusals("planning the read"):
            return tensorfs.plan(
                self.header_bytes, list(traversal), list(components), window=window_bytes
            )

    def acquire(self) -> ReadLease:
        """A verified read lease over this snapshot's CozyTensors runtime closure."""
        with store_refusals(f"acquiring a lease on {self.manifest_id[:23]}"):
            lease = self.store.acquire_cozytensors(self.manifest_id)
        return lease

    def model_assets(self) -> dict[str, bytes]:
        """Read the header-declared model files through one verified closure lease."""
        names = sorted(self.header["assets"])
        if not names:
            return {}
        with (
            store_refusals(f"reading model assets from {self.manifest_id[:23]}"),
            self.acquire() as lease,
        ):
            assets = {
                name: bytes(lease.read_asset(self.header_bytes, name, max_bytes=64 << 20))
                for name in names
            }
        return assets


def logical_weight_bytes(root: str | Path, snapshots: Mapping[str, str]) -> dict[str, int]:
    """Each constructed component's LOGICAL bytes, from the header of the snapshot it fills
    from: what its destinations hold. An encoded checkpoint stores fewer bytes than that
    (fp8 SDXL: 4.71 GB stored, 6.94 GB logical), so the stored closure never prices one. A
    component its header lacks has no entry; model-code-fit judges that mismatch."""
    checkpoints = {snapshot: Checkpoint(root, snapshot) for snapshot in set(snapshots.values())}
    return {
        component: sum(row.nbytes for row in checkpoints[snapshot].rows(component))
        for component, snapshot in snapshots.items()
        if checkpoints[snapshot].has(component)
    }


@dataclass(frozen=True, slots=True)
class Servability:
    """This runtime's OWN answer to "can I serve these bytes", given before any of them move.

    #501f's runtime twin. That ruling makes a PRODUCER declare, in its artifact's manifest,
    that its encoding has no runtime provider. This is the other half, and it is the half
    that cannot be gamed: the runtime never reads an encoding CLAIM. It reads the per-tensor
    spec DIGESTS the header cites, looks each one up in the provider set this build was
    compiled with, and asks the capability table — whose records exist only where measured
    numerics minted them — whether this device serves it. A manifest that says "servable"
    changes nothing here, and neither does a manifest that says nothing at all.

    `serves` is therefore never a claim about the artifact; it is a claim about THIS worker
    standing on THIS card, which is the only thing a worker is entitled to claim.
    """

    serves: bool
    by_alias: dict[str, int]
    """Every encoding alias the header actually cites, and how many tensors cite it."""
    routes: dict[str, int]
    refusal: str
    code: str

    def line(self) -> str:
        cited = ", ".join(f"{alias} x{n}" for alias, n in sorted(self.by_alias.items()))
        if self.serves:
            served = ", ".join(f"{route} x{n}" for route, n in sorted(self.routes.items()))
            return f"serves {cited} via {served}"
        return f"CANNOT serve {cited}: [{self.code}] {self.refusal}"

    def require(self) -> None:
        """Raise the refusal, in the fill plane's vocabulary. A caller that wants to CHOOSE
        another variant reads `serves` instead — which is what the delivery lane does."""
        if not self.serves:
            raise _refuse(self.code, self.refusal)


def servability(rows: Sequence[PlanRow], answer: ResolvedModelPlan | PlanRefusal) -> Servability:
    """The honesty answer, DERIVED from plan resolution — never from a second selector.

    `answer` is what `resolution.resolve()` (or `script_plan()`) produced for these rows: a
    plan, in which case every route it names is executable here by construction, or the
    `PlanRefusal` it raised. A refusal is spoken through the walk's first typed verdict
    (`encoding_unqualified`, `role_mismatch`, ...) so the answer names the ENCODING and
    the remedy rather than the resolver's summary; a plan that refused for a reason the
    walk does not type is `plan_unresolved`.
    """
    by_alias: dict[str, int] = {}
    for row in rows:
        by_alias[row.encoded.alias] = by_alias.get(row.encoded.alias, 0) + 1
    if isinstance(answer, ResolvedModelPlan):
        return Servability(True, by_alias, answer.route_census(), "", "")
    summary = {"no_rows", "unconsented", "unscored", "eligible", "chosen", "chosen_uncalibrated"}
    for step in answer.steps:
        if step.verdict in summary:
            continue
        code = "encoding_unqualified" if step.verdict == "unqualified" else step.verdict
        if code in REFUSALS:
            return Servability(False, by_alias, {}, step.why, code)
    return Servability(False, by_alias, {}, answer.detail, "plan_unresolved")


def servability_for_script(
    rows: Sequence[PlanRow],
    *,
    release: str,
    providers: Mapping[str, tuple[Provider, ...]],
    capabilities: Capabilities,
    device: DeviceFacts,
    store: str = "synthetic",
    snapshot: str = "synthetic",
    objective: str = "latency",
    encoded_leaves: str = "accept",
) -> Servability:
    """The honesty answer for a DRIVER SCRIPT: resolve a `script_plan`, then derive.

    Header work only — no tensor bytes move — through the same `resolve()` the executor
    uses, so what a script prints as servable is what a worker would execute.
    """
    try:
        answer: ResolvedModelPlan | PlanRefusal = script_plan(
            rows=rows,
            components=sorted({row.component for row in rows}),
            release=release,
            store=store,
            snapshot=snapshot,
            providers=providers,
            capabilities=capabilities,
            device=device,
            runtime=measure_runtime(torch_module(), release),
            facts=PlanFacts(),
            objective=objective,
            encoded_leaves=encoded_leaves,
            dtype_name=dtype_name,
        )
    except PlanRefusal as exc:
        answer = exc
    return servability(rows, answer)


def tensor_schema_of(rows: Sequence[PlanRow]) -> dict[str, TensorSpec]:
    """The checkpoint tensor schema the loader matches the code against.

    The spec's `encoding` carries the tensor's real ALIAS rather than a constant `plain`:
    the match itself is over logical shape (an encoding route changes no tensor schema), but
    a table that lied about the
    encoding would be the only place in the runtime that did.
    """
    return {
        row.key: TensorSpec(row.shape, dtype_name(row.dtype), row.encoded.alias, row.nbytes)
        for row in rows
    }


def _traversal_digest(keys: Sequence[str]) -> str:
    """Identity of one exact destination sequence, without importing a document codec."""
    digest = hashlib.sha256()
    for key in keys:
        raw = key.encode("utf-8")
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
    return "sha256:" + digest.hexdigest()


def _require_construction_order(value: object, component: str) -> str:
    """Admit TensorFS's construction-order token and no benchmark/header order."""
    order = str(value)
    if order != "construction":
        raise _refuse(
            "plan_divergence",
            f"{component}: TensorFS returned order {order!r} for the construction's "
            "destination traversal; header/key order is not executable",
            [component],
        )
    return order


@dataclass(frozen=True, slots=True)
class LeafReplacement:
    """One destination that became a MODULE instead of a tensor. The substitution, as data.

    `materialize`'s law is that placement moves bytes and never tensor schema, and it is checked
    key-for-key. Leaf replacement moves tensor schema, which is why it is not placement and why
    it cannot simply relax that check: what replaces it is a DECLARED substitution, and
    this record is the declaration. `commit` re-censuses and requires that the module tree
    holds exactly `contract_key` gone and `role_keys` arrived — nothing else moved, nothing
    else vanished. A leaf that quietly dropped a bias, or a provider that registered a
    buffer nobody expected, is a `leaf_schema` refusal rather than a surprise at the
    first forward pass.
    """

    contract_key: str
    """The logical destination the checkpoint supplies, e.g. `unet.…qkv_proj.weight`."""
    module_path: str
    """The module that was REPLACED, e.g. `unet.…qkv_proj` — the parent of the tensor."""
    was: str
    now: str
    role_keys: tuple[str, ...]
    """The census keys the leaf's own buffers present, fully qualified."""
    stored_bytes: int
    original: Any
    owner: Any
    attribute: str
    freed_bytes: int
    """Device bytes the float destination gave back. The other half of the VRAM claim: a
    replacement that freed nothing bought nothing, whatever the payload cost."""


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
        inflight: int,
        fence: bool = True,
    ) -> None:
        # Double buffering needs two slots and one free slot beyond the copies in flight;
        # a smaller request is raised to that minimum rather than refused.
        slots = max(int(slots), 2)
        self.torch = torch
        self.device = device
        self.window_bytes = int(window_bytes)
        self.fence = fence
        self.in_flight_peak = 0
        self.stream = torch.cuda.Stream(device=device)
        self.buffers: list[Any] = []
        self.views: list[memoryview] = []
        pinned, locked = True, 0
        while len(self.buffers) < slots:
            # device="cpu" is NOT redundant: construction runs under an ambient meta
            # device, and a meta staging buffer has data_ptr() == 0 — the ring would read
            # into address zero. Every allocation this plane makes names its device.
            try:
                buffer = torch.empty(
                    self.window_bytes, dtype=torch.uint8, pin_memory=pinned, device="cpu"
                )
            except RuntimeError:
                # A host that cannot page-lock more gets a smaller ring, and one that
                # cannot page-lock two slots stages through pageable memory: slower, never
                # a refusal (host RAM is whatever the machine has).
                if len(self.buffers) >= 2:
                    break
                if not pinned:
                    raise
                pinned = False
                continue
            locked += pinned
            block = (ctypes.c_ubyte * buffer.numel()).from_address(buffer.data_ptr())
            self.buffers.append(buffer)
            self.views.append(memoryview(block).cast("B"))
        self.keep = min(int(inflight), len(self.buffers) - 1)
        #: The page-locked bytes this ring REGISTERED with the driver, and the high-water
        #: of slots DMA-live at once. Facts for the worker's ledger, never a budget.
        self.registered_bytes = self.window_bytes * locked
        self.inflight: collections.deque[tuple[Any, Any]] = collections.deque()
        self.waits = 0
        self.charged = 0

    def hold(self, event: Any, batch: Any) -> None:
        """This batch's slot is now DMA-live; it is charged until its event fires."""
        self.inflight.append((event, batch))
        self.charged += 1
        self.in_flight_peak = max(self.in_flight_peak, self.charged)

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

    def track(self, tensor: Any) -> None:
        """The destination is written on OUR stream; the allocator must know."""
        tensor.record_stream(self.stream)

    def finish(self) -> None:
        self.drain(0)
        self.stream.synchronize()
        self.torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def close(self) -> None:
        """Drop the pinned slots. No counter to settle — the ring reports facts."""
        self.inflight.clear()
        self.buffers.clear()
        self.views.clear()


# -------------------------------------------------------------------------- backend


State = Literal["open", "staging", "ready", "poisoned"]


class ReadLeases:
    """Every verified read lease one generation holds, keyed by WHAT A LEASE COVERS.

    A lease is a hold on a SNAPSHOT. The fill plane asks for one per COMPONENT, and the
    components of a single-snapshot model all read from the same snapshot — so a map keyed
    by component name made the identical hold be taken once per component. Measured on
    sdxl: four acquisitions of the same 2,602-object lease, 1091/1104/1160/1257 ms, and
    10,408 pinned file descriptors for 2,602 distinct objects — against the FD_HEADROOM
    admission `acquire_cozytensors` performs on its way in (cr-102/cr-103).

    It lives here, apart from the backend, because the backend's constructor measures a
    device and runs the qualification suite: lease bookkeeping that could only be exercised
    behind those could only be proved by a fill on a card, and this is arithmetic over a
    real store that a test can drive directly. A component-per-snapshot binding (cr-008b)
    still takes one lease per distinct snapshot, which is the point — the key is the
    snapshot, not the model.
    """

    def __init__(self, checkpoints: Mapping[str, Checkpoint]) -> None:
        self.checkpoints = checkpoints
        self.held: dict[str, ReadLease] = {}
        #: How many times this generation went to the store. The EFFECT the keying has to
        #: show, reported rather than asserted about the code: one per distinct manifest.
        self.acquisitions = 0
        #: HIGH-WATER, like `pinned`: what was held at the peak, not what is held now. A
        #: full-resident generation releases at its first commit, so a live count read
        #: from the prepare facts is always zero and says nothing about the descriptors
        #: this fill actually spent against FD_HEADROOM.
        self.peak = {"objects": 0, "bytes": 0}

    def acquire(self, component: str) -> ReadLease:
        """The lease covering this component's manifest, taken once and reused."""
        checkpoint = self.checkpoints.get(component)
        if checkpoint is None:
            raise _refuse(
                "destination_absent",
                f"no checkpoint is bound for component {component!r} (bound: "
                f"{', '.join(sorted(self.checkpoints)) or 'none'})",
            )
        manifest = checkpoint.manifest_id
        lease = self.held.get(manifest)
        if lease is None or not lease.live:
            lease = self.held[manifest] = checkpoint.acquire()
            self.acquisitions += 1
            self.peak = {
                "objects": max(self.peak["objects"], self._objects()),
                "bytes": max(self.peak["bytes"], self._bytes()),
            }
        return lease

    def _objects(self) -> int:
        return sum(len(one.objects) for one in self.held.values())

    def _bytes(self) -> int:
        return sum(int(one.bytes) for one in self.held.values())

    def release(self) -> None:
        """Give every hold back. Idempotent: a poison and a commit may both reach here."""
        held, self.held = self.held, {}
        for lease in held.values():
            if lease.live:
                lease.release()

    def document(self) -> dict[str, int]:
        """What this generation leased, in the units the FD_HEADROOM admission is spent in.

        `leases` is what is held right now; `objects` and `bytes` are the peak, because a
        commit releases and a report of the after state would say a fill leased nothing.
        """
        return {
            "leases": len(self.held),
            "acquisitions": self.acquisitions,
            "objects": max(self.peak["objects"], self._objects()),
            "bytes": max(self.peak["bytes"], self._bytes()),
        }


class StreamOptions(TypedDict, total=False):
    """The stream and instrumentation knobs a driver script may pass through `for_script`."""

    window_bytes: int
    slots: int
    readers: int
    inflight: int
    fence: bool
    on_position: Callable[[int], None] | None
    decode_scratch_bytes: int
    forensics_dir: str
    custody: str


class StreamingFillBackend:
    """The `author.Backend` implementation that moves real bytes (§1.1's ONE fill path).

    `fill` VALIDATES and enqueues; `commit` moves, fences and freezes. Splitting them is
    what makes the transaction real: the whole component's destinations are known before
    a single byte lands, so a refusal has nothing to undo and a failure has one thing.
    """

    def __init__(
        self,
        checkpoint: Checkpoint | Mapping[str, Checkpoint],
        rows: Sequence[PlanRow],
        *,
        plan: ResolvedModelPlan,
        # THE CURRENT DEVICE OF A SEALED PROCESS (cr-066): the worker sealed this executor
        # to its lane's devices, so "cuda" resolves to the lane's card and no ordinal is
        # chosen here. A driver script that constructs this backend directly inherits the
        # same rule from its own environment.
        device: str = "cuda",
        window_bytes: int = 16 << 20,
        slots: int = 16,
        readers: int = 16,
        inflight: int = 8,
        fence: bool = True,
        on_position: Callable[[int], None] | None = None,
        decode_scratch_bytes: int = 512 << 20,
        release: str = "cozy-runtime/0.0.2",
        qualified: tuple[Capabilities, QualificationResult] | None = None,
        forensics_dir: str = "",
        custody: str = "canonical",
    ) -> None:
        self.torch = torch_module()
        # ONE CHECKPOINT PER COMPONENT (cr-008b). A TensorFS read plan must name every
        # tensor its header carries — tfs-007's completeness rule — so a whole-pipeline
        # snapshot cannot be read one component at a time, and reading 6.4 GiB to stage
        # 0.16 GiB is not staging. Per-component snapshots turn the partial read into the
        # store's ordinary COMPLETE read; the objects dedup, so it costs no extra disk. A
        # single-component binding passes one Checkpoint and nothing below changes.
        self.checkpoints: dict[str, Checkpoint] = (
            dict(checkpoint)
            if isinstance(checkpoint, Mapping)
            else {row.component: checkpoint for row in rows}
        )
        # Lookup only. Header iteration order must not become execution order: canonical JSON
        # sorted it lexically (`blocks.10` before `blocks.2`), and the deterministic CBOR
        # hardcut requires the ordered tensor rows to agree with the construction traversal.
        # In both cases `_stream` receives the exact order from `fill`'s enqueue transaction.
        self.plan = {row.key: row for row in rows}
        #: How this machine came to hold the bytes (model-code-fit D5): `canonical` refuses
        #: a stored key the code does not consume, `local` lists and ignores it. The fit
        #: reads it; nothing else here does.
        self.custody = custody
        self.device = self.torch.device(device)
        self.device_str = str(self.torch.empty(0, device=self.device).device)
        #: The BACKEND this generation is filled on. Memory metering and reclaim go through
        #: `accel` with it (#447); the stream/event machinery below is this boundary's CUDA
        #: IMPLEMENTATION, which the MPS lane replaces behind the same names (cr-021).
        self.kind = str(self.device.type)
        # The window is the store's PLANNING unit and the slot is this plane's staging unit;
        # tfs-007 measured 4 MiB windows into 16 MiB slots as the shape that works.
        self.window_bytes = int(window_bytes)
        self.slot_bytes = slot_bytes(window_bytes)
        self.slots = int(slots)
        self.readers = int(readers)
        self.inflight = int(inflight)
        #: The staging ring's page-locked facts, read off the last ring this plane ran:
        #: bytes REGISTERED with the driver and the peak of slots DMA-live at once. They
        #: ride the prepare facts for the worker's ledger; nothing here budgets on them.
        self.pinned: dict[str, int] = {"registered_bytes": 0, "in_flight_peak_bytes": 0}
        self.fence = fence
        #: cr-011's liveness feed: the MONOTONE integral-MiB position of the logical stream,
        #: reported per batch. It is the only thing the wedge detector reads, and it is a
        #: position rather than a heartbeat — a stalled transfer stops moving it, while a
        #: slow one keeps moving it however slowly.
        self.on_position = on_position
        self.leases = ReadLeases(self.checkpoints)
        self.state: State = "open"
        self.enqueued: dict[str, Any] = {}
        self.expected: dict[str, frozenset[str]] = {}
        self.components: dict[str, Any] = {}
        #: components this generation OWNS but is not holding on device right now. Their
        #: modules live on `meta`, so re-staging is a reservation plus a fill, never a copy
        #: back from a host tier that does not exist (§3.2: eviction FREES).
        self.parked: dict[str, Any] = {}
        self.vram_charge: dict[str, int] = {}
        #: One accounting authority keeps the float construction peak distinct from the
        #: settled encoded tree. Eviction never overwrites the bytes needed to restore it.
        self.component_memory: dict[str, ComponentMemory] = {}
        #: cr-008b: a STAGED plan re-reads the store after construction, so the GC-safe hold
        #: must outlive the first commit. It is released by `close()` with the generation.
        self.hold_lease = False
        self.stage_log: list[StageEvent] = []
        #: DERIVED tensor values per component, captured once at construction. A
        #: non-persistent buffer is a pure function of config that no checkpoint supplies,
        #: so an eviction that dropped it could never get it back — the module's `__init__`
        #: is not re-runnable. Held here, a re-stage restores exactly what construction
        #: computed. Kilobytes, and the alternative is serving uninitialized memory.
        self.derived: dict[str, dict[tuple[Any, str], Any]] = {}
        #: The device bytes construction may leave RESIDENT. Components are filled in the
        #: contract's own declared order while the cumulative total fits; the rest are
        #: PARKED on `meta` and staged in on first use. 0 means "no ceiling", which is what
        #: an all-resident plan means. This is the DEPLOYMENT's declared budget, deliberately
        #: not a measurement: construction must be reproducible, and the attempt-time ladder
        #: is where measured free VRAM decides things (cr-008b).
        self.resident_budget = 0
        self.paging: dict[str, BlockLayout] = {}
        self.paging_hooks: list[Any] = []
        self.paging_specs: dict[str, TensorSpec] = {}
        self.paging_plans: dict[str, ReadPlan] = {}
        self.page_resident: dict[str, PageUnit] = {}
        self.stats: FillStats = {"windows": 0, "waits": 0}
        #: Why this generation was poisoned and what the allocator held at that instant.
        #: Empty until a failure lands; the refusal reply carries it as evidence. Nothing
        #: is freed by this plane afterwards — the process is replaced (#613).
        self.poison_report: PoisonReport = {}
        #: THE MEMORY FORENSICS (§3.2's meters, at the boundaries). None unless a capture
        #: directory was configured; every call below is a no-op then, and the serving path
        #: is byte-for-byte the same. See `internal/forensics.py` for why it is permanent.
        self.forensics = FillForensics.open(
            self.torch,
            self.kind,
            self.device,
            f"fill-{os.getpid()}-{int(time.time())}",
            forensics_dir,
        )
        # cr-006. The device is measured and the qualification suite runs at PREPARE time;
        # the request path runs neither. Selection is an O(1) lookup per key inside `fill`.
        self.device_facts: DeviceFacts = measure_device(self.torch, self.device.index or 0)
        self.providers = launch_providers()
        self.capabilities: Capabilities
        # cr-008c hands the suite's result IN. The delivery choice has to ask the capability
        # table which variants this device can serve BEFORE it knows which artifact to fill,
        # so the suite runs once above this constructor and its result is passed down —
        # rather than being re-run here against the variant the choice already made.
        if qualified is not None:
            self.capabilities, self.probe = qualified
        else:
            self.capabilities, self.probe = qualify(
                self.torch,
                self.providers,
                self.device_facts,
                sorted({dtype_name(row.dtype) for row in rows}),
                release=release,
            )
        #: THE RESOLVED PLAN IS THE EXECUTABLE AUTHORITY (cr-025). Every route, provider and
        #: geometry this plane will execute is read from it by contract key; nothing below
        #: this line runs a `Selector`. Its digest is what entered generation identity, so a
        #: mismatch between it and what this checkpoint/build/device can do is
        #: `plan_divergence`, refused before a byte moves — never re-chosen.
        self.execution = plan
        self.resolved: dict[str, TensorResolution] = plan.by_key()
        planned, held = set(self.resolved), set(self.plan)
        if planned != held:
            raise _refuse(
                "plan_divergence",
                f"the resolved plan [{plan.digest()[:19]}…] names {len(planned)} destinations "
                f"and this checkpoint set supplies {len(held)}: missing from the plan "
                f"{sorted(held - planned)[:3]}, absent from the checkpoint "
                f"{sorted(planned - held)[:3]} — the plan was resolved over different rows",
                sorted(planned ^ held)[:6],
            )
        #: THE SERVABILITY ANSWER, derived from the plan (#501f's runtime twin): a resolved
        #: plan exists only where every tensor has a qualified route on this device.
        self.servability = servability(rows, plan)
        #: EVERY destination's provider, matched by digest HERE — at construction, before a
        #: lease is taken, a destination reserved or the consent gate consulted. The match is
        #: pure (this build's providers, this device's records), so a divergence refuses with
        #: nothing allocated; `fill` reads this map and never selects.
        self.selected: dict[str, Selection] = {
            key: self._select(key, row) for key, row in self.plan.items()
        }
        #: Transient device buffers holding STORED role bytes. They exist between the
        #: stream and the decode and never outlive one component's fill.
        self.scratch: dict[str, dict[str, Any]] = {}
        self.decode_scratch_bytes = int(decode_scratch_bytes)
        self.decoded = DecodeStats(tensors=0, bytes=0, flushes=0, ms=0, leaves=0, resident_bytes=0)
        #: The model class's `encoded_leaves=` declaration, handed in by `materialize` and
        #: `"refuse"` until it is. Not a default the runtime picked — the ABSENCE of a
        #: materialize call, which is a state no fill reaches.
        self.encoded_leaves = "refuse"
        #: What leaf replacement actually did, per replaced destination. It is the
        #: SUBSTITUTION MAP the census reconciliation is checked against, and the only
        #: record that a generation's tensor schema moved — so it is data a reader can print,
        #: not an effect they have to infer from the module tree.
        self.replaced: dict[str, LeafReplacement] = {}
        #: contract key -> the data pointer of the destination a leaf will stand for. It
        #: exists to catch a TIE: one tensor under two names, where replacing the first
        #: module frees memory the second still points at.
        self.leaf_sites: dict[str, int] = {}
        #: The census key set of each resident component, as of the last time this plane
        #: agreed with it. `materialize` writes it, `_reconcile_leaves` checks the
        #: substitution against it and writes the new one.
        self.census_keys: dict[str, frozenset[str]] = {}

    # -- plan execution (cr-025) --------------------------------------------------

    def _select(self, key: str, row: PlanRow) -> Selection:
        """The plan's answer for one destination, MATCHED BY DIGEST against this build.

        The plan names an implementation digest, a route, a validated geometry and an output
        dtype. This finds the provider in this build whose `implementation_digest` is exactly
        that, and the capability record on THIS device for it. Anything that does not match
        is `plan_divergence`: the tensor is not the tensor the plan priced, this build does
        not carry the implementation the plan chose, or this card was never measured under
        it. The plane refuses; it never selects a substitute.

        `Selection.record` is therefore the RE-MATCHED record, not the one resolution ranked
        — the two coincide whenever the plan was resolved on this device with this build,
        and a record that is absent now is a divergence rather than a permission.
        """
        resolved = self.resolved.get(key)
        if resolved is None:
            raise _refuse(
                "plan_divergence", f"{key}: the resolved plan names no such destination", [key]
            )
        actual = (row.encoded.encoding, tuple(row.shape), dtype_name(row.dtype))
        planned = (resolved.encoding, tuple(resolved.geometry), resolved.output_dtype)
        if actual != planned:
            raise _refuse(
                "plan_divergence",
                f"{key}: the plan priced encoding/geometry/dtype {planned} and the checkpoint "
                f"holds {actual} — the plan was resolved over different bytes",
                [key],
            )
        provider = next(
            (
                c
                for c in self.providers.get(row.encoded.encoding, ())
                if implementation_digest(c) == resolved.implementation_digest
            ),
            None,
        )
        if provider is None or str(provider.route) != resolved.route:
            raise _refuse(
                "plan_divergence",
                f"{key}: the plan executes {resolved.implementation} "
                f"[{resolved.implementation_digest[:19]}…, {resolved.route}] and this build "
                + (
                    "carries no implementation with that digest"
                    if provider is None
                    else f"has it on route {provider.route!r}"
                )
                + " — the runtime that resolved the plan is not the runtime executing it",
                [key],
            )
        records = self.capabilities.qualified(
            row.encoded.encoding,
            self.device_facts,
            resolved.output_dtype,
            {resolved.implementation_digest: provider},
        )
        if not records:
            raise _refuse(
                "plan_divergence",
                f"{key}: the plan executes {resolved.implementation} and "
                f"{device_line(self.device_facts)} holds no capability record for it — the "
                "plan was resolved on a device this one is not",
                [key],
            )
        return Selection(provider, records[0], row.encoded, resolved.reason)

    @classmethod
    def for_script(
        cls,
        checkpoint: Checkpoint | Mapping[str, Checkpoint],
        rows: Sequence[PlanRow],
        *,
        release: str,
        store: str,
        snapshot: str,
        objective: str = "latency",
        placement: str = "all_resident",
        encoded_leaves: str = "accept",
        device: str = "cuda",
        **kwargs: Unpack[StreamOptions],
    ) -> StreamingFillBackend:
        """A backend for a DRIVER SCRIPT with no model class: resolve, then construct.

        The small script-facing plan builder cr-025 asked for. It measures the device, launches
        the providers, runs the qualification suite once, resolves a `script_plan` over these
        rows under the stated objective, and hands that plan to the constructor — so a script
        fills through exactly the authority the executor does. A refusal is the resolver's
        `PlanRefusal`, surfaced untranslated: the script asked a plan question.
        """
        torch = torch_module()
        index = torch.device(device).index or 0
        facts = measure_device(torch, index)
        providers = launch_providers()
        qualified = qualify(
            torch,
            providers,
            facts,
            sorted({dtype_name(row.dtype) for row in rows}),
            release=release,
        )
        components = sorted({row.component for row in rows})
        plan = script_plan(
            rows=rows,
            components=components,
            release=release,
            store=store,
            snapshot=snapshot,
            snapshots=(
                {name: book.manifest_id for name, book in checkpoint.items()}
                if isinstance(checkpoint, Mapping)
                else {}
            ),
            providers=providers,
            capabilities=qualified[0],
            device=facts,
            runtime=measure_runtime(torch, release),
            facts=PlanFacts(),
            objective=objective,
            placement=placement,
            encoded_leaves=encoded_leaves,
            dtype_name=dtype_name,
        )
        return cls(
            checkpoint,
            rows,
            plan=plan,
            device=device,
            release=release,
            qualified=qualified,
            **kwargs,
        )

    # -- reservation -------------------------------------------------------------

    def expect(self, sets: Mapping[str, Sequence[str]]) -> None:
        """The checkpoint's complete destination key sets, and the verified read lease.

        These are membership facts for commit, not a traversal. The ordered DestinationSet
        arrives later from the construction census and must agree with them. The LEASE is
        acquired here rather than at commit: TensorFS verifies every object in the snapshot
        when the hold registers, so a corrupt or truncated artifact refuses BEFORE this plane
        reserves a single byte of device memory. The interim reader could only discover that
        mid-stream, with gigabytes of destinations already live.
        """
        # Completeness is a SET question. Retaining header iteration here would create a
        # second, accidental order authority even though commit only asks which keys arrived.
        self.expected = {name: frozenset(keys) for name, keys in sets.items()}
        for name in sets:
            self._acquire(name)
        # ONE ROW PER MANIFEST, so these are the objects and bytes actually held rather
        # than the same snapshot counted once per component that reads from it.
        held = self.leases.document()
        self.stats["lease_bytes"] = held["bytes"]
        self.stats["lease_objects"] = held["objects"]
        self.stats["lease_acquisitions"] = held["acquisitions"]
        self.stats["leases"] = held["leases"]

    def _acquire(self, component: str) -> ReadLease:
        """The verified read lease covering this component's manifest (see `ReadLeases`)."""
        return self.leases.acquire(component)

    def fit(self, walked: Census, *, encoded_leaves: str) -> Mapping[str, object] | None:
        """THE ONE VERDICT (model-code-fit §3): `tensorfs.fit` over the serving census and
        the checkpoint header, with this executor's device class and its own qualification
        records, before a destination is reserved. One fit per checkpoint the census reads:
        a component-per-snapshot binding (cr-008b) is judged against each header it fills
        from. The document comes back verbatim; on `ok` the keys `local` custody ignored
        leave the fill's completeness set so the commit does not miss them.
        """
        by_checkpoint: dict[str, tuple[Checkpoint, list[str]]] = {}
        for component in walked.components:
            checkpoint = self.checkpoints.get(component)
            if checkpoint is None:
                raise _refuse(
                    "destination_absent",
                    f"the construction built component {component!r} and the binding names "
                    f"no checkpoint for it (bound: {', '.join(sorted(self.checkpoints)) or 'none'})"
                    " — the derive's constructed components and the serving census disagree",
                    [component],
                )
            by_checkpoint.setdefault(checkpoint.manifest_id, (checkpoint, []))[1].append(component)
        observations = canonical.write(observed_capability_records(self.probe))
        verdict: dict[str, object] | None = None
        ignored: list[str] = []
        routes = {"verbatim": 0, "decoded_float": 0, "encoded_gemm": 0}
        for checkpoint, constructed in by_checkpoint.values():
            members = set(constructed)
            rows = [
                (d.component, d.key[len(d.component) + 1 :], d.spec.dtype, d.spec.shape)
                for d in walked.destinations
                if d.component in members
            ]
            requirements = [
                tensorfs.TensorRequirement(
                    component=component,
                    key=key,
                    shape=list(shape),
                    logical_dtype=tensorfs_requirement_dtype(dtype),
                )
                for component, key, dtype, shape in rows
            ]
            with store_refusals(f"fitting {checkpoint.manifest_id[:23]}"):
                fit = dict(
                    tensorfs.fit(
                        requirements,
                        checkpoint.header_bytes,
                        custody=self.custody,
                        encoded_leaves=encoded_leaves == "accept",
                        device=self.device_facts.predicate,
                        observations=observations,
                    )
                )
            if not fit.get("ok"):
                return fit
            verdict = fit
            ignored.extend(str(k) for k in cast("Sequence[object]", fit.get("ignored", ())))
            counted = cast("Mapping[str, int]", fit.get("routes", {}))
            for route in routes:
                routes[route] += int(counted.get(route, 0))
        if verdict is None:
            return None
        for full in ignored:
            component, _, key = full.partition("/")
            expected = self.expected.get(component)
            if expected is not None:
                self.expected[component] = expected - {f"{component}.{key}"}
        if len(by_checkpoint) > 1:
            verdict = {**verdict, "ignored": ignored, "routes": routes}
        return verdict

    def materialize(
        self, obj: object, walked: Census, *, encoded_leaves: str = "refuse"
    ) -> Mapping[str, TensorLike]:
        """Reserve ordinary CUDA destinations for the constructed component roots.

        Baseline residency is ordinary CUDA allocation and that is sufficient (§3.2); a
        promoted Varena adapter would replace this method and nothing else. The census is
        re-taken afterwards and must agree key-for-key: a materialization that changes the
        tensor schema is not a placement, it is a different construction.

        THE CONSENT GATE IS FIRST, before a byte of device memory is reserved. Every row's
        route is the resolved plan's — so an
        package that did not declare `encoded_leaves="accept"` and was handed a plan
        that replaces modules refuses HERE, with nothing allocated
        and nothing to undo. Discovering it at the decode, with the component resident,
        would mean rolling back gigabytes to deliver an answer that was knowable from the
        header.
        """
        self.encoded_leaves = encoded_leaves
        native = sorted(k for k, t in self.resolved.items() if t.route == "encoded_gemm")
        if native and encoded_leaves != "accept":
            raise _refuse(
                "leaf_unconsented",
                f"{len(native)} of this artifact's tensors ({', '.join(native[:3])}…) serve "
                f"through a NATIVE ENCODED LEAF on {device_line(self.device_facts)}, and this "
                f"package declares encoded_leaves={encoded_leaves!r}. That route does not "
                "hand the module a float weight: the stored payload and its scales ARE the "
                "weight and the linear op is REPLACED, so code that reads `.weight`, walks "
                "submodule dtypes or rebuilds a Linear from one finds something that is not "
                'there. The package states `class …(Model[…], encoded_leaves="accept")` '
                "when its components are touched through their forward only — the runtime "
                "cannot know that, and reading silence as consent is how a serving path "
                "starts raising AttributeError in production",
                native[:6],
            )
        roots = dict(_members(obj))
        self.paging_specs = {
            destination.key: destination.spec for destination in walked.destinations
        }
        started = time.perf_counter_ns()
        self._mark(
            "materialize_begin", walked_components=list(walked.components), roots=sorted(roots)
        )
        held = 0
        header_memory = self.device_envelope()["components"]
        with placing():
            for name in walked.components:
                member = roots.get(name)
                if member is None or not hasattr(member, "to_empty"):
                    continue
                size = sum(t.numel() * t.element_size() for _, t in _tensors(member))
                memory = self.component_memory[name] = (
                    _component_memory(header_memory[name])
                    if name in header_memory
                    else ComponentMemory()
                )
                memory.destinations = size
                # Captured BEFORE the park decision: a parked component is staged in later
                # and its derived tables have to survive to be restored, which they cannot
                # if the capture happens only on the resident branch.
                derived = self.derived[name] = _derived_values(member)
                if (
                    self.resident_budget
                    and size > self.resident_budget
                    and not any(
                        key.startswith(f"{name}.") and resolution.route != "verbatim"
                        for key, resolution in self.resolved.items()
                    )
                ):
                    try:
                        layout = partition(member)
                    except ValueError as exc:
                        raise _refuse(
                            "leaf_schema", f"{name}: unsafe paging layout: {exc}", [name]
                        ) from exc
                    if layout is not None:
                        self.paging[name] = layout
                        self.hold_lease = True
                if self.resident_budget and held + memory.base > self.resident_budget:
                    # PARKED, not dropped. Its tensor schema is built and its destinations are
                    # known; what it does not have is device bytes, and `stage()` supplies
                    # those on first declared use. A construction that silently skipped it
                    # would be a different model.
                    self.parked[name] = member
                    self.stage_log.append({"component": name, "action": "park", "bytes": size})
                    continue
                held += memory.base
                want = size
                self.vram_charge[name] = want
                # A `to_empty` that runs out of device memory halfway leaves the half it
                # moved held by the module. Nothing here frees it: the prepare that owns
                # this construction is poisoned by the raise and the process is replaced,
                # which is the one reclaim that cannot miss a byte (#613).
                member.to_empty(device=self.device, recurse=True)
                _restore_derived(member, derived)
                self.components[name] = member
                self._mark(
                    "component_reserved",
                    component=name,
                    priced=want,
                    derived_restored=len(derived),
                )
            # Inside the placement window too: censusing a REAL module detaches every
            # tensor, and a detach on cuda is exactly the "real allocation during
            # construction" the derive guard exists to catch in author code.
            fresh = census(obj)
        self.stats["materialize_ms"] = (time.perf_counter_ns() - started) // 1_000_000
        if [d.key for d in fresh.destinations] != [d.key for d in walked.destinations]:
            raise _refuse(
                "destination_absent",
                "the census after materialization is not the census before it — "
                "placement moves bytes, never tensor schema",
            )
        # The baseline the SUBSTITUTION law is checked against later. Taken here, from the
        # census this method just agreed with, so leaf replacement is measured against what
        # placement actually produced rather than against the plan's own idea of it.
        for destination in fresh.destinations:
            self.census_keys.setdefault(destination.component, frozenset())
        for component in {d.component for d in fresh.destinations}:
            self.census_keys[component] = frozenset(
                d.key for d in fresh.destinations if d.component == component
            )
        self._mark("materialize_end", destinations=len(fresh.destinations))
        return fresh.live

    # -- the forensics view ------------------------------------------------------

    def _forensic_state(self) -> dict[str, object]:
        """What the PLANE believes, in the same row as what the allocator holds.

        The two halves of a byte gap have to be read at one instant or the difference is an
        argument rather than a measurement. `charged` is this plane's arithmetic per class;
        `held` is the module trees' own distinct-storage bytes; `charge` is what each
        component was priced at. Subtracting `held` from `charge` names the census's error
        and subtracting their sum from the allocator's live bytes names everything else.
        """
        held: dict[str, int] = {}
        for name, member in list(self.components.items()):
            try:
                held[name] = distinct_storage_bytes(t for _, t in _tensors(member))
            except Exception:
                continue
        scratch = 0
        for parts in self.scratch.values():
            scratch += distinct_storage_bytes(parts.values())
        return {
            "charge": dict(self.vram_charge),
            "component_bytes": dict(self.component_bytes),
            "held": held,
            "held_total": sum(held.values()),
            "scratch_live_bytes": scratch,
            "enqueued": len(self.enqueued),
            "components": sorted(self.components),
            "parked": sorted(self.parked),
            "replaced": len(self.replaced),
        }

    def _mark(self, boundary: str, **fields: object) -> None:
        if self.forensics is None:
            return
        self.forensics.mark(boundary, **{**self._forensic_state(), **fields})

    # -- the arithmetic, before a byte moves -------------------------------------

    @property
    def component_bytes(self) -> dict[str, int]:
        return {name: memory.settled for name, memory in self.component_memory.items()}

    def stage_bytes(self, name: str, headroom: int = 0) -> int:
        if name in self.paging:
            return self.paging[name].working_bytes + headroom
        return self.component_memory[name].admission(headroom)

    def device_envelope(self) -> DeviceEnvelope:
        """Header costs for the selected implementation; the module census adds derived buffers."""
        rows = []
        for row in self.plan.values():
            route = self.resolved[row.key].route
            resident = scratch = 0
            provider = self.selected[row.key].provider
            if isinstance(provider, LeafProvider):
                parts = {part.role: part for part in row.encoded.parts}
                resident = provider.resident_bytes(parts)
                scratch = provider.fill_scratch_bytes(parts)
            rows.append(
                (row.component, int(row.nbytes), int(row.stored_nbytes), route, resident, scratch)
            )
        return price_envelope(rows, self.decode_scratch_bytes)

    # -- the Backend protocol ----------------------------------------------------

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None:
        """Validate one destination against the stored row and enqueue it. No I/O here:
        every refusal in the matrix must fire before the first byte moves."""
        row = self.plan.get(key)
        if row is not None and row.component in self.parked:
            # A PARKED component's destinations are on `meta` and have no bytes to fill.
            # The check is FIRST because every check below is about a device destination,
            # and a parked one legitimately has none — `stage()` fills exactly these keys
            # on first declared use, and `commit` requires completeness only for what is
            # resident.
            return
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
        if dtype_name(tensor.dtype) != dtype_name(row.dtype):
            raise _refuse(
                "dtype_mismatch",
                f"{key}: the checkpoint's LOGICAL dtype is {row.dtype} and the destination "
                f"is {dtype_name(tensor.dtype)} — a decoder targets the logical dtype the "
                "border assigned, it does not choose a different one (cr-006). A "
                "destination that wants another dtype is a different construction",
                [key],
            )
        # PLAN EXECUTION (cr-025): the provider was matched by digest at construction and is
        # read here, never selected. A key the plan does not name cannot reach this line —
        # `row` came from the same table the construction-time match covered.
        assert key in self.selected, key
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
                f"{key}: the tensor requirements says {spec.nbytes} B, the checkpoint header "
                f"{row.nbytes} B",
                [key],
            )
        if self.selected[key].provider.route == "encoded_gemm":
            # THE LEAF'S SITE, resolved before a byte moves and for the same reason every
            # other check here is: the decode leg has nothing to undo only if the whole
            # component's destinations were validated first. A native row that has no
            # module to replace is not a late surprise, it is a refusal at the same moment
            # a dtype disagreement is.
            self._leaf_site(key, row, tensor)
        self.enqueued[key] = tensor
        self.state = "staging"

    def _leaf_site(self, key: str, row: PlanRow, tensor: TensorLike) -> tuple[Any, Any, str]:
        """`(owner, replaced, attribute)` — where a leaf would stand for this destination.

        The unit of replacement is the MODULE, not the tensor: `nn.Module.__setattr__`
        refuses to accept anything that is not a Module as a child, so the payload cannot
        be assigned where the float weight was and the linear op has to be swapped whole.
        That is the machinery the `encoded_gemm` route waited on, and this is where it is
        checked to be possible.

        Four things have to hold, each a different way for a leaf to be wrong:
        a `.weight` (a leaf stands for a linear op, which its weight names), a rank-2
        logical shape (a scaled GEMM has two outer dimensions), an OWNER to hold the new
        child (a component root has no parent to be replaced on), and the module's own
        declared geometry if it has one (a tied or re-wired weight resolves to a module
        whose shape is not this tensor's, and installing there would compute a different
        model in silence).
        """
        component, _, local = key.partition(".")
        if not local.endswith(".weight"):
            raise _refuse(
                "leaf_schema",
                f"{key}: the native route replaces a LINEAR OP and names it by its weight; "
                f"this destination is not a `.weight`, so there is no module for the stored "
                "roles to become. The artifact encodes a tensor whose consumer this runtime "
                "cannot identify",
                [key],
            )
        if len(row.shape) != 2:
            raise _refuse(
                "leaf_schema",
                f"{key}: the native route computes a scaled GEMM, whose operands have two "
                f"outer dimensions; this destination is rank {len(row.shape)} "
                f"{row.shape}. A reshape here would be a different model",
                [key],
            )
        owner_local, _, attribute = local[: -len(".weight")].rpartition(".")
        root = self.components.get(component)
        if root is None:
            raise _refuse(
                "destination_absent",
                f"{key}: component {component!r} is not resident, so its module tree cannot "
                "take a leaf",
                [key],
            )
        if not attribute:
            raise _refuse(
                "leaf_schema",
                f"{key}: the weight belongs to the component ROOT, which has no parent to be "
                "replaced on — a leaf is installed by `setattr` on the module that owns it, "
                "and the root is owned by the census",
                [key],
            )
        owner: Any = root
        for step in owner_local.split(".") if owner_local else ():
            owner = getattr(owner, step, None)
            if owner is None:
                raise _refuse(
                    "leaf_schema",
                    f"{key}: no module at {component}.{owner_local} to hold a replaced leaf",
                    [key],
                )
        replaced = getattr(owner, attribute, None)
        held = getattr(replaced, "weight", None)
        # STORAGE identity, not object identity: the census reads destinations out of
        # `state_dict()`, which hands back detached views, so the tensor this method is
        # given is never the same Python object as the module's Parameter. What has to
        # match is the memory — that is what "this module owns this destination" means.
        if held is None or _data_ptr(held) != _data_ptr(tensor) or _data_ptr(tensor) == 0:
            raise _refuse(
                "leaf_schema",
                f"{key}: {component}.{local[: -len('.weight')]} does not own the destination "
                "this key names — a re-wired weight resolves to a module whose `.weight` is "
                "different memory, and installing a leaf there would compute a different "
                "model without saying so",
                [key],
            )
        tied = [
            other
            for other, site in self.leaf_sites.items()
            if other != key and site == _data_ptr(tensor)
        ]
        if tied:
            raise _refuse(
                "leaf_schema",
                f"{key}: this destination shares storage with {tied[0]}, and both serve "
                "through a native leaf. A tie is ONE tensor under two names, so replacing "
                "the first leaf frees memory the second module still points at — the alias "
                "group has to become one shared leaf, which this build does not do and will "
                "not fake",
                [key, tied[0]],
            )
        self.leaf_sites[key] = _data_ptr(tensor)
        declared_out = getattr(replaced, "out_features", None)
        declared_in = getattr(replaced, "in_features", None)
        if (declared_out, declared_in) not in ((None, None), (row.shape[0], row.shape[1])):
            raise _refuse(
                "leaf_schema",
                f"{key}: {type(replaced).__name__} declares "
                f"({declared_out}, {declared_in}) and the checkpoint supplies {row.shape}",
                [key],
            )
        return owner, replaced, attribute

    def commit(self) -> None:
        """Move every enqueued destination, fence the copies, then freeze.

        `ready` is set after the last completion event and not one statement earlier: the
        freeze point is what makes "no partially filled generation is ever observable"
        a property of the code rather than of the caller's discipline.
        """
        if self.state == "open":
            self.state = "ready"
            return
        missing = sorted(
            key
            for name, keys in self.expected.items()
            if name in self.components
            for key in keys
            if key not in self.enqueued
        )
        if missing:
            raise _refuse(
                "incomplete_fill",
                f"{len(missing)} destination(s) of a declared component never reached the "
                f"fill plane ({', '.join(missing[:4])}) — a DestinationSet is COMPLETE or "
                "the component does not commit",
                missing[:6],
            )
        self._commit_enqueued()
        self.state = "ready"

    def _commit_enqueued(self) -> None:
        """Move whatever is enqueued right now, fence it, and clear the queue.

        Split out of `commit` so a component STAGE is the same transaction over a smaller
        DestinationSet rather than a second copy of it — there is one code path that moves
        bytes to the device, and this is it.
        """
        started = time.perf_counter_ns()
        with placing():
            ring = StagingRing(
                self.torch,
                self.device,
                window_bytes=self.slot_bytes,
                slots=self.slots,
                inflight=self.inflight,
                fence=self.fence,
            )
            try:
                for tensor in self.enqueued.values():
                    ring.track(tensor)
                # THE COPY LEG, SPLIT OFF THE RING (cr-103). `stream_ms` is the whole
                # transaction: allocate the page-locked ring, track every destination,
                # read, copy, decode, fence, release. Only the middle of that scales with
                # BYTES; the rest scales with the number of destinations and slots. A swap
                # whose cost is per-object rather than per-byte is invisible in one total,
                # and which of the two it is decides whether the fix is faster I/O or
                # bigger batches.
                tracked = time.perf_counter_ns()
                windows = self._stream(ring)
                copied = time.perf_counter_ns()
                ring.finish()
                accel.synchronize(self.torch, self.kind)
                self.stats["ring_ms"] = (tracked - started) // 1_000_000
                self.stats["copy_ms"] = (copied - tracked) // 1_000_000
                self.stats["fence_ms"] = (time.perf_counter_ns() - copied) // 1_000_000
                self.stats["windows"] = windows
                self.stats["waits"] = ring.waits
                self.pinned = {
                    "registered_bytes": ring.registered_bytes,
                    "in_flight_peak_bytes": ring.in_flight_peak * ring.window_bytes,
                }
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
        self.stats["decode"] = self.decoded.copy()
        self._reconcile_leaves()
        # Logical destination addresses are retired after replacement. Keeping them
        # across commits would mistake allocator address reuse for a new tied weight.
        self.leaf_sites.clear()
        self.enqueued.clear()
        self._mark("fill_end", decoded=dict(self.decoded))
        if self.forensics is not None:
            self.forensics.dump("fill_end")

    def _reconcile_leaves(self) -> None:
        """The census law for a generation whose tensor schema MOVED, and priced bytes settle.

        `materialize` holds the placement law — the census after is the census before, key
        for key, because placement moves bytes and never tensor schema. Leaf replacement is not
        placement and does move tensor schema, so it cannot ride that law and must not relax it
        either. What holds here instead is the SUBSTITUTION law: the component's census is
        exactly what it was, minus every replaced destination, plus exactly the role keys
        each leaf DECLARED. A key that vanished unannounced, a declared role that never
        arrived, or a replaced weight still standing are all `leaf_schema` — because each
        one means the module tree is not what the fill plane priced.

        Then the component's priced bytes settle to the MEASURED truth rather than to
        arithmetic: its real device bytes, read off its own tensors, exactly as
        `materialize` read them before the replacement.
        """
        if not self.replaced:
            return
        for name, member in self.components.items():
            rows = [r for r in self.replaced.values() if r.contract_key.startswith(f"{name}.")]
            if not rows:
                continue
            before = self.census_keys.get(name, frozenset())
            gone = {r.contract_key for r in rows}
            arrived = {key for r in rows for key in r.role_keys}
            present = {f"{name}.{key}" for key in member.state_dict()}
            want = (before - gone) | arrived
            if present != want:
                raise _refuse(
                    "leaf_schema",
                    f"{name}: leaf replacement declared {len(gone)} destination(s) replaced "
                    f"by {len(arrived)} role key(s), and the module tree does not hold that. "
                    f"Missing {sorted(want - present)[:4]}, unexpected "
                    f"{sorted(present - want)[:4]}. A substitution the census cannot confirm "
                    "is a tensor schema priced wrong",
                    sorted(want ^ present)[:6],
                )
            self.census_keys[name] = frozenset(present)
            after = sum(int(t.numel() * t.element_size()) for _, t in _tensors(member))
            charged = self.vram_charge.get(name, 0)
            self.vram_charge[name] = after
            self.component_memory[name].resident = after
            self.stage_log.append(
                {
                    "component": name,
                    "action": "leaves",
                    "count": len(rows),
                    "bytes": after,
                    "was": charged,
                }
            )
            del self.stage_log[:-64]

    def release_lease(self) -> None:
        """Give the hold back. Idempotent: a poison and a commit may both reach here.

        A STAGING generation keeps it: re-staging an evicted component reads the store
        again, and reading objects the GC may already consider free is the one thing a hold
        exists to prevent. `close()` is where a held lease ends.
        """
        if self.hold_lease:
            return
        self.leases.release()

    def close(self) -> None:
        """End the generation's hold. Called when the executor tears the generation down."""
        for hook in self.paging_hooks:
            hook.remove()
        self.paging_hooks.clear()
        self.hold_lease = False
        self.release_lease()

    # -- component staging (cr-008b) ---------------------------------------------

    def evict(self, name: str) -> int:
        """FREE one component's device bytes. Returns what came back.

        Eviction is demand-pull and it FREES — there is no demote-copy and no host
        residence tier (§3.2). Re-staging is an ordinary cr-005 fill from the page-cached
        CAS mapping, which is both simpler and, on a warm page cache, faster than a D2H
        copy plus an H2D copy would be. The module is moved to `meta` rather than dropped,
        so its tensor schema survives and the restage is a reservation, not a construction.
        """
        if name in self.paging:
            if name in self.page_resident:
                raise _refuse(
                    "leaf_schema", "cannot evict a component during a block forward", [name]
                )
            if name not in self.components:
                return 0
            self._park_page(self.paging[name].common)
            self.parked[name] = self.components.pop(name)
            return self.vram_charge.pop(name, 0)
        member = self.components.pop(name, None)
        if member is None:
            return 0
        charged = self.vram_charge.pop(name, 0)
        with placing():
            self._free_storages(t for _, t in _tensors(member))
            member.to_empty(device=self.torch.device("meta"), recurse=True)
            rows = [r for r in self.replaced.values() if r.contract_key.startswith(f"{name}.")]
            for row in rows:
                # The original weight storage was retired by _decode; bias may alias the
                # encoded leaf, whose storage was just freed. Keep topology, never bytes.
                row.original.to_empty(device=self.torch.device("meta"), recurse=True)
                setattr(row.owner, row.attribute, row.original)
                self.replaced.pop(row.contract_key)
                self.leaf_sites.pop(row.contract_key, None)
            if rows:
                self.census_keys[name] = frozenset(
                    (self.census_keys[name] - {key for row in rows for key in row.role_keys})
                    | {row.contract_key for row in rows}
                )
        self.parked[name] = member
        self.stage_log.append({"component": name, "action": "evict", "bytes": charged})
        del self.stage_log[:-64]
        return charged

    def stage(self, name: str) -> StageRow | None:
        """Re-reserve and RE-FILL one evicted component. The same transaction, scoped.

        A paged component fills only its common unit and reports no stage legs (None).

        Every refusal the whole-generation path has fires here too, because it IS that
        path: `fill` validates each destination against the stored row, `_stream` splices
        the store's batches, and the completion events fence before the component is
        observable. A staged component that half-arrives POISONS the generation through
        the residency plane and the process is replaced; it is never served and never
        unwound in-process (#613).
        """
        member = self.parked.get(name)
        if member is None:
            raise _refuse(
                "destination_absent",
                f"component {name!r} is not parked on this generation (parked: "
                f"{', '.join(sorted(self.parked)) or 'none'}) — staging moves bytes for a "
                "component the construction already built, never for a new one",
            )
        if name in self.paging:
            common = self.paging[name].common
            self.parked.pop(name)
            self._fill_page(name, common)
            self.components[name] = member
            self.vram_charge[name] = common.nbytes
            return None
        started = time.perf_counter_ns()
        memory = self.component_memory[name]
        want = memory.destinations
        # NO UNWIND PAST THIS LINE. A stage that fails after reserving, or after the first
        # batch lands, raises; the residency plane poisons the generation and the process
        # is replaced (#613). Freeing the half that arrived would be an in-process rollback
        # engine, which is exactly what this hardcut deletes.
        with placing():
            member.to_empty(device=self.device, recurse=True)
            _restore_derived(member, self.derived.get(name, {}))
            fresh = census(_Holder({name: member}))
        reserved = time.perf_counter_ns()
        self.vram_charge[name] = want
        objects = len(self._acquire(name).objects)
        leased = time.perf_counter_ns()
        # OUT OF `parked` BEFORE THE FILL. `fill` skips a parked component's destinations —
        # they are on `meta` and have no bytes — so staging while still marked parked
        # enqueues NOTHING and commits an allocation full of whatever was in that memory.
        # It is the worst possible failure: on a warm allocator the freed pages often come
        # back with the old contents, so the weights LOOK right and only the output is
        # wrong. Measured exactly that — a black image from a "successful" 43 ms stage of
        # 4.78 GiB, a rate no disk can produce.
        self.parked.pop(name, None)
        # Own the tree before _leaf_site validates replacements. It is not observable
        # until the scope admission commits; a failure poisons the executor generation.
        self.components[name] = member
        for destination in fresh.destinations:
            self.fill(destination.key, destination.spec, fresh.live[destination.key])
        enqueued = time.perf_counter_ns()
        expected = self.expected.get(name, frozenset())
        missing = sorted(key for key in expected if key not in self.enqueued)
        if missing:
            raise _refuse(
                "incomplete_fill",
                f"staging {name!r} enqueued {len(self.enqueued)} of {len(expected)} "
                f"declared destinations ({', '.join(missing[:4])}) — a staged component "
                "is COMPLETE or it does not become resident",
                missing[:6],
            )
        self._commit_enqueued()
        self.state = "ready"
        ended = time.perf_counter_ns()
        took = (ended - started) // 1_000_000
        # THE SAME SPLIT `prepare` REPORTS (cr-100): `ms` is the whole stage, `stream_ms`
        # is the leg that actually moves bytes to the device — reserve, read, copy, fence.
        # A swap's cost has only ever been inferred from a construction's fill leg, which
        # is a number measured under different conditions; this is the swap's own.
        #
        # AND ITS OWN SEGMENT LEDGER (cr-103). cr-100 reported `stream_ms` inside `ms` and
        # nothing about the difference, so a 9734 ms restore whose fill leg was 716 ms had
        # 9018 ms nobody could attribute — the same ambiguity `prepare` carried until its
        # legs were named. Four legs PARTITION `ms`: reserving the device bytes and walking
        # the re-placed module (`reserve_ms`), taking the read lease (`lease_ms`),
        # validating each destination against its stored row (`enqueue_ms`), and the
        # transaction that moves them (`commit_ms`). `ring_ms`/`copy_ms`/`fence_ms` split
        # that last one again; `stream_ms` is unchanged and still the tfs-018-comparable
        # number.
        row: StageRow = {
            "component": name,
            "action": "stage",
            "bytes": memory.settled,
            "ms": took,
            "stream_ms": self.stats.get("stream_ms", 0),
            "reserve_ms": (reserved - started) // 1_000_000,
            "lease_ms": (leased - reserved) // 1_000_000,
            "enqueue_ms": (enqueued - leased) // 1_000_000,
            "commit_ms": (ended - enqueued) // 1_000_000,
            # The copy leg alone, and the two counts a per-object cost would scale with.
            # `objects` is the STORED objects the leased snapshot holds; `destinations` is
            # the tensors this component put back. Reported rather than divided here: the
            # ratio is the reader's question and this plane does not decide on it.
            "copy_ms": self.stats.get("copy_ms", 0),
            "ring_ms": self.stats.get("ring_ms", 0),
            "fence_ms": self.stats.get("fence_ms", 0),
            "objects": objects,
            "destinations": len(fresh.destinations),
        }
        self.stage_log.append(row)
        del self.stage_log[:-64]
        return row

    def _fill_page(self, name: str, unit: PageUnit) -> None:
        """Reserve only this unit, then use the ordinary validated TensorFS fill transaction."""
        if self.enqueued:
            raise _refuse("incomplete_fill", "another fill is already pending", [name])
        with placing():
            owners = {id(module) for _, module in unit.owners}
            for _, module in unit.owners:
                module.to_empty(device=self.device, recurse=False)
            _restore_derived(
                self.components.get(name),
                {
                    key: value
                    for key, value in self.derived.get(name, {}).items()
                    if id(key[0]) in owners
                },
            )
            live = unit.live(name)
        expected = {f"{name}.{key}" for key in unit.keys}
        if set(live) != expected or not expected <= self.expected.get(name, frozenset()):
            raise _refuse("incomplete_fill", "paging unit destinations changed", [name, unit.path])
        for key, tensor in live.items():
            spec = self.paging_specs[key]
            if tuple(tensor.shape) != spec.shape:
                raise _refuse("length_mismatch", f"{key}: paging destination shape changed", [key])
            self.fill(key, spec, tensor)
        if live:
            self._commit_enqueued()
        self.state = "ready"

    def _park_page(self, unit: PageUnit) -> None:
        with placing():
            for _, module in unit.owners:
                self._free_storages(
                    tensor
                    for _, tensor in (
                        *module.named_parameters(recurse=False),
                        *module.named_buffers(recurse=False),
                    )
                )
                module.to_empty(device=self.torch.device("meta"), recurse=False)

    def stage_page(self, name: str, unit: PageUnit) -> None:
        if name not in self.components or name in self.page_resident:
            raise _refuse(
                "leaf_schema", "paging requires one admitted component and no live block", [name]
            )
        self._fill_page(name, unit)
        self.page_resident[name] = unit
        self.vram_charge[name] += unit.nbytes

    def evict_page(self, name: str, unit: PageUnit) -> None:
        if self.page_resident.get(name) is not unit:
            raise _refuse("leaf_schema", "paging eviction does not name the live block", [name])
        self._park_page(unit)
        self.page_resident.pop(name)
        self.vram_charge[name] -= unit.nbytes

    def poison(self, keys: Sequence[str]) -> None:
        """Abort the new generation: mark it poisoned and free NOTHING (#613).

        Verified CAS is untouched by construction (this plane never writes to the store).
        The device bytes a failed fill holds are reclaimed by the one mechanism that cannot
        miss a reference — the executor process dies and the worker replaces it (cr-024).
        An in-process rollback used to free storages, prove its own return against the
        allocator and report `rollback_incomplete` when it could not; every one of those
        mechanisms was a second authority over byte liveness, and #613 deleted them.

        What survives is EVIDENCE: the allocator's live bytes at the instant of failure,
        how many destinations were in flight, and the read lease's end — a hold kept for a
        generation that no longer exists would be the store's reaper's problem.
        """
        self.hold_lease = False
        self.release_lease()
        self.poison_report = {
            "keys_in_flight": len(keys),
            "enqueued": len(self.enqueued),
            "resident": sorted(self.components),
            "parked": sorted(self.parked),
            "allocator_bytes": accel.allocated(self.torch, self.kind, self.device),
        }
        self._mark("poison", **self.poison_report)
        if self.forensics is not None:
            self.forensics.dump("poison")
        self.state = "poisoned"

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

    # -- the decode leg (cr-006) --------------------------------------------------

    def _decode(self, pending: list[PlanRow]) -> int:
        """Turn every landed role set into its logical tensor, then free the scratch.

        Called only behind `ring.drain(0)`, so every H2D for these roles has completed.
        The provider writes into the destination the census reserved — there is no moment
        where a decoded tensor exists outside a priced component.

        Returns the scratch bytes released, so the caller's rolling window stays exact.
        """
        if not pending:
            return 0
        started = time.perf_counter_ns()
        freed = 0
        spent: list[Any] = []
        #: Freed behind the same fence and NOT counted against the rolling scratch window:
        #: a replaced float destination and a swizzled-away scale grid are not decode
        #: scratch, and subtracting them from a scratch budget would let the next component
        #: hold more live scratch than the envelope priced.
        retired: list[Any] = []
        for row in pending:
            provider = self.selected[row.key].provider
            parts = self.scratch.pop(row.key, {})
            missing = sorted(provider.required - set(parts))
            if missing:
                raise _refuse(
                    "incomplete_fill",
                    f"{row.key}: role(s) {missing} never arrived, so the {provider.route}"
                    " provider has no complete role set — a partially delivered encoded tensor "
                    "is rolled back, never served from what happened to land",
                    missing[:6],
                )
            if isinstance(provider, LeafProvider):
                # THE ROLES DO NOT DECODE, THEY STAY. `spent` is deliberately not extended:
                # these buffers are the resident weight from here on, and freeing them is
                # what the whole rung exists not to do.
                retired.append(self._install_leaf(row, provider, parts))
                self.decoded["leaves"] += 1
                self.decoded["resident_bytes"] += row.stored_nbytes
                continue
            provider.decode(self.torch, parts, self.enqueued[row.key])
            spent.append(parts)
            self.decoded["tensors"] += 1
            self.decoded["bytes"] += row.stored_nbytes
        # THE DECODE'S OWN FENCE, before its scratch goes back to the allocator. `decode`
        # ENQUEUES kernels on this stream and returns; `resize_(0)` frees the block those
        # kernels are still reading, and the very next role's H2D on the ring's stream —
        # which is where the scratch was allocated, so it is that pool the block returns to
        # — writes over it. An e4m3 byte of 0xff IS NaN, so the damage arrives as a handful
        # of NaN elements rather than as a fault: measured on the real fp8 UNet at 2 of 6
        # stage-ins, 9 to 13 tensors of 1,680 with 9 to 16 NaN elements each, and never on
        # a whole-generation fill (a fresh allocator does not hand the block straight back).
        # A package without an output-integrity floor SERVES that (se-008).
        self.torch.cuda.current_stream().synchronize()
        for parts in spent:
            freed += self._free_storages(parts.values())
        for parts in retired:
            self._free_storages(parts.values())
        pending.clear()
        self.decoded["flushes"] += 1
        self.decoded["ms"] += (time.perf_counter_ns() - started) // 1_000_000
        return freed

    def _install_leaf(
        self, row: PlanRow, provider: LeafProvider, parts: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Replace one module with the leaf that holds its encoded weight.

        Returns what is now RETIRED — the float destination whose storage the leaf made
        redundant, plus any role buffer the leaf transformed rather than kept (the mxfp8
        leaf swizzles its scale grid once at fill and holds the swizzled one). The caller
        frees them behind the decode's own fence, which is exactly as load-bearing here as
        it is on the decode path: `to_blocked` ENQUEUES kernels that read the grid, and
        resizing its storage to zero before they run hands the next H2D that block.

        The float destination is freed rather than dropped, for cr-005's finding-3 reason:
        a dropped reference frees nothing while the exception's frames, the census's `live`
        map, or `enqueued` still hold one — and here `enqueued` certainly does.
        """
        destination = self.enqueued[row.key]
        owner, replaced, attribute = self._leaf_site(row.key, row, destination)
        leaf = provider.leaf(self.torch, parts, replaced, destination.dtype)
        # Hooks are authored computation. Share their registries so existing handles
        # still remove them after replacement and after a component is staged again.
        for name in (
            "_forward_pre_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_forward_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_hooks_always_called",
        ):
            setattr(leaf, name, getattr(replaced, name))
        # `nn.Module.__setattr__` is the whole reason this is machinery: it accepts the leaf
        # here precisely because the leaf is a Module, and would have refused the payload.
        setattr(owner, attribute, leaf)
        # BY ADDRESS, never by `id()`. `untyped_storage()` builds a fresh Python wrapper on
        # every call, so the wrappers in a set comprehension are garbage by the time the
        # next comprehension runs — and CPython reuses the addresses `id()` returns. A role
        # the leaf kept could therefore test as retired and have its storage freed under a
        # resident weight, which is the worst possible spelling of this loop.
        kept = {_data_ptr(t) for t in leaf.roles().values()}
        retired: dict[str, Any] = {
            role: tensor for role, tensor in parts.items() if _data_ptr(tensor) not in kept
        }
        retired["#destination"] = destination
        module_path = row.key[: -len(".weight")]
        self.replaced[row.key] = LeafReplacement(
            contract_key=row.key,
            module_path=module_path,
            was=type(replaced).__name__,
            now=type(leaf).__name__,
            original=replaced,
            owner=owner,
            attribute=attribute,
            role_keys=tuple(sorted(f"{module_path}.{role}" for role in leaf.roles())),
            stored_bytes=row.stored_nbytes,
            freed_bytes=int(destination.numel() * destination.element_size()),
        )
        with placing():
            replaced.to_empty(device=self.torch.device("meta"), recurse=True)
        return retired

    def _drop_scratch(self) -> int:
        """Give back every scratch buffer a failed stream left behind. Positive act."""
        freed = 0
        for parts in self.scratch.values():
            freed += self._free_storages(parts.values())
        self.scratch.clear()
        return freed

    # -- the unload/release path -------------------------------------------------

    def release(self, component: str) -> int:
        """Component release RETURNS VRAM. Release makes bytes evictable and this plane's
        baseline backend frees them outright (§3.2: device eviction FREES, never copies to
        host). Returns the allocator's own delta, never arithmetic."""
        member = self.components.pop(component, None)
        if member is None:
            raise _refuse(
                "destination_absent",
                f"{component!r} is not a component this generation holds",
                [component],
            )
        before = accel.allocated(self.torch, self.kind, self.device)
        with placing():
            self._free_storages(t for _, t in _tensors(member))
            member.to_empty(device=self.torch.device("meta"), recurse=True)
        for key in [k for k in self.enqueued if k.split(".")[0] == component]:
            del self.enqueued[key]
        accel.release_cached(self.torch, self.kind)
        self.vram_charge.pop(component, None)
        return before - accel.allocated(self.torch, self.kind, self.device)

    # -- the stream ---------------------------------------------------------------

    def _stream(self, ring: StagingRing) -> int:
        """The store streams; this plane splices and fences.

        cr-005's rule survives the reader swap and is split across the boundary exactly where
        it belongs: ORDER IS THE CONSTRUCTION'S — `Loader._fill` enqueues the exact census
        traversal — and BYTE LOCATION IS THE STORE'S. TensorFS validates and plans that
        DestinationSet traversal, and every batch item carries the destination it belongs to.

        Queue depth is a MEASURED requirement, not a preference: tfs-018 found 3.5x between
        depth 1 and depth 4 on this store, tfs-005 confirmed it, and tfs-007 confirmed it a
        third time from a third language. What tfs-007 added is that COPIES IN FLIGHT share
        the ring with readers, so the caller sizes both or reads at the depth the leftovers
        allow.

        cr-006 widened the destination of one hop and nothing else. A VERBATIM role still
        lands directly in the module's parameter; an ENCODED tensor's roles land in
        transient device scratch, and the provider turns them into the same parameter once
        the copies have fenced. The scratch is allocated ON FIRST TOUCH and freed as soon
        as its tensor decodes, so the decode envelope is a bounded rolling window rather
        than a second whole copy of the component.
        """
        # `dict` preserves insertion order. `Loader._fill` calls `fill` in the construction
        # census order, and component restage does the same from its fresh census. Reading the
        # lookup table here instead silently changed H3's order to 0, 1, 10, 11, 2 under the
        # canonical JSON header.
        rows = [self.plan[key] for key in self.enqueued]
        if not rows:
            return 0
        by_component: dict[str, list[PlanRow]] = {}
        for row in rows:
            by_component.setdefault(row.component, []).append(row)

        batches = [0]
        cursor = [0]
        #: bytes already moved by EARLIER components. The liveness position must stay
        #: monotone across the whole fill, and each component's plan restarts at zero.
        moved = [0]
        torch = self.torch
        flat: dict[str, Any] = {}
        base: dict[str, int] = {}
        owner: dict[str, PlanRow] = {}
        remaining: dict[str, int] = {}
        pending: list[PlanRow] = []
        held = [0]
        #: which component's plan the callbacks are servicing right now — a forensics row
        #: with no component names a boundary nobody can place in the fill.
        current = [""]

        def slot_for(what: str) -> Any:
            """The uint8 view one role's bytes land in, allocated on first touch.

            For a verbatim role that is the destination itself — the cr-005 path, byte for
            byte. For an encoded role it is a fresh device buffer of exactly the role's
            declared dtype and shape, so `view(uint8)` is aligned by construction rather
            than by an offset that happened to come out even.
            """
            view = flat.get(what)
            if view is not None:
                return view
            row = owner[what]
            role = what.rsplit("#", 1)[1]
            if self.selected[row.key].provider.route == "verbatim":
                view = self.enqueued[row.key].view(-1).view(torch.uint8)
            else:
                part = row.encoded.part(role)
                buffer = torch.empty(
                    part.numel,
                    dtype=getattr(torch, TORCH_DTYPES[part.dtype]),
                    device=self.device,
                ).reshape(part.shape)
                self.scratch.setdefault(row.key, {})[role] = buffer
                if self.selected[row.key].provider.route == "decoded_float":
                    # ONLY a decode route's roles are scratch. A native role's buffer is the
                    # weight, so counting it toward the rolling flush budget would trigger
                    # flushes for bytes that are never coming back.
                    held[0] += part.nbytes
                view = buffer.view(-1).view(torch.uint8)
            flat[what] = view
            return view

        def on_batch(batch: Batch) -> None:
            """Splice one filled slot across the destinations it covers.

            A batch item carries `off` — where its bytes sit IN THE SLOT — and the store
            delivers batches in plan order, so the cursor is what turns a slot offset back
            into a position in the logical stream. `base` then turns that into an offset
            inside one destination tensor. Every arithmetic step here comes from a number
            the store handed over; none of it is re-derived from a naming rule.
            """
            start = cursor[0]
            done: list[PlanRow] = []
            with torch.cuda.stream(ring.stream):
                for item in batch.items():
                    what = str(item["what"])
                    source = int(item["off"])
                    length = int(item["len"])
                    target = start + source - base[what]
                    slot_for(what)[target : target + length].copy_(
                        ring.buffers[batch.slot][source : source + length], non_blocking=True
                    )
                    remaining[what] -= length
                    if remaining[what] == 0:
                        row = owner[what]
                        if self.selected[row.key].provider.route != "verbatim" and all(
                            remaining[row.role_what(p.role)] == 0 for p in row.encoded.parts
                        ):
                            done.append(row)
                event = torch.cuda.Event()
                event.record(ring.stream)
            cursor[0] = start + int(batch.nbytes)
            ring.hold(event, batch)
            batches[0] += 1
            if self.on_position is not None:
                self.on_position((moved[0] + cursor[0]) >> 20)
            # A slot whose H2D has not completed is NOT free. Query first, block last.
            ring.drain(ring.keep)
            pending.extend(done)
            if held[0] >= self.decode_scratch_bytes:
                # The rolling flush. Decoding needs the copies LANDED, so this fences the
                # whole ring — which is why it is triggered by a byte budget and not per
                # tensor: 743 fences would cost more than the scratch it saves.
                ring.drain(0)
                held[0] -= self._decode(pending)
                self._mark(
                    "decode_flush",
                    component=current[0],
                    scratch_held=held[0],
                    batches=batches[0],
                )

        started = time.perf_counter_ns()
        stats: dict[str, object] = {}
        for component, group in by_component.items():
            traversal = [(row.component, row.name) for row in group]
            if component in self.paging:
                complete = self.paging_plans.get(component)
                if complete is None:
                    complete_traversal = [
                        (component, self.plan[key].name)
                        for key in self.paging_specs
                        if key in self.plan and self.plan[key].component == component
                    ]
                    complete = self.checkpoints[component].read_plan(
                        complete_traversal, self.window_bytes, (component,)
                    )
                    self.paging_plans[component] = complete
                # Native selection keeps its windows and order after the complete
                # constructor/header traversal has passed validation.
                plan = complete.select(
                    [row.role_what(part.role) for row in group for part in row.encoded.parts]
                )
            else:
                plan = self.checkpoints[component].read_plan(
                    traversal, self.window_bytes, (component,)
                )
            order = _require_construction_order(plan.order, component)
            traversal_digest = _traversal_digest([row.key for row in group])
            history = self.stats.setdefault("traversals", [])
            history.append(
                {
                    "component": component,
                    "order": order,
                    "tensors": len(group),
                    "digest": traversal_digest,
                    "first": group[0].key,
                    "last": group[-1].key,
                }
            )
            del history[:-64]
            expect = sum(row.stored_nbytes for row in group)
            if int(plan.bytes) != expect:
                raise _refuse(
                    "length_mismatch",
                    f"{component}: the store's read plan carries {plan.bytes} B for a "
                    f"traversal this contract prices at {expect} STORED B",
                )
            flat = {}
            owner = {r.role_what(p.role): r for r in group for p in r.encoded.parts}
            remaining = {r.role_what(p.role): p.nbytes for r in group for p in r.encoded.parts}
            base = {}
            for item in plan.items():
                base.setdefault(str(item["what"]), int(item["dest_off"]))
            missing = sorted(set(owner) - set(base))
            if missing:
                raise _refuse(
                    "incomplete_fill",
                    f"{len(missing)} stored role(s) of {component!r} are absent "
                    f"from the store's read plan ({', '.join(missing[:4])})",
                    missing[:6],
                )
            # The decode envelope and native payload of this component, for the forensics
            # row only. ADMISSION priced them before the first allocation (`price_envelope`:
            # `min(total, budget + largest)` for scratch, stored bytes for native leaves);
            # nothing is granted or released per component any more (#613).
            routes = {r.key: self.selected[r.key].provider.route for r in group}
            encoded = [r for r in group if routes[r.key] == "decoded_float"]
            envelope = 0
            if encoded:
                total = sum(r.stored_nbytes for r in encoded)
                largest = max(r.stored_nbytes for r in encoded)
                envelope = min(total, self.decode_scratch_bytes + largest)
            native_bytes = sum(r.stored_nbytes for r in group if routes[r.key] == "encoded_gemm")
            # The cursor is per COMPONENT because each component gets its own read plan,
            # and `dest_off` is an offset inside that plan's own logical stream.
            moved[0] += cursor[0]
            cursor[0] = 0
            current[0] = component
            self._mark(
                "component_stream_begin",
                component=component,
                order=order,
                traversal_digest=traversal_digest,
                rows=len(group),
                stored_bytes=expect,
                logical_bytes=sum(row.nbytes for row in group),
                decode_envelope=envelope,
                native_bytes=native_bytes,
                routes={
                    route: sum(1 for r in routes.values() if r == route)
                    for route in set(routes.values())
                },
            )
            try:
                with store_refusals(f"streaming {component}"):
                    one = self._acquire(component).stream(
                        plan, ring.views, on_batch, readers=self.readers
                    )
                # FENCE BETWEEN COMPONENTS. `stream` returns with up to `keep` slots still
                # DMA-live, and the next component's stream starts by taking the whole ring
                # as free — so the store's reader threads overwrite slots whose H2D has not
                # landed, and the corruption is SILENT. Measured before the fence: 12 of
                # `text_encoder_2`'s 517 tensors differed from the safetensors source in a
                # four-component fill, while the same component filled alone was
                # byte-perfect. A slot is the store's to hand out only when the ring owns
                # it, and the ring's ownership has to be whole again before a new stream
                # begins. It is ALSO the decode's fence: a provider reading scratch whose
                # H2D has not completed reads whatever the allocator last left there.
                ring.drain(0)
                held[0] -= self._decode(pending)
            finally:
                dropped = self._drop_scratch()
                self._mark(
                    "component_stream_end",
                    component=component,
                    scratch_dropped=dropped,
                    scratch_held=held[0],
                )
            for key, value in one.items():
                prior = stats.get(key, 0)
                stats[key] = (
                    prior + value if isinstance(value, int) and isinstance(prior, int) else value
                )
        self.stats["read_ns"] = time.perf_counter_ns() - started
        self.stats["store"] = stats
        return batches[0]


# ------------------------------------------------------------------------- helpers


#: TensorFS dtype name -> torch dtype name, imported from the FORMAT seam rather than kept
#: here (#549.10): `probe.py` needs the same map to build role tensors out of the vendored
#: spec vectors, and two copies of one dtype table is the #497-class duplication.
#: `Checkpoint.rows` still CHECKS it against `tensorfs.DTYPES` in both directions, and every
#: consumer imports it from `cozy_runtime.internal.encoding` now — one definition, one door.


def dtype_name(dtype: object) -> str:
    """The one dtype spelling. The store writes TensorFS names (`f16`), torch prints its
    own (`torch.float16`), and a fill that guesses which is which is a silent wrong copy."""
    name = str(getattr(dtype, "name", dtype)).removeprefix("torch.")
    return TORCH_DTYPES.get(name, name)


def tensorfs_requirement_dtype(dtype: object | None) -> str | None:
    """Constraint spelling for TensorFS fit; ordinary bf16/f16 remain convertible.

    `canonical_dtype` also covers a census derived by an OLDER runtime inside the package
    environment, whose wire rows still spell torch names — the boundary canonicalizes."""

    if dtype is None:
        return None
    name = canonical_dtype(dtype)
    return None if name in ("bf16", "f16") else name


def _data_ptr(tensor: Any) -> int:
    """The tensor's own storage address, or 0 when it has none. The one comparison that
    survives `state_dict()`'s detach and still means "the same memory"."""
    try:
        return int(tensor.untyped_storage().data_ptr()) + int(tensor.storage_offset())
    except Exception:
        return 0


def _tensors(module: Any) -> Iterator[tuple[str, Any]]:
    yield from module.named_parameters(remove_duplicate=False)
    yield from module.named_buffers(remove_duplicate=False)


def _derived_values(root: Any) -> dict[tuple[Any, str], Any]:
    """The DERIVED tensors' real values, captured before the destinations are reserved.

    A non-persistent buffer is a pure function of config that no checkpoint supplies, and
    the serving substrate lets its construction EXECUTE (on the host, where it is
    kilobytes) precisely so there is a value to carry. `to_empty` moves shapes and drops
    contents, so the values are held here across it and copied onto the device after.

    Keyed by (module, local name) rather than by path: a path is re-derived and this is
    the object itself, so a module renamed between the two statements cannot lose a table.
    """
    held: dict[tuple[Any, str], Any] = {}
    for _, sub in root.named_modules():
        for local in getattr(sub, "_non_persistent_buffers_set", ()):
            value = sub._buffers.get(local)
            if value is not None and str(value.device) != "meta":
                held[(sub, local)] = value.detach().clone()
    return held


def _restore_derived(root: Any, held: Mapping[tuple[Any, str], Any]) -> int:
    """Materialize the derived tables ON THE TARGET DEVICE. Returns how many landed."""
    landed = 0
    for (sub, local), value in held.items():
        destination = sub._buffers.get(local)
        if destination is None:
            continue
        # Eviction frees the module's storage in place. Never alias the retained
        # reconstruction copy, even when it already lives on the target device.
        sub._buffers[local] = value.to(destination.device, non_blocking=False, copy=True)
        landed += 1
    return landed
