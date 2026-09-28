"""Ledger v0 — the WORKER-scoped unified byte ledger, and the attempt-close reconciliation.

§3.2 asks for ONE unified resource ledger where every byte counter declares HOW it was
measured. This is its first real body, deliberately v0: the classes one all-resident BF16
model on one device actually moves bytes through, and no more. The full class set (encoded
weights, decode scratch, adapter memory, transfers), the pools and the demand pipeline are
cr-008b's, and inventing them here would be rows with no producer.

Three things make it a ledger rather than a counter bag:

**Every counter names its method and its kind.** `arithmetic` is the worker's own
bookkeeping — a number it computed, not one it observed. `measured` came from a device or
kernel interface, named exactly. `unreadable` is the honest third state: a class this
worker could not read right now, kept as a NAMED absence instead of a zero (§3.6's
present/absent/unreadable rule — a killed executor's device numbers are unreadable, not 0).

**An attempt closes by RECONCILING, not by dropping a reference.** `open_attempt` snapshots
every class; `close_attempt` re-reads them and refuses to call the difference zero unless it
is. A leak is a typed fact on the terminal path, on success and on failure alike — cr-005's
finding that an interrupted fill released 0 bytes until rollback freed storages explicitly is
exactly the class of defect this catches, and it caught it only because someone measured.

RSS is measured by the WORKER, out of `/proc/<pid>/status`, for both processes. That is
deliberate: the executor's own report of its footprint is a claim the worker cannot
check, and here it does not have to — the parent can read the child's kernel-owned number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import msgspec

from cozy_runtime.internal import proctree
from cozy_runtime.internal.canonical import Json

#: Every byte class this ledger carries, in report order. A class is added here only when a
#: real producer exists for it; a row nothing writes is not a ledger entry, it is decoration.
CLASSES = (
    "host_available",
    "pinned_registered",
    "active_weights",
    "activations_scratch",
    "device_allocated",
    "device_reserved",
    "device_out_of_allocator",
    "device_allocatable",
    "device_free",
    "rss_worker",
    "rss_executor",
)

#: How each class's number is produced. This table IS the "declares how it was measured"
#: requirement — one sentence per counter, naming the interface, never the intent.
METHODS: dict[str, tuple[str, str]] = {
    "pinned_registered": (
        "measured",
        "the staging ring's page-locked PEAK, the bytes actually handed to the driver; "
        "kernel-unreclaimable while registered",
    ),
    "active_weights": (
        "measured",
        "the fill plane's committed destination bytes at the generation fence "
        "(cr-005 record.filled_bytes) - what the model's parameters occupy on device",
    ),
    "activations_scratch": (
        "measured",
        "torch.cuda.max_memory_allocated() across the attempt minus the post-fill "
        "allocator baseline - everything the kernel allocated that the weights did not. "
        "UNREADABLE when the attempt staged or evicted: the baseline is only a line to "
        "measure against while the resident set stands still",
    ),
    "device_allocated": (
        "measured",
        "torch.cuda.memory_allocated() in the executor - the caching allocator's live "
        "bytes, not the driver's reservation",
    ),
    "device_free": (
        "measured",
        "torch.cuda.mem_get_info() - the driver's own free/total, which is the only "
        "number that sees other processes",
    ),
    "rss_worker": (
        "measured",
        "/proc/<pid>/status VmRSS read BY THE WORKER for itself",
    ),
    "rss_executor": (
        "measured",
        "/proc/<pid>/status VmRSS read BY THE WORKER for its child - a parent reading "
        "the kernel's number, never the child's claim about itself",
    ),
    "host_available": (
        "measured",
        "the LIVE host budget: cgroup memory.max minus memory.current when this process is "
        "in a bounded cgroup, else /proc/meminfo MemAvailable - re-read every time it is "
        "reported, because a host budget fixed at boot is a claim about a machine nobody "
        "else is on",
    ),
    "device_reserved": (
        "measured",
        "torch.cuda.memory_reserved() in the executor - the caching allocator's SEGMENTS, "
        "which is what the driver sees; allocated is what the tensors see, and the gap "
        "between them is reusable without asking the driver",
    ),
    "device_out_of_allocator": (
        "measured",
        "NVML per-process device memory for the executor pid MINUS "
        "torch.cuda.memory_reserved() - the CUDA context and the cuDNN/cuBLAS workspaces, "
        "which max_memory_allocated() structurally cannot see (the SS3.2 out-of-allocator meter)",
    ),
    "device_allocatable": (
        "measured",
        "driver-free plus the allocator's reserved-but-unallocated segments - what a new "
        "reservation can ACTUALLY get, never arithmetic free bytes",
    ),
}


def read_host_available() -> int:
    """The LIVE host budget — §3.2's host-RAM rule, -1 UNREADABLE.

    The measurement lives behind the ProcessTree boundary (#447): the ledger's SEMANTICS
    are platform-neutral, the reader is not.
    """
    return proctree.available_host_bytes()


def read_vmrss(pid: int) -> int:
    """Current resident bytes for one pid, or -1 when the process is gone (UNREADABLE)."""
    return proctree.rss_bytes(pid)


class Counter(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """One byte class, its number, and the method that produced it. -1 is unset."""

    name: str = msgspec.field(name="class")
    bytes: int
    kind: str
    method: str
    budget: int = -1
    peak: int = -1

    def document(self) -> dict[str, Json]:
        document: dict[str, Json] = msgspec.to_builtins(self)
        return document


class ReconciledClass(Counter, frozen=True, kw_only=True, omit_defaults=True):
    """One class across an attempt: its snapshot before and after, and the verdict."""

    before: int
    after: int
    reconciled: bool
    delta: int | msgspec.UnsetType = msgspec.UNSET
    verdict: str | msgspec.UnsetType = msgspec.UNSET
    declared_transition: int | msgspec.UnsetType = msgspec.UNSET
    residual: int | msgspec.UnsetType = msgspec.UNSET


class Reconciliation(msgspec.Struct, frozen=True, kw_only=True):
    """One attempt close: every class reconciled, and what did not reconcile."""

    attempt: str
    classes: tuple[ReconciledClass, ...]
    leaks: tuple[str, ...]
    baseline_shifts: tuple[str, ...]
    declared_residency_delta: int
    unreadable: tuple[str, ...]
    closed: bool


class Measurement(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """The ledger as a report: its classes and the conditions that qualify them."""

    classes: tuple[Counter, ...]
    unreadable: tuple[str, ...] = ()
    unreconciled: str = ""
    unsuitable_placements: dict[str, str] = {}
    activation_scopes_by_cell: dict[str, dict[str, int]] = {}


class _Pinned(msgspec.Struct, frozen=True):
    registered_bytes: int = 0


class ConstructionFacts(msgspec.Struct, frozen=True, kw_only=True):
    """The prepare reply's `facts` as the ledger reads them; the executor sends more."""

    filled_bytes: int = 0
    filled: int = 0
    allocator_bytes: int = -1
    reserved_bytes: int = -1
    device_free_bytes: int = -1
    device_total_bytes: int = -1
    resident: dict[str, int] = {}
    constructed: dict[str, int] | msgspec.UnsetType = msgspec.UNSET
    """Absent: the construction holds everything it built."""
    parked: tuple[str, ...] = ()
    evicted: dict[str, int] = {}
    gpu_name: str = ""
    declared_scopes: dict[str, tuple[str, ...]] = {}
    paging: dict[str, dict[str, int]] = {}
    pinned: _Pinned = msgspec.field(default_factory=_Pinned)


class AttemptMetrics(msgspec.Struct, frozen=True, kw_only=True):
    activation_peaks: dict[str, int] = {}
    activation_peak_bytes: int = 0
    peak_vram_bytes: int = 0


class _Transition(msgspec.Struct, frozen=True):
    allocator_delta_bytes: int = 0


class _Resident(msgspec.Struct, frozen=True):
    resident: dict[str, int] = {}
    evicted: dict[str, int] = {}


class Probe(msgspec.Struct, frozen=True, kw_only=True):
    """The executor's device after-state. -1 is unreadable; no `residency`, no residency map."""

    allocator_bytes: int = -1
    reserved_bytes: int = -1
    device_free_bytes: int = -1
    device_total_bytes: int = -1
    residency: _Resident | None = None


class _Freed(msgspec.Struct, frozen=True):
    freed_bytes: int = 0


class _Refilled(msgspec.Struct, frozen=True):
    restored_bytes: int = 0


class VacateReply(Probe, frozen=True, kw_only=True):
    vacated: _Freed = msgspec.field(default_factory=_Freed)


class RestoreReply(Probe, frozen=True, kw_only=True):
    restored: _Refilled = msgspec.field(default_factory=_Refilled)


@dataclass(frozen=True, slots=True)
class DeviceFacts:
    """ONE device of the lane, as the DRIVER reports it to the worker (cr-068).

    A group lane holds K of these under one ledger: every rank holds the full weights on
    its own card, so residency is per device while the generation is one. `free`/`total`
    are NVML's numbers read from outside every CUDA context; `process` is the executor's
    per-process bytes on THIS device, -1 when unreadable.
    """

    ordinal: int
    entry: str
    free: int
    total: int
    process: int = -1

    def document(self) -> dict[str, Json]:
        return {
            "ordinal": self.ordinal,
            "entry": self.entry,
            "free": self.free,
            "total": self.total,
            "process": self.process,
        }


@dataclass(slots=True)
class Ledger:
    """The worker's one ledger. Worker-owned, torch-free, atomic over host classes."""

    #: measured facts, refreshed from the executor's replies and from /proc
    registered: int = 0
    pinned_in_flight: int = 0
    weights_bytes: int = 0
    weights_destinations: int = 0
    #: the allocator's live bytes right after the fill committed — the resident BASELINE
    #: every attempt must return to. -1 until a generation is prepared.
    device_baseline: int = -1
    device_allocated: int = -1
    device_reserved: int = -1
    device_process: int = -1
    """NVML per-process device bytes for the executor. -1 = UNREADABLE, never 0."""
    device_free: int = -1
    device_total: int = -1
    activations_peak: int = 0
    #: the device this generation was BUILT on, named by the process that owns it
    accelerator: str = ""
    #: shape cell -> the activation peak measured on THAT cell. A generation-wide maximum
    #: prices a 512px request against whatever a 1024px one needed, which is a refusal or a
    #: worse rung charged to a request that never asked for it (codex audit, adopted
    #: 2026-08-25). `activations_for` reads this and falls back, LABELLED, when a cell has
    #: no serve behind it yet.
    activations_by_cell: dict[str, int] = field(default_factory=dict)
    #: shape cell -> component-use method -> measured scratch peak while that method's
    #: declared component set was resident. Staged admission prices matching pairs only.
    activation_scopes_by_cell: dict[str, dict[str, int]] = field(default_factory=dict)
    #: THE SAME TWO MAPS, ACROSS GENERATIONS (cr-097).
    #:
    #: §3.2 says reservations are executor-generation-scoped and the ledger restarts WITH
    #: the new generation, never reconstructed from DEVICE STATE. That invariant is not
    #: weakened here, and the distinction is the reason: §3.2 protects against claiming to
    #: know what a device holds after the process that filled it was replaced. An
    #: activation envelope is not device state. It is a property of a MODEL and a REQUEST
    #: SHAPE — what these kernels allocate at this shape — and it would be equally true on
    #: another card in another datacenter. Nothing about it describes the replaced
    #: process's residency, so carrying it forward asserts nothing §3.2 forbids.
    #:
    #: The shape cell is what keeps that honest: a measurement read for the wrong shape
    #: would be the same lie by another route, so these are keyed by cell and nothing else
    #: is inferred across cells that `activations_for` does not already label as an upper
    #: bound. The placement id and the lane hold the model and the card fixed for the life
    #: of the row (`lane.forget` drops it when the placement leaves), so a carried number
    #: can never be read for a different model.
    #:
    #: It has to. `plan.py` may not refuse on an UNCALIBRATED envelope, so the 1 GiB
    #: placeholder deepens the ladder and runs. When the real demand is 3.9 GiB and a
    #: co-tenant holds the slack that used to hide the gap, the attempt OOMs, poisons the
    #: executor, and the rebuild wiped the measurement that would have refused the retry —
    #: so the retry was uncalibrated again. Measured live: four identical 38.3 s failures,
    #: each with a 25.8 s rebuild, until the requeue budget ran out.
    carried_activations_by_cell: dict[str, int] = field(default_factory=dict)
    carried_activation_scopes_by_cell: dict[str, dict[str, int]] = field(default_factory=dict)
    #: cell -> THE DRIVER-FREE BYTES THE MEASUREMENT WAS TAKEN UNDER (cr-097, third cut).
    #: An activation peak is not interpretable without it. Kernels expand into the headroom
    #: they are given — cuBLAS/cuDNN workspace selection, allocator behaviour — so the same
    #: model at the same shape measured 3.91 GiB alone on a 15.6 GiB card and 1.92 GiB with
    #: a co-tenant beside it, reproducibly. A peak observed with the card empty is therefore
    #: an upper bound the kernels CHOSE, not a requirement they HAVE, and a bare scalar
    #: cannot say which it is. This is what lets a reader tell.
    carried_activation_free_by_cell: dict[str, int] = field(default_factory=dict)
    #: cell -> the free bytes under which an attempt for it ALREADY FAILED on the device
    #: (cr-097). A lower bound on demand, never an envelope; see `observe_device_shortfall`.
    carried_shortfall_by_cell: dict[str, int] = field(default_factory=dict)
    #: cell -> THE LEAST DRIVER-FREE BYTES THIS CELL HAS COMPLETED IN WITH NOTHING MOVING
    #: (cr-100). The exact counterpart of the row above, in the same units and comparable
    #: with the card the same way: one says "at this much room, this shape FAILED", this
    #: one says "at this much room, this shape FINISHED, with its whole resident set
    #: standing still". Two direct observations of the question a rung actually asks.
    #:
    #: It exists because an activation PEAK cannot answer that question. A peak is what the
    #: kernels took given the room they had, so a peak from a roomier card is an upper
    #: bound they chose (decisions 698) and a peak from a tighter one says nothing about a
    #: roomier one. A completed serve at F free bytes is not an inference at all — it is
    #: the outcome the plan is trying to predict, already observed, and it outranks the
    #: envelope wherever the two disagree.
    carried_resident_free_by_cell: dict[str, int] = field(default_factory=dict)
    #: set when an attempt staged or evicted: the activation basis moved under it
    activations_moved: bool = False
    #: attempts closed since the device baseline was last measured. Zero means the next
    #: close is the first on this generation, which is the only one allowed to shift it.
    closed_since_baseline: int = 0
    executor_pid: int = 0
    worker_pid: int = 0
    #: the residence ceiling the WORKER assigned this generation before its prepare (cr-025's
    #: `authorized_device_limit_bytes`; cr-066 assigns it from the lane's measured free
    #: bytes). -1 until assigned. It is a policy fact the executor is handed, never one it
    #: derives: the executor fits its prepare under the smaller of this and what it measures.
    authorized_device_limit_bytes: int = -1
    #: ordinal -> that device's driver-reported row (cr-068). One entry on a device lane;
    #: K on a group lane. `device_free`/`device_total` above are the MIN over these, which
    #: is what a plan priced once for every rank is priced against.
    devices: dict[int, DeviceFacts] = field(default_factory=dict)
    #: classes this worker could not read, by name. Never silently zero.
    unreadable: list[str] = field(default_factory=list)
    #: per-component RESIDENT device bytes, by component name. The staging plane's own
    #: arithmetic, and the input the chooser's rung prices are computed from.
    resident: dict[str, int] = field(default_factory=dict)
    #: WHAT THE CONSTRUCTION FILLED, kept whole for the life of the generation (cr-097).
    #: `resident` follows the device and a vacate empties it; this does not move, so the
    #: worker can tell "this generation never held it" from "the lane took it away" and
    #: ask for it back. A generation that legitimately parked a component never had it
    #: here, so a restore can never contradict the construction's own ceiling.
    constructed: dict[str, int] = field(default_factory=dict)
    #: components this generation knows about but is not holding right now, with their size
    evicted: dict[str, int] = field(default_factory=dict)
    #: components CONSTRUCTION parked, because the deployment's resident ceiling did not
    #: hold them (cr-008b). It is a fact about the generation that exists, not a rung the
    #: attempt ladder chose — so a plan may deepen BELOW it and may never claim to be above
    #: it. Found by cr-008c's cross: a generation with a parked UNet was reporting
    #: `all_resident` because the headroom arithmetic fit, which made the plan's placement
    #: disagree with the generation it was planning for.
    parked: tuple[str, ...] = ()
    #: method -> the component set that method DECLARES it may touch, reported by the
    #: construction that read the class. A scope holds its whole set resident at once, so
    #: this is what a staged rung must raise; pricing the largest single component instead
    #: under-charged every multi-component scope (codex audit, adopted 2026-08-25).
    declared_scopes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Runtime-derived block working sets; nominal checkpoint bytes remain separate.
    paging: dict[str, dict[str, int]] = field(default_factory=dict)
    #: every re-baselining and its size, so a moved line is visible rather than assumed
    baseline_notes: list[str] = field(default_factory=list)
    reconciliations: list[Reconciliation] = field(default_factory=list)
    #: THE ATTEMPT'S DECLARED RESIDENCY TRANSITION, signed, in the same allocator bytes
    #: `device_allocated` is measured in. A staged rung moves gigabytes on purpose; that
    #: movement is a fact the attempt DECLARES, so the reconciliation subtracts it and holds
    #: the remainder to zero. Reset at every open.
    residency_delta: int = 0
    #: WHY this ledger is not a truth right now, or empty. A non-empty reason is a LOUD
    #: condition: the chooser refuses to price a rung against it, because an unreconciled
    #: ledger is exactly a ledger whose residency numbers are known to be wrong, and pricing
    #: against wrong residency is how a 1024px decode came to be planned `all_resident` on a
    #: card holding one component (cl-003). It clears only when the ledger is re-established
    #: from a real construction — never by another attempt agreeing with itself.
    unreconciled: str = ""
    #: Placement rungs this executor generation falsified with measured capacity evidence.
    #: The fact dies with the generation; a different construction/device proves itself.
    unsuitable_placements: dict[str, str] = field(default_factory=dict)

    @property
    def out_of_allocator(self) -> int:
        """CUDA context + library workspaces: what the allocator's own numbers cannot see."""
        if self.device_process < 0 or self.device_reserved < 0:
            return -1
        return max(self.device_process - self.device_reserved, 0)

    @property
    def allocatable(self) -> int:
        """What a NEW reservation can actually get (§3.2: the allocator view, never
        arithmetic free bytes). Driver-free plus the allocator's own reserved-but-unused."""
        if self.device_free < 0:
            return -1
        slack = max(self.device_reserved - self.device_allocated, 0)
        if self.device_reserved < 0 or self.device_allocated < 0:
            slack = 0
        return self.device_free + slack

    # ------------------------------------------------------------------ observation

    def begin_generation(self, executor_pid: int) -> None:
        """Discard every fact owned by the previous executor generation.

        Reconciliation history belongs to the worker, so it survives an executor
        replacement. Everything below is either reported by one executor or
        derived from its resident set and must start unknown or empty. ``executor_pid == 0``
        represents the interval after retirement and before a successor is published.
        """
        self.registered = 0
        self.pinned_in_flight = 0
        self.weights_bytes = 0
        self.weights_destinations = 0
        self.device_baseline = -1
        self.device_allocated = -1
        self.device_reserved = -1
        self.device_process = -1
        self.device_free = -1
        self.device_total = -1
        self.activations_peak = 0
        self.accelerator = ""
        self.activations_by_cell.clear()
        self.activation_scopes_by_cell.clear()
        # `carried_activations_by_cell` / `carried_activation_scopes_by_cell` are NOT
        # cleared: they outlive the generation deliberately (cr-097, see their field docs).
        self.activations_moved = False
        self.closed_since_baseline = 0
        self.executor_pid = executor_pid
        self.authorized_device_limit_bytes = -1
        self.devices.clear()
        self.unreadable.clear()
        self.resident.clear()
        self.constructed.clear()
        self.evicted.clear()
        self.parked = ()
        self.declared_scopes.clear()
        self.paging.clear()
        self.baseline_notes.clear()
        self.residency_delta = 0
        self.unreconciled = ""
        self.unsuitable_placements.clear()

    def observe_construction(self, facts: Mapping[str, object]) -> None:
        """The generation's resident baseline, from the fill that just committed."""
        read = msgspec.convert(facts, ConstructionFacts)
        self.weights_bytes = read.filled_bytes
        self.weights_destinations = read.filled
        self.device_baseline = read.allocator_bytes
        self.device_allocated = self.device_baseline
        self.device_reserved = read.reserved_bytes
        self.resident = dict(read.resident)
        self.constructed = dict(
            read.resident if read.constructed is msgspec.UNSET else read.constructed
        )
        self.parked = read.parked
        # Parked components WITH their sizes, from the construction that parked them: what
        # the next attempt may have to stage in is priced from the first plan, not from
        # the first probe.
        self.evicted = dict(read.evicted)
        self.accelerator = read.gpu_name
        self.declared_scopes = dict(read.declared_scopes)
        self.paging = {name: dict(layout) for name, layout in read.paging.items()}
        self.device_free = read.device_free_bytes
        self.device_total = read.device_total_bytes
        self.registered = read.pinned.registered_bytes
        # The ring is closed by the time facts cross the seam: nothing is DMA-live.
        self.pinned_in_flight = 0
        # A generation is the ONLY thing that re-establishes this ledger. §3.2's rule is that
        # reservations are executor-generation-scoped and the ledger restarts WITH the new
        # generation, never reconstructed from device state. A successful construction
        # replaces the new generation's unknown facts; it never repairs the old numbers.

    def observe_attempt(
        self,
        metrics: Mapping[str, object],
        resident_moved: bool = False,
        cell: str = "",
        succeeded: bool = False,
    ) -> None:
        """What one attempt's kernel allocated ON TOP of the resident weights.

        A residency-aware executor supplies `activation_peak_bytes` from a fresh allocator
        high-water mark inside each component-use scope. That measurement survives staging.
        Without it, `peak_allocated - the post-fill baseline` is valid only while the resident
        set stands still; a moved basis is named absent rather than reported as a small number.
        """
        # A SERVE THAT FINISHED WITHOUT MOVING A BYTE IS A FACT ABOUT THE CARD, not about
        # the kernels' appetite (cr-100). It is banked before anything below, because it
        # does not depend on an activation peak being readable: whatever the peak was, the
        # whole resident set stood still and the work completed in the room that was left.
        if succeeded and cell and not resident_moved and self.device_free >= 0:
            previous = self.carried_resident_free_by_cell.get(cell)
            if previous is None or self.device_free < previous:
                self.carried_resident_free_by_cell[cell] = self.device_free
        read = msgspec.convert(metrics, AttemptMetrics)
        scopes = {method: value for method, value in read.activation_peaks.items() if value > 0}
        direct = max(scopes.values(), default=read.activation_peak_bytes)
        if direct > 0:
            measured = direct
        elif resident_moved:
            self.activations_moved = True
            if "activations_scratch" not in self.unreadable:
                self.unreadable.append("activations_scratch")
            return
        else:
            peak = read.peak_vram_bytes
            measured = peak - self.device_baseline if peak and self.device_baseline >= 0 else 0
        if measured > 0:
            self.activations_peak = max(self.activations_peak, measured)
            if cell:
                self.activations_by_cell[cell] = max(
                    self.activations_by_cell.get(cell, 0), measured
                )
                held = self.activation_scopes_by_cell.setdefault(cell, {})
                for method, value in scopes.items():
                    held[method] = max(held.get(method, 0), value)
                # ONLY A SUCCEEDING SERVE MAY CALIBRATE (cr-097). The generation-scoped
                # maps above still record a fault's peak, which is what an operator wants
                # to read; the CARRIED map may not, because a peak observed on an attempt
                # that died on its way up is a TRUNCATED LOWER BOUND, not a measurement.
                # Banking it would make the retry calibrated AND WRONG — priced too low,
                # `enforceable`, and therefore able to refuse nothing while running
                # straight back into the same OOM. That is strictly worse than
                # uncalibrated, whose may-not-refuse rule is at least honest about not
                # knowing. A failed attempt's fact goes to `observe_device_shortfall`,
                # which is a different slot holding a different KIND of number.
                if succeeded:
                    previous = self.carried_activations_by_cell.get(cell, 0)
                    self.carried_activations_by_cell[cell] = max(previous, measured)
                    if measured >= previous:
                        # The CONDITIONS travel with the number, always together: a peak
                        # whose free-bytes context was dropped is exactly the category
                        # error this issue is about.
                        self.carried_activation_free_by_cell[cell] = self.device_free
                    carried = self.carried_activation_scopes_by_cell.setdefault(cell, {})
                    for method, value in scopes.items():
                        carried[method] = max(carried.get(method, 0), value)

    def observe_device_shortfall(self, cell: str, free_bytes: int) -> None:
        """A failed attempt's REFUSAL-GRADE fact (cr-097): at `free_bytes` free on this
        lane, this shape cell did not fit. It is a LOWER BOUND on demand and never an
        envelope — the attempt died climbing, so its true peak is strictly greater than
        anything it was observed to reach. Kept apart from `carried_activations_by_cell`
        in the TYPE, not by convention, because the one thing that must never happen is a
        truncated peak being read as a measurement.
        """
        if not cell or free_bytes < 0:
            return
        self.carried_shortfall_by_cell[cell] = max(
            self.carried_shortfall_by_cell.get(cell, 0), int(free_bytes)
        )

    def carried_conditions_apply(self, cell: str) -> bool:
        """Whether a CARRIED measurement for `cell` may be enforced right now (cr-097).

        True when it was observed with no more room than there is now, so the number is a
        requirement the kernels actually had; false when it was observed on a roomier card,
        where it is an upper bound they chose. A basis of 0 means nothing was banked, and
        the uncalibrated path owns that case with its own may-not-refuse rule.
        """
        basis = self.activation_basis_free(cell)
        return basis == 0 or basis <= self.device_free

    def activation_basis_free(self, cell: str) -> int:
        """The driver-free bytes the carried activation measurement for `cell` was observed
        under, or 0 when there is none (cr-097).

        A caller compares it with the free bytes NOW. Measured under conditions at least as
        tight as the present ones, the number is a requirement and may refuse. Measured on a
        roomier card, it is an upper bound the kernels chose and may only DEEPEN the ladder
        — never refuse, and never evict a co-tenant. Getting that backwards would move the
        defect from "this tenant is slow" to "the OTHER tenant is thrown off the card for no
        reason", which is quieter and worse: the victim is not even the tenant being served.
        """
        exact = self.carried_activation_free_by_cell.get(cell, 0)
        if exact:
            return exact
        # A cross-cell read already labels itself an upper bound; its conditions are the
        # widest cell's, so the same comparison still applies.
        if self.carried_activation_free_by_cell:
            widest = max(
                self.carried_activations_by_cell,
                key=lambda k: self.carried_activations_by_cell[k],
                default="",
            )
            return self.carried_activation_free_by_cell.get(widest, 0)
        return 0

    def insufficient_for(self, cell: str, free_now: int) -> int:
        """The banked floor when this cell is KNOWN not to fit in `free_now`, else 0.

        The comparison is deliberately `<=`: a picture no better than the one that already
        failed cannot produce a different answer, which is the whole of the retry rule. A
        picture with MORE room is not answered here — it may or may not fit, and finding
        out is exactly what a genuinely different attempt is for.
        """
        floor = self.carried_shortfall_by_cell.get(cell, 0)
        if floor and free_now <= floor:
            return floor
        return 0

    def resident_sufficiency(self, cell: str) -> int:
        """The least free bytes this cell has COMPLETED in with nothing moving, else 0.

        The mirror of `insufficient_for` and read the same way: a caller compares it with
        the free bytes it has now, and a number no larger than what it has is a serve that
        already happened under conditions no roomier than these. It PERMITS; it never
        refuses, because the absence of an observation is not evidence against one.
        """
        return self.carried_resident_free_by_cell.get(cell, 0)

    def activations_for(self, cell: str) -> tuple[int, str]:
        """The activation envelope for ONE shape cell, and where the number came from.

        A cell with its own serve behind it is measured. A cell with none takes the largest
        cell measured on this generation — conservative, and NAMED as an upper bound over
        other shapes rather than passed off as this request's measurement.
        """
        exact = self.activations_by_cell.get(cell, 0)
        if exact:
            return exact, (
                f"scope-local max_memory_allocated() minus its resident component baseline, "
                f"on cell {cell}"
            )
        if self.activations_by_cell:
            widest = max(self.activations_by_cell, key=lambda k: self.activations_by_cell[k])
            return self.activations_by_cell[widest], (
                f"no serve has been banked on cell {cell}; the widest cell measured on this "
                f"generation ({widest}) is the upper bound this request is priced against"
            )
        # NOTHING ON THIS GENERATION — which is the normal state of a generation the last
        # attempt's OOM just rebuilt. A measurement carried from an earlier generation of
        # the same placement on the same lane is still a measurement (cr-097): it prices
        # the retry against what the model actually allocated instead of against the 1 GiB
        # placeholder that may not refuse.
        exact = self.carried_activations_by_cell.get(cell, 0)
        if exact:
            return exact, (
                f"scope-local max_memory_allocated() minus its resident component baseline, "
                f"on cell {cell}, measured on an earlier generation of this placement"
            )
        if self.carried_activations_by_cell:
            widest = max(
                self.carried_activations_by_cell,
                key=lambda k: self.carried_activations_by_cell[k],
            )
            return self.carried_activations_by_cell[widest], (
                f"no serve has been banked on cell {cell} in any generation; the widest "
                f"cell measured on an earlier generation of this placement ({widest}) is "
                f"the upper bound this request is priced against"
            )
        return 0, ""

    def activation_scopes_for(self, cell: str) -> dict[str, int]:
        # ONE PREDICATE, AT THE TOP, FOR EVERY SCOPE MEASUREMENT — carried or this
        # generation's own (cr-100). cr-097 gated only the carried maps, which left the
        # same over-estimate live inside the generation that measured it: a serve run
        # alone banks per-scope peaks, a co-tenant arrives, and `admit` then refuses
        # `device_shortfall` on `size + headroom` INSIDE the executor, past every upstream
        # check, against a number measured on a card that no longer exists. A scope
        # measurement that does not apply is ABSENT, whoever measured it.
        if not self.carried_conditions_apply(cell):
            return {}
        exact = self.activation_scopes_by_cell.get(cell)
        if exact:
            return dict(exact)
        if self.activation_scopes_by_cell:
            widest = max(
                self.activation_scopes_by_cell,
                key=lambda key: max(self.activation_scopes_by_cell[key].values(), default=0),
            )
            return dict(self.activation_scopes_by_cell[widest])
        # THE CARRIED SCOPES OBEY THE SAME CONDITIONS RULE AS THE CARRIED ENVELOPE
        # (cr-097). This is the consumer that reaches furthest: `plan.py` turns these into
        # `scope_headrooms`, the worker sends them to the executor on `open_attempt`, and
        # `admit` refuses `device_shortfall` on `size + headroom` INSIDE the executor —
        # past `Coefficient.enforceable`, which is where every other over-estimate is
        # caught. A per-scope peak measured on a roomier card would refuse an attempt there
        # that would have succeeded, and no gate upstream would see it. So a carried scope
        # measurement that does not apply is simply absent, and the caller falls back to
        # the same bound it uses when nothing was ever measured. The gate itself is at the
        # top of this function now (cr-100), where it covers the generation's own map too.
        exact = self.carried_activation_scopes_by_cell.get(cell)
        if exact:
            return dict(exact)
        if self.carried_activation_scopes_by_cell:
            widest = max(
                self.carried_activation_scopes_by_cell,
                key=lambda key: max(
                    self.carried_activation_scopes_by_cell[key].values(), default=0
                ),
            )
            return dict(self.carried_activation_scopes_by_cell[widest])
        return {}

    def measured_scopes(self, cell: str) -> frozenset[str]:
        """Methods whose scratch peak was measured at exactly `cell`, applies on this card
        now, and never ran out of room here: the scopes staged admission may keep other
        components beside. Another cell's peak says nothing about this one's demand."""
        if not cell or cell in self.carried_shortfall_by_cell:
            return frozenset()
        if not self.carried_conditions_apply(cell):
            return frozenset()
        exact = self.activation_scopes_by_cell.get(cell)
        return frozenset(exact or self.carried_activation_scopes_by_cell.get(cell) or ())

    def observe_residency(self, residency: Mapping[str, object]) -> None:
        """The attempt's DECLARED residency transition, from the plane that performed it."""
        self.residency_delta = msgspec.convert(residency, _Transition).allocator_delta_bytes

    def observe_probe(self, probe: Mapping[str, object]) -> None:
        """The device's after-state, read through the executor that owns the context."""
        self._observe(msgspec.convert(probe, Probe))

    def _observe(self, probe: Probe) -> None:
        self.device_allocated = probe.allocator_bytes
        self.device_reserved = probe.reserved_bytes
        self.device_free = probe.device_free_bytes
        self.device_total = probe.device_total_bytes
        if probe.residency is not None:
            self.resident = dict(probe.residency.resident)
            self.evicted = dict(probe.residency.evicted)

    def observe_vacate(self, reply: Mapping[str, object], why: str) -> int:
        """The lane took this generation's device bytes for another tenant (cr-022).

        The reply is the executor's probe after the eviction, so every device class is
        re-read from the process that owns the context, and the resident/evicted maps say
        what is left. The BASELINE MOVES to the new allocated line and the move is
        recorded: the next attempt on this generation re-stages on purpose and declares
        that transition, so it reconciles against where the vacate left the device, not
        against the construction's fill. Returns the bytes the executor reports freed.
        """
        read = msgspec.convert(reply, VacateReply)
        freed = read.vacated.freed_bytes
        self._rebaseline(read, f"vacated ({why})", f"{freed} B freed")
        return freed

    def observe_restore(self, reply: Mapping[str, object], why: str) -> int:
        """This generation took back bytes the lane had vacated (cr-097).

        The mirror of `observe_vacate` and it moves the baseline the same way, for the same
        reason: the next attempt reconciles against where the device IS, not against the
        construction's fill or against the hole the vacate left. Returns the bytes the
        executor reports restored.
        """
        read = msgspec.convert(reply, RestoreReply)
        gained = read.restored.restored_bytes
        self._rebaseline(read, f"restored ({why})", f"{gained} B refilled")
        return gained

    def _rebaseline(self, probe: Probe, what: str, moved: str) -> None:
        before = self.device_allocated
        self._observe(probe)
        if self.device_allocated >= 0:
            shift = self.device_allocated - before if before >= 0 else 0
            self.device_baseline = self.device_allocated
            self.baseline_notes.append(f"{what}: {shift:+d} B allocator, {moved}")
            del self.baseline_notes[:-8]

    def missing_resident(self) -> dict[str, int]:
        """What the construction filled and the device no longer holds (cr-097): the exact
        set a restore asks for. Empty for a generation that holds everything it built."""
        return {name: size for name, size in self.constructed.items() if name not in self.resident}

    def observe_device_free(self, free: int, total: int) -> None:
        """The driver's view of the whole device, read by the WORKER through NVML now.

        `device_free` is otherwise a number the executor reported at its last probe, and
        between probes other tenants of the lane move gigabytes. A chooser pricing a rung
        against that number would price against a device that no longer exists; the lane
        refreshes it before pricing (cr-022), with the same meter cr-066 assigns ceilings
        from. The allocator classes stay the executor's own.
        """
        self.device_free = int(free)
        self.device_total = int(total)

    def observe_devices(self, rows: Mapping[int, DeviceFacts]) -> None:
        """Every device of the lane, per ordinal, and the MIN as the generation's number.

        A group lane's plan is priced ONCE for all ranks (cr-018: the degrade ladder is
        refused, never adapted per rank), so the free bytes the chooser reads are the
        tightest card's. The per-device rows stay readable beside it.
        """
        self.devices = dict(rows)
        if rows:
            self.observe_device_free(
                min(row.free for row in rows.values()), min(row.total for row in rows.values())
            )

    def refuse_placement(self, placement: str, reason: str) -> None:
        """Bank one measured generation/rung capacity falsification."""

        self.unsuitable_placements[placement] = reason[:400]

    def placement_refusal(self, placement: str) -> str:
        return self.unsuitable_placements.get(placement, "")

    def device_unreadable(self, why: str) -> None:
        """The executor is gone: its device classes are UNREADABLE, never zero."""
        self.device_allocated = -1
        self.device_free = -1
        for name in ("activations_scratch", "device_allocated", "device_free"):
            if name not in self.unreadable:
                self.unreadable.append(name)
        if why and why not in self.unreadable:
            self.unreadable.append(why)

    # ------------------------------------------------------------------ the counters

    def counters(self) -> list[Counter]:
        values = {
            "host_available": (read_host_available(), -1, -1),
            "pinned_registered": (self.registered, -1, -1),
            "active_weights": (self.weights_bytes, -1, -1),
            "activations_scratch": (
                -1
                if (self.device_baseline < 0 or self.activations_moved)
                else self.activations_peak,
                -1,
                -1,
            ),
            "device_allocated": (self.device_allocated, self.device_total, -1),
            "device_reserved": (self.device_reserved, self.device_total, -1),
            "device_out_of_allocator": (self.out_of_allocator, -1, -1),
            "device_allocatable": (self.allocatable, self.device_total, -1),
            "device_free": (self.device_free, self.device_total, -1),
            "rss_worker": (read_vmrss(self.worker_pid), -1, -1),
            "rss_executor": (read_vmrss(self.executor_pid) if self.executor_pid else -1, -1, -1),
        }
        out = []
        for name in CLASSES:
            nbytes, budget, peak = values[name]
            kind, method = METHODS[name]
            out.append(
                Counter(
                    name=name,
                    bytes=nbytes,
                    kind="unreadable" if nbytes < 0 else kind,
                    method=method if nbytes >= 0 else f"{method} (UNREADABLE right now)",
                    budget=budget,
                    peak=peak,
                )
            )
        return out

    def measurement(self) -> Measurement:
        """Current typed ledger fields for records, triage, and local reporting."""

        return Measurement(
            classes=tuple(self.counters()),
            unreadable=tuple(sorted(self.unreadable)),
            unreconciled=self.unreconciled,
            unsuitable_placements=dict(self.unsuitable_placements),
            activation_scopes_by_cell={
                cell: dict(scopes)
                for cell, scopes in sorted(self.activation_scopes_by_cell.items())
            },
        )

    def rows(self) -> dict[str, Json]:
        """The compact projection carried by the worker summary and process records."""
        return {
            "pinned": {"registered": self.registered},
            "device": {
                "active_weights": self.weights_bytes,
                "authorized_limit": self.authorized_device_limit_bytes,
                "baseline": self.device_baseline,
                "allocated": self.device_allocated,
                "reserved": self.device_reserved,
                "out_of_allocator": self.out_of_allocator,
                "allocatable": self.allocatable,
                "activations_peak": self.activations_peak,
                "activation_scopes_by_cell": {
                    cell: dict(scopes)
                    for cell, scopes in sorted(self.activation_scopes_by_cell.items())
                },
                "free": self.device_free,
            },
            "devices": {
                str(ordinal): facts.document() for ordinal, facts in sorted(self.devices.items())
            },
            "components": {"resident": dict(self.resident), "evicted": dict(self.evicted)},
            "unreconciled": self.unreconciled,
            "unsuitable_placements": dict(self.unsuitable_placements),
        }

    # ------------------------------------------------------------------ reconciliation

    def snapshot(self) -> dict[str, int]:
        """Every class as one flat mapping — the before half of a reconciliation."""
        return {counter.name: counter.bytes for counter in self.counters()}

    def close_attempt(self, key: str, before: Mapping[str, int]) -> dict[str, Json]:
        """Reconcile EVERY class against the snapshot taken BEFORE the attempt was accepted.

        The claim, and the only one: an attempt that reached a terminal holds no byte it did
        not hold before it started. `device_allocated` returns to the resident baseline
        because the kernel's activations were freed. RSS is reported and NOT reconciled: an
        allocator
        legitimately retains freed pages, so a higher RSS is a fact to look at rather than a
        leak to claim.

        One measured exception, named rather than swept up. `device_allocated` can end the
        FIRST attempt of an executor generation ABOVE the fill's baseline, because the linear
        algebra libraries allocate their workspaces lazily on the first kernel launch and keep
        them for the life of the PROCESS, not the attempt. That is a baseline SHIFT, not a
        leak: the ledger records it with its size, re-baselines, and holds every later attempt
        to the new line — where a genuine per-attempt leak would keep growing and be caught on
        attempt two. An author's `Model.warm` that launches kernels pays it inside the
        prepare, before this baseline is read (cr-110).

        THE DECLARED RESIDENCY TRANSITION (cl-003's second-1024px defect). Under a staged
        rung the attempt evicts and stages components ON PURPOSE, so `device_allocated`
        legitimately ends somewhere else entirely — measured once at -6,770,424,832 B, which
        this reconciliation called a leak and then did nothing about, leaving the next
        attempt's chooser pricing rungs against a residency the ledger knew was stale. The
        claim is therefore not "the same bytes" but "the bytes the attempt SAID it would
        move, and not one more": `residency_delta` is subtracted, the remainder must be zero,
        and the baseline MOVES TO THE NEW TRUTH so the ledger keeps describing the device.
        A transition is a reconciliation, never a leak — and never a silence either.
        """
        after = self.snapshot()
        classes: list[ReconciledClass] = []
        leaks: list[str] = []
        shifts: list[str] = []
        moved = self.residency_delta
        for counter in self.counters():
            name = counter.name
            was = before.get(name, -1)
            fields = msgspec.structs.asdict(counter)
            # Only `device_allocated` is reconciled; the declared transition is its own.
            if name != "device_allocated" or after[name] < 0 or was < 0:
                classes.append(
                    ReconciledClass(
                        **fields,
                        before=was,
                        after=after[name],
                        reconciled=name == "device_allocated",
                    )
                )
                continue
            delta = after[name] - was
            residual = delta - moved
            verdict = "residency_transition" if moved else "reconciled"
            first = self.closed_since_baseline == 0
            if residual < 0:
                # Returned MORE than the attempt took: the device has more free memory
                # than the ledger priced, which can only make choices conservative. The
                # baseline moves down to the truth; nothing is refused.
                verdict = "baseline_shift"
                shifts.append(
                    f"device_allocated: {residual:+d} B released beyond the attempt's "
                    "own allocation; the baseline moves down to the measured device"
                )
                self.device_baseline += residual
            elif residual > 0 and first:
                verdict = "baseline_shift"
                shifts.append(
                    f"device_allocated: +{residual} B on the first attempt of this "
                    "executor generation - lazily allocated library workspace, held by "
                    "the process rather than the attempt; the baseline moves and every "
                    "later attempt is held to the new line"
                )
                self.device_baseline += residual
            elif residual > 0:
                # KEPT: an unexplained growth means the chooser would price rungs against
                # memory the device no longer has, which is an OOM mid-attempt (cl-003).
                verdict = "leak"
                leaks.append(
                    f"{name}: {residual:+d} B unaccounted across the attempt "
                    f"({was} -> {after[name]}, of which {moved:+d} B was a "
                    f"declared residency transition), measured by {counter.kind}"
                )
            classes.append(
                ReconciledClass(
                    **fields,
                    before=was,
                    after=after[name],
                    reconciled=True,
                    delta=delta,
                    verdict=verdict,
                    declared_transition=moved or msgspec.UNSET,
                    residual=residual if moved else msgspec.UNSET,
                )
            )
        if moved and not leaks:
            # THE LINE MOVES WITH THE DEVICE. Leaving the baseline where the construction put
            # it is what made every later attempt measure against a residency that no longer
            # existed. It is recorded with its size, like every other baseline move.
            self.device_baseline += moved
            self.baseline_notes.append(f"declared residency transition on {key}: {moved:+d} B")
            del self.baseline_notes[:-8]
        self.closed_since_baseline += 1
        if leaks:
            self.unreconciled = "; ".join(leaks)
        record = Reconciliation(
            attempt=key,
            classes=tuple(classes),
            leaks=tuple(leaks),
            baseline_shifts=tuple(shifts),
            declared_residency_delta=moved,
            unreadable=tuple(sorted(self.unreadable)),
            closed=not leaks,
        )
        self.reconciliations.append(record)
        del self.reconciliations[:-32]
        # The transition belonged to THIS attempt. Carrying it into the next one would
        # forgive that one's device delta by exactly the size of a move it never made.
        self.residency_delta = 0
        document: dict[str, Json] = msgspec.to_builtins(record)
        return document
