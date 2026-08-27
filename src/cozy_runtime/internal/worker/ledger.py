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

**Host grants are atomic over classes.** Host RAM and pinned memory are shared by every
executor on the worker, so the torch-free worker grants each attempt's host needs whole,
before anything moves. No incremental cross-executor acquisition exists, so no hold-and-wait
cycle can form, and `allocated` and `registered` stay two numbers (the v1 hostslab lesson).

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

from dataclasses import dataclass, field
from typing import Any

from cozy_runtime.internal import proctree

#: Every byte class this ledger carries, in report order. A class is added here only when a
#: real producer exists for it; a row nothing writes is not a ledger entry, it is decoration.
CLASSES = (
    "host_ram",
    "host_available",
    "pinned_allocated",
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
    "host_ram": (
        "arithmetic",
        "worker grant arithmetic over accepted attempt envelopes; no host allocation "
        "is observed, so this is bookkeeping and says so",
    ),
    "pinned_allocated": (
        "arithmetic",
        "the executor staging ring's own slot arithmetic - page-locked bytes IN FLIGHT",
    ),
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
        "in a bounded cgroup, else /proc/meminfo MemAvailable - re-read at every grant, "
        "because a host budget fixed at boot is a claim about a machine nobody else is on",
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
    """The LIVE host budget — §3.2's host-RAM rule, re-read at every grant, -1 UNREADABLE.

    The measurement lives behind the ProcessTree boundary (#447): the ledger's SEMANTICS
    are platform-neutral, the reader is not.
    """
    return proctree.available_host_bytes()


def read_vmrss(pid: int) -> int:
    """Current resident bytes for one pid, or -1 when the process is gone (UNREADABLE)."""
    return proctree.rss_bytes(pid)


class LedgerRefusal(Exception):
    """A class cannot be granted. Carries the structured shortfall the terminal reports."""

    def __init__(self, code: str, detail: str, shortfall: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.shortfall = shortfall


@dataclass(frozen=True, slots=True)
class Counter:
    """One byte class, its number, and the method that produced it."""

    name: str
    bytes: int
    kind: str
    method: str
    budget: int = -1
    peak: int = -1

    def document(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "class": self.name,
            "bytes": self.bytes,
            "kind": self.kind,
            "method": self.method,
        }
        if self.budget >= 0:
            row["budget"] = self.budget
        if self.peak >= 0:
            row["peak"] = self.peak
        return row


@dataclass(slots=True)
class Ledger:
    """The worker's one ledger. Worker-owned, torch-free, atomic over host classes."""

    host_budget: int
    pinned_budget: int
    #: charged host bytes per attempt key, so a release can never guess an amount
    grants: dict[str, dict[str, int]] = field(default_factory=dict)
    peak_host: int = 0

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
    #: set when an attempt staged or evicted: the activation basis moved under it
    activations_moved: bool = False
    #: attempts closed since the device baseline was last measured. Zero means the next
    #: close is the first on this generation, which is the only one allowed to shift it.
    closed_since_baseline: int = 0
    executor_pid: int = 0
    worker_pid: int = 0
    #: classes this worker could not read, by name. Never silently zero.
    unreadable: list[str] = field(default_factory=list)
    #: per-component RESIDENT device bytes, by component name. The staging plane's own
    #: arithmetic, and the input the chooser's rung prices are computed from.
    resident: dict[str, int] = field(default_factory=dict)
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
    #: every re-baselining and its size, so a moved line is visible rather than assumed
    baseline_notes: list[str] = field(default_factory=list)
    #: TRUE once the generation's first-call tax has been paid by something that is not a
    #: paying request — the boot warm pass. It closes the one hole in `close_attempt`'s
    #: first-attempt forgiveness: with a warm pass on, there IS no legitimate library-
    #: workspace shift left for a paying attempt to hide a leak behind, so none is allowed.
    #: Measured: a planted 64 MiB persistent allocation on the first paying attempt of a
    #: warmed generation was absorbed as a `baseline_shift` and named nothing.
    baseline_paid: bool = False
    reconciliations: list[dict[str, Any]] = field(default_factory=list)
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

    # ------------------------------------------------------------------ host grants

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

    @property
    def host_charged(self) -> int:
        return sum(row["host"] for row in self.grants.values())

    @property
    def pinned_charged(self) -> int:
        return sum(row["pinned"] for row in self.grants.values())

    #: A host move at or above this size is checked against LIVE availability before it is
    #: attempted. Below it the check costs more than the move (§3.2's ">= 1 GiB" rule).
    GUARD_FLOOR = 1 << 30
    #: What must stay for the rest of the machine. A move that would consume the floor is
    #: doomed: the OOM killer is the alternative, and it does not name its victim.
    HOST_FLOOR = 512 << 20

    def host_move_guard(self, nbytes: int, why: str) -> None:
        """Refuse a DOOMED host move BEFORE the OOM killer, not after (§3.2).

        The v1 shape this replaces detected growth AFTER the fact: RSS climbed, the kernel
        picked a victim, and the postmortem was a dmesg line naming a process nobody chose.
        This reads cgroup-derived availability at the moment of the grant — the same number
        `host_available` reports — and refuses with both numbers while there is still
        something to refuse with.
        """
        if nbytes < self.GUARD_FLOOR:
            return
        available = read_host_available()
        if available < 0:
            return  # UNREADABLE is not "plenty", but it is also not evidence to refuse on
        if nbytes > available - self.HOST_FLOOR:
            raise LedgerRefusal(
                "host_move_doomed",
                f"{why}: a {nbytes} B host move against {available} B available leaves "
                f"less than the {self.HOST_FLOOR} B floor. This move is doomed and the "
                "alternative is the OOM killer, which picks a victim nobody chose and "
                "explains itself in dmesg - so it refuses HERE, with both numbers",
                {
                    "resource": "host_ram",
                    "scope": "host_move",
                    "needed_bytes": nbytes + self.HOST_FLOOR,
                    "available_bytes": available,
                    "evidence_class": "measured",
                },
            )

    def grant(self, key: str, host: int, pinned: int, why: str) -> None:
        """All classes or none — the envelope is granted whole, before anything moves."""
        if key in self.grants:
            raise LedgerRefusal("double_grant", f"{key} already holds a host grant")
        self.host_move_guard(host, why)
        free_host = self.host_budget - self.host_charged
        free_pinned = self.pinned_budget - self.pinned_charged
        for name, need, free, budget in (
            ("host_ram", host, free_host, self.host_budget),
            ("pinned_allocated", pinned, free_pinned, self.pinned_budget),
        ):
            if need > free:
                raise LedgerRefusal(
                    "host_shortfall",
                    f"{why}: {name} needs {need} B, {free} B free of {budget} B "
                    f"(short by {need - free} B)",
                    {
                        "resource": name,
                        "scope": "attempt_envelope",
                        "needed_bytes": need,
                        "available_bytes": free,
                        "evidence_class": "measured",
                    },
                )
        self.grants[key] = {"host": host, "pinned": pinned}
        self.peak_host = max(self.peak_host, self.host_charged)

    def release(self, key: str) -> None:
        self.grants.pop(key, None)

    # ------------------------------------------------------------------ observation

    def begin_generation(self, executor_pid: int) -> None:
        """Discard every fact owned by the previous executor generation.

        Host grants and reconciliation history belong to the worker, so they survive an
        executor replacement. Everything below is either reported by one executor or
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
        self.activations_moved = False
        self.closed_since_baseline = 0
        self.executor_pid = executor_pid
        self.unreadable.clear()
        self.resident.clear()
        self.evicted.clear()
        self.parked = ()
        self.declared_scopes.clear()
        self.baseline_notes.clear()
        self.baseline_paid = False
        self.residency_delta = 0
        self.unreconciled = ""

    def observe_construction(self, facts: dict[str, Any]) -> None:
        """The generation's resident baseline, from the fill that just committed."""
        self.weights_bytes = int(facts.get("filled_bytes", 0))
        self.weights_destinations = int(facts.get("filled", 0))
        self.device_baseline = int(facts.get("allocator_bytes", -1))
        self.device_allocated = self.device_baseline
        self.device_reserved = int(facts.get("reserved_bytes", -1))
        self.resident = {k: int(v) for k, v in (facts.get("resident") or {}).items()}
        self.parked = tuple(str(name) for name in (facts.get("parked") or ()))
        self.accelerator = str(facts.get("gpu_name", ""))
        self.declared_scopes = {
            str(method): tuple(str(name) for name in names)
            for method, names in (facts.get("declared_scopes") or {}).items()
        }
        self.device_free = int(facts.get("device_free_bytes", -1))
        self.device_total = int(facts.get("device_total_bytes", -1))
        pinned = facts.get("pinned", {})
        self.registered = int(pinned.get("registered_bytes", 0))
        # The ring is closed by the time facts cross the seam: nothing is DMA-live.
        self.pinned_in_flight = 0
        # A generation is the ONLY thing that re-establishes this ledger. §3.2's rule is that
        # reservations are executor-generation-scoped and the ledger restarts WITH the new
        # generation, never reconstructed from device state. A successful construction
        # replaces the new generation's unknown facts; it never repairs the old numbers.

    def observe_attempt(
        self, metrics: dict[str, Any], resident_moved: bool = False, cell: str = ""
    ) -> None:
        """What one attempt's kernel allocated ON TOP of the resident weights.

        `resident_moved` makes the number UNREADABLE rather than wrong. The figure is
        `peak_allocated - the post-fill baseline`, and that baseline is only a line to
        measure against while the resident set STANDS STILL. Under a staged rung the
        attempt evicts and re-stages gigabytes underneath itself, so the same subtraction
        produced 2,048 B for a 1024px pipeline — a number that is not small, it is
        meaningless. A measurement whose basis moved is named absent (§3.6), never printed.
        """
        if resident_moved:
            self.activations_moved = True
            if "activations_scratch" not in self.unreadable:
                self.unreadable.append("activations_scratch")
            return
        peak = int(metrics.get("peak_vram_bytes", 0))
        if peak and self.device_baseline >= 0:
            measured = peak - self.device_baseline
            self.activations_peak = max(self.activations_peak, measured)
            if cell:
                self.activations_by_cell[cell] = max(
                    self.activations_by_cell.get(cell, 0), measured
                )

    def activations_for(self, cell: str) -> tuple[int, str]:
        """The activation envelope for ONE shape cell, and where the number came from.

        A cell with its own serve behind it is measured. A cell with none takes the largest
        cell measured on this generation — conservative, and NAMED as an upper bound over
        other shapes rather than passed off as this request's measurement.
        """
        exact = self.activations_by_cell.get(cell, 0)
        if exact:
            return exact, f"max_memory_allocated() minus the post-fill baseline, on cell {cell}"
        if self.activations_by_cell:
            widest = max(self.activations_by_cell, key=lambda k: self.activations_by_cell[k])
            return self.activations_by_cell[widest], (
                f"no serve has been banked on cell {cell}; the widest cell measured on this "
                f"generation ({widest}) is the upper bound this request is priced against"
            )
        return 0, ""

    def observe_residency(self, residency: dict[str, Any]) -> None:
        """The attempt's DECLARED residency transition, from the plane that performed it."""
        self.residency_delta = int(residency.get("allocator_delta_bytes", 0))

    def observe_probe(self, probe: dict[str, Any]) -> None:
        """The device's after-state, read through the executor that owns the context."""
        self.device_allocated = int(probe.get("allocator_bytes", -1))
        self.device_reserved = int(probe.get("reserved_bytes", -1))
        self.device_free = int(probe.get("device_free_bytes", -1))
        self.device_total = int(probe.get("device_total_bytes", -1))
        residency = probe.get("residency")
        if isinstance(residency, dict):
            self.resident = {k: int(v) for k, v in (residency.get("resident") or {}).items()}
            self.evicted = {k: int(v) for k, v in (residency.get("evicted") or {}).items()}

    def rebaseline(self, why: str) -> int:
        """Re-establish the resident baseline after real work that is NOT an attempt.

        The boot warm pass is the case this exists for. It runs one real case per warmable
        entrypoint through ordinary dispatch, so the library workspaces it causes to be
        allocated belong to the executor GENERATION, not to the first paying request —
        which is the whole point of paying the first-call tax on boot. Re-baselining here is
        what makes "the first paying request reconciles to zero" a true statement instead of
        an excused one. Returns the shift, which is recorded rather than swallowed.
        """
        if self.device_baseline < 0 or self.device_allocated < 0:
            return 0
        shift = self.device_allocated - self.device_baseline
        self.device_baseline = self.device_allocated
        self.closed_since_baseline = 0
        self.baseline_paid = True
        self.baseline_notes.append(f"{why}: {shift:+d} B")
        del self.baseline_notes[:-8]
        return shift

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
            "host_ram": (self.host_charged, self.host_budget, self.peak_host),
            "host_available": (read_host_available(), -1, -1),
            "pinned_allocated": (self.pinned_charged, self.pinned_budget, -1),
            "pinned_registered": (self.registered, self.pinned_budget, -1),
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

    def document(self) -> dict[str, Any]:
        return {
            "format": "cozy.runtime.Ledger/0",
            "classes": [counter.document() for counter in self.counters()],
            "unreadable": sorted(self.unreadable),
            "unreconciled": self.unreconciled,
            "grants": len(self.grants),
        }

    def rows(self) -> dict[str, Any]:
        """The compact projection the worker summary and the journal carry."""
        return {
            "host_ram": {
                "charged": self.host_charged,
                "budget": self.host_budget,
                "peak": self.peak_host,
            },
            "pinned": {
                "allocated": self.pinned_charged,
                "registered": self.registered,
                "budget": self.pinned_budget,
            },
            "device": {
                "active_weights": self.weights_bytes,
                "baseline": self.device_baseline,
                "allocated": self.device_allocated,
                "reserved": self.device_reserved,
                "out_of_allocator": self.out_of_allocator,
                "allocatable": self.allocatable,
                "activations_peak": self.activations_peak,
                "free": self.device_free,
            },
            "components": {"resident": dict(self.resident), "evicted": dict(self.evicted)},
            "unreconciled": self.unreconciled,
            "grants": len(self.grants),
        }

    # ------------------------------------------------------------------ reconciliation

    def snapshot(self) -> dict[str, int]:
        """Every class as one flat mapping — the before half of a reconciliation."""
        return {counter.name: counter.bytes for counter in self.counters()}

    def close_attempt(self, key: str, before: dict[str, int]) -> dict[str, Any]:
        """Reconcile EVERY class against the snapshot taken BEFORE the attempt was accepted.

        The claim, and the only one: an attempt that reached a terminal holds no byte it did
        not hold before it started — where "started" means before its envelope was granted,
        not after. `host_ram` and `pinned_allocated` return to their opening values because
        the grant was released; `device_allocated` returns to the resident baseline because
        the kernel's activations were freed. RSS is reported and NOT reconciled: an allocator
        legitimately retains freed pages, so a higher RSS is a fact to look at rather than a
        leak to claim.

        One measured exception, named rather than swept up. `device_allocated` can end the
        FIRST attempt of an executor generation ABOVE the fill's baseline, because the linear
        algebra libraries allocate their workspaces lazily on the first kernel launch and keep
        them for the life of the PROCESS, not the attempt. That is a baseline SHIFT, not a
        leak: the ledger records it with its size, re-baselines, and holds every later attempt
        to the new line — where a genuine per-attempt leak would keep growing and be caught on
        attempt two. With the boot warm pass on, the shift lands on the warm case and the
        first paying request already reconciles to zero.

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
        reconciled = ("host_ram", "pinned_allocated", "device_allocated")
        classes: list[dict[str, Any]] = []
        leaks: list[str] = []
        shifts: list[str] = []
        moved = self.residency_delta
        for counter in self.counters():
            name = counter.name
            row = counter.document()
            row["before"] = before.get(name, -1)
            row["after"] = after[name]
            row["reconciled"] = name in reconciled
            if name in reconciled and after[name] >= 0 and before.get(name, -1) >= 0:
                delta = after[name] - before[name]
                row["delta"] = delta
                row["verdict"] = "reconciled"
                declared = moved if name == "device_allocated" else 0
                residual = delta - declared
                if declared:
                    row["declared_transition"] = declared
                    row["residual"] = residual
                    row["verdict"] = "residency_transition"
                first = self.closed_since_baseline == 0 and not self.baseline_paid
                if residual > 0 and name == "device_allocated" and first:
                    row["verdict"] = "baseline_shift"
                    shifts.append(
                        f"device_allocated: +{residual} B on the first attempt of this "
                        "executor generation - lazily allocated library workspace, held by "
                        "the process rather than the attempt; the baseline moves and every "
                        "later attempt is held to the new line"
                    )
                    self.device_baseline += residual
                elif residual != 0:
                    row["verdict"] = "leak"
                    leaks.append(
                        f"{name}: {residual:+d} B unaccounted across the attempt "
                        f"({before[name]} -> {after[name]}, of which {declared:+d} B was a "
                        f"declared residency transition), measured by {counter.kind}"
                    )
            classes.append(row)
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
        record = {
            "attempt": key,
            "classes": classes,
            "leaks": leaks,
            "baseline_shifts": shifts,
            "declared_residency_delta": moved,
            "unreadable": sorted(self.unreadable),
            "closed": not leaks,
        }
        self.reconciliations.append(record)
        del self.reconciliations[:-32]
        # The transition belonged to THIS attempt. Carrying it into the next one would
        # forgive that one's device delta by exactly the size of a move it never made.
        self.residency_delta = 0
        return record
