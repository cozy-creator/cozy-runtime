"""Ledger v0 — the WORKER-scoped unified byte ledger, and the attempt-close reconciliation.

§3.2 asks for ONE unified resource ledger where every byte counter declares HOW it was
measured. This is its first real body, deliberately v0: the classes one all-resident BF16
model on one device actually moves bytes through, and no more. The full class set (encoded
weights, decode scratch, adapter memory, transfers), the pools and the demand pipeline are
cr-008b's, and inventing them here would be rows with no producer.

Three things make it a ledger rather than a counter bag:

**Every counter names its method and its kind.** `arithmetic` is the supervisor's own
bookkeeping — a number it computed, not one it observed. `measured` came from a device or
kernel interface, named exactly. `unreadable` is the honest third state: a class this
worker could not read right now, kept as a NAMED absence instead of a zero (§3.6's
present/absent/unreadable rule — a killed executor's device numbers are unreadable, not 0).

**Host grants are atomic over classes.** Host RAM and pinned memory are shared by every
executor on the worker, so the torch-free supervisor grants each attempt's host needs whole,
before anything moves. No incremental cross-executor acquisition exists, so no hold-and-wait
cycle can form, and `allocated` and `registered` stay two numbers (the v1 hostslab lesson).

**An attempt closes by RECONCILING, not by dropping a reference.** `open_attempt` snapshots
every class; `close_attempt` re-reads them and refuses to call the difference zero unless it
is. A leak is a typed fact on the terminal path, on success and on failure alike — cr-005's
finding that an interrupted fill released 0 bytes until rollback freed storages explicitly is
exactly the class of defect this catches, and it caught it only because someone measured.

RSS is measured by the SUPERVISOR, out of `/proc/<pid>/status`, for both processes. That is
deliberate: the executor's own report of its footprint is a claim the supervisor cannot
check, and here it does not have to — the parent can read the child's kernel-owned number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Every byte class this ledger carries, in report order. A class is added here only when a
#: real producer exists for it; a row nothing writes is not a ledger entry, it is decoration.
CLASSES = (
    "host_ram",
    "pinned_allocated",
    "pinned_registered",
    "active_weights",
    "activations_scratch",
    "device_allocated",
    "device_free",
    "rss_supervisor",
    "rss_executor",
)

#: How each class's number is produced. This table IS the "declares how it was measured"
#: requirement — one sentence per counter, naming the interface, never the intent.
METHODS: dict[str, tuple[str, str]] = {
    "host_ram": (
        "arithmetic",
        "supervisor grant arithmetic over accepted attempt envelopes; no host allocation "
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
        "allocator baseline - everything the kernel allocated that the weights did not",
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
    "rss_supervisor": (
        "measured",
        "/proc/<pid>/status VmRSS read BY THE SUPERVISOR for itself",
    ),
    "rss_executor": (
        "measured",
        "/proc/<pid>/status VmRSS read BY THE SUPERVISOR for its child - a parent reading "
        "the kernel's number, never the child's claim about itself",
    ),
}


def read_vmrss(pid: int) -> int:
    """Current resident bytes for one pid, or -1 when the process is gone (UNREADABLE)."""
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return -1
    return -1


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
    """The worker's one ledger. Supervisor-owned, torch-free, atomic over host classes."""

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
    device_free: int = -1
    device_total: int = -1
    activations_peak: int = 0
    #: attempts closed since the device baseline was last measured. Zero means the next
    #: close is the first on this generation, which is the only one allowed to shift it.
    closed_since_baseline: int = 0
    executor_pid: int = 0
    supervisor_pid: int = 0
    #: classes this worker could not read, by name. Never silently zero.
    unreadable: list[str] = field(default_factory=list)
    #: every re-baselining and its size, so a moved line is visible rather than assumed
    baseline_notes: list[str] = field(default_factory=list)
    reconciliations: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ host grants

    @property
    def host_charged(self) -> int:
        return sum(row["host"] for row in self.grants.values())

    @property
    def pinned_charged(self) -> int:
        return sum(row["pinned"] for row in self.grants.values())

    def grant(self, key: str, host: int, pinned: int, why: str) -> None:
        """All classes or none — the envelope is granted whole, before anything moves."""
        if key in self.grants:
            raise LedgerRefusal("double_grant", f"{key} already holds a host grant")
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

    def observe_construction(self, facts: dict[str, Any]) -> None:
        """The generation's resident baseline, from the fill that just committed."""
        self.weights_bytes = int(facts.get("filled_bytes", 0))
        self.weights_destinations = int(facts.get("filled", 0))
        self.device_baseline = int(facts.get("allocator_bytes", -1))
        self.device_allocated = self.device_baseline
        self.device_free = int(facts.get("device_free_bytes", -1))
        self.device_total = int(facts.get("device_total_bytes", -1))
        rows = facts.get("ledger", {})
        self.registered = int(rows.get("pinned", {}).get("peak", 0))
        self.pinned_in_flight = int(rows.get("staging", {}).get("charged", 0))
        self.closed_since_baseline = 0
        self.unreadable = [c for c in self.unreadable if not c.startswith("device")]

    def observe_attempt(self, metrics: dict[str, Any]) -> None:
        """What one attempt's kernel allocated ON TOP of the resident weights."""
        peak = int(metrics.get("peak_vram_bytes", 0))
        if peak and self.device_baseline >= 0:
            self.activations_peak = max(self.activations_peak, peak - self.device_baseline)

    def observe_probe(self, probe: dict[str, Any]) -> None:
        """The device's after-state, read through the executor that owns the context."""
        self.device_allocated = int(probe.get("allocator_bytes", -1))
        self.device_free = int(probe.get("device_free_bytes", -1))
        self.device_total = int(probe.get("device_total_bytes", -1))
        rows = probe.get("ledger", {})
        if rows:
            self.pinned_in_flight = int(rows.get("staging", {}).get("charged", 0))

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
            "pinned_allocated": (self.pinned_charged, self.pinned_budget, -1),
            "pinned_registered": (self.registered, self.pinned_budget, -1),
            "active_weights": (self.weights_bytes, -1, -1),
            "activations_scratch": (
                -1 if self.device_baseline < 0 else self.activations_peak,
                -1,
                -1,
            ),
            "device_allocated": (self.device_allocated, self.device_total, -1),
            "device_free": (self.device_free, self.device_total, -1),
            "rss_supervisor": (read_vmrss(self.supervisor_pid), -1, -1),
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
            "grants": len(self.grants),
        }

    def rows(self) -> dict[str, Any]:
        """The compact projection the supervisor summary and the journal carry."""
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
                "activations_peak": self.activations_peak,
                "free": self.device_free,
            },
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
        """
        after = self.snapshot()
        reconciled = ("host_ram", "pinned_allocated", "device_allocated")
        classes: list[dict[str, Any]] = []
        leaks: list[str] = []
        shifts: list[str] = []
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
                if delta > 0 and name == "device_allocated" and self.closed_since_baseline == 0:
                    row["verdict"] = "baseline_shift"
                    shifts.append(
                        f"device_allocated: +{delta} B on the first attempt of this executor "
                        "generation - lazily allocated library workspace, held by the "
                        "process rather than the attempt; the baseline moves and every "
                        "later attempt is held to the new line"
                    )
                    self.device_baseline += delta
                elif delta != 0:
                    row["verdict"] = "leak"
                    leaks.append(
                        f"{name}: {delta:+d} B across the attempt "
                        f"({before[name]} -> {after[name]}), measured by {counter.kind}"
                    )
            classes.append(row)
        self.closed_since_baseline += 1
        record = {
            "attempt": key,
            "classes": classes,
            "leaks": leaks,
            "baseline_shifts": shifts,
            "unreadable": sorted(self.unreadable),
            "closed": not leaks,
        }
        self.reconciliations.append(record)
        del self.reconciliations[:-32]
        return record
