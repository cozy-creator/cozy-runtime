"""Component-use ADMISSION and MATERIALIZATION — the runtime body of the §1.1 contract.

The package declares, per public model operation, the possible heavyweight component set it
may touch. It declares nothing else: no device, no movement, no offload, no pinning, no
eviction, no backend. This module is what makes that declaration mean something at runtime.

    admit(method, components)   BEFORE method entry: every declared component is resident,
                                or the call refuses typed. Making room is demand-pull
                                eviction of evictable components in LRU order.
    release(method, components) AFTER the body returns: record the completion event and
                                close the scope. Release makes bytes EVICTABLE. It does not
                                evict, and it imposes no global scope sequence (§3.2).

**Scopes are SERIAL and the active scope is a SET, not a refcount** (decisions #613,
cr-025). Under law 8 — one attempt per device — exactly one component-use scope is active
per instance at a time, so a component is either in the active scope's set or evictable.
There is no `held` counter to leak: the refcount machinery this replaces let a refusal on
member two of a composite admission strand member one permanently unevictable (cr-034's
class), and the symmetric release path clamped underflow with `max`. Both defects are
unrepresentable here because the state they lived in does not exist.

**The executor process is the transaction** (#613, law 19). A capacity shortfall is checked
on MEASURED numbers before a byte moves and refuses typed — the executor stays Ready and
the ladder deepens. A failure AFTER mutation began (a stage that half-arrived, an allocator
surprise the pre-check did not predict) POISONS this plane and, through it, the executor:
the worker replaces the process and external reclaim frees the card. Nothing here unwinds
device state on a failure path. So do double release, a release of a scope that is not the
active one, and a release naming a component the generation does not hold — accounting that
no longer matches reality is a poison, never a clamp.

Three properties are structural rather than promised:

**Eviction FREES; it never demotes.** There is no host residence tier for weights. A
re-stage is an ordinary cr-005 fill from the page-cached CAS mapping — the same transaction,
the same refusals, the same freeze point. A demote-copy would need a pinned host tier sized
for the whole pipeline, and §3.2 rules pinned a staging RING, never a residence tier.

**A fence outlives its statement.** `release` records a CUDA event on the executor's
stream; the component's bytes become evictable only after that event has fired. Evicting
bytes a kernel is still reading is the defect this exists to make unrepresentable, and
querying the event is how it is known rather than assumed.

**Refusal is typed and quantified.** When the declared set cannot be made resident even
after evicting everything evictable, the answer is a `device_shortfall` naming what was
needed, what was there, and by how much it was short — never an OOM walk, and never a
reactive load of something the method did not declare.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cozy_runtime.internal import accel
from cozy_runtime.internal.fill import FillRefusal, StreamingFillBackend


class ResidencyRefusal(Exception):
    """The declared set cannot be admitted. Carries the structured shortfall."""

    def __init__(self, code: str, detail: str, shortfall: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.shortfall = shortfall


@dataclass(slots=True)
class Fence:
    """One component's completion fence and LRU stamp. No refcount exists (#613)."""

    event: Any = None
    last_used: int = 0

    def passed(self) -> bool:
        """The last completion event fired, or none was ever recorded."""
        return self.event is None or bool(self.event.query())


@dataclass(slots=True)
class ComponentResidency:
    """The `author.Residency` body: admit, materialize, fence, make evictable."""

    backend: StreamingFillBackend
    torch: Any
    #: the placement rung the PlanChooser fixed for this attempt. `all_resident` means
    #: nothing may be evicted; the staged rungs are what make eviction legal at all.
    placement: str = "all_resident"
    #: The device bytes the ATTEMPT's kernels need on top of the weights — the plan's
    #: activations envelope. Under a staged rung, admission frees down to it; under
    #: `all_resident` it is recorded and nothing is freed, because that rung's whole claim
    #: is that it did not have to.
    headroom: int = 0
    #: THE ACTIVE SCOPE — (method, its declared set) — or None between scopes. Serial by
    #: law 8: a component is in this set or it is evictable (its fence permitting).
    active: tuple[str, frozenset[str]] | None = None
    fences: dict[str, Fence] = field(default_factory=dict)
    #: Set on the first mid-mutation failure or accounting violation and never cleared.
    #: The executor reads it after every attempt and converts it into its own poison, so
    #: the worker replaces the process (cr-024) — the transaction IS the process (#613).
    poisoned: str = ""
    clock: int = 0
    admissions: int = 0
    stages: int = 0
    evictions: int = 0
    staged_ms: float = 0.0
    evicted_ms: float = 0.0
    #: refused ADMISSIONS, kept so a bundle can say the ladder ran out rather than crashed
    refusals: list[str] = field(default_factory=list)
    #: the structured shortfall of the LAST refusal, so the terminal carries both numbers
    #: instead of an exception string (cr-008a finding 8: a Fault has no shortfall slot,
    #: a TerminalCause does, and this is what fills it).
    last_shortfall: dict[str, Any] | None = None
    #: THE ATTEMPT'S DECLARED RESIDENCY TRANSITION, signed, MEASURED on the same meter the
    #: ledger reconciles `device_allocated` with (`torch.cuda.memory_allocated()`, read
    #: either side of the move). The worker subtracts exactly this and holds the
    #: remainder to zero, so the meter that reports the class must be the meter that
    #: explains it — plan arithmetic would leave the allocator's rounding as "leak"
    #: (cr-008b).
    allocator_delta_bytes: int = 0
    attempt_stages: int = 0
    attempt_evictions: int = 0
    attempt_staged_ms: float = 0.0
    attempt_evicted_ms: float = 0.0

    @property
    def kind(self) -> str:
        """The BACKEND this residency plane is operating, taken from the device the fill
        backend actually bound. Every device operation below goes through `accel` with it,
        so a second backend needs an implementation there and nothing here (#447)."""
        return str(self.backend.device.type)

    # ------------------------------------------------------------------ the contract

    def open_attempt(self, placement: str, headroom: int) -> None:
        """Apply the attempt's rung and ZERO the per-attempt transition counters.

        The cumulative counters stay cumulative — they are the generation's history — but a
        reconciliation is about ONE attempt, and `stages`/`evictions` were being read as if
        they were. Measured: after any staged attempt, every later one on that generation
        reported `resident_moved`, so the activation basis was called unreadable on attempts
        that never moved a byte.
        """
        self.placement = placement
        self.headroom = headroom
        self.last_shortfall = None
        self.allocator_delta_bytes = 0
        self.attempt_stages = 0
        self.attempt_evictions = 0
        self.attempt_staged_ms = 0.0
        self.attempt_evicted_ms = 0.0

    def admit(self, method: str, components: tuple[str, ...]) -> None:
        """Make the declared set resident BEFORE the method body runs, or refuse typed.

        Admission is atomic against the ACTIVE-SCOPE SET: the scope becomes active only
        after every member is resident. A capacity shortfall refuses BEFORE the shortfall
        member mutates anything — earlier members staged by this admission are simply
        resident and evictable, because no counter pinned them (#613 absorbs cr-034). A
        failure after mutation began poisons; nothing unwinds.
        """
        self.admissions += 1
        self._require_clean(method)
        if self.active is not None:
            # UNREACHABLE BY CONSTRUCTION: the only caller (`author/_model._Scope`) refuses
            # `concurrent_scope` before it ever gets here. Reaching it means a Runtime bug,
            # and a Runtime bug is not a reason to latch poison and make the worker respawn
            # and refill multiple GB of weights. It is recorded as a fault and the stale
            # scope is closed.
            self._fault(
                "scope_overlap",
                f"{method}() entered while {self.active[0]}() holds the active "
                "component-use scope — scopes are serial (law 8)",
            )
            self.active = None
        # Under a staged rung, admission first frees ROOM FOR THE WORK, not just room for
        # the weights: every evictable component this method does not declare goes, so the
        # scope gets the whole card minus its own set. That is the difference between a
        # rung that exists and one that helps — a 1024px VAE decode next to 6.46 GiB of
        # resident weights runs out of memory INSIDE the kernel, where nothing can decide.
        self._make_headroom(components)
        for name in components:
            fence = self.fences.setdefault(name, Fence())
            self.clock += 1
            fence.last_used = self.clock
            if name in self.backend.components:
                continue
            self._make_room(name, components)
            # THE PRE-MUTATION CAPACITY CHECK. Everything evictable is already gone; if the
            # measured allocatable room still does not fit this component plus the attempt's
            # headroom, the answer is a typed shortfall with nothing to undo — the executor
            # stays Ready and the ladder chooses a deeper rung. Past this line, a failure is
            # a surprise the arithmetic did not predict, and surprises poison.
            if not self._fits(self._size(name) + self.headroom):
                refusal = self._shortfall(
                    method, name, "refused on measured capacity before any byte moved"
                )
                self.last_shortfall = refusal.shortfall
                self.refusals.append(refusal.detail[:400])
                raise refusal
            started = time.perf_counter()
            allocated = self._allocated()
            try:
                self.backend.stage(name)
            except FillRefusal as exc:
                self.refusals.append(f"{exc.code}: {exc}"[:400])
                raise self._poison(
                    exc.code,
                    f"{method}() declares component {name!r} and its stage failed after "
                    f"mutation began: {exc}. The executor process is the transaction — "
                    "this generation is poisoned and the worker replaces the process "
                    "(#613); no in-process unwind repairs a half-arrived component",
                ) from exc
            except accel.oom_error(self.torch, self.kind) as exc:
                refusal = self._shortfall(method, name, str(exc))
                self.last_shortfall = refusal.shortfall
                self.refusals.append(refusal.detail[:400])
                self.poisoned = self.poisoned or (
                    f"device_shortfall: staging {name!r} hit the allocator after the "
                    "measured pre-check passed — a mid-mutation surprise, so the "
                    "generation is poisoned and the process is replaced (#613)"
                )
                raise refusal from exc
            self.stages += 1
            self.attempt_stages += 1
            self.allocator_delta_bytes += self._allocated() - allocated
            took = (time.perf_counter() - started) * 1000
            self.staged_ms += took
            self.attempt_staged_ms += took
        self.active = (method, frozenset(components))

    def release(self, method: str, components: tuple[str, ...]) -> None:
        """Record the fence and close the active scope. Nothing is evicted here.

        The three scope-bookkeeping conditions below — no scope open, a scope naming a
        different method or set, a component the generation does not hold — are all
        UNREACHABLE given the only caller (`author/_model._Scope`, whose `__enter__` and
        `__exit__` pass the same method and set and cannot nest). They used to latch poison,
        which forced a full worker respawn and a multi-GB refill: an impossible internal bug
        turned into a production outage. They are recorded as faults on the residency
        observation and the release completes.
        """
        self._require_clean(method)
        if self.active is None:
            self._fault(
                "scope_over_release",
                f"{method}() released with no component-use scope active",
            )
        else:
            held_method, held_set = self.active
            if held_method != method or held_set != frozenset(components):
                self._fault(
                    "scope_release_mismatch",
                    f"{method}({', '.join(components)}) released while the active scope is "
                    f"{held_method}({', '.join(sorted(held_set))})",
                )
        unknown = sorted(n for n in components if n not in self.backend.components)
        if unknown:
            self._fault(
                "scope_release_unknown",
                f"{method}() releases {unknown} and this generation does not hold "
                f"{'it' if len(unknown) == 1 else 'them'} resident",
            )
        event = accel.completion_event(self.torch, self.kind)
        for name in components:
            if name in self.backend.components:
                self.fences.setdefault(name, Fence()).event = event
        self.active = None

    # ------------------------------------------------------------------ poison

    def _fault(self, code: str, detail: str) -> None:
        """An impossible-by-construction bookkeeping violation: RECORD it, do not refuse.

        It rides `document()["refusals"]` into the residency observation and the triage
        bundle, so it is visible to whoever reads the attempt — without evicting a live
        generation over a condition its own caller makes unreachable.
        """
        self.refusals.append(f"fault {code}: {detail}"[:400])

    def _poison(self, code: str, detail: str) -> ResidencyRefusal:
        """Mark this plane poisoned and hand back the typed refusal to raise.

        Poison is latched: every later admit/release refuses on it, and the executor
        converts it into its own poison after the attempt, which is what makes the worker
        replace the process. The refusal still carries the structured code so the terminal
        names the violation rather than the death.
        """
        self.poisoned = self.poisoned or f"{code}: {detail[:300]}"
        return ResidencyRefusal(code, detail)

    def _require_clean(self, what: str) -> None:
        if self.poisoned:
            raise ResidencyRefusal(
                "residency_poisoned",
                f"{what}() against a poisoned residency plane ({self.poisoned[:300]}) — "
                "a poisoned generation has no recovery edge; the worker replaces the "
                "process",
            )

    # ------------------------------------------------------------------ room

    def _evictable(self, name: str) -> bool:
        """Not in the active scope's set AND the completion fence passed. Both, or the
        bytes stay. This is the whole of the eviction law now — no counter exists."""
        if self.active is not None and name in self.active[1]:
            return False
        return self.fences.get(name, Fence()).passed()

    def _make_headroom(self, declared: tuple[str, ...]) -> None:
        """Under a STAGED rung, the declared set is what is resident. Everything else goes.

        The first version freed only down to the plan's activation envelope, and that
        envelope is an AUTHORED BOUND until a serve has been banked — so on a fresh
        generation it is a guess, and a guess that is 1 GiB when a 1024px VAE decode wants
        three is a kernel OOM with nothing left to decide. Measured exactly that: latents
        healthy, `Tried to allocate 512.00 MiB` with 98 MiB free, twice.

        So the rung means what its name says: `component_staged` keeps the component-use set
        resident AT A TIME, and every other evictable component is freed at admission. Each
        scope gets the whole card minus its own declared set, which makes the memory
        behaviour PREDICTABLE instead of dependent on an uncalibrated number. It is the
        sequential plan, it is what an 8 GiB card can actually do with a 6.46 GiB pipeline,
        and its price is measured rather than hidden: one component swap per scope change.

        `all_resident` frees NOTHING — that rung's whole claim is that it did not have to,
        and a rung that quietly starts swapping is a rung whose price was a lie.
        """
        if self.placement == "all_resident":
            return
        for victim in self._victims(declared):
            self._evict(victim)

    def _victims(self, declared: tuple[str, ...]) -> list[str]:
        """Every resident component the method does not declare, LRU first.

        Evictable means outside the active scope's set with the completion fence FIRED. A
        component whose kernels may still be reading it is never a victim, which is what
        the event fence is for, and a component the CURRENT method declares is never a
        victim of its own admission.
        """
        candidates = [
            name
            for name in list(self.backend.components)
            if name not in declared and self._evictable(name)
        ]
        return sorted(candidates, key=lambda n: self.fences.get(n, Fence()).last_used)

    def _make_room(self, wanted: str, declared: tuple[str, ...]) -> None:
        """Demand-pull eviction, LRU over evictable components whose fences have fired.

        `all_resident` evicts NOTHING: it is the rung that promised everything fits, and a
        rung that quietly starts swapping is a rung whose price was a lie. Under a staged
        rung this frees just enough for what is being admitted, and a component the CURRENT
        method declares is never a victim of its own admission.
        """
        need = sum(row.nbytes for row in self.backend.plan.values() if row.component == wanted)
        if self.placement == "all_resident":
            # The rung that promised everything fits does not free anything to keep the
            # promise. If a component is not resident under it, admission checks the room
            # that is actually there and a shortage is a typed refusal with numbers.
            return
        while not self._fits(need + self.headroom):
            victim = self._victim(declared)
            if victim is None:
                return  # nothing left to give; the pre-mutation check refuses with numbers
            self._evict(victim)

    def _evict(self, victim: str) -> None:
        """One eviction, timed and MEASURED against the allocator on both sides."""
        started = time.perf_counter()
        allocated = self._allocated()
        self.evictions += 1
        self.attempt_evictions += 1
        self.backend.evict(victim)
        accel.release_cached(self.torch, self.kind)
        self.allocator_delta_bytes += self._allocated() - allocated
        took = (time.perf_counter() - started) * 1000
        self.evicted_ms += took
        self.attempt_evicted_ms += took

    def _allocated(self) -> int:
        return accel.allocated(self.torch, self.kind)

    def _size(self, name: str) -> int:
        """The component's DECLARED bytes, from the read plan — resident or not.

        This is what makes a non-resident component priceable. A component that has been
        evicted still costs exactly this much to bring back, and a chooser that reads only
        the RESIDENT map cannot see that cost at all (cl-003's second-1024px defect).
        """
        known = self.backend.component_bytes.get(name)
        if known:
            return known
        return sum(row.nbytes for row in self.backend.plan.values() if row.component == name)

    def _fits(self, need: int) -> bool:
        """ALLOCATABLE capacity, from the allocator's own view (§3.2), never arithmetic."""
        rows = accel.allocation(self.torch, self.kind)
        slack = rows["reserved_bytes"] - rows["allocated_bytes"]
        return rows["driver_free_bytes"] + max(slack, 0) >= need

    def _victim(self, declared: tuple[str, ...]) -> str | None:
        victims = self._victims(declared)
        return victims[0] if victims else None

    def _shortfall(self, method: str, name: str, detail: str) -> ResidencyRefusal:
        need = sum(row.nbytes for row in self.backend.plan.values() if row.component == name)
        rows = accel.allocation(self.torch, self.kind)
        free, total = rows["driver_free_bytes"], rows["driver_total_bytes"]
        resident = {n: self.backend.vram_charge.get(n, 0) for n in self.backend.components}
        return ResidencyRefusal(
            "device_shortfall",
            f"{method}() declares component {name!r} needing {need} B; {free} B of "
            f"{total} B are free with {sorted(resident)} resident, short by "
            f"{max(need - int(free), 0)} B. Every evictable component was already freed, so "
            "this is a capacity fact, not an allocation to retry - the ladder chooses a "
            f"deeper rung or the request refuses. ({detail[:160]})",
            {
                "resource": "vram",
                "scope": "component_use",
                "needed_bytes": int(need),
                "available_bytes": int(free),
                "evidence_class": "measured",
                "request_shape": method,
            },
        )

    # ------------------------------------------------------------------ facts

    def document(self) -> dict[str, Any]:
        return {
            "placement": self.placement,
            "headroom_bytes": self.headroom,
            "admissions": self.admissions,
            "stages": self.stages,
            "evictions": self.evictions,
            "attempt_stages": self.attempt_stages,
            "attempt_evictions": self.attempt_evictions,
            "attempt_staged_ms": round(self.attempt_staged_ms, 2),
            "attempt_evicted_ms": round(self.attempt_evicted_ms, 2),
            "allocator_delta_bytes": self.allocator_delta_bytes,
            "staged_ms": round(self.staged_ms, 2),
            "evicted_ms": round(self.evicted_ms, 2),
            "resident": {n: self.backend.vram_charge.get(n, 0) for n in self.backend.components},
            # WITH THEIR REAL SIZES. An evicted component reported at 0 B is a component the
            # planner cannot price, and pricing a rung against a residency map that says the
            # absent weights are free is exactly how `all_resident` came to "fit" on a card
            # holding one 167 MB VAE out of a 6.46 GiB pipeline.
            "evicted": {n: self._size(n) for n in self.backend.parked},
            "active": (
                {"method": self.active[0], "components": sorted(self.active[1])}
                if self.active is not None
                else None
            ),
            "fences": {
                n: {"passed": f.passed(), "last_used": f.last_used} for n, f in self.fences.items()
            },
            "poisoned": self.poisoned,
            "log": list(self.backend.stage_log[-16:]),
            "refusals": list(self.refusals[-8:]),
        }
