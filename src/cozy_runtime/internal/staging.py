"""Component-use ADMISSION and MATERIALIZATION — the runtime body of the §1.1 contract.

The endpoint declares, per public model operation, the possible heavyweight component set it
may touch. It declares nothing else: no device, no movement, no offload, no pinning, no
eviction, no backend. This module is what makes that declaration mean something at runtime.

    admit(method, components)   BEFORE method entry: every declared component is resident,
                                or the call refuses typed. Making room is demand-pull
                                eviction of refcount-0 components in LRU order.
    release(method, components) AFTER the body returns: record the completion event, drop
                                the refcount. Release makes bytes EVICTABLE. It does not
                                evict, and it imposes no global scope sequence (§3.2).

Three properties are structural rather than promised:

**Eviction FREES; it never demotes.** There is no host residence tier for weights. A
re-stage is an ordinary cr-005 fill from the page-cached CAS mapping — the same transaction,
the same refusals, the same freeze point — which is why a staged component that half-arrives
is rolled back rather than served. A demote-copy would need a pinned host tier sized for the
whole pipeline, and §3.2 rules pinned a staging RING, never a residence tier.

**A lease outlives its statement.** `release` records a CUDA event on the executor's stream;
the component's bytes become evictable only after that event has fired. Evicting bytes a
kernel is still reading is the defect this exists to make unrepresentable, and querying the
event is how it is known rather than assumed.

**Refusal is typed and quantified.** When the declared set cannot be made resident even
after evicting everything evictable, the answer is a `device_shortfall` naming what was
needed, what was there, and by how much it was short — never an OOM walk, and never a
reactive load of something the method did not declare.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cozy_runtime.internal.fill import FillRefusal, StreamingFillBackend


class ResidencyRefusal(Exception):
    """The declared set cannot be admitted. Carries the structured shortfall."""

    def __init__(self, code: str, detail: str, shortfall: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.shortfall = shortfall


@dataclass(slots=True)
class Lease:
    """One component's live use. `held` is a refcount; `event` is when the kernels finished."""

    component: str
    held: int = 0
    event: Any = None
    last_used: int = 0

    def evictable(self) -> bool:
        """Refcount zero AND the completion event fired. Both, or the bytes stay."""
        if self.held:
            return False
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
    leases: dict[str, Lease] = field(default_factory=dict)
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

    # ------------------------------------------------------------------ the contract

    def admit(self, method: str, components: tuple[str, ...]) -> None:
        """Make the declared set resident BEFORE the method body runs, or refuse typed."""
        self.admissions += 1
        # Under a staged rung, admission first frees ROOM FOR THE WORK, not just room for
        # the weights. This is the difference between a rung that exists and one that
        # helps: a 1024px VAE decode next to 6.46 GiB of resident weights runs out of
        # memory INSIDE the kernel, where there is nothing left to decide. Freeing the LRU
        # component the method does not declare, down to the plan's activation envelope,
        # is the decision — made before entry, where it is still a decision.
        self._make_headroom(components)
        for name in components:
            lease = self.leases.setdefault(name, Lease(name))
            self.clock += 1
            lease.last_used = self.clock
            if name in self.backend.components:
                lease.held += 1
                continue
            self._make_room(name, components)
            started = time.perf_counter()
            try:
                self.backend.stage(name)
            except FillRefusal as exc:
                lease.held = 0
                self.refusals.append(f"{exc.code}: {exc}"[:400])
                raise ResidencyRefusal(
                    exc.code,
                    f"{method}() declares component {name!r} and it could not be staged: {exc}",
                ) from exc
            except self.torch.cuda.OutOfMemoryError as exc:
                lease.held = 0
                refusal = self._shortfall(method, name, str(exc))
                self.last_shortfall = refusal.shortfall
                self.refusals.append(refusal.detail[:400])
                raise refusal from exc
            self.stages += 1
            self.staged_ms += (time.perf_counter() - started) * 1000
            lease.held += 1

    def release(self, method: str, components: tuple[str, ...]) -> None:
        """Record the fence and drop the refcount. Nothing is evicted here."""
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream())
        for name in components:
            lease = self.leases.get(name)
            if lease is None:
                continue
            lease.held = max(lease.held - 1, 0)
            # The event replaces the previous one only when the lease is fully released:
            # a nested or repeated use is still live and its earlier fence is not the one
            # that matters.
            if lease.held == 0:
                lease.event = event

    # ------------------------------------------------------------------ room

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
            started = time.perf_counter()
            self.evictions += 1
            self.backend.evict(victim)
            self.torch.cuda.empty_cache()
            self.evicted_ms += (time.perf_counter() - started) * 1000

    def _victims(self, declared: tuple[str, ...]) -> list[str]:
        """Every resident component the method does not declare, LRU first.

        Evictable means refcount ZERO and the completion event FIRED. A component whose
        kernels may still be reading it is never a victim, which is what the event fence is
        for, and a component the CURRENT method declares is never a victim of its own
        admission.
        """
        candidates = [
            name
            for name in list(self.backend.components)
            if name not in declared and self.leases.get(name, Lease(name)).evictable()
        ]
        return sorted(candidates, key=lambda n: self.leases.get(n, Lease(n)).last_used)

    def _make_room(self, wanted: str, declared: tuple[str, ...]) -> None:
        """Demand-pull eviction, LRU over refcount-0 leases whose events have fired.

        `all_resident` evicts NOTHING: it is the rung that promised everything fits, and a
        rung that quietly starts swapping is a rung whose price was a lie. Under a staged
        rung this frees just enough for what is being admitted, and a component the CURRENT
        method declares is never a victim of its own admission.
        """
        need = sum(row.nbytes for row in self.backend.plan.values() if row.component == wanted)
        if self.placement == "all_resident":
            # The rung that promised everything fits does not free anything to keep the
            # promise. If a component is not resident under it, staging is tried on the
            # room that is actually there and a shortage is a typed refusal with numbers.
            return
        while not self._fits(need + self.headroom):
            victim = self._victim(declared)
            if victim is None:
                return  # nothing left to give; `stage` will refuse with the real numbers
            started = time.perf_counter()
            self.evictions += 1
            self.backend.evict(victim)
            self.torch.cuda.empty_cache()
            self.evicted_ms += (time.perf_counter() - started) * 1000

    def _fits(self, need: int) -> bool:
        """ALLOCATABLE capacity, from the allocator's own view (§3.2), never arithmetic."""
        free, _ = self.torch.cuda.mem_get_info()
        slack = self.torch.cuda.memory_reserved() - self.torch.cuda.memory_allocated()
        return int(free) + max(int(slack), 0) >= need

    def _victim(self, declared: tuple[str, ...]) -> str | None:
        victims = self._victims(declared)
        return victims[0] if victims else None

    def _shortfall(self, method: str, name: str, detail: str) -> ResidencyRefusal:
        need = sum(row.nbytes for row in self.backend.plan.values() if row.component == name)
        free, total = self.torch.cuda.mem_get_info()
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
            "staged_ms": round(self.staged_ms, 2),
            "evicted_ms": round(self.evicted_ms, 2),
            "resident": {n: self.backend.vram_charge.get(n, 0) for n in self.backend.components},
            "evicted": {n: 0 for n in self.backend.parked},
            "leases": {
                n: {"held": v.held, "last_used": v.last_used} for n, v in self.leases.items()
            },
            "log": list(self.backend.stage_log[-16:]),
            "refusals": list(self.refusals[-8:]),
        }
