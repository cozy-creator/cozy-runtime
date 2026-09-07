"""Component-use ADMISSION and MATERIALIZATION — the runtime body of the §1.1 contract.

The package declares, per public model operation, the possible heavyweight component set it
may touch. It declares nothing else: no device, no movement, no offload, no pinning, no
eviction, no backend. This module is what makes that declaration mean something at runtime.

    admit(method, components)   BEFORE method entry: reserve the declared component working
                                set, or refuse typed. Plain oversized transformers keep
                                common weights plus one declared no-split block resident.
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

import functools
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from cozy_runtime.internal import accel
from cozy_runtime.internal.fill import FillRefusal, StreamingFillBackend
from cozy_runtime.internal.paging import PageUnit


def _read_bytes() -> int:
    """Bytes THIS process has actually pulled from the block layer, or -1 (cr-100).

    The one cheap way to tell a warm page-cache re-stage from a cold read: page faults on
    the mapped CAS objects are charged here, cache hits are not. Observation only, and it
    is allowed to be unavailable — every non-Linux host reports -1 rather than a zero that
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
    #: The same demand split by authored component-use method. A staged rung never combines
    #: one method's resident set with another method's scratch peak.
    scope_headrooms: dict[str, int] = field(default_factory=dict)
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
    #: Per-component-use activation measurement. A staged attempt moves the resident baseline,
    #: so only a scope-local baseline can separate its kernels from the weights it currently holds.
    scope_baseline_bytes: int = -1
    attempt_activation_peaks: dict[str, int] = field(default_factory=dict)
    attempt_absolute_peak_bytes: int = 0
    page_call: tuple[str, PageUnit] | None = None
    page_weight_delta: int = 0
    page_autocast_cache: bool | None = None
    page_bytes: int = 0
    attempt_page_bytes: int = 0

    def __post_init__(self) -> None:
        for name, layout in self.backend.paging.items():
            for unit in layout.blocks:
                module = unit.owners[0][1]
                self.backend.paging_hooks.extend(
                    (
                        module.register_forward_pre_hook(
                            functools.partial(self._page_enter, name, unit)
                        ),
                        module.register_forward_hook(
                            functools.partial(self._page_leave, name, unit), always_call=True
                        ),
                    )
                )

    def _paging_peak(self, weight_delta: int) -> None:
        peak = accel.peak_allocated(self.torch, self.kind)
        self.attempt_absolute_peak_bytes = max(self.attempt_absolute_peak_bytes, peak)
        if self.active is not None and self.scope_baseline_bytes >= 0:
            method = self.active[0]
            self.attempt_activation_peaks[method] = max(
                self.attempt_activation_peaks.get(method, 0),
                max(peak - self.scope_baseline_bytes - weight_delta, 0),
            )

    def _page_enter(self, name: str, unit: PageUnit, module: Any, args: Any) -> None:
        self._require_clean(unit.path)
        if self.active is None or name not in self.active[1] or self.page_call is not None:
            raise self._poison(
                "paging_scope", "a paging block requires one declared, nonnested forward"
            )
        if self.torch.is_grad_enabled():
            raise ResidencyRefusal(
                "paging_autograd", "weight paging requires inference with gradients disabled"
            )
        self._paging_peak(0)
        headroom = self.scope_headrooms.get(self.active[0], self.headroom)
        remaining = max(headroom - max(self._allocated() - self.scope_baseline_bytes, 0), 0)
        if not self._fits(unit.nbytes + remaining):
            raise self._shortfall(
                self.active[0],
                name,
                remaining,
                "paging block does not fit before reservation",
                needed_weight_bytes=unit.nbytes,
            )
        allocated = self._allocated()
        started = time.perf_counter()
        try:
            self.backend.stage_page(name, unit)
        except Exception as exc:
            raise self._poison(
                "paging_fill", f"{name}.{unit.path}: fill failed after reservation"
            ) from exc
        # A surrounding autocast scope may span hundreds of tiles. Its ordinary
        # weight cache would retain a fresh cast for every rematerialized Parameter,
        # defeating eviction even though the original FP32 storage was released.
        # Change only caching, never the caller's enabled state or compute dtype.
        self.page_autocast_cache = self.torch.is_autocast_cache_enabled()
        self.torch.set_autocast_cache_enabled(False)
        self.page_call = (name, unit)
        self.page_weight_delta = self._allocated() - allocated
        moved = sum(self.backend.plan[f"{name}.{key}"].stored_nbytes for key in unit.keys)
        self.page_bytes += moved
        self.attempt_page_bytes += moved
        self.allocator_delta_bytes += self.page_weight_delta
        self.stages += 1
        self.attempt_stages += 1
        elapsed = (time.perf_counter() - started) * 1000
        self.staged_ms += elapsed
        self.attempt_staged_ms += elapsed
        self.attempt_absolute_peak_bytes = max(
            self.attempt_absolute_peak_bytes, accel.peak_allocated(self.torch, self.kind)
        )
        accel.reset_peak(self.torch, self.kind)

    def _page_leave(self, name: str, unit: PageUnit, module: Any, args: Any, output: Any) -> None:
        if self.page_call != (name, unit):
            return  # an always-call hook also runs after a pre-hook refusal
        try:
            self._finish_page(name, unit, output)
        finally:
            if self.page_autocast_cache is not None:
                self.torch.set_autocast_cache_enabled(self.page_autocast_cache)
                self.page_autocast_cache = None

    def _finish_page(self, name: str, unit: PageUnit, output: Any) -> None:
        if output is None and sys.exception() is not None:
            self.poisoned = f"paging forward failed in {name}.{unit.path}"
            self.page_call = None
            return  # the process, not an in-process unwind, owns failed computation
        try:
            storages = {
                tensor.untyped_storage().data_ptr()
                for _, owner in unit.owners
                for tensor in (*owner.parameters(recurse=False), *owner.buffers(recurse=False))
            } - {0}
            leaves = self.torch.utils._pytree.tree_leaves(output)
            opaque = any(
                not isinstance(value, self.torch.Tensor)
                and type(value) not in (bool, int, float, complex, str, bytes, type(None))
                for value in leaves
            )
            aliases = any(
                isinstance(value, self.torch.Tensor)
                and value.untyped_storage().data_ptr() in storages
                for value in leaves
            )
        except Exception as exc:
            raise self._poison(
                "paging_output_storage", "block output storage could not be verified"
            ) from exc
        if opaque:
            raise self._poison(
                "paging_output_schema", "block outputs must expose tensor leaves through a PyTree"
            )
        if aliases:
            raise self._poison(
                "paging_output_alias",
                "block output retains a view of weights selected for eviction",
            )
        self._paging_peak(self.page_weight_delta)
        allocated = self._allocated()
        started = time.perf_counter()
        try:
            event = accel.completion_event(self.torch, self.kind)
            # Conservative first path: finish all device work before freeing weights.
            # There is no prefetch stream or second liveness authority.
            accel.synchronize(self.torch, self.kind)
            if event is not None and not event.query():
                raise RuntimeError("block completion was not observed")
            self.backend.evict_page(name, unit)
        except Exception as exc:
            raise self._poison(
                "paging_eviction", f"{name}.{unit.path}: completion or eviction failed"
            ) from exc
        self.allocator_delta_bytes += self._allocated() - allocated
        self.evictions += 1
        self.attempt_evictions += 1
        elapsed = (time.perf_counter() - started) * 1000
        self.evicted_ms += elapsed
        self.attempt_evicted_ms += elapsed
        self.page_call = None
        self.page_weight_delta = 0
        accel.reset_peak(self.torch, self.kind)

    @property
    def kind(self) -> str:
        """The BACKEND this residency plane is operating, taken from the device the fill
        backend actually bound. Every device operation below goes through `accel` with it,
        so a second backend needs an implementation there and nothing here (#447)."""
        return str(self.backend.device.type)

    # ------------------------------------------------------------------ the contract

    def open_attempt(
        self, placement: str, headroom: int, scope_headrooms: dict[str, int] | None = None
    ) -> None:
        """Apply the attempt's rung and ZERO the per-attempt transition counters.

        The cumulative counters stay cumulative — they are the generation's history — but a
        reconciliation is about ONE attempt, and `stages`/`evictions` were being read as if
        they were. Measured: after any staged attempt, every later one on that generation
        reported `resident_moved`, so the activation basis was called unreadable on attempts
        that never moved a byte.
        """
        self.placement = placement
        self.headroom = headroom
        self.scope_headrooms = dict(scope_headrooms or {})
        self.last_shortfall = None
        self.allocator_delta_bytes = 0
        self.attempt_stages = 0
        self.attempt_evictions = 0
        self.attempt_staged_ms = 0.0
        self.attempt_evicted_ms = 0.0
        self.scope_baseline_bytes = -1
        self.attempt_activation_peaks.clear()
        self.attempt_absolute_peak_bytes = 0
        self.attempt_page_bytes = 0

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
        headroom = self.scope_headrooms.get(method, self.headroom)
        for name in components:
            if name in self.backend.paging and self._size(name) > self.backend.resident_budget:
                raise ResidencyRefusal(
                    "device_shortfall",
                    f"{name}: common weights and one block exceed the assigned ceiling",
                    {
                        "resource": "vram",
                        "needed_bytes": self._size(name),
                        "available_bytes": self.backend.resident_budget,
                        "scope": method,
                    },
                )
            fence = self.fences.setdefault(name, Fence())
            self.clock += 1
            fence.last_used = self.clock
            if name in self.backend.components:
                continue
            self._make_room(name, components, headroom)
            # THE PRE-MUTATION CAPACITY CHECK. Everything evictable is already gone; if the
            # measured allocatable room still does not fit this component plus the attempt's
            # headroom, the answer is a typed shortfall with nothing to undo — the executor
            # stays Ready and the ladder chooses a deeper rung. Past this line, a failure is
            # a surprise the arithmetic did not predict, and surprises poison.
            if not self._fits(self.backend.stage_bytes(name, headroom)):
                refusal = self._shortfall(
                    method,
                    name,
                    headroom,
                    "refused on measured capacity before any byte moved",
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
                refusal = self._shortfall(method, name, headroom, str(exc))
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
        self.scope_baseline_bytes = self._allocated()
        accel.reset_peak(self.torch, self.kind)
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
        peak = accel.peak_allocated(self.torch, self.kind)
        self.attempt_absolute_peak_bytes = max(self.attempt_absolute_peak_bytes, peak)
        if self.scope_baseline_bytes >= 0:
            self.attempt_activation_peaks[method] = max(
                self.attempt_activation_peaks.get(method, 0),
                max(peak - self.scope_baseline_bytes, 0),
            )
        self.scope_baseline_bytes = -1
        event = accel.completion_event(self.torch, self.kind)
        for name in components:
            if name in self.backend.components:
                self.fences.setdefault(name, Fence()).event = event
        self.active = None

    # ------------------------------------------------------------------ the lane's call

    def vacate(self) -> dict[str, Any]:
        """FREE every evictable component: the LANE handed the device to another tenant.

        Not an attempt's admission and not a rung's doing (cr-022's swap arm). A rung is
        an attempt's promise about its OWN weights — `all_resident` frees nothing to keep
        that promise — while this is the worker's decision, between attempts, about whose
        weights hold the card; the seat guarantees no scope is active here. Two things do
        not bend for it: the completion fence (a component a kernel may still be reading
        stays, and is reported as held) and the original module topology needed for
        checkpoint refill after encoded-leaf eviction. Eviction
        FREES — weights are immutable, there is no copy-out — and the next admission on
        this generation re-stages from the store like any other.
        """
        self._require_clean("vacate")
        freed: dict[str, int] = {}
        held: dict[str, str] = {}
        allocated = self._allocated()
        started = time.perf_counter()
        for name in sorted(
            list(self.backend.components), key=lambda n: self.fences.get(n, Fence()).last_used
        ):
            if not self._evictable(name):
                held[name] = "the completion fence has not fired"
                continue
            try:
                freed[name] = self.backend.evict(name)
            except FillRefusal as exc:
                held[name] = f"{exc.code}: {exc}"[:200]
                continue
            self.evictions += 1
        accel.release_cached(self.torch, self.kind)
        took = (time.perf_counter() - started) * 1000
        self.evicted_ms += took
        return {
            "freed": freed,
            "held": held,
            "freed_bytes": sum(freed.values()),
            "allocator_delta_bytes": self._allocated() - allocated,
            "vacate_ms": round(took, 2),
        }

    def restore(self, names: tuple[str, ...] = ()) -> dict[str, Any]:
        """RE-FILL what a vacate took, TAKING THE CARD FROM NOBODY (cr-097).

        `vacate` had no counterpart, and that asymmetry was the whole defect: a tenant the
        lane emptied for a co-tenant stayed empty for the life of its generation. Its own
        later attempts re-admitted per scope and settled into a partial residency the card
        had room to end -- measured at +118% handler time (8739 ms -> 18909 ms) across
        seven consecutive attempts on a device holding 10 GiB free, ended only when an
        unrelated rebuild re-prepared the generation by accident.

        THIS EVICTS NOTHING. There is no `_make_headroom` and no `_make_room` here, and
        that is the point rather than an omission: restoring by displacing a neighbour
        would be the evict-refill oscillation the owner named a defect, and it would make
        an alternating two-model workload pay a fill on every attempt. A component that
        does not fit RIGHT NOW is left absent and named in `held`; the next admission
        stages it on demand exactly as it does today, so the worst case is what happens
        already.

        The fence and the mid-mutation rule do not bend. A stage that fails after mutation
        began poisons for the same reason it does in `admit` -- a half-arrived component
        is the same hazard whoever asked for it -- while a measured shortfall is simply a
        component not restored, because this is an optimization and refusing an attempt
        for it would be worse than the state it repairs.
        """
        self._require_clean("restore")
        # THE SAME RULE PREPARE USES, AND DELIBERATELY NO OTHER (cr-097, second cut).
        #
        # The first cut gated each component on `size + the attempt's activation envelope`.
        # That looked careful and was wrong, on measured evidence: the envelope is banked
        # from a SUCCEEDING serve, and a serve run on an empty card takes more than the same
        # serve run beside a co-tenant — 3.91 GiB alone against 1.92 GiB co-resident, same
        # model, same shape. Kernels expand into the headroom they are given (workspace
        # selection, allocator behaviour); the envelope is an upper bound they CHOSE, not a
        # requirement they have. Gating on it made restore refuse `unet` by 1.1 GiB forever
        # and left the tenant at 2x its solo handler time — the exact degradation restore
        # exists to end.
        #
        # So this asks only what PREPARE asks: does the component fit on the card right now,
        # taking the card from nobody. `constructed` is by definition a set prepare already
        # held under its authorized ceiling, so restoring back to it can never exceed what
        # this generation has already been proven to hold; the only thing that changed since
        # is the co-tenant, and `_fits` measures that directly through the allocator. Prepare
        # and restore are the same act — putting a generation's weights on a card — and an
        # act with two different rules was the asymmetry this issue is about.
        wanted = tuple(names) if names else tuple(sorted(self.backend.checkpoints))
        restored: dict[str, int] = {}
        held: dict[str, str] = {}
        allocated = self._allocated()
        started = time.perf_counter()
        # THE SWAP'S OWN COST, MEASURED (cr-100). Until now the only host->device number
        # anyone had was a CONSTRUCTION's fill leg, and a swap's cost was inferred from it
        # — a number measured under one set of conditions read as a fact about another,
        # which is the class this whole cycle keeps finding. `stream_ms` is the copy leg
        # per staged component, `read_bytes` is what the block layer actually served, so a
        # cold re-read is distinguishable from a page-cache hit instead of assumed.
        fill_ms = 0
        # THE LEGS OF THE OTHER HALF (cr-103). `restore_ms` minus `fill_ms` was 9018 ms of
        # a 9734 ms restore with nothing said about it, because the swap path had never
        # been instrumented the way `prepare` now is. Each staged component reports the
        # legs that partition its own `ms`, and they are summed here.
        legs = {
            "reserve_ms": 0,
            "lease_ms": 0,
            "enqueue_ms": 0,
            "commit_ms": 0,
            "ring_ms": 0,
            "copy_ms": 0,
            "fence_ms": 0,
            "destinations": 0,
        }
        # OBJECTS ARE COUNTED PER SNAPSHOT, NOT PER COMPONENT. Four components of one
        # manifest touch one set of stored objects, so summing each component's count
        # would report four times what the store actually holds — the same mistake the
        # component-keyed lease map made, one plane up. `destinations` above IS summed:
        # tensors put back are per component and do not overlap.
        objects: dict[str, int] = {}
        read_before = _read_bytes()
        for name in wanted:
            if name in self.backend.components:
                continue
            if name not in self.backend.checkpoints:
                held[name] = "this generation has no checkpoint for it"
                continue
            size = self._size(name)
            if not self._fits(size):
                # Measured, before a byte moves, against the SAME meter admission uses.
                held[name] = "the device has no room for it without evicting a neighbour"
                continue
            try:
                staged = self.backend.stage(name)
                fill_ms += int(staged.get("stream_ms", 0))
                for leg in legs:
                    legs[leg] += int(staged.get(leg, 0))
                objects[self.backend.checkpoints[name].manifest_id] = int(staged.get("objects", 0))
            except FillRefusal as exc:
                raise self._poison(
                    exc.code,
                    f"restoring component {name!r} failed after mutation began: {exc}. "
                    "The executor process is the transaction -- this generation is "
                    "poisoned and the worker replaces the process (#613)",
                ) from exc
            except accel.oom_error(self.torch, self.kind) as exc:
                # The pre-check passed and the allocator disagreed. Nothing half-arrived
                # is recoverable, so this poisons exactly as the same surprise does in
                # `admit`; it is not downgraded to `held` merely because restore is
                # optional work.
                self.poisoned = self.poisoned or (
                    f"device_shortfall: restoring {name!r} hit the allocator after the "
                    "measured pre-check passed -- a mid-mutation surprise, so the "
                    "generation is poisoned and the process is replaced (#613)"
                )
                held[name] = f"device_out_of_memory: {exc}"[:200]
                break
            self.stages += 1
            self.attempt_stages += 1
            restored[name] = size
            fence = self.fences.setdefault(name, Fence())
            self.clock += 1
            fence.last_used = self.clock
        took = (time.perf_counter() - started) * 1000
        self.staged_ms += took
        self.attempt_staged_ms += took
        self.allocator_delta_bytes += self._allocated() - allocated
        read_after = _read_bytes()
        return {
            "restored": restored,
            "held": held,
            "restored_bytes": sum(restored.values()),
            "allocator_delta_bytes": self._allocated() - allocated,
            "restore_ms": round(took, 2),
            "fill_ms": fill_ms,
            "legs": {**legs, "objects": sum(objects.values())},
            "read_bytes": max(read_after - read_before, 0) if read_after >= 0 else -1,
        }

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

    def _make_room(self, wanted: str, declared: tuple[str, ...], headroom: int) -> None:
        """Demand-pull eviction, LRU over evictable components whose fences have fired.

        `all_resident` evicts NOTHING: it is the rung that promised everything fits, and a
        rung that quietly starts swapping is a rung whose price was a lie. Under a staged
        rung this frees just enough for what is being admitted, and a component the CURRENT
        method declares is never a victim of its own admission.
        """
        need = self.backend.stage_bytes(wanted, headroom)
        if self.placement == "all_resident":
            # The rung that promised everything fits does not free anything to keep the
            # promise. If a component is not resident under it, admission checks the room
            # that is actually there and a shortage is a typed refusal with numbers.
            return
        while not self._fits(need):
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
        if name in self.backend.paging:
            return self.backend.paging[name].working_bytes
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

    def _shortfall(
        self,
        method: str,
        name: str,
        headroom: int,
        detail: str,
        *,
        needed_weight_bytes: int | None = None,
    ) -> ResidencyRefusal:
        need = self._size(name) if needed_weight_bytes is None else needed_weight_bytes
        rows = accel.allocation(self.torch, self.kind)
        free, total = rows["driver_free_bytes"], rows["driver_total_bytes"]
        allocatable = int(free) + max(int(rows["reserved_bytes"]) - int(rows["allocated_bytes"]), 0)
        required = (
            self.backend.stage_bytes(name, headroom)
            if needed_weight_bytes is None
            else int(need) + headroom
        )
        resident = {n: self.backend.vram_charge.get(n, 0) for n in self.backend.components}
        return ResidencyRefusal(
            "device_shortfall",
            f"{method}() declares component {name!r} needing {need} B plus "
            f"{headroom} B of scope headroom; {allocatable} B are allocatable "
            f"({free} B driver-free of {total} B) with {sorted(resident)} resident, short by "
            f"{max(required - allocatable, 0)} B. Every evictable component was already freed, so "
            "this is a capacity fact, not an allocation to retry - the ladder chooses a "
            f"deeper rung or the request refuses. ({detail[:160]})",
            {
                "resource": "vram",
                "scope": "component_use",
                "needed_bytes": required,
                "available_bytes": allocatable,
                "evidence_class": "measured",
                "request_shape": method,
            },
        )

    # ------------------------------------------------------------------ facts

    def document(self) -> dict[str, Any]:
        return {
            "placement": self.placement,
            "headroom_bytes": self.headroom,
            "scope_headroom_bytes": dict(self.scope_headrooms),
            "admissions": self.admissions,
            "stages": self.stages,
            "evictions": self.evictions,
            "attempt_stages": self.attempt_stages,
            "attempt_evictions": self.attempt_evictions,
            "attempt_staged_ms": round(self.attempt_staged_ms, 2),
            "attempt_evicted_ms": round(self.attempt_evicted_ms, 2),
            "page_bytes": self.page_bytes,
            "attempt_page_bytes": self.attempt_page_bytes,
            "paging": {name: layout.document() for name, layout in self.backend.paging.items()},
            "attempt_activation_peak_bytes": max(self.attempt_activation_peaks.values(), default=0),
            "attempt_activation_peaks": dict(self.attempt_activation_peaks),
            "attempt_absolute_peak_bytes": self.attempt_absolute_peak_bytes,
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
