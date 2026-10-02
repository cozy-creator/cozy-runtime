"""The machine's device memory policy: per-GPU, per-tenant budgets (weight-plane.md §4-5).

ONE manager per Worker decides; each executor's weight plane moves the bytes. A TENANT is one
placement's executor over its lane's devices; a rank group is one tenant with rank-symmetric
budgets. The stage scheduler grants turns; a tenant from its turn's `start` to its `end` is
active and never cut; every other tenant there is idle, and one mid-call is never cut either.

`start` reads the driver once and gives activations first: the plane gets what is free after
the shape's activation growth, banked from what executors report and never sampled. A shape
never measured is admitted optimistically: a plane executor retries an OOM at a block boundary
under a lower budget. Idle tenants give room least recently used first, and only while the
active one can still use it: a plane executor (`weight_plane/1`) by a budget cut, an unmap
whose bytes stay in its host tier or the page cache; an older executor by `Vacate` (degraded,
never refused); a dead one by being ended. Below the window-1 floor `start` answers a typed
`Shortfall` with both numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence

import msgspec

from cozy_runtime.internal import accel, executor_commands, tolerant, weight_policy
from cozy_runtime.internal.accel import DeviceMemory
from cozy_runtime.internal.execution_evidence import gpu_name, gpu_names
from cozy_runtime.internal.executor_replies import AttemptReply, PlaneFacts
from cozy_runtime.internal.worker import child
from cozy_runtime.internal.worker.child import ExecutorGone
from cozy_runtime.internal.worker.lanes import DeviceLane, LaneRow, LaneSet
from cozy_runtime.internal.worker.ledger import Ledger, read_host_available, read_host_memory

#: The executor capability that runs its weights on the plane.
PLANE = "weight_plane/1"
#: An entrypoint and its request's shape cell (`plan.shape_cell`): what growth is banked by.
Shape = tuple[str, str]
Victim = tuple[str, DeviceLane, LaneRow, child.Executor]


class StageNeed(msgspec.Struct, frozen=True, kw_only=True):
    """What one stage asks of its devices."""

    entrypoint: str
    cell: str = ""
    #: the components it touches; () is every component of the active construction
    components: tuple[str, ...] = ()
    #: its activation growth; -1 takes the bank's measurement for the shape
    activation_bytes: int = -1


class Budget(msgspec.Struct, frozen=True, kw_only=True):
    """A granted turn. `vram` is the plane's device budget per ordinal, empty when the executor
    derives its own (one before the plane, an unreadable device); `whole` says the stage's
    weights and activations fit without streaming."""

    vram: dict[int, int] = {}
    whole: bool = True
    evicted: tuple[str, ...] = ()

    @property
    def plane_bytes(self) -> int:
        """The one number a rank-symmetric executor applies: its tightest device's, else -1."""
        return min(self.vram.values(), default=-1)


class Shortfall(msgspec.Struct, frozen=True, kw_only=True):
    """Below the refusal floor with every idle byte given: both numbers per ordinal."""

    needed: dict[int, int]
    available: dict[int, int]
    detail: str


class TenantView(msgspec.Struct, frozen=True, kw_only=True):
    tenant: str
    gpus: tuple[int, ...]
    active: bool
    last_used: int
    #: the executor's latest plane document; None before the plane
    plane: PlaneFacts | None


class MemoryView(msgspec.Struct, frozen=True, kw_only=True):
    #: ordinal -> (free, total) bytes as the driver read them at the latest admission
    devices: dict[int, tuple[int, int]]
    tenants: tuple[TenantView, ...]
    host_available_bytes: int


class _BudgetReply(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool = False
    code: str = ""
    detail: str = ""
    freed_bytes: int = 0
    plane: PlaneFacts | None = None


def plane_capable(executor: child.Executor) -> bool:
    return PLANE in executor.hello.get("memory", ())


def axes(cell: str) -> dict[str, int] | None:
    """A shape cell's axes (`plan.shape_cell`), or None when one is not a count."""
    try:
        return {
            axis: int(value)
            for axis, value in (part.split("=", 1) for part in cell.split(",") if part != "-")
        }
    except ValueError:
        return None


def growth(shape: Shape, bank: Mapping[Shape, int]) -> int | None:
    """`shape`'s banked activation growth, else the least among measured shapes of the same
    entrypoint at least as large on every axis (memory does not shrink as a shape grows)."""
    if shape in bank:
        return bank[shape]
    entrypoint, cell = shape
    mine = axes(cell) if cell else None
    if mine is None:
        return None
    return min(
        (
            grown
            for (other_entrypoint, other), grown in bank.items()
            if other_entrypoint == entrypoint
            and (theirs := axes(other)) is not None
            and theirs.keys() == mine.keys()
            and all(theirs[axis] >= value for axis, value in mine.items())
        ),
        default=None,
    )


def reach(free: int, committed: int, slack: int, grow: int) -> int:
    """What the active tenant can use on one device: the driver's free bytes, what its plane
    already maps, and the part of torch's cache its activations reuse."""
    return free + max(committed, 0) + min(max(slack, 0), grow)


def weights(ledger: Ledger, components: Collection[str] = ()) -> int:
    """The active construction's weight bytes per device, of `components` or all of them."""
    if ledger.layouts:
        return sum(
            layout.total
            for name, layout in ledger.layouts.items()
            if not components or name in components
        )
    built = ledger.constructed
    return sum(built.get(name, 0) for name in components or built) or ledger.weights_bytes


def floor_bytes(layouts: Mapping[str, weight_policy.Layout], components: Collection[str]) -> int:
    """The weights' refusal floor: a stage's components' floors at their finest grain. Without
    components (one attempt, its components in turn) the largest single one's."""
    per = {name: weight_policy.floor(layout) for name, layout in layouts.items()}
    if components:
        return sum(per.get(name, 0) for name in components)
    return max(per.values(), default=0)


def pinned_split(
    available: int, tenants: Sequence[tuple[str, int, int]], held: int = -1
) -> dict[str, int]:
    """Pinned host-tier budgets over `(tenant, weights, pinned now)`, most recently used first.

    The tenants share the machine's one pinned total (`weight_policy.pinned_total`: half of
    what the host has free plus `held`, everything pinned now; without it, what the tenants
    pin). No tenant rises by more than half of what is free now.
    """
    free = max(available, 0)
    pinned = held if held >= 0 else sum(max(pinned, 0) for _, _, pinned in tenants)
    left = weight_policy.pinned_total(free, pinned)
    split: dict[str, int] = {}
    for tenant, size, pinned in tenants:
        split[tenant] = min(max(size, 0), left, max(pinned, 0) + free // 2)
        left -= split[tenant]
    return split


def holds(row: LaneRow, executor: child.Executor, ordinals: Collection[int]) -> bool:
    """Whether an idle tenant holds weights a cut would free on `ordinals`."""
    if plane_capable(executor):
        return row.ledger.plane is None or row.ledger.plane.committed_bytes != 0
    return row.resident_bytes() > 0 and any(o not in row.emptied for o in ordinals)


class MemoryManager:
    """Budgets, eviction and reclaim over every tenant of every device."""

    def __init__(
        self,
        lanes: LaneSet,
        note: Callable[[str, str], None],
        reclaim: Callable[[str, child.Executor, str], bool],
    ) -> None:
        self.lanes = lanes
        self.note = note
        #: kill and reap one tenant's executor now (True once proved gone); it rebuilds later
        self.reclaim = reclaim
        self.kind = accel.host_backend_family()
        #: ordinal -> the driver's reading at the latest admission
        self.seen: dict[int, DeviceMemory] = {}
        #: tenants in their turn: never cut while alive
        self.active: set[str] = set()
        #: releases the pinned tiers the Worker keeps beyond the host's room (`HostTiers.trim`)
        self.trim: Callable[[], int] | None = None
        #: device bytes a starting executor needs free, by tenant: nobody's budget takes them
        self.wanted: dict[str, dict[int, int]] = {}

    def tenants(self, ordinals: set[int] | None = None) -> list[tuple[str, DeviceLane, LaneRow]]:
        """Every model-bearing tenant with a device among `ordinals` (None: any), each once."""
        found: dict[str, tuple[str, DeviceLane, LaneRow]] = {}
        for lane in self.lanes.lanes:
            if ordinals is None or ordinals & set(lane.ordinals):
                for slot, row in lane.rows.items():
                    if row.model_bearing:
                        found.setdefault(slot, (slot, lane, row))
        return list(found.values())

    # ------------------------------------------------------------------ the scheduler's calls

    def view(self) -> MemoryView:
        """Every device's latest reading and every tenant's plane facts; reads no driver."""
        return MemoryView(
            devices={
                o: (m.free_bytes, m.total_bytes)
                for o, m in sorted(self.seen.items())
                if m.state == "measured"
            },
            tenants=tuple(
                TenantView(
                    tenant=slot,
                    gpus=lane.ordinals,
                    active=slot in self.active,
                    last_used=row.last_used,
                    plane=row.ledger.plane,
                )
                for slot, lane, row in self.tenants()
            ),
            host_available_bytes=read_host_available(),
        )

    def start(
        self, tenant: str, stage: StageNeed, gpus: Sequence[int], why: str = ""
    ) -> Budget | Shortfall:
        """`tenant`'s turn on `gpus`, which the caller holds: idle tenants give room while it
        can still use it, then its plane budget per device, or a `Shortfall` below the floor."""
        lane = self._lane(tenant, gpus)
        lane.touch(tenant)
        self.active.add(tenant)
        row = lane.row(tenant)
        executor = row.executor()
        why = why or f"{tenant!r}'s turn"
        shape = (stage.entrypoint, stage.cell)
        grow = (
            stage.activation_bytes if stage.activation_bytes >= 0 else growth(shape, row.activation)
        )
        if executor is not None and plane_capable(executor):
            return self._start_plane(lane, tenant, row, stage, grow, why)
        return self._start_legacy(lane, tenant, row, grow, why)

    def context_need(self, lane: DeviceLane) -> dict[int, int]:
        """What a new executor needs free on each of `lane`'s devices before its first stage:
        the largest context any executor there measured, as much again for the library
        workspaces its first kernels allocate, and the margin. Empty before any was measured."""
        need = {o: self._context(o) for o in lane.ordinals}
        return need if all(need.values()) else {}

    def _context(self, ordinal: int) -> int:
        measured = max(
            (
                row.ledger.plane.context_bytes
                for _slot, _lane, row in self.tenants({ordinal})
                if row.ledger.plane is not None
            ),
            default=0,
        )
        return 2 * measured + weight_policy.MARGIN if measured > 0 else 0

    def _reserved(self, tenant: str, ordinal: int) -> int:
        """Device bytes `tenant`'s budget leaves free: what executors starting beside it want."""
        return sum(need.get(ordinal, 0) for other, need in self.wanted.items() if other != tenant)

    def room(self, lane: DeviceLane, slot: str, need: dict[int, int], why: str) -> bool:
        """Cut idle tenants until `need` is free on each of `lane`'s devices. Whether it is,
        as the driver measures it now: another process's memory counts as it stands."""
        measured = self.make_room(lane, slot, need, why, contexts=True)
        return all(
            m.state == "measured" and m.free_bytes >= need.get(o, 0) for o, m in measured.items()
        )

    def short(self, tenant: str, stage: StageNeed, gpus: Sequence[int]) -> bool:
        """Whether `tenant`'s turn on `gpus` would find less room than its weights and
        activations want, as the devices stand now. What cannot be told is short."""
        row = self._row(tenant)
        executor = row.executor() if row is not None else None
        if row is None or executor is None or not plane_capable(executor):
            return True
        ledger = row.ledger
        size = weights(ledger, stage.components)
        if not size or not any(
            lane.ordinals == tuple(gpus) and tenant in lane.rows for lane in self.lanes
        ):
            return True
        grow = growth((stage.entrypoint, stage.cell), row.activation) or 0
        committed = ledger.plane.committed_bytes if ledger.plane else 0
        return any(
            memory.state != "measured"
            or reach(memory.free_bytes, committed, ledger.slack, grow)
            < size + grow + weight_policy.MARGIN
            for memory in self._lane(tenant, gpus).measure(self.kind).values()
        )

    def yielded(
        self, tenant: str, gpus: Sequence[int], next_stage: StageNeed | None = None
    ) -> None:
        """A stage exit: its regions become evictable inside the plane; the tenant is MRU."""
        self._lane(tenant, gpus).touch(tenant)

    def prefill(self, tenant: str, contents: Sequence[str]) -> bool:
        """Fill the host tier of `contents` (loaded construction keys, in priority order) while
        `tenant` waits for its devices: its pinned budget, then `Prefetch`. Only its own
        executor is commanded; one before the plane has no host tier and answers False."""
        row = self._row(tenant)
        executor = row.executor() if row is not None else None
        if row is None or executor is None or not plane_capable(executor):
            return False
        if executor.busy():
            return False  # its previous attempt still runs, over a tier already filled
        keys = [key for key in contents if key in executor.loaded]
        if not keys:
            return False
        # Idle tenants over their share give pinned memory back first: this tenant's fill
        # then lands in room that exists, never over the host's limit.
        self.shed(keep=tenant)
        pinned = self._pinned(tenant, first=True)
        current = row.ledger.plane.pinned_budget_bytes if row.ledger.plane else -1
        try:
            with executor.watched("prefill"):
                # Only ever raised here: a tenant about to run keeps the tier it has (shrinking
                # it sent SDXL's streamed blocks to disk run after run).
                if pinned >= 0 and pinned > current:
                    # no device grant: its attempt carries one when it holds the device
                    command = executor_commands.Budget(vram_bytes=-1, pinned_bytes=pinned)
                    self.observe(row, executor, executor.call(command, timeout=None))
                for key in keys:
                    executor.call(executor_commands.Prefetch(construction=key), timeout=None)
        except ExecutorGone:
            return False  # device entry finds it dead and says so
        except Exception as exc:  # a prefill is an overlap, never the attempt's failure
            self.note("residency", f"{tenant!r}: prefill skipped: {exc}"[:400])
            return False
        return True

    def prefetch(self, tenant: str, gpu: int, stage: StageNeed) -> bool:
        """A stage's first regions into spare device budget: no executor command does this
        before `stage/1`, so nothing moves."""
        return False

    def end(self, tenant: str) -> None:
        """The turn is over. Device budgets stay; its bytes stay mapped until a tenant that can
        use them cuts it, least recently used first. The pinned host tiers are held to what
        the host has now (`shed`): process memory may have grown during the turn."""
        self.active.discard(tenant)
        self.shed()

    # ------------------------------------------------------------------ today's attempt path

    def observe(self, row: LaneRow, executor: child.Executor, reply: Mapping[str, object]) -> None:
        """Keep a plane executor's latest `plane` document (load, activate, budget replies)."""
        if plane_capable(executor):
            row.ledger.observe_plane(reply.get("plane"))

    def settle(
        self,
        lane: DeviceLane,
        slot: str,
        shape: Shape,
        reply: AttemptReply,
        *,
        ok: bool,
        forget: bool,
    ) -> None:
        """Bank one attempt: its plane facts, and on success its activation growth for `shape`
        (the plane's measurement, else what the ledger read from an older executor's metrics at
        release). A failure on the device forgets the shape: its next call is unmeasured."""
        row = lane.row(slot)
        if reply.plane is not None:
            row.ledger.plane = reply.plane
        if ok:
            grew = (
                reply.plane.activation_peak_bytes
                if reply.plane is not None
                else row.ledger.activations_for(shape[1])
            )
            if grew > 0:
                row.activation[shape] = max(row.activation.get(shape, 0), grew)
            row.emptied.clear()
        elif forget:
            row.activation.pop(shape, None)

    def make_room(
        self,
        lane: DeviceLane,
        slot: str,
        need: dict[int, int] | None,
        why: str,
        *,
        contexts: bool = False,
    ) -> dict[int, DeviceMemory]:
        """Cut idle tenants of `lane`'s devices until `need` is free on each (None: the whole
        device), for a load or an older executor that asks for room. With `contexts`, a
        shortfall that remains ends idle processes too, for the device contexts they hold."""

        def short(o: int, memory: DeviceMemory) -> bool:
            return need is None or memory.free_bytes < need.get(o, 0)

        measured, _ = self._give(lane, slot, short, why, contexts=contexts)
        if need is not None and (
            left := [o for o, m in measured.items() if m.state == "measured" and short(o, m)]
        ):
            self.note(
                "residency",
                f"{lane.lane_id}: no idle tenant is left to evict for {slot!r} on {why}: "
                + ", ".join(
                    f"{self._gpu(lane, o)} needs {need.get(o, 0)} B free, has "
                    f"{measured[o].free_bytes} of {measured[o].total_bytes}"
                    for o in sorted(left)
                ),
            )
        return measured

    # ------------------------------------------------------------------ policy

    def _start_plane(
        self,
        lane: DeviceLane,
        tenant: str,
        row: LaneRow,
        stage: StageNeed,
        banked: int | None,
        why: str,
    ) -> Budget | Shortfall:
        ledger = row.ledger
        facts = ledger.plane or PlaneFacts()
        grow = banked or 0
        size = weights(ledger, stage.components)
        want = size + grow + weight_policy.MARGIN if size else None

        def usable(memory: DeviceMemory) -> int:
            return reach(memory.free_bytes, facts.committed_bytes, ledger.slack, grow)

        # A stage whose activations were never measured cannot say what it needs: every idle
        # tenant of its GPUs gives room, least recently used first.
        read, evicted = self._give(
            lane, tenant, lambda o, m: banked is None or want is None or usable(m) < want, why
        )
        if any(m.state != "measured" for m in read.values()):
            self.note("residency", f"{lane.lane_id}: unreadable; {tenant!r} derives its budget")
            return Budget(evicted=tuple(evicted))
        ledger.observe_devices(lane.device_facts(read))
        if banked is None:
            # Activations unmeasured: the executor keeps only the stage's own weights and
            # measures them first; a grant here would hand their room to weights.
            self.note(
                "residency",
                f"{lane.lane_id}: {tenant!r} starts {why} ({stage.entrypoint} "
                f"{stage.cell or '-'}, activations unmeasured): it derives its budget; "
                f"cut {evicted or 'nobody'}",
            )
            return Budget(evicted=tuple(evicted))
        # The grant must hold the weights' floor itself: that is the refusal line.
        least = floor_bytes(ledger.layouts, stage.components)

        def below(o: int, memory: DeviceMemory) -> bool:
            return weight_policy.plane_budget(usable(memory), grow) < least

        if any(below(o, m) for o, m in read.items()):
            # Full residency is a wish and streams when unmet; the floor is a need. With every
            # idle weight byte cut and the floor still unmet, idle processes end for the
            # device contexts they hold.
            read, ended = self._give(lane, tenant, below, why, contexts=True)
            evicted += [name for name in ended if name not in evicted]
            if any(m.state != "measured" for m in read.values()):
                return Budget(evicted=tuple(evicted))
        # What an executor waiting to start beside it wants stays free, down to the floor.
        budget = {
            o: max(
                weight_policy.plane_budget(usable(m) - self._reserved(tenant, o), grow),
                min(weight_policy.plane_budget(usable(m), grow), least),
            )
            for o, m in read.items()
        }
        if any(value < least for value in budget.values()):
            floor = least + grow + weight_policy.MARGIN
            available = {o: usable(m) for o, m in read.items()}
            detail = (
                f"{tenant!r} needs {floor} B per GPU for {why} ({least} B of weights at their "
                f"finest grain, {grow} B of activations) with every idle tenant cut: "
                + ", ".join(f"{self._gpu(lane, o)} has {v} B" for o, v in sorted(available.items()))
            )
            self.note("residency", f"{lane.lane_id}: refused {detail}"[:512])
            return Shortfall(
                needed=dict.fromkeys(available, floor), available=available, detail=detail
            )
        if evicted:
            self.note(
                "residency",
                f"{lane.lane_id}: {tenant!r} starts {why} ({stage.entrypoint} {stage.cell or '-'}, "
                + f"{grow} B of activations"
                + "): budget "
                + ", ".join(f"{self._gpu(lane, o)} {b} B" for o, b in sorted(budget.items()))
                + f"; cut {evicted or 'nobody'}",
            )
        whole = want is not None and all(usable(m) >= want for m in read.values())
        return Budget(vram=budget, whole=whole, evicted=tuple(evicted))

    def _start_legacy(
        self, lane: DeviceLane, tenant: str, row: LaneRow, grow: int | None, why: str
    ) -> Budget:
        """An executor before the plane needs its measured growth free beside what a restore
        refills; a shape never measured has the whole of its devices."""
        ledger = row.ledger
        missing = sum(ledger.missing_resident().values())
        built = sum(ledger.constructed.values())

        def short(o: int, memory: DeviceMemory) -> bool:
            refill = built if o in row.emptied else missing
            return grow is None or memory.free_bytes + min(ledger.slack, grow) < grow + refill

        read, evicted = self._give(lane, tenant, short, why, contexts=grow is not None)
        fits = grow is None or all(
            m.state == "measured" and not short(o, m) for o, m in read.items()
        )
        if evicted or grow is None:
            self.note(
                "residency",
                f"{lane.lane_id}: admitted {tenant!r} for {why}: "
                + (
                    "never measured, so it has the whole of its devices"
                    if grow is None
                    else f"needs {grow} B"
                )
                + f"; evicted {evicted or 'nobody'}; free "
                + ", ".join(
                    f"{m.free_bytes} B on {self._gpu(lane, o)}" for o, m in sorted(read.items())
                )
                + ("" if fits else "; it runs staged in what is left"),
            )
        return Budget(whole=fits, evicted=tuple(evicted))

    def _give(
        self,
        lane: DeviceLane,
        slot: str,
        short: Callable[[int, DeviceMemory], bool],
        why: str,
        *,
        contexts: bool = False,
    ) -> tuple[dict[int, DeviceMemory], list[str]]:
        """Cut idle tenants from `lane`'s short devices until none is short or nobody is left:
        one driver read, and one more after each cut. With `contexts`, a measured need that
        cuts did not meet then ends idle processes, least recently used first, for the device
        contexts they hold: an idle executor stays warm only while there is room for it."""
        tried: set[str] = set()
        ended: set[str] = set()
        evicted: list[str] = []
        while True:
            measured = lane.measure(self.kind)
            self.seen.update(measured)
            if any(m.state != "measured" for m in measured.values()):
                return measured, evicted
            devices = {o for o, m in measured.items() if short(o, m)}
            victim = self._victim(slot, devices, tried) if devices else None
            if victim is not None:
                tried.add(victim[0])
                evicted.append(victim[0])
                self._cut(victim, slot, why, devices)
                continue
            idle = self._idle_context(slot, devices, ended) if devices and contexts else None
            if idle is None:
                return measured, evicted
            other, other_lane, _row, executor = idle
            ended.add(other)
            evicted += [other] if other not in evicted else []
            self.note(
                "residency",
                f"{other_lane.lane_id}: {other!r} (epoch {executor.epoch} pid {executor.pid}) is "
                f"idle and holds only its device context; ended for {slot!r} on {why}",
            )
            self.reclaim(other, executor, f"idle context reclaimed for {slot!r} on {why}")

    def _idle_context(self, slot: str, short: set[int], ended: set[str]) -> Victim | None:
        """The least recently used idle process on `short` that holds a device context and
        nothing a cut could still take. Never `slot`, never a tenant in its turn or mid-call."""
        idle: list[Victim] = []
        for other, lane, row in self.tenants(short):
            executor = row.supervision.current if row.supervision is not None else None
            if other == slot or other in ended or other in self.active or executor is None:
                continue
            if executor.started and executor.alive() and not executor.busy():
                idle.append((other, lane, row, executor))
        return min(idle, key=lambda item: item[2].last_used, default=None)

    def _victim(self, slot: str, short: set[int], tried: set[str]) -> Victim | None:
        """The next tenant to give room on `short`: a dead one, else the least recently used
        idle one holding weights there, else a busy one outside its turn (a load, an attempt's
        tail) that can be cut mid-call through its budget cell. Never `slot`, never one in its
        turn (a computing stage keeps its room), never twice."""
        dead: list[Victim] = []
        idle: list[Victim] = []
        busy: list[Victim] = []
        for other, lane, row in self.tenants(short):
            executor = row.supervision.current if row.supervision is not None else None
            if other == slot or other in tried or executor is None:
                continue
            if executor.poisoned or not executor.alive():
                dead.append((other, lane, row, executor))
            elif other in self.active or not holds(row, executor, short & set(lane.ordinals)):
                continue
            elif not executor.busy():
                idle.append((other, lane, row, executor))
            elif executor.cell is not None:
                busy.append((other, lane, row, executor))
        for tier in (dead, idle, busy):
            if tier:
                return min(tier, key=lambda item: item[2].last_used)
        return None

    def _cut(self, victim: Victim, slot: str, why: str, short: set[int]) -> None:
        other, _lane, _row, executor = victim
        if executor.poisoned or not executor.alive():
            dead = executor.poisoned or "exited"
            self.reclaim(other, executor, f"reclaimed for {slot!r} on {why}: {dead}")
        elif executor.busy() and executor.cell is not None:
            self._cut_running(victim, slot, why)
        elif plane_capable(executor):
            self._cut_plane(victim, slot, why)
        else:
            self._vacate(victim, slot, why, short)

    def unmap(self, tenant: str, why: str) -> None:
        """Unmap every device byte of `tenant`'s plane (they stay pinned): what a refused load
        left mapped would otherwise count against its own next try."""
        row = self._row(tenant)
        executor = row.executor() if row is not None else None
        if row is None or executor is None or not plane_capable(executor) or executor.busy():
            return
        lane = next(lane for lane in self.lanes if lane.ordinals and tenant in lane.rows)
        self._cut_plane((tenant, lane, row, executor), tenant, why)

    def _cut_running(self, victim: Victim, slot: str, why: str) -> None:
        """A busy tenant outside its turn gives its device bytes at its next block boundary,
        down to what its open stages need. Its call goes on; its next grant restores it."""
        other, lane, row, executor = victim
        applied = executor.cut(0)
        if applied is None:  # its call ended first: idle now, cut as one
            if executor.alive() and not executor.busy():
                self._cut_plane(victim, slot, why)
            return
        if row.ledger.plane is not None:
            committed = min(row.ledger.plane.committed_bytes, applied)
            row.ledger.plane = msgspec.structs.replace(row.ledger.plane, committed_bytes=committed)
        self.note(
            "residency",
            f"{lane.lane_id}: {other!r} (epoch {executor.epoch} pid {executor.pid}) cut mid-call "
            f"to {applied} B for {slot!r} on {why}",
        )

    def _cut_plane(self, victim: Victim, slot: str, why: str) -> None:
        """Budget 0: the plane unmaps at once and keeps the bytes in its host tier (lowered to
        its LRU share of the host) or the page cache. One that refuses or dies is ended."""
        other, lane, row, executor = victim
        pinned = self._pinned(other, first=False)
        held = row.ledger.plane.pinned_budget_bytes if row.ledger.plane else -1
        command = executor_commands.Budget(
            vram_bytes=0, pinned_bytes=pinned if 0 <= pinned < held else -1
        )
        try:
            with executor.watched("budget"):
                raw = executor.call(command, timeout=None)
        except ExecutorGone as exc:
            raw = {"ok": False, "code": "executor_gone", "detail": str(exc)}
        reply, _ = tolerant.read(raw, _BudgetReply, ("code", "detail", "freed_bytes", "plane"))
        if not reply.ok:
            self.note(
                "residency",
                f"{other!r} could not lower its budget ({reply.code}: {reply.detail[:200]}); "
                "ending it instead",
            )
            self.reclaim(other, executor, f"evicted for {slot!r} on {why}")
            return
        if reply.plane is not None:
            row.ledger.plane = reply.plane
        # Its cut released torch's cache too: no slack is left for its next grant to count.
        row.ledger.device_reserved = row.ledger.device_allocated
        self.note(
            "residency",
            f"{lane.lane_id}: {other!r} (epoch {executor.epoch} pid {executor.pid}) cut to 0 B "
            f"for {slot!r} on {why}: {reply.freed_bytes} B unmapped"
            + (f", pinned tier {command.pinned_bytes} B" if command.pinned_bytes >= 0 else ""),
        )

    def _vacate(self, victim: Victim, slot: str, why: str, short: set[int]) -> None:
        """An executor before the plane: `Vacate` keeps the process warm and moves its weights
        off, only on `short` cards when it is a group that can say which ranks. One that
        refuses or dies is ended: death frees all."""
        other, lane, row, executor = victim
        on = [o for o in lane.ordinals if o in short]
        ranks = (
            tuple(lane.ordinals.index(o) for o in on)
            if lane.group and "vacate_ranks" in executor.hello.get("memory", ())
            else ()
        )
        try:
            with executor.watched("vacate"):
                reply = executor.call(executor_commands.Vacate(ranks=ranks), timeout=None)
        except ExecutorGone as exc:
            reply = {"ok": False, "code": "executor_gone", "detail": str(exc)}
        if not reply.get("ok"):
            self.note(
                "residency",
                f"{other!r} could not vacate ({reply.get('code')}: "
                f"{str(reply.get('detail', ''))[:200]}); ending it instead",
            )
            self.reclaim(other, executor, f"evicted for {slot!r} on {why}")
            return
        current = row.supervision is not None and row.supervision.current is executor
        freed = (
            row.ledger.observe_vacate(reply, why) if current and (not ranks or 0 in ranks) else -1
        )
        row.emptied.update(on if ranks else lane.ordinals)
        self.note(
            "residency",
            f"{lane.lane_id}: {other!r} (epoch {executor.epoch} pid {executor.pid}) vacated "
            + (f"{gpu_names([lane.entries[r] for r in ranks])} " if ranks else "")
            + f"for {slot!r} on {why}"
            + (f": {freed} B freed" if freed >= 0 else ""),
        )

    def _pinned(self, tenant: str, *, first: bool) -> int:
        """`tenant`'s share of the pinned host tier (`pinned_split`), most recently used first
        and `first` ahead of all; -1 when the host or its weights are unreadable."""
        return self._shares(tenant if first else "").get(tenant, -1)

    def _shares(self, first: str = "") -> dict[str, int]:
        """Every plane tenant's share of the machine's pinned total, most recently used first
        and `first` ahead of all. Empty when the host is unreadable."""
        memory = read_host_memory()
        if memory.available < 0:
            return {}
        rows = sorted(
            ((slot, row) for slot, _lane, row in self.tenants() if row.ledger.plane),
            key=lambda item: (item[0] != first, -item[1].last_used),
        )
        sizes = {slot: weights(row.ledger) for slot, row in rows}
        split = pinned_split(
            memory.available,
            [
                (slot, sizes[slot], row.ledger.plane.pinned_bytes if row.ledger.plane else 0)
                for slot, row in rows
            ],
            held=memory.shmem,
        )
        return {slot: share for slot, share in split.items() if sizes[slot] > 0}

    def shed(self, keep: str = "") -> None:
        """Hold the machine's pinned host tiers inside what the host has now: every idle
        tenant pinning more than its share is lowered to it (the plane punches its least
        recently used regions; page cache and disk stay beneath), then the tiers the Worker
        keeps are trimmed. Before a tenant's tier is raised (`keep`), at the end of every
        turn, and never a tenant in its turn."""
        shares = self._shares(keep)
        for slot, _lane, row in self.tenants():
            executor = row.executor()
            plane = row.ledger.plane
            share = shares.get(slot, -1)
            if (
                slot == keep
                or slot in self.active
                or executor is None
                or plane is None
                or not plane_capable(executor)
                or executor.busy()
                or not 0 <= share < plane.pinned_budget_bytes
            ):
                continue
            # its device budget stays what it is; only the pinned tier is lowered
            command = executor_commands.Budget(vram_bytes=plane.budget_bytes, pinned_bytes=share)
            try:
                with executor.watched("budget"):
                    self.observe(row, executor, executor.call(command, timeout=None))
            except ExecutorGone:
                continue  # its tier goes with its process
            except Exception as exc:  # the host tier is optional: never a turn's failure
                self.note("residency", f"{slot!r}: pinned tier not lowered: {exc}"[:400])
        if self.trim is not None:
            self.trim()

    def _lane(self, tenant: str, gpus: Sequence[int]) -> DeviceLane:
        ordinals = tuple(gpus)
        for lane in self.lanes:
            if lane.ordinals == ordinals and tenant in lane.rows:
                return lane
        raise LookupError(f"{tenant!r} holds no lane over {list(ordinals)}")

    def _row(self, tenant: str) -> LaneRow | None:
        return next(
            (lane.rows[tenant] for lane in self.lanes if lane.ordinals and tenant in lane.rows),
            None,
        )

    @staticmethod
    def _gpu(lane: DeviceLane, ordinal: int) -> str:
        return gpu_name(lane.entries, lane.ordinals.index(ordinal))
