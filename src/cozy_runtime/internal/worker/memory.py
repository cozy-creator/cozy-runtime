"""The machine's device memory policy: per-GPU, per-tenant budgets (weight-plane.md §4-5).

ONE manager per Worker decides; each executor's weight plane moves the bytes. A TENANT is one
placement's executor over its lane's devices; a rank group is one tenant with rank-symmetric
budgets. A tenant in its turn (`start` .. `end`) holds its devices; every other tenant there
is idle.

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

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager

import msgspec

from cozy_runtime.internal import accel, executor_commands, tolerant, weight_policy
from cozy_runtime.internal.accel import DeviceMemory
from cozy_runtime.internal.execution_evidence import gpu_name, gpu_names
from cozy_runtime.internal.executor_replies import AttemptReply, PlaneFacts
from cozy_runtime.internal.worker import child
from cozy_runtime.internal.worker.child import ExecutorGone
from cozy_runtime.internal.worker.lanes import DeviceLane, LaneRow, LaneSet
from cozy_runtime.internal.worker.ledger import Ledger, read_host_available

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


def pinned_split(available: int, tenants: Sequence[tuple[str, int, int]]) -> dict[str, int]:
    """Pinned host-tier budgets over `(tenant, weights, pinned now)`, most recently used first.

    Pinned memory is unreclaimable, so the tier takes at most half of what the host has (free
    plus what the tier already pins); the rest stays for processes and the page cache, the
    tier's own fallback. No tenant rises by more than half of what is free now.
    """
    free = max(available, 0)
    left = (free + sum(max(pinned, 0) for _, _, pinned in tenants)) // 2
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

    @contextmanager
    def starting(self, lane: DeviceLane, slot: str) -> Iterator[None]:
        """Admit a new device context under the same physical locks as a call: idle tenants
        of these devices are cut before CUDA or the rank group initializes."""
        with lane.device:
            self.make_room(lane, slot, None, "executor startup")
            yield

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
        pinned = self._pinned(tenant, first=True)
        current = row.ledger.plane.pinned_budget_bytes if row.ledger.plane else -1
        try:
            with executor.watched("prefill"):
                if pinned >= 0 and pinned != current:
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
        """The turn is over. Budgets stay; its bytes stay mapped until a tenant that can use
        them cuts it, least recently used first."""
        self.active.discard(tenant)

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
        self, lane: DeviceLane, slot: str, need: dict[int, int] | None, why: str
    ) -> dict[int, DeviceMemory]:
        """Cut idle tenants of `lane`'s devices until `need` is free on each (None: the whole
        device), for a load or an older executor that asks for room."""

        def short(o: int, memory: DeviceMemory) -> bool:
            return need is None or memory.free_bytes < need[o]

        measured, _ = self._give(lane, slot, short, why)
        if need is not None and (
            left := [o for o, m in measured.items() if m.state == "measured" and short(o, m)]
        ):
            self.note(
                "residency",
                f"{lane.lane_id}: no idle tenant is left to evict for {slot!r} on {why}: "
                + ", ".join(
                    f"{self._gpu(lane, o)} needs {need[o]} B free, has "
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

        read, evicted = self._give(lane, tenant, lambda o, m: want is None or usable(m) < want, why)
        if any(m.state != "measured" for m in read.values()):
            self.note("residency", f"{lane.lane_id}: unreadable; {tenant!r} derives its budget")
            return Budget(evicted=tuple(evicted))
        ledger.observe_devices(lane.device_facts(read))
        # The grant must hold the weights' floor itself: that is the refusal line.
        least = floor_bytes(ledger.layouts, stage.components)
        budget = {o: weight_policy.plane_budget(usable(m), grow) for o, m in read.items()}
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
        if evicted or banked is None:
            self.note(
                "residency",
                f"{lane.lane_id}: {tenant!r} starts {why} ({stage.entrypoint} {stage.cell or '-'}, "
                + (f"{grow} B of activations" if banked is not None else "activations unmeasured")
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

        read, evicted = self._give(lane, tenant, short, why)
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
        self, lane: DeviceLane, slot: str, short: Callable[[int, DeviceMemory], bool], why: str
    ) -> tuple[dict[int, DeviceMemory], list[str]]:
        """Cut idle tenants from `lane`'s short devices until none is short or nobody is left:
        one driver read, and one more after each cut."""
        tried: set[str] = set()
        evicted: list[str] = []
        while True:
            measured = lane.measure(self.kind)
            self.seen.update(measured)
            if any(m.state != "measured" for m in measured.values()):
                return measured, evicted
            devices = {o for o, m in measured.items() if short(o, m)}
            victim = self._victim(slot, devices, tried) if devices else None
            if victim is None:
                return measured, evicted
            tried.add(victim[0])
            evicted.append(victim[0])
            self._cut(victim, slot, why, devices)

    def _victim(self, slot: str, short: set[int], tried: set[str]) -> Victim | None:
        """The next tenant to give room on `short`: a dead one, else the least recently used
        idle one holding weights there. Never `slot`, never one in its turn, never twice."""
        dead: list[Victim] = []
        idle: list[Victim] = []
        for other, lane, row in self.tenants(short):
            executor = row.supervision.current if row.supervision is not None else None
            if other == slot or other in tried or executor is None:
                continue
            if executor.poisoned or not executor.alive():
                dead.append((other, lane, row, executor))
            elif other not in self.active and holds(row, executor, short & set(lane.ordinals)):
                idle.append((other, lane, row, executor))
        for tier in (dead, idle):
            if tier:
                return min(tier, key=lambda item: item[2].last_used)
        return None

    def _cut(self, victim: Victim, slot: str, why: str, short: set[int]) -> None:
        other, _lane, _row, executor = victim
        if executor.poisoned or not executor.alive():
            dead = executor.poisoned or "exited"
            self.reclaim(other, executor, f"reclaimed for {slot!r} on {why}: {dead}")
        elif plane_capable(executor):
            self._cut_plane(victim, slot, why)
        else:
            self._vacate(victim, slot, why, short)

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
        available = read_host_available()
        rows = sorted(
            ((slot, row) for slot, _lane, row in self.tenants() if row.ledger.plane),
            key=lambda item: (not (first and item[0] == tenant), -item[1].last_used),
        )
        sizes = {slot: weights(row.ledger) for slot, row in rows}
        split = pinned_split(
            available,
            [
                (slot, sizes[slot], row.ledger.plane.pinned_bytes if row.ledger.plane else 0)
                for slot, row in rows
            ],
        )
        return split[tenant] if available >= 0 and sizes.get(tenant, 0) > 0 else -1

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
