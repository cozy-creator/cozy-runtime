"""Every device-memory decision the Runtime makes, in one place.

The WORKER half (no torch) owns every tenant of every device: one placement's executor over
its lane's devices. The driver's numbers decide, capped by the configured budget. A call gets
room on every device it needs from idle tenants, least recently used first, dead ones first
of all; the tenant holding the devices' locks is never a victim.

The EXECUTOR half (`ComponentResidency`) owns one construction's components. A scope makes
its declared components resident; a component whose forward runs while it is absent is
staged first, so weights follow forwards. A component too large for the room left pages its
declared blocks through the forward. An out-of-memory error inside a component forward is
caught, the failed step's allocations are dropped, the plane descends one rung (room from
idle tenants, idle components, block paging, a split batch, a tiled decode) and runs THAT
forward again. Only when the lowest rung cannot fit does the attempt fail, with the numbers.
That is recovery inside one execution; a failed run is still never run again.
"""

from __future__ import annotations

import functools
import gc
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict

from cozy_runtime.author._errors import DEVICE_OOM, RuntimeFailure, is_device_oom
from cozy_runtime.internal import accel, fill, proctree
from cozy_runtime.internal.accel import DeviceMemory
from cozy_runtime.internal.execution_evidence import gpu_name, gpu_names

if TYPE_CHECKING:
    from cozy_runtime.internal.worker import child
    from cozy_runtime.internal.worker.lanes import DeviceLane, LaneRow, LaneSet

# ---------------------------------------------------------------------------- out of memory


_UNITS = {"b": 1, "kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30, "tib": 1 << 40}


def requested_bytes(exc: BaseException) -> int:
    """What the failed allocation asked for, from the allocator's own message; 0 unknown."""
    match = re.search(r"Tried to allocate (\d+(?:\.\d+)?) ([KMGT]?i?B)", str(exc))
    if match is None:
        return 0
    return int(float(match.group(1)) * _UNITS.get(match.group(2).lower(), 1))


def recover(torch: Any, kind: str, exc: BaseException) -> None:
    """Drop what a failed step still pins and hand the cache back. The traceback's frames
    hold the failed step's activations until they are cleared."""
    exc.__traceback__ = None
    for linked in (exc.__cause__, exc.__context__):
        if linked is not None:
            linked.__traceback__ = None
    gc.collect()
    if kind == "cuda":
        # An asynchronous allocation failure stays pending on the context until drained.
        try:
            torch.cuda.synchronize()
        except Exception as pending:
            if not is_device_oom(pending):
                raise
    accel.release_cached(torch, kind)


# ---------------------------------------------------------------------------- block paging


#: One tensor a unit moves: its state-dict key, the module holding it, and its attribute.
Site = tuple[str, Any, str]


@dataclass(frozen=True)
class PageUnit:
    """A set of tensor sites that arrive and leave together."""

    path: str
    sites: tuple[Site, ...]
    nbytes: int

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for key, _, _ in self.sites)

    def live(self, component: str) -> dict[str, Any]:
        return {
            f"{component}.{key}": getattr(module, attribute).detach()
            for key, module, attribute in self.sites
        }


@dataclass(frozen=True)
class BlockLayout:
    common: PageUnit
    blocks: tuple[PageUnit, ...]

    @property
    def working_bytes(self) -> int:
        return self.common.nbytes + max(block.nbytes for block in self.blocks)

    @property
    def total_bytes(self) -> int:
        return self.common.nbytes + sum(block.nbytes for block in self.blocks)

    def document(self) -> dict[str, int]:
        return {
            "total_bytes": self.total_bytes,
            "common_bytes": self.common.nbytes,
            "largest_block_bytes": max(block.nbytes for block in self.blocks),
            "working_bytes": self.working_bytes,
            "blocks": len(self.blocks),
        }


def partition(root: Any) -> BlockLayout | None:
    """Blocks are the REPEATED instances of the declared no-split classes, innermost where
    they nest, and they page parameters only: buffers and one-off modules stay common, so a
    device read of an embedding or an index table outside the forward stays valid.

    Storage shared across units, or state-dict keys that are not registered destinations,
    refuse: those units cannot move independently."""
    classes = getattr(root, "_no_split_modules", None)
    if not classes:
        return None
    if not isinstance(classes, (tuple, list, set, frozenset)) or any(
        not isinstance(name, str) for name in classes
    ):
        raise ValueError("no-split module declarations must contain class names")
    modules = list(root.named_modules(remove_duplicate=False))
    declared = [path for path, module in modules if path and type(module).__name__ in classes]
    innermost = [a for a in declared if not any(b.startswith(a + ".") for b in declared)]
    counts: dict[str, int] = {}
    for path in innermost:
        kind = type(root.get_submodule(path)).__name__
        counts[kind] = counts.get(kind, 0) + 1
    blocks = [path for path in innermost if counts[type(root.get_submodule(path)).__name__] > 1]
    if not blocks:
        return None
    sites: dict[str, list[Site]] = {"": [], **{path: [] for path in blocks}}
    sizes = dict.fromkeys(sites, 0)
    storages: dict[object, str] = {}
    for path, module in modules:
        block = next((b for b in blocks if path == b or path.startswith(b + ".")), "")
        tensors = [(name, tensor, block) for name, tensor in module._parameters.items()]
        tensors += [(name, tensor, "") for name, tensor in module._buffers.items()]
        for name, tensor, unit in tensors:
            if tensor is None:
                continue
            if tensor.numel():
                prior = storages.setdefault(tensor.untyped_storage(), unit)
                if prior != unit:
                    raise ValueError("one tensor storage is shared across paging units")
            sizes[unit] += int(tensor.numel() * tensor.element_size())
            if name not in module._non_persistent_buffers_set:
                sites[unit].append((f"{path}.{name}" if path else name, module, name))
    keys = {key for group in sites.values() for key, _, _ in group}
    if keys != set(root.state_dict()):
        raise ValueError("custom state-dict keys do not map to registered paging destinations")
    units = {path: PageUnit(path, tuple(sites[path]), sizes[path]) for path in sites}
    return BlockLayout(units[""], tuple(units[path] for path in blocks))


# ---------------------------------------------------------------------------- the worker half

#: How often the driver is read while a call runs, to see its peak. A sampling cadence;
#: nothing waits on it.
SAMPLE_SECONDS = 0.01

#: Bytes per ordinal a tenant needs free, or None: the whole device (never measured).
Need = dict[int, int] | None
#: An entrypoint and its request's shape cell (`plan.shape_cell`).
Shape = tuple[str, str]


@dataclass(slots=True)
class Watch:
    """One transition's device picture: free bytes before, the lowest seen, and after."""

    before: dict[int, int]
    low: dict[int, int]
    after: dict[int, int] = field(default_factory=dict)


def axes(cell: str) -> dict[str, int] | None:
    try:
        return {
            axis: int(value)
            for axis, value in (part.split("=", 1) for part in cell.split(",") if part != "-")
        }
    except ValueError:
        return None


def bound(shape: Shape, peaks: Mapping[Shape, dict[int, int]]) -> dict[int, int] | None:
    """The least peak among measured shapes of the same entrypoint at least as large as
    `shape` on every axis."""
    entrypoint, cell = shape
    mine = axes(cell) if cell else None
    if mine is None:
        return None
    covering = [
        peak
        for (other_entrypoint, other), peak in peaks.items()
        if other_entrypoint == entrypoint
        and (theirs := axes(other)) is not None
        and theirs.keys() == mine.keys()
        and all(theirs[axis] >= value for axis, value in mine.items())
    ]
    if not covering:
        return None
    return {o: min(peak.get(o, 0) for peak in covering) for o in covering[0]}


def _descends_from(pid: int, ancestor: int) -> bool:
    seen = 0
    while pid > 1 and seen < 64:
        if pid == ancestor:
            return True
        try:
            with open(f"/proc/{pid}/stat", "rb") as stat:
                pid = int(stat.read().rsplit(b")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        seen += 1
    return False


class MemoryManager:
    """Admission, eviction and reclaim over every tenant of every device."""

    def __init__(
        self,
        lanes: LaneSet,
        note: Callable[[str, str], None],
        end: Callable[[str, child.Executor, str], bool],
        vacate: Callable[[child.Executor, tuple[int, ...]], dict[str, object]],
        budget: Mapping[str, int] | None = None,
    ) -> None:
        self.lanes = lanes
        self.note = note
        #: kill and reap one tenant's executor now (True once proved gone)
        self.end = end
        #: one Vacate round trip to an executor; `executor_gone` in the reply when it died
        self.vacate = vacate
        #: GPU entry (or "*") -> the bytes this Runtime may hold there
        self.budget = dict(budget or {})
        self.kind = accel.host_backend_family()

    def tenants(self, ordinals: set[int]) -> list[tuple[str, DeviceLane, LaneRow]]:
        found: dict[str, tuple[str, DeviceLane, LaneRow]] = {}
        for lane in self.lanes.lanes:
            if ordinals & set(lane.ordinals):
                for slot, row in lane.rows.items():
                    if row.model_bearing:
                        found.setdefault(slot, (slot, lane, row))
        return list(found.values())

    # ------------------------------------------------------------------ measurement

    def measure(self, lane: DeviceLane) -> dict[int, DeviceMemory]:
        """Each device of `lane` as the driver reports it, with the budget applied: the free
        bytes this Runtime may still take there."""
        measured = lane.measure(self.kind)
        if not self.budget:
            return measured
        for ordinal, entry in zip(lane.ordinals, lane.entries, strict=True):
            cap = self.budget.get(entry, self.budget.get("*"))
            memory = measured[ordinal]
            if cap is None or memory.state != "measured":
                continue
            used = self.runtime_bytes(entry, memory)
            measured[ordinal] = replace(
                memory,
                free_bytes=max(min(memory.free_bytes, cap - used), 0),
                total_bytes=min(memory.total_bytes, cap),
            )
        return measured

    def runtime_bytes(self, entry: str, memory: DeviceMemory) -> int:
        """What this Runtime's processes hold on the GPU: the driver's per-process rows of
        this worker's descendants, or everything used where the driver's pids are not this
        namespace's (a pod, where the card is the Runtime's alone)."""
        table = accel.device_processes(entry)
        worker = os.getpid()
        if table:
            mine = [used for pid, used in table.items() if _descends_from(pid, worker)]
            if mine or any(os.path.exists(f"/proc/{pid}") for pid in table):
                return sum(max(used, 0) for used in mine)
        return memory.total_bytes - memory.free_bytes

    # ------------------------------------------------------------------ admission

    def need(self, lane: DeviceLane, slot: str, shape: Shape) -> Need:
        row = lane.row(slot)
        peak = row.peak.get(shape)
        if peak is None:
            peak = bound(shape, row.peak)
        if peak is None:
            return None
        return {o: max(peak.get(o, 0) - row.held.get(o, 0), 0) for o in lane.ordinals}

    def admit(self, lane: DeviceLane, slot: str, shape: Shape, why: str) -> bool:
        """Make room for one call of `slot` on every device of its lane (the caller holds
        them). True when its measured need fits everywhere, or when it was never measured and
        every idle tenant has gone."""
        lane.touch(slot)
        need = self.need(lane, slot, shape)
        evicted: list[str] = []
        measured = self.make_room(lane, slot, need, why, evicted)
        fits = need is None or all(
            measured[o].state == "measured" and measured[o].free_bytes >= n for o, n in need.items()
        )
        if evicted or need is None:
            self.note(
                "residency",
                f"{lane.lane_id}: admitted {slot!r} for {why} ({shape[0]} {shape[1] or '-'}): "
                + (
                    "never measured, so it has the whole of its GPUs"
                    if need is None
                    else "needs "
                    + ", ".join(
                        f"{need[o]} B on {gpu_name(lane.entries, i)}"
                        for i, o in enumerate(lane.ordinals)
                    )
                )
                + f"; evicted {evicted or 'nobody'}; free "
                + ", ".join(
                    f"{measured[o].free_bytes} B on {gpu_name(lane.entries, i)}"
                    for i, o in enumerate(lane.ordinals)
                )
                + ("" if fits else "; it runs staged in what is left"),
            )
        return fits

    def free_bytes(self, lane: DeviceLane) -> dict[int, int]:
        """What each device of `lane` has free for this Runtime now; -1 unreadable."""
        return {
            o: m.free_bytes if m.state == "measured" else -1 for o, m in self.measure(lane).items()
        }

    def make_room(
        self, lane: DeviceLane, slot: str, need: Need, why: str, evicted: list[str] | None = None
    ) -> dict[int, DeviceMemory]:
        """Evict idle tenants from `lane`'s short devices until `need` fits on every one:
        dead tenants first, then weights, then whole idle processes (their CUDA context,
        NCCL and kernel bytes), least recently used first."""
        tried: set[str] = set()
        ended: set[str] = set()
        while True:
            measured = self.measure(lane)
            if any(memory.state != "measured" for memory in measured.values()):
                return measured
            short = {o for o in lane.ordinals if need is None or measured[o].free_bytes < need[o]}
            if not short:
                return measured
            victim = self._victim(slot, short, tried)
            if (
                victim is None
                and need is not None
                and all(need[o] <= measured[o].total_bytes for o in short)
            ):
                # Weights are all gone and it could still fit an empty card: idle processes
                # hold the rest (contexts, NCCL, kernels), and only their death returns it.
                victim = self._process(slot, short, ended)
                if victim is not None:
                    other, _, row, executor = victim
                    ended.add(other)
                    if evicted is not None:
                        evicted.append(other)
                    if self.end(other, executor, f"its process ended for {slot!r} on {why}"):
                        row.held.clear()
                        row.emptied.clear()
                    continue
            if victim is None:
                if need is not None:
                    self.note(
                        "residency",
                        f"{lane.lane_id}: no idle tenant is left to evict for {slot!r} on {why}: "
                        + ", ".join(
                            f"{gpu_name(lane.entries, lane.ordinals.index(o))} needs {need[o]} B "
                            f"free, has {measured[o].free_bytes} of {measured[o].total_bytes}"
                            for o in sorted(short)
                        ),
                    )
                return measured
            other, other_lane, row, executor = victim
            tried.add(other)
            if evicted is not None:
                evicted.append(other)
            if executor.poisoned or not executor.alive():
                why_dead = executor.poisoned or "exited"
                if self.end(other, executor, f"reclaimed for {slot!r} on {why}: {why_dead}"):
                    row.held.clear()
                continue
            self._evict(other, other_lane, row, executor, slot, why, measured, short)

    def _victim(
        self, slot: str, short: set[int], tried: set[str]
    ) -> tuple[str, DeviceLane, LaneRow, child.Executor] | None:
        """A dead tenant, else the least recently used idle one holding bytes on `short`."""
        dead = []
        idle = []
        for other, lane, row in self.tenants(short):
            executor = row.supervision.current if row.supervision is not None else None
            if other == slot or other in tried or executor is None:
                continue
            if executor.poisoned or not executor.alive():
                dead.append((other, lane, row, executor))
            elif any(row.held.get(o, 0) for o in short) or (not row.held and row.resident_bytes()):
                idle.append((other, lane, row, executor))
        for tier in (dead, idle):
            if tier:
                return min(tier, key=lambda item: item[2].last_used)
        return None

    def _process(
        self, slot: str, short: set[int], ended: set[str]
    ) -> tuple[str, DeviceLane, LaneRow, child.Executor] | None:
        """The least recently used idle process with a context on `short`: its weights are
        already gone, and only its death returns the rest."""
        idle = []
        for other, lane, row in self.tenants(short):
            executor = row.supervision.current if row.supervision is not None else None
            if other == slot or other in ended or executor is None or not executor.alive():
                continue
            idle.append((other, lane, row, executor))
        return min(idle, key=lambda item: item[2].last_used) if idle else None

    def _evict(
        self,
        victim: str,
        lane: DeviceLane,
        row: LaneRow,
        executor: child.Executor,
        slot: str,
        why: str,
        before: Mapping[int, DeviceMemory],
        short: set[int],
    ) -> None:
        """Vacate one idle tenant, only on `short` cards when its group can say which. An
        executor that refuses or dies is ended instead: death frees all."""
        on = [o for o in lane.ordinals if o in short]
        ranks = (
            tuple(lane.ordinals.index(o) for o in on)
            if lane.group and "vacate_ranks" in executor.hello.get("memory", ())
            else ()
        )
        reply = self.vacate(executor, ranks)
        if not reply.get("ok"):
            self.note(
                "residency",
                f"{victim!r} could not vacate ({reply.get('code')}: "
                f"{str(reply.get('detail', ''))[:200]}); ending it instead",
            )
            if self.end(victim, executor, f"evicted for {slot!r} on {why}"):
                row.held.clear()
            return
        current = row.supervision is not None and row.supervision.current is executor
        if current and (not ranks or 0 in ranks):
            row.ledger.observe_vacate(reply, why)
        emptied = on if ranks else list(lane.ordinals)
        for o in emptied:
            row.held.pop(o, None)
        row.emptied.update(emptied)
        after = self.measure(lane)
        self.note(
            "residency",
            f"{lane.lane_id}: {victim!r} (epoch {executor.epoch} pid {executor.pid}) vacated "
            + (f"{gpu_names([lane.entries[r] for r in ranks])} " if ranks else "")
            + f"for {slot!r} on {why}: "
            + ", ".join(
                f"{gpu_name(lane.entries, i)} "
                f"{before[o].free_bytes} -> {after[o].free_bytes} B free"
                for i, o in enumerate(lane.ordinals)
                if o in before and after[o].state == "measured"
            ),
        )

    @contextmanager
    def watch(self, lane: DeviceLane, *, sample: bool = True) -> Iterator[Watch]:
        """`lane`'s devices before and after the body; sampled while it runs, the lowest
        free bytes are its peak."""
        before = {
            o: m.free_bytes for o, m in lane.measure(self.kind).items() if m.state == "measured"
        }
        seen = Watch(before=before, low=dict(before))
        stop = threading.Event()

        def sampling() -> None:
            while not stop.wait(SAMPLE_SECONDS):
                for o, memory in lane.measure(self.kind).items():
                    if o in seen.low and memory.state == "measured":
                        seen.low[o] = min(seen.low[o], memory.free_bytes)

        sampler = threading.Thread(target=sampling, name=f"memory:{lane.lane_id}", daemon=True)
        if sample and before:
            sampler.start()
        try:
            yield seen
        finally:
            stop.set()
            if sampler.is_alive():
                sampler.join()
            seen.after = {
                o: m.free_bytes for o, m in lane.measure(self.kind).items() if m.state == "measured"
            }
            for o, free in seen.after.items():
                if o in seen.low:
                    seen.low[o] = min(seen.low[o], free)

    def moved(self, lane: DeviceLane, slot: str, seen: Watch) -> None:
        """A load, restore or unload `slot` made under its lock: bank what it took."""
        row = lane.row(slot)
        for o, free in seen.before.items():
            if o in seen.after:
                row.held[o] = max(row.held.get(o, 0) + free - seen.after[o], 0)

    def settle(
        self, lane: DeviceLane, slot: str, shape: Shape, seen: Watch, *, ok: bool, forget: bool
    ) -> None:
        """Bank one call. A success raises its shape's peak; a failure on the device forgets
        it, so the next call of that shape has the whole device again."""
        row = lane.row(slot)
        if ok:
            peak = row.peak.setdefault(shape, {})
            for o, free in seen.before.items():
                peak[o] = max(peak.get(o, 0), row.held.get(o, 0) + free - seen.low.get(o, free))
        elif forget:
            row.peak.pop(shape, None)
        if ok:
            row.emptied.clear()
        self.moved(lane, slot, seen)


# ---------------------------------------------------------------------------- the executor half


class Shortfall(TypedDict):
    """The structured capacity fact a refusal carries to the terminal."""

    resource: str
    scope: str
    needed_bytes: int
    available_bytes: int
    evidence_class: str
    request_shape: str


class VacateReport(TypedDict):
    freed: dict[str, int]
    held: dict[str, str]
    freed_bytes: int
    allocator_delta_bytes: NotRequired[int]
    vacate_ms: NotRequired[float]


class RestoreReport(TypedDict):
    restored: dict[str, int]
    held: dict[str, str]
    restored_bytes: int
    allocator_delta_bytes: NotRequired[int]
    restore_ms: NotRequired[float]
    fill_ms: NotRequired[int]
    legs: NotRequired[dict[str, int]]
    read_bytes: NotRequired[int]


class ActiveScope(TypedDict):
    method: str
    components: list[str]


class FenceState(TypedDict):
    passed: bool
    last_used: int


class ResidencyDocument(TypedDict):
    """One construction's residency facts, as the executor reports them."""

    placement: str
    headroom_bytes: int
    scope_headroom_bytes: dict[str, int]
    measured_scopes: list[str]
    admissions: int
    stages: int
    evictions: int
    attempt_stages: int
    attempt_evictions: int
    attempt_staged_ms: float
    attempt_evicted_ms: float
    page_bytes: int
    attempt_page_bytes: int
    paging: dict[str, dict[str, int]]
    attempt_activation_peak_bytes: int
    attempt_activation_peaks: dict[str, int]
    attempt_absolute_peak_bytes: int
    allocator_delta_bytes: int
    staged_ms: float
    evicted_ms: float
    resident: dict[str, int]
    evicted: dict[str, int]
    hosted: list[str]
    active: ActiveScope | None
    fences: dict[str, FenceState]
    poisoned: str
    log: list[fill.StageEvent]
    refusals: list[str]
    recoveries: list[str]
    cap_bytes: int
    kept_bytes: int


class ConstructionRow(TypedDict):
    key: str
    state: str
    resident_bytes: int
    last_used_ns: int
    active: bool
    parked: NotRequired[list[str]]


class ResidencyRefusal(RuntimeFailure):
    """The declared set cannot be admitted. Carries the structured shortfall."""

    def __init__(self, code: str, detail: str, shortfall: Shortfall | None = None) -> None:
        super().__init__(detail, code=code)
        self.detail = detail
        self.shortfall = shortfall


class _Event(Protocol):
    def query(self) -> bool: ...


class _Tensor(Protocol):
    shape: tuple[int, ...]
    dtype: object
    _version: int

    def is_inference(self) -> bool: ...

    def dim(self) -> int: ...


@dataclass(slots=True)
class Fence:
    """One component's completion fence and LRU stamp."""

    event: _Event | None = None
    last_used: int = 0

    def passed(self) -> bool:
        return self.event is None or bool(self.event.query())


#: A capacity outcome: the attempt could not fit, and nothing in the process is broken.
CAPACITY_CODES = frozenset({DEVICE_OOM, "device_shortfall"})

#: The rungs a component forward descends, cheapest first. `room`: idle tenants of the
#: device give theirs back; `evict`: this process's idle components leave; `page`: the
#: component itself pages its blocks; `split`: the leading batch runs in halves.
FORWARD_RUNGS = ("room", "evict", "page", "split")
#: A decode or encode slices and tiles before paging: its work, not its weights, is large.
CODEC_RUNGS = ("room", "evict", "slice", "tile", "page")


@dataclass(slots=True)
class Op:
    """What one kind of call (component, argument shapes) needed: the rungs that made it
    fit and the most the allocator held during it."""

    rungs: tuple[str, ...] = ()
    peak: int = 0
    #: the leading batch of the call, and how many pieces it runs in
    batch: int = 0
    splits: int = 1
    #: the codec tile the call last fit in: attribute -> value
    tile: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ComponentResidency:
    """The component-use contract's runtime body: admit, stage, page, fence, recover."""

    backend: fill.StreamingFillBackend
    torch: Any
    #: the worker's placement label; the plane decides residency itself, per call
    placement: str = "all_resident"
    #: the attempt's measured activation envelope, and per authored method
    headroom: int = 0
    scope_headrooms: dict[str, int] = field(default_factory=dict)
    measured: frozenset[str] = frozenset()
    attempt_used: set[str] = field(default_factory=set)
    previous_used: frozenset[str] = frozenset()
    #: components another rank of the group keeps resident
    hosted: frozenset[str] = frozenset()
    active: tuple[str, frozenset[str]] | None = None
    spare: bool = False
    fences: dict[str, Fence] = field(default_factory=dict)
    #: set on an accounting violation or a fill that failed for a reason other than memory
    poisoned: str = ""
    clock: int = 0
    admissions: int = 0
    stages: int = 0
    evictions: int = 0
    staged_ms: float = 0.0
    evicted_ms: float = 0.0
    refusals: list[str] = field(default_factory=list)
    recoveries: list[str] = field(default_factory=list)
    last_shortfall: Shortfall | None = None
    allocator_delta_bytes: int = 0
    attempt_stages: int = 0
    attempt_evictions: int = 0
    attempt_staged_ms: float = 0.0
    attempt_evicted_ms: float = 0.0
    scope_baseline_bytes: int = -1
    attempt_activation_peaks: dict[str, int] = field(default_factory=dict)
    attempt_absolute_peak_bytes: int = 0
    page_call: tuple[str, PageUnit] | None = None
    page_weight_delta: int = 0
    page_autocast_cache: bool | None = None
    page_bytes: int = 0
    attempt_page_bytes: int = 0
    #: asks the worker for allocatable bytes this process cannot free (idle co-tenants)
    room: Callable[[int], None] | None = None
    #: False in a rank group: a sharded forward cannot run again on one rank alone
    recoverable: bool = True
    #: the allocator bytes this process may hold, or 0 for the whole device
    cap: int = 0
    #: (component, argument shapes) -> what it needed
    ops: dict[tuple[str, tuple[object, ...]], Op] = field(default_factory=dict)
    #: the components whose forward is on the stack now: never victims
    running: list[str] = field(default_factory=list)
    windows: list[list[int]] = field(default_factory=list)
    #: host bytes holding paged units' weights
    kept_bytes: int = 0
    #: the order paged blocks ran in: (component, block) -> the block after it
    next_unit: dict[tuple[str, str], str] = field(default_factory=dict)
    last_unit: dict[str, str] = field(default_factory=dict)
    copy_stream: Any = None

    def __post_init__(self) -> None:
        for name, layout in self.backend.layouts.items():
            root = self.backend.components.get(name) or self.backend.parked[name]
            for unit in layout.blocks:
                block = root.get_submodule(unit.path)
                self.backend.paging_hooks.extend(
                    (
                        block.register_forward_pre_hook(
                            functools.partial(self._page_enter, name, unit)
                        ),
                        block.register_forward_hook(
                            functools.partial(self._page_leave, name, unit), always_call=True
                        ),
                    )
                )
        for name, member in {**self.backend.components, **self.backend.parked}.items():
            if hasattr(member, "_call_impl"):
                self._guard(name, member)

    @property
    def kind(self) -> str:
        return str(self.backend.device.type)

    # ------------------------------------------------------------------ the contract

    def open_attempt(
        self,
        placement: str,
        headroom: int,
        scope_headrooms: dict[str, int] | None = None,
        measured: tuple[str, ...] | frozenset[str] = (),
    ) -> None:
        """Start one attempt's counters; the cumulative ones stay the generation's history."""
        self.placement = placement
        self.headroom = headroom
        self.scope_headrooms = dict(scope_headrooms or {})
        self.measured = frozenset(measured)
        if self.attempt_used:
            self.previous_used = frozenset(self.attempt_used)
        self.attempt_used = set()
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
        self.recoveries.clear()

    def admit(self, method: str, components: tuple[str, ...]) -> None:
        """Make the declared set resident before the method body runs, paging what does not
        fit whole, or refuse typed once nothing is left to give."""
        self.admissions += 1
        self._require_clean(method)
        if self.active is not None:
            self._fault("scope_overlap", f"{method}() entered while {self.active[0]}() is active")
            self.active = None
        local = tuple(name for name in components if name not in self.hosted)
        if self.spare:
            self._require_spare_room(local, method)
        if local:
            self._make_headroom(local, method)
        headroom = self.scope_headrooms.get(method, self.headroom)
        for name in local:
            self._touch(name)
            if name not in self.backend.components:
                self._stage(name, components, headroom, method)
        self._fold()
        self.scope_baseline_bytes = self._allocated()
        self.windows.append([0])
        self.active = (method, frozenset(components))
        self.attempt_used.update(components)

    def release(self, method: str, components: tuple[str, ...]) -> None:
        """Record the completion fence and close the scope. Nothing is evicted here."""
        self._require_clean(method)
        if self.active is None:
            self._fault("scope_over_release", f"{method}() released with no scope active")
        elif self.active != (method, frozenset(components)):
            self._fault("scope_release_mismatch", f"{method}() released {self.active[0]}()")
        peak = max(self._fold(), *(self.windows.pop() if self.windows else [0]))
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

    # ------------------------------------------------------------------ staging

    def _stage(
        self, name: str, declared: tuple[str, ...], headroom: int, method: str
    ) -> None:
        """Bring one absent component in: whole when it fits beside `headroom` after idle
        components and co-tenants gave room, else its common weights with blocks paged."""
        if name not in self.backend.paging:
            whole = self.backend.stage_bytes(name, headroom)
            self._free(whole, declared)
            if not self._fits(whole) and self.room is not None:
                self.room(whole - self._allocatable())
            if not self._fits(whole) and name in self.backend.layouts:
                self.backend.paging[name] = self.backend.layouts[name]
                self.recoveries.append(f"{name}: pages its blocks (whole needs {whole} B)")
        need = self.backend.stage_bytes(name, headroom)
        self._free(need, declared)
        if not self._fits(need) and self.room is not None:
            self.room(need - self._allocatable())
        if not self._fits(need):
            refusal = self._shortfall(method, name, headroom, "nothing left to give before staging")
            self.last_shortfall = refusal.shortfall
            self.refusals.append(refusal.detail[:400])
            raise refusal
        started = time.perf_counter()
        allocated = self._allocated()
        try:
            self.backend.stage(name)
        except fill.FillRefusal as exc:
            self.refusals.append(f"{exc.code}: {exc}"[:400])
            raise self._poison(exc.code, f"staging {name!r} failed: {exc}") from exc
        except Exception as exc:
            if not is_device_oom(exc):
                raise
            self.backend.abandon(name)
            recover(self.torch, self.kind, exc)
            refusal = self._shortfall(method, name, headroom, f"staging ran out: {exc}")
            self.last_shortfall = refusal.shortfall
            self.refusals.append(refusal.detail[:400])
            raise refusal from None
        self.stages += 1
        self.attempt_stages += 1
        self.allocator_delta_bytes += self._allocated() - allocated
        took = (time.perf_counter() - started) * 1000
        self.staged_ms += took
        self.attempt_staged_ms += took

    def _touch(self, name: str) -> None:
        self.clock += 1
        self.fences.setdefault(name, Fence()).last_used = self.clock

    # ------------------------------------------------------------------ recovery

    def _guard(self, name: str, member: object) -> None:
        """Wrap the component's forward (and a codec's decode/encode) without touching its
        signature: stage it if an eviction took it, and on an out-of-memory error descend
        one rung and run it again."""
        call = getattr(member, "_call_impl")
        setattr(member, "_call_impl", functools.partial(self._call, name, call, FORWARD_RUNGS))
        for method in ("decode", "encode"):
            original = getattr(member, method, None)
            if callable(original) and callable(getattr(member, "enable_tiling", None)):
                wrapped = functools.partial(self._call, name, original, CODEC_RUNGS)
                setattr(member, method, functools.update_wrapper(wrapped, original))

    def _call(
        self,
        name: str,
        call: Callable[..., object],
        ladder: tuple[str, ...],
        *args: object,
        **kwargs: object,
    ) -> object:
        if name in self.running or name in self.hosted or self.torch.is_grad_enabled():
            return call(*args, **kwargs)
        op = self.ops.setdefault((name, self._signature(args, kwargs)), Op())
        op.batch = self._batch(args, kwargs)
        self.running.append(name)
        try:
            while True:
                self._resident(name)
                for rung in op.rungs:
                    self._apply(name, rung, op)
                if op.peak:
                    self._clear(name, op.peak - self._allocated())
                versions = self._versions(args, kwargs)
                rng = self.torch.cuda.get_rng_state() if self.kind == "cuda" else None
                # A call that may not fit keeps its modules' counters (a step index a
                # forward advances), so running it again sees what the first run saw.
                armed = not op.peak or op.peak - self._allocated() > self._allocatable()
                counters = _counters(self.backend.components.get(name)) if armed else None
                self.windows.append([0])
                failure: Exception | None = None
                try:
                    result = self._split(call, args, kwargs, op)
                except Exception as exc:
                    if not self.recoverable or not is_device_oom(exc):
                        raise
                    failure = exc
                finally:
                    window = self.windows.pop()
                op.peak = max(op.peak, window[0], self._fold())
                if failure is None:
                    return result
                asked = requested_bytes(failure)
                detail = str(failure).splitlines()[0][:200]
                recover(self.torch, self.kind, failure)
                del failure
                if self._versions(args, kwargs) != versions:
                    raise self.torch.cuda.OutOfMemoryError(
                        f"{name}: out of memory after writing its inputs, so it cannot run "
                        f"again ({detail})"
                    )
                if rng is not None:
                    self.torch.cuda.set_rng_state(rng)
                for (module, attribute), value in (counters or {}).items():
                    setattr(module, attribute, value)
                step = self._descend(name, ladder, op, asked)
                if step is None:
                    raise self.torch.cuda.OutOfMemoryError(self._exhausted(name, op, asked, detail))
                self.recoveries.append(
                    f"{name}: out of memory (asked {asked} B, {self._allocatable()} B "
                    f"allocatable after recovery) -> {step}"
                )
                print(f"memory: {self.recoveries[-1]}", file=sys.stderr, flush=True)
        finally:
            self.running.pop()

    def _resident(self, name: str) -> None:
        """A component an eviction took comes back before its forward runs."""
        if name in self.backend.components or name not in self.backend.parked:
            return
        method = self.active[0] if self.active is not None else name
        declared = (*(self.active[1] if self.active is not None else ()), *self.running)
        self._touch(name)
        self._stage(name, declared, self.scope_headrooms.get(method, self.headroom), method)

    def _clear(self, name: str, need: int) -> None:
        """Room for a call whose peak is known, before it runs."""
        if need <= 0 or self._fits(need):
            return
        self._free(need, (name, *self.running))
        if not self._fits(need) and self.room is not None:
            self.room(need)

    def _descend(self, name: str, ladder: tuple[str, ...], op: Op, asked: int) -> str | None:
        """The first rung that still changes something, applied."""
        for rung in ladder:
            if self._apply(name, rung, op, asked=asked, descend=True):
                if rung not in op.rungs:
                    op.rungs = (*op.rungs, rung)
                return rung
        return None

    def _apply(
        self, name: str, rung: str, op: Op, *, asked: int = 0, descend: bool = False
    ) -> bool:
        """One rung. `descend` steps down from what failed; otherwise the rung is repeated
        for a call that needed it before. True when it changed something."""
        member = self.backend.components.get(name) or self.backend.parked.get(name)
        if rung == "room":
            if not descend or self.room is None:
                return False
            before = self._allocatable()
            self.room(before + max(asked, 1))
            return self._allocatable() > before
        if rung == "evict":
            victims = self._victims((name, *self.running))
            for victim in victims:
                self._evict(victim)
            return bool(victims)
        if rung == "page":
            if name in self.backend.paging or name not in self.backend.layouts:
                return False
            self._evict(name)
            self.backend.paging[name] = self.backend.layouts[name]
            self._resident(name)
            return True
        if rung == "split":
            if not descend:
                return False
            if op.splits * 2 > op.batch:
                return False
            op.splits *= 2
            return True
        if rung == "slice":
            enable = getattr(member, "enable_slicing", None)
            if not callable(enable) or getattr(member, "use_slicing", False) or op.batch < 2:
                return False
            enable()
            return True
        if rung == "tile":
            return _tile(member, op, halve=descend)
        return False

    def _leaves(self, args: tuple[object, ...], kwargs: dict[str, object]) -> list[_Tensor]:
        return [
            leaf
            for leaf in self.torch.utils._pytree.tree_leaves((args, kwargs))
            if isinstance(leaf, self.torch.Tensor)
        ]

    def _signature(self, args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[object, ...]:
        """A call's structural identity: the shape and dtype of every tensor it is given."""
        return tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in self._leaves(args, kwargs))

    def _versions(self, args: tuple[object, ...], kwargs: dict[str, object]) -> list[int]:
        """In-place writes bump a tensor's version: equal versions mean the inputs are
        intact. Inference-mode tensors keep no version; an inference forward that writes its
        own inputs is not one this Runtime serves."""
        return [
            leaf._version for leaf in self._leaves(args, kwargs) if not leaf.is_inference()
        ]

    def _batch(self, args: tuple[object, ...], kwargs: dict[str, object]) -> int:
        """The leading dimension of the first tensor argument: the batch a split divides."""
        return next((int(t.shape[0]) for t in self._leaves(args, kwargs) if t.dim()), 0)

    def _split(
        self,
        call: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        op: Op,
    ) -> object:
        """Run `call` with its leading batch in `op.splits` pieces and join the outputs."""
        if op.splits <= 1:
            return call(*args, **kwargs)
        pytree = self.torch.utils._pytree
        leaves, spec = pytree.tree_flatten((args, kwargs))
        bounds = [op.batch * i // op.splits for i in range(op.splits + 1)]
        outputs = []
        for start, stop in zip(bounds, bounds[1:], strict=False):
            piece = [
                leaf[start:stop]
                if isinstance(leaf, self.torch.Tensor) and leaf.dim() and leaf.shape[0] == op.batch
                else leaf
                for leaf in leaves
            ]
            part_args, part_kwargs = pytree.tree_unflatten(piece, spec)
            outputs.append(call(*part_args, **part_kwargs))
        flat = [pytree.tree_flatten(output) for output in outputs]
        joined = [
            self.torch.cat(values)
            if isinstance(values[0], self.torch.Tensor) and values[0].dim()
            else values[0]
            for values in zip(*(leaves for leaves, _ in flat), strict=True)
        ]
        return pytree.tree_unflatten(joined, flat[0][1])

    def _exhausted(self, name: str, op: Op, asked: int, detail: str) -> str:
        rows = accel.allocation(self.torch, self.kind)
        layout = self.backend.layouts.get(name)
        floor = f"; its floor is {layout.working_bytes} B of weights" if layout else ""
        refusal: Shortfall = {
            "resource": "vram",
            "scope": "component_forward",
            "needed_bytes": asked,
            "available_bytes": self._allocatable(),
            "evidence_class": "measured",
            "request_shape": name,
        }
        self.last_shortfall = refusal
        return (
            f"{name}: out of memory at every rung ({', '.join(op.rungs) or 'none applied'}): "
            f"asked {asked} B with {self._allocatable()} B allocatable, "
            f"{rows['driver_free_bytes']} B driver-free of {rows['driver_total_bytes']} B"
            f"{f', capped at {self.cap} B' if self.cap else ''}{floor} ({detail})"
        )

    def set_limit(self, limit: int) -> None:
        """The most this process's allocator may reserve on the device (0: no limit). Past
        it the allocator raises, and the guard asks the worker for room before descending."""
        if self.kind != "cuda":
            return
        total = int(self.torch.cuda.mem_get_info()[1])
        self.cap = limit if 0 < limit < total else 0
        self.torch.cuda.set_per_process_memory_fraction(self.cap / total if self.cap else 1.0)

    # ------------------------------------------------------------------ paging hooks

    def _fold(self) -> int:
        """Fold the allocator's high-water mark into every open measurement, then reset it."""
        peak = accel.peak_allocated(self.torch, self.kind)
        self.attempt_absolute_peak_bytes = max(self.attempt_absolute_peak_bytes, peak)
        for window in self.windows:
            window[0] = max(window[0], peak)
        accel.reset_peak(self.torch, self.kind)
        return peak

    def _page_enter(self, name: str, unit: PageUnit, module: object, args: object) -> None:
        if name not in self.backend.paging:
            return
        self._require_clean(unit.path)
        if self.page_call is not None:
            raise self._poison("paging_scope", "paging blocks do not nest")
        if self.torch.is_grad_enabled():
            raise ResidencyRefusal("paging_autograd", "weight paging requires inference mode")
        self._fold()
        self._free(unit.nbytes, (name, *self.running))
        if not self._fits(unit.nbytes) and self.room is not None:
            self.room(unit.nbytes - self._allocatable())
        allocated = self._allocated()
        started = time.perf_counter()
        try:
            self.backend.stage_page(name, unit)
        except Exception as exc:
            self.backend.abandon_page(name, unit)
            if is_device_oom(exc):
                raise
            raise self._poison("paging_fill", f"{name}.{unit.path}: {exc}") from exc
        # An autocast scope would keep a cast copy of every paged weight alive.
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
        self._keep(name, unit)
        self._ahead(name, unit)
        self._fold()

    def _ahead(self, name: str, unit: PageUnit) -> None:
        """Learn the order blocks run in, and start copying the next one while this one
        computes, when it is kept in host memory and fits beside this one."""
        previous = self.last_unit.get(name)
        if previous is not None:
            self.next_unit[(name, previous)] = unit.path
        self.last_unit[name] = unit.path
        following = self.next_unit.get((name, unit.path))
        layout = self.backend.paging.get(name)
        if following is None or layout is None or self.kind != "cuda":
            return
        block = next((b for b in layout.blocks if b.path == following), None)
        if block is None or not self._fits(block.nbytes):
            return
        if self.copy_stream is None:
            self.copy_stream = self.torch.cuda.Stream()
        self.backend.prefetch(name, block, self.copy_stream)

    def _keep(self, name: str, unit: PageUnit) -> None:
        """Keep a paged unit in host memory while the host has room for it: the next
        page-in is a copy, not a store read. Measured each time; no host floor."""
        if (name, unit.path) in self.backend.mirror:
            return
        if proctree.available_host_bytes() > unit.nbytes:
            self.kept_bytes += self.backend.keep(name, unit)

    def _page_leave(
        self, name: str, unit: PageUnit, module: object, args: object, output: object
    ) -> None:
        if self.page_call != (name, unit):
            return
        try:
            self._finish_page(name, unit, output)
        finally:
            self.page_call = None
            self.page_weight_delta = 0
            if self.page_autocast_cache is not None:
                self.torch.set_autocast_cache_enabled(self.page_autocast_cache)
                self.page_autocast_cache = None

    def _finish_page(self, name: str, unit: PageUnit, output: object) -> None:
        if output is None and sys.exception() is not None:
            # The forward failed; the component's guard decides. The block leaves.
            self.backend.abandon_page(name, unit)
            return
        storages = {
            getattr(module, attribute).untyped_storage().data_ptr()
            for _, module, attribute in unit.sites
        } - {0}
        leaves = self.torch.utils._pytree.tree_leaves(output)
        if any(
            not isinstance(value, self.torch.Tensor)
            and type(value) not in (bool, int, float, complex, str, bytes, type(None))
            for value in leaves
        ):
            raise self._poison("paging_output_schema", "block outputs must be a tensor PyTree")
        if any(
            isinstance(value, self.torch.Tensor)
            and value.untyped_storage().data_ptr() in storages
            for value in leaves
        ):
            raise self._poison("paging_output_alias", "a block output views a paged weight")
        self._fold()
        allocated = self._allocated()
        started = time.perf_counter()
        # Freed in stream order: the next block's copy waits for this block's kernels.
        self.backend.evict_page(name, unit)
        self.allocator_delta_bytes += self._allocated() - allocated
        self.evictions += 1
        self.attempt_evictions += 1
        elapsed = (time.perf_counter() - started) * 1000
        self.evicted_ms += elapsed
        self.attempt_evicted_ms += elapsed

    # ------------------------------------------------------------------ the lane's calls

    def vacate(self) -> VacateReport:
        """Free every evictable component: the worker handed the device to another tenant."""
        self._require_clean("vacate")
        freed: dict[str, int] = {}
        held: dict[str, str] = {}
        allocated = self._allocated()
        started = time.perf_counter()
        for name in sorted(self.backend.components, key=lambda n: self.fences.get(n, Fence()).last_used):
            if not self._evictable(name):
                held[name] = "the completion fence has not fired"
                continue
            try:
                freed[name] = self.backend.evict(name)
            except fill.FillRefusal as exc:
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

    def restore(self, names: tuple[str, ...] = ()) -> RestoreReport:
        """Refill what a vacate took, evicting nobody; what does not fit stays absent."""
        self._require_clean("restore")
        wanted = tuple(names) if names else tuple(sorted(self.backend.checkpoints))
        restored: dict[str, int] = {}
        held: dict[str, str] = {}
        allocated = self._allocated()
        started = time.perf_counter()
        fill_ms = 0
        legs: dict[str, int] = dict.fromkeys(fill.STAGE_LEGS, 0)
        objects: dict[str, int] = {}
        read_before = fill.block_read_bytes()
        for name in wanted:
            if name in self.backend.components or name in self.hosted:
                continue
            if name not in self.backend.checkpoints:
                held[name] = "this generation has no checkpoint for it"
                continue
            if not self._fits(self.backend.stage_bytes(name)):
                held[name] = "the device has no room for it without evicting a neighbour"
                continue
            try:
                staged = self.backend.stage(name)
            except fill.FillRefusal as exc:
                raise self._poison(exc.code, f"restoring {name!r} failed: {exc}") from exc
            except Exception as exc:
                if not is_device_oom(exc):
                    raise
                self.backend.abandon(name)
                recover(self.torch, self.kind, exc)
                held[name] = "the allocator ran out while it arrived"
                break
            if staged is not None:
                fill_ms += staged["stream_ms"]
                for leg in fill.STAGE_LEGS:
                    legs[leg] += staged[leg]
                objects[self.backend.checkpoints[name].manifest_id] = staged["objects"]
            self.stages += 1
            self.attempt_stages += 1
            restored[name] = self._size(name)
            self._touch(name)
        took = (time.perf_counter() - started) * 1000
        self.staged_ms += took
        self.attempt_staged_ms += took
        self.allocator_delta_bytes += self._allocated() - allocated
        read_after = fill.block_read_bytes()
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
        self.refusals.append(f"fault {code}: {detail}"[:400])

    def _poison(self, code: str, detail: str) -> ResidencyRefusal:
        self.poisoned = self.poisoned or f"{code}: {detail[:300]}"
        return ResidencyRefusal(code, detail)

    def _require_clean(self, what: str) -> None:
        if self.poisoned:
            raise ResidencyRefusal(
                "residency_poisoned", f"{what}() against a poisoned plane ({self.poisoned[:300]})"
            )

    # ------------------------------------------------------------------ room

    def _evictable(self, name: str) -> bool:
        """Not running, and its completion fence has fired. A component of the active scope
        may leave between forwards: its guard brings it back."""
        if name in self.running or (self.page_call is not None and self.page_call[0] == name):
            return False
        if not self.recoverable and self.active is not None and name in self.active[1]:
            return False
        return self.fences.get(name, Fence()).passed()

    def host(self, name: str) -> None:
        """`name` is kept resident by another rank of the group: free it here for good."""
        if name in self.backend.components:
            self._evict(name)
        self.hosted = self.hosted | {name}

    @contextmanager
    def sparing(self) -> Iterator[None]:
        """Admit under this block only what fits beside everything already held."""
        self.spare = True
        try:
            yield
        finally:
            self.spare = False

    def _require_spare_room(self, local: tuple[str, ...], method: str) -> None:
        if method not in self.measured:
            raise self._spared(method, "its scratch is not measured at this request's shape")
        absent = sum(self._size(name) for name in local if name not in self.backend.components)
        headroom = max([self.headroom, *self.scope_headrooms.values()])
        allocatable = self._allocatable()
        if allocatable < absent + headroom:
            raise self._spared(
                method,
                f"needs {absent} B of weights and {headroom} B of scratch; "
                f"{allocatable} B are allocatable",
            )

    def _spared(self, method: str, why: str) -> ResidencyRefusal:
        return ResidencyRefusal("device_spared", f"{method}() declined: {why}")

    def observe_remote(self, method: str, working_bytes: int) -> None:
        """A hosted call's scratch on its home rank is this scope's demand there."""
        peaks = self.attempt_activation_peaks
        peaks[method] = max(peaks.get(method, 0), int(working_bytes))

    def shed(self) -> None:
        """Free every component outside the active scope."""
        declared = tuple(self.active[1]) if self.active is not None else ()
        self._wait_for_pending_victims(declared)
        for victim in self._victims(declared):
            self._evict(victim)

    def _make_headroom(self, declared: tuple[str, ...], method: str) -> None:
        """Free room for the scope's work before its weights move. A scope measured at this
        shape keeps what fits beside its peak. An unmeasured one frees everything else in a
        rank group, where a forward cannot run again; alone, its forwards recover instead."""
        self._wait_for_pending_victims(declared)
        if method not in self.measured:
            if not self.recoverable:
                for victim in self._victims(declared):
                    self._evict(victim)
            return
        absent = sum(self._size(name) for name in declared if name not in self.backend.components)
        self._free(absent + max(self.scope_headrooms.get(method, self.headroom), self.headroom), declared)

    def _victims(self, declared: tuple[str, ...]) -> list[str]:
        """Resident components outside `declared` whose fences fired, least recent first."""
        candidates = [
            name for name in list(self.backend.components) if name not in declared and self._evictable(name)
        ]
        return sorted(candidates, key=lambda n: self.fences.get(n, Fence()).last_used)

    def _free(self, need: int, declared: tuple[str, ...]) -> None:
        if self._fits(need):
            return
        self._wait_for_pending_victims(declared)
        while not self._fits(need):
            victim = self._victim(declared, need)
            if victim is None:
                return
            self._evict(victim)

    def _victim(self, declared: tuple[str, ...], need: int) -> str | None:
        """Stale components first, then those this attempt is done with; then the smallest
        one the previous attempt used that covers the shortfall."""
        victims = self._victims(declared)
        stale = [n for n in victims if n not in self.attempt_used and n not in self.previous_used]
        done = [n for n in victims if n in self.attempt_used]
        if stale or done:
            return (stale or done)[0]
        shortfall = need - self._allocatable()
        covering = [n for n in victims if self._charge(n) >= shortfall]
        if covering:
            return min(covering, key=self._charge)
        return victims[0] if victims else None

    def _charge(self, name: str) -> int:
        return self.backend.vram_charge.get(name) or self._size(name)

    def _wait_for_pending_victims(self, declared: tuple[str, ...]) -> None:
        """A scope's release records an event rather than synchronizing; wait for it only
        when a candidate victim's fence is still pending."""
        if any(
            name not in declared and not self.fences.get(name, Fence()).passed()
            for name in self.backend.components
        ):
            accel.synchronize(self.torch, self.kind)

    def _evict(self, victim: str) -> None:
        if self.spare:
            raise self._spared(
                self.active[0] if self.active else "admission", f"it would evict {victim!r}"
            )
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
        """The component's declared bytes, resident or not."""
        if name in self.backend.paging:
            return self.backend.paging[name].working_bytes
        known = self.backend.component_bytes.get(name)
        if known:
            return known
        return sum(row.nbytes for row in self.backend.plan.values() if row.component == name)

    def _fits(self, need: int) -> bool:
        return self._allocatable() >= need

    def _allocatable(self) -> int:
        """What a new reservation can get: driver-free plus the allocator's unused cache,
        within the budget's cap."""
        rows = accel.allocation(self.torch, self.kind)
        slack = max(rows["reserved_bytes"] - rows["allocated_bytes"], 0)
        free = rows["driver_free_bytes"]
        if self.cap:
            free = min(free, max(self.cap - rows["reserved_bytes"], 0))
        return free + slack

    def _shortfall(self, method: str, name: str, headroom: int, detail: str) -> ResidencyRefusal:
        need = self._size(name)
        rows = accel.allocation(self.torch, self.kind)
        free, total = rows["driver_free_bytes"], rows["driver_total_bytes"]
        allocatable = self._allocatable()
        required = self.backend.stage_bytes(name, headroom)
        resident = sorted(self.backend.components)
        return ResidencyRefusal(
            "device_shortfall",
            f"{method}() declares component {name!r} requiring {required} B ({need} B of "
            f"weights, {headroom} B of forward headroom, including fill storage); {allocatable} "
            f"B are allocatable ({free} B driver-free of {total} B"
            f"{f', capped at {self.cap} B' if self.cap else ''}) with {resident} resident, "
            f"short by {max(required - allocatable, 0)} B after every idle component and "
            f"co-tenant gave room ({detail[:160]})",
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

    def document(self) -> ResidencyDocument:
        return {
            "placement": self.placement,
            "headroom_bytes": self.headroom,
            "scope_headroom_bytes": dict(self.scope_headrooms),
            "measured_scopes": sorted(self.measured),
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
            # With their real sizes: an absent component still costs this much to return.
            "evicted": {n: self._size(n) for n in self.backend.parked if n not in self.hosted},
            "hosted": sorted(self.hosted),
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
            "recoveries": list(self.recoveries[-16:]),
            "cap_bytes": self.cap,
            "kept_bytes": self.kept_bytes,
        }


def _counters(root: object) -> dict[tuple[object, str], int | float]:
    """Every public number a module tree holds, by (module, attribute)."""
    modules = getattr(root, "modules", None)
    if not callable(modules):
        return {}
    return {
        (module, attribute): value
        for module in modules()
        for attribute, value in vars(module).items()
        if not attribute.startswith("_") and type(value) in (int, float)
    }


#: The tile attributes of the two codec families diffusers ships.
_TILES = (
    ("tile_sample_min_size",),
    ("tile_sample_min_height", "tile_sample_min_width"),
)


def _tile(member: object, op: Op, *, halve: bool) -> bool:
    """Tile a codec at the op's remembered size; descending halves it, down to the codec's
    own latent scale. False when there is nothing smaller to try."""
    enable = getattr(member, "enable_tiling", None)
    names = next((names for names in _TILES if all(hasattr(member, n) for n in names)), ())
    if not callable(enable) or not names:
        return False
    if not halve:
        if not op.tile:
            return False
        enable()
    else:
        scale = _scale(member)
        current = op.tile or {n: int(getattr(member, n)) for n in names}
        if min(current.values()) // 2 < scale * scale:
            return False
        op.tile = {n: value // 2 for n, value in current.items()}
        enable()
    tiles = dict(op.tile)
    if "tile_sample_min_size" in tiles:
        tiles["tile_latent_min_size"] = tiles["tile_sample_min_size"] // _scale(member)
    else:
        tiles["tile_sample_stride_height"] = tiles["tile_sample_min_height"] * 3 // 4
        tiles["tile_sample_stride_width"] = tiles["tile_sample_min_width"] * 3 // 4
    for attribute, value in tiles.items():
        setattr(member, attribute, value)
    return True


def _scale(member: object) -> int:
    """A codec's latent scale: two to the number of its down-sampling stages."""
    config = getattr(member, "config", None)
    return int(2 ** (len(getattr(config, "block_out_channels", ()) or (0,)) - 1))


# ---------------------------------------------------------------------------- constructions


class _Residency(Protocol):
    def vacate(self) -> VacateReport: ...

    def set_limit(self, limit: int) -> None: ...

    def restore(self, names: tuple[str, ...] = ()) -> RestoreReport: ...

    def document(self) -> ResidencyDocument: ...


@dataclass(slots=True)
class Construction:
    """One loaded construction as the process holds it."""

    residency: _Residency | None
    #: bytes to bring `names` back: their settled sum plus the widest one's fill transient
    need: Callable[[tuple[str, ...]], int]
    last_used: int = 0
    last_used_ns: int = 0
    #: the resident set a park vacated, restored by `restore`; None while resident
    parked: tuple[str, ...] | None = None

    def resident_bytes(self) -> int:
        if self.residency is None or self.parked is not None:
            return 0
        return sum(self.residency.document()["resident"].values())


@dataclass(slots=True)
class ConstructionArbiter:
    """Whole-construction LRU parking inside one executor process."""

    rows: dict[str, Construction] = field(default_factory=dict)
    clock: int = 0

    def add(self, key: str, row: Construction) -> None:
        self.rows[key] = row
        self.touch(key)

    def touch(self, key: str) -> None:
        self.clock += 1
        self.rows[key].last_used = self.clock
        self.rows[key].last_used_ns = time.monotonic_ns()

    def remove(self, key: str) -> Construction | None:
        return self.rows.pop(key, None)

    def victims(self, exclude: str) -> list[str]:
        held = [
            key
            for key, row in self.rows.items()
            if key != exclude and row.parked is None and row.resident_bytes() > 0
        ]
        return sorted(held, key=lambda key: self.rows[key].last_used)

    def park(self, key: str) -> VacateReport:
        row = self.rows[key]
        if row.residency is None or row.parked is not None:
            return {"freed": {}, "held": {}, "freed_bytes": 0}
        report = row.residency.vacate()
        row.parked = tuple(sorted(report["freed"]))
        return report

    def make_room(self, exclude: str, fits: Callable[[], bool]) -> list[str]:
        parked: list[str] = []
        while not fits():
            victims = self.victims(exclude)
            if not victims:
                break
            self.park(victims[0])
            parked.append(victims[0])
        return parked

    def restore(self, key: str, names: tuple[str, ...] = ()) -> RestoreReport:
        row = self.rows[key]
        wanted = names or row.parked or ()
        row.parked = None
        if row.residency is None or not wanted:
            return {"restored": {}, "held": {}, "restored_bytes": 0}
        return row.residency.restore(names=tuple(wanted))

    def need(self, key: str) -> int:
        row = self.rows[key]
        return row.need(row.parked or ()) if row.parked is not None else 0

    def document(self, active: str = "") -> list[ConstructionRow]:
        rows: list[ConstructionRow] = []
        for key, row in sorted(self.rows.items(), key=lambda item: -item[1].last_used):
            document: ConstructionRow = {
                "key": key,
                "state": "resident" if row.parked is None else "parked",
                "resident_bytes": row.resident_bytes(),
                "last_used_ns": row.last_used_ns,
                "active": key == active,
            }
            if row.parked is not None:
                document["parked"] = list(row.parked)
            rows.append(document)
        return rows
