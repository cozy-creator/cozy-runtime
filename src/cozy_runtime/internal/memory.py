"""Every memory decision the Runtime makes, in one place.

`Planner` runs in each executor process (one device). Before every op (a managed component
root's forward, encode, decode, or a method a package spreads) it chooses where each weight
unit lives and how the op's activations are bounded, from measured unit sizes and the op's
measured working memory, and only then runs it. An out-of-memory error inside an op is
recovered only when the op's state can be put back; otherwise it fails typed with the numbers.

`Machine` runs in the Worker. It grants each executor the device bytes it may hold and makes
room for one by moving idle tenants' weights off the device, least recently used first.

Mechanism lives elsewhere: `tiers` moves bytes between device, host and disk; the fill plane
reads the store; the executor and session carry the messages.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NoReturn, Protocol

from cozy_runtime.author._errors import DEVICE_OOM, RuntimeFailure
from cozy_runtime.author._retry import is_declared_pure
from cozy_runtime.internal import proctree, tiers
from cozy_runtime.internal.tiers import DEVICE, DISK, HOST, Device, Module, Tensor, Torch

SHORTFALL = "device_shortfall"
SPARED = "device_spared"
_ASKED = re.compile(r"Tried to allocate ([0-9.]+) ?([KMGT]?i?B)")
_SCALAR = (bool, int, float, str, type(None))


class MemoryRefusal(RuntimeFailure):
    """A typed memory outcome carrying the numbers the terminal reports."""

    def __init__(
        self,
        code: str,
        message: str,
        needed: int = 0,
        available: int = 0,
        *,
        reusable: bool = True,
    ) -> None:
        super().__init__(message, code=code)
        self.needed, self.available = int(needed), int(available)
        self.reusable = reusable
        self.shortfall = {
            "resource": "vram",
            "scope": "component_use",
            "needed_bytes": int(needed),
            "available_bytes": int(available),
            "evidence_class": "measured",
            "request_shape": "",
        }


def is_oom(exc: BaseException) -> bool:
    """A device allocation failure: torch's, an asynchronous one, or cuBLAS/cuDNN's."""
    name, text = type(exc).__name__, str(exc)
    torch_error = type(exc).__module__.split(".")[0] == "torch"
    if torch_error and name == "OutOfMemoryError":
        return "DefaultCPUAllocator" not in text
    if torch_error and name == "AcceleratorError":
        return getattr(exc, "error_code", None) == 2 or "out of memory" in text
    return isinstance(exc, RuntimeError) and any(
        mark in text
        for mark in (
            "CUDA error: out of memory",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDNN_STATUS_ALLOC_FAILED",
        )
    )


def _asked(text: str) -> int:
    match = _ASKED.search(text)
    if match is None:
        return 0
    unit = match[2][0] if match[2] != "B" else " "
    return int(float(match[1]) * 1024 ** " KMGT".index(unit))


def _leaves(value: object) -> Iterator[object]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _leaves(value[key])
    else:
        yield value


# --------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class OpKey:
    component: str
    method: str
    tensors: tuple[tuple[tuple[int, ...], str], ...]
    flags: tuple[bool | str | None, ...] = ()
    mode: tuple[tuple[str, object], ...] = ()

    def ratio(self, other: OpKey) -> float | None:
        """How much larger this op's inputs are than `other`'s, argument by argument."""
        if (
            other.method != self.method
            or other.flags != self.flags
            or other.mode != self.mode
            or len(other.tensors) != len(self.tensors)
            or any(
                mine[1] != theirs[1]
                for mine, theirs in zip(self.tensors, other.tensors, strict=True)
            )
            or not self.tensors
        ):
            return None
        return max(
            math.prod(mine) / max(math.prod(theirs), 1)
            for (mine, _), (theirs, _) in zip(self.tensors, other.tensors, strict=True)
        )


@dataclass(slots=True)
class Record:
    #: most working bytes a success took; -1 until one succeeds
    ok: int = -1
    #: least working bytes a failure proved it needs
    fail: int = 0


# --------------------------------------------------------------------------- the planner


@dataclass(eq=False)
class Held:
    """One component this process holds, and what the planner knows about it."""

    key: str
    construction: str
    part: tiers.Component
    #: the architecture: checkpoints of one architecture share their measurements
    kind: str = ""
    hosted: bool = False
    compiled: bool = False
    #: the package reads its tensors itself (`author.read_whole`): whole on the device while
    #: a scope declaring it is open
    whole: bool = False
    last_used: int = 0
    ops: dict[str, Callable[..., object]] = field(default_factory=dict)


@dataclass(eq=False)
class Frame:
    held: Held
    base: int
    working: int = 0
    loaded: int = 0


@dataclass(slots=True)
class Plan:
    keep: int
    evict: list[Held]
    #: its victims include the scope's other components' common units
    deep: bool = False


@dataclass(eq=False)
class Snapshot:
    tensors: list[tuple[Tensor, Tensor]]
    rng: list[tuple[object, object]]
    state: tuple[Tensor, ...]
    geometry: tuple[tuple[object, ...], ...]
    resolve: Callable[[], Iterable[Tensor]]
    custom: tuple[Callable[[object], None], object] | None


class Planner:
    """The executor's memory policy. One per process; the process owns one device."""

    def __init__(
        self,
        torch: Torch,
        device: Device,
        *,
        room: Callable[[int, int, bool], int | None] | None = None,
        host_room: Callable[[str, int, bool], bool] | None = None,
        sharded: Callable[[str], bool] = lambda key: False,
        grouped: bool = False,
        agree: Callable[[int], int] | None = None,
    ) -> None:
        self.torch = torch
        self.device = device
        self.cuda = device.type == "cuda"
        #: asks the Worker for (driver-free bytes, total bytes to hold, measured need): the grant
        self.room = room
        self.host_room = host_room
        self._host_sequence = 0
        self._host_pending = 0
        self.sharded = sharded
        self.grouped = grouped
        #: the group's largest shortfall, given this rank's (a collective)
        self.agree = agree
        self.grant: int | None = None
        #: this process's non-allocator device bytes: context, modules, communicators
        self.overhead = 0
        self.held: dict[str, Held] = {}
        self.records: dict[OpKey, Record] = {}
        self.stack: list[Frame] = []
        self.scopes: list[tuple[str, frozenset[str]]] = []
        self.seq: list[str] = []
        self.last_seq: dict[str, list[str]] = {}
        self.clock = 0
        self.spare = False
        self.deep = False
        self.asked_room = False
        self.room_asks = 0
        self.peak = 0
        self.peak_allocated = 0
        self.working = 0
        self.loads = self.offloads = self.moved = 0
        #: bytes sent to the store and read back from it this attempt: the disk tier's traffic
        self.stored = self.refilled = 0
        #: the allocator's counters when the attempt began: its churn is their growth
        self.counters: dict[str, int] = {}
        #: allocator bytes the attempt's weight moves added (negative: gave back)
        self.transition = 0
        #: weights left the device since the last cap: their bytes sit in the allocator's cache
        self.dropped = False
        self.log: list[str] = []
        self.plans: dict[str, list[str]] = {}
        self.operations: dict[str, dict[str, int]] = {}
        self.oom_asked = 0
        #: the typed refusal this attempt ended on, for the reply's shortfall numbers
        self.refusal: MemoryRefusal | None = None
        #: A failed rollback or device fault must never be cleared by begin().
        self.unusable = ""
        #: A group router owns this one-shot boundary. It runs after all preparation,
        #: while inputs, placement and snapshot remain live, before authored hooks/body.
        self._gates: dict[str, Callable[[], None]] = {}
        self.preparation_failure: str = ""
        self.group_phase = ""
        if self.cuda:
            attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
            if attach is not None:
                attach(self._observe_oom)

    # ------------------------------------------------------------------ measurement

    def _observe_oom(self, device: int, asked: int, total: int, free: int) -> None:
        self.oom_asked = int(asked)

    def checkpoint(self) -> tuple[int, int, int]:
        """Driver free, allocator reserve and room requests now: the start of a measurement."""
        if not self.cuda:
            return 0, 0, self.room_asks
        return self._driver()[0], self._reserved(), self.room_asks

    def measure_overhead(self, free_before: int, reserved_before: int, asks: int = 0) -> None:
        """Non-allocator bytes this process took since `free_before` was read: called only at
        controlled points (init, warm, a communicator join), and discarded when a room
        request moved other tenants' bytes in between."""
        if not self.cuda or free_before <= 0 or self.room_asks != asks:
            return
        free, _ = self._driver()
        grown = free_before - free - (self._reserved() - reserved_before)
        self.overhead += max(grown, 0)

    def _allocated(self) -> int:
        return int(self.torch.cuda.memory_allocated(self.device)) if self.cuda else 0

    def _reserved(self) -> int:
        return int(self.torch.cuda.memory_reserved(self.device)) if self.cuda else 0

    def _driver(self) -> tuple[int, int]:
        free, total = self.torch.cuda.mem_get_info(self.device)
        return int(free), int(total)

    def allocatable(self) -> int:
        """Bytes the allocator may still reserve: inside the grant, and never the driver's
        headroom (one `overhead` kept free on the device). Its cache is no credit: holes
        between live blocks cannot be released, and the cap counts them. Negative when the
        process holds more than its grant, and the plan must give bytes back."""
        if not self.cuda:
            return 1 << 62
        free, total = self._driver()
        granted = (total if self.grant is None else self.grant) - self.overhead - self._reserved()
        return min(granted, free - self.overhead)

    def _cap(self) -> None:
        """The allocator stops at the grant, so an overrun is a clean OOM, not a driver fault."""
        if not self.cuda:
            return
        if self.dropped:
            tiers.trim(self.torch)
            self.dropped = False
        free, total = self._driver()
        limit = min(
            (total if self.grant is None else self.grant) - self.overhead,
            free + self._reserved() - self.overhead,
        )
        self.torch.cuda.set_per_process_memory_fraction(
            min(max(limit, 0) / total, 1.0), self.device
        )

    def _sample(self) -> None:
        if not self.cuda:
            return
        cuda = self.torch.cuda
        peak = int(cuda.max_memory_allocated(self.device))
        self.peak = max(self.peak, int(cuda.max_memory_reserved(self.device)))
        self.peak_allocated = max(self.peak_allocated, peak)
        for frame in self.stack:
            frame.working = max(frame.working, peak - frame.base - frame.loaded)
        cuda.reset_peak_memory_stats(self.device)

    # ------------------------------------------------------------------ components

    def adopt(
        self, construction: str, name: str, root: Module, store: tiers.Store | None, tier: str
    ) -> Held:
        """Take a constructed component under management: units, hooks, op wrappers."""
        key = f"{construction}/{name}"
        part = tiers.partition(name, root, tier, store)
        held = Held(
            key,
            construction,
            part,
            kind=f"{type(root).__module__}.{type(root).__qualname__}:{name}:{part.nbytes}",
            compiled=_compiled(root),
            whole=bool(getattr(root, "_cozy_read_whole", False)),
        )
        self.held[key] = held
        self._pin_state(held)
        if self.cuda:
            tiers.install(part, self._enter_unit, self._leave_unit)
            tiers.wrap(root, functools.partial(self.run, held))
            tiers.intercept_apply(
                self.torch,
                part,
                lambda: None,
                host_copy=lambda size: self._host_copy(size, protected=part.units),
            )
            root.__dict__["_cozy_op"] = functools.partial(self.op, held)
        return held

    def op(self, held: Held, method: str) -> Callable[..., object]:
        """A method the package spreads, run in its unchanged authored numerical mode."""
        if method not in held.ops:
            bound = getattr(held.part.root, method)
            run = functools.partial(self.run, held)
            held.ops[method] = tiers.op(method, bound, run)
        return held.ops[method]

    def forget(self, construction: str) -> None:
        for key in [key for key, held in self.held.items() if held.construction == construction]:
            tiers.release(self.held.pop(key).part)

    def host(self, key: str) -> None:
        """Another GPU of the group keeps this component: it never occupies this one."""
        held = self.held[key]
        if held.part.store is not None:
            tiers.to_disk(self.torch, held.part, None)
        held.hosted = True

    # ------------------------------------------------------------------ attempts and scopes

    def begin(self, grant: int | None) -> None:
        if self.unusable:
            raise MemoryRefusal("device_fault", self.unusable, reusable=False)
        if grant is not None:
            self.grant = grant
        self.asked_room = False
        self.peak = self.peak_allocated = self.working = 0
        self.loads = self.offloads = self.moved = self.transition = 0
        self.stored = self.refilled = 0
        self.counters = self._counters()
        self.log = []
        self.plans = {}
        self.operations = {}
        self.refusal = None
        self.preparation_failure = ""
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)
            self._cap()

    def end(self) -> dict[str, int]:
        """Close the attempt: return the allocator's cache and report this process's bytes."""
        for construction in {self.held[key].construction for key in self.seq if key in self.held}:
            self.last_seq[construction] = [
                key for key in self.seq if key.startswith(construction + "/")
            ]
        self.seq = []
        self._sample()
        if self.cuda:
            tiers.trim(self.torch)
        return self.facts()

    def document(self, construction: str) -> dict[str, object]:
        """The attempt's residency facts for one generation, in the shape readers of the
        executor's `residency` reply know."""
        mine = [
            held
            for held in self.held.values()
            if held.construction.split("#", 1)[0] == construction and not held.hosted
        ]
        return {
            "placement": "planned",
            # All constructions in this executor, including explicit zeroes. Absent
            # components remain unknown to older workers/executors.
            "disk": {h.key: h.part.on(DISK) for h in self.held.values() if not h.hosted},
            "disk_epoch": {h.key: h.part.disk_epoch for h in self.held.values() if not h.hosted},
            "resident": {h.key: h.part.on(DEVICE) for h in mine if h.part.on(DEVICE)},
            "evicted": {
                h.key: h.part.nbytes - h.part.on(DEVICE)
                for h in mine
                if h.part.nbytes > h.part.on(DEVICE)
            },
            "hosted": sorted(h.key for h in self.held.values() if h.hosted),
            "attempt_stages": self.loads,
            "attempt_evictions": self.offloads,
            "attempt_moved_bytes": self.moved,
            "attempt_stored_bytes": self.stored,
            "attempt_refilled_bytes": self.refilled,
            **{
                f"attempt_{name}": count - self.counters.get(name, 0)
                for name, count in self._counters().items()
            },
            "attempt_activation_peak_bytes": self.working,
            "attempt_absolute_peak_bytes": self.peak_allocated,
            "allocator_delta_bytes": self.transition,
            "overhead_bytes": self.overhead,
            "grant_bytes": self.grant if self.grant is not None else -1,
            "log": list(self.log[-8:]),
            "plans": {op: list(seen) for op, seen in self.plans.items()},
            "operations": {op: dict(counts) for op, counts in self.operations.items()},
            "poisoned": self.unusable,
        }

    @contextmanager
    def _traffic(self, counts: dict[str, int], phase: str) -> Iterator[None]:
        """Counter deltas, inclusive of nested operations; never sum across nesting.

        Placement covers _apply; invoke covers the authored callable, including its
        hooks and failed calls. Rollback/recovery remain in the old attempt totals.
        """
        before = (self.loads, self.offloads, self.moved)
        counts[f"{phase}_calls"] = counts.get(f"{phase}_calls", 0) + 1
        try:
            yield
        finally:
            for name, old, new in zip(
                ("stages", "evictions", "moved_bytes"),
                before,
                (self.loads, self.offloads, self.moved),
                strict=True,
            ):
                key = f"{phase}_{name}"
                counts[key] = counts.get(key, 0) + new - old

    def _counters(self) -> dict[str, int]:
        """Device allocations and frees, allocator retries (a cache flush to satisfy one
        request) and out-of-memory errors so far."""
        if not self.cuda:
            return {}
        stats = self.torch.cuda.memory_stats(self.device)
        names = ("num_device_alloc", "num_device_free", "num_alloc_retries", "num_ooms")
        return {name: int(stats.get(name, 0)) for name in names}

    def facts(self) -> dict[str, int]:
        reserved, allocated = self._reserved(), self._allocated()
        host_backings: dict[tuple[str, int], int] = {}
        for held in self.held.values():
            for unit in held.part.units:
                tensors = list(unit.host.values())
                if unit.tier == HOST:
                    tensors.extend(tensor for *_, tensor in unit.slots())
                for tensor in tensors:
                    if tensor.device.type == "cpu":
                        storage = tensor.untyped_storage()
                        host_backings[tiers._backing(tensor)] = storage.nbytes()
        return {
            "overhead_bytes": self.overhead,
            "reserved_bytes": reserved,
            "allocated_bytes": allocated,
            "reserve_room_bytes": self.allocatable() if self.cuda else 0,
            # Visible cache is not a planning credit: fragmented cached blocks cannot
            # necessarily satisfy the next allocation or be returned to the driver.
            "reusable_cache_bytes": max(reserved - allocated, 0),
            "peak_bytes": self.peak + self.overhead,
            "peak_allocated_bytes": self.peak_allocated,
            "working_bytes": self.working,
            "host_bytes": sum(host_backings.values()),
            "disk_weight_bytes": sum(h.part.on(DISK) for h in self.held.values() if not h.hosted),
        }

    def enter(self, construction: str, method: str, components: tuple[str, ...]) -> None:
        """A declared method begins. Each declared component comes onto the device whole when
        it fits beside the rest of the scope, since a package may read its tensors outside
        any forward; otherwise its common unit does, so `.device` reads true, and its blocks
        arrive with its ops. One the package reads itself (`read_whole`) must fit whole."""
        before = len(self.scopes)
        self.scopes.append((construction, frozenset(components)))
        try:
            self._enter_scope(construction, method, components)
        except BaseException:
            # Model._cozy_scope has not entered yet, so its __exit__ will never run.
            # A failed transfer must release these pins as well as a planned refusal.
            del self.scopes[before:]
            raise

    def _enter_scope(self, construction: str, method: str, components: tuple[str, ...]) -> None:
        if not self.cuda:
            return
        scope = {f"{construction}/{name}" for name in components}
        for name in components:
            held = self.held.get(f"{construction}/{name}")
            if held is None or held.hosted:
                continue
            # A package may have replaced a forward since (a cache hook): it is an op again.
            tiers.wrap(held.part.root, functools.partial(self.run, held))
            held.compiled = _compiled(held.part.root)
            part = held.part
            victims = [] if self.spare else [v for v in self._victims(held) if v.key not in scope]
            whole = part.nbytes - part.on(DEVICE)
            if part.on(DISK) and part.store is not None:
                whole += part.store.transient(part.name)
            common = part.common.nbytes if part.common.tier != DEVICE else 0
            chosen = self._room_from(victims, whole) if whole else []
            if chosen is None and (held.whole or not common):
                chosen = (
                    self._room_from(victims, whole)
                    if self._ask_room(whole, victims, measured=True)
                    else None
                )
            if chosen is not None:
                self._stage_whole(held, chosen)
                continue
            if held.whole:
                self._refuse_scope(held, method, whole, victims)
            chosen = self._room_from(victims, common)
            if chosen is None and self._ask_room(common, victims, measured=True):
                chosen = self._room_from(victims, common)
            if chosen is None:
                self._refuse_scope(held, method, common, victims)
            for victim in chosen:
                self._offload(victim)
            self._load_common(held)
            self._cap()

    def _stage_whole(self, held: Held, victims: list[Held]) -> None:
        for victim in victims:
            self._offload(victim)
        part = held.part
        if part.on(DISK) == part.nbytes and part.store is not None:
            with self._moving():
                tiers.from_disk(self.torch, part, None)
            self.loads += 1
            self.refilled += part.nbytes
        else:
            self._load_common(held)
            for unit in part.blocks:
                self._load(part, unit)
        self._cap()

    def _refuse_scope(self, held: Held, method: str, need: int, victims: list[Held]) -> NoReturn:
        available = self.allocatable() + sum(self._evictable(v) for v in victims)
        self.refusal = MemoryRefusal(
            SPARED if self.spare else SHORTFALL,
            f"{held.key}: {method}() needs {need} B of it on the device before it runs, and "
            f"{available} B can be made allocatable",
            need,
            available,
        )
        raise self.refusal

    def exit(self) -> None:
        if self.scopes:
            self.scopes.pop()

    @contextmanager
    def sparing(self) -> Iterator[None]:
        """Run on the room that is already free, or decline: never evict for a spread call."""
        self.spare = True
        try:
            yield
        finally:
            self.spare = False

    def trim(self) -> dict[str, object]:
        """Only the allocator's cache goes back to the driver."""
        before = self._reserved()
        if self.cuda:
            tiers.trim(self.torch)
        return {"freed": {}, "held": {}, "freed_bytes": before - self._reserved()}

    def vacate(self) -> dict[str, object]:
        """The Worker hands this device to another tenant: every weight leaves it."""
        freed: dict[str, int] = {}
        before = self._allocated()
        for held in self.held.values():
            moved = held.part.on(DEVICE)
            if moved:
                self._offload(held, everything=True)
                released = moved - held.part.on(DEVICE)
                if released:
                    freed[held.key] = released
        if self.cuda:
            tiers.trim(self.torch)
        return {"freed": freed, "held": {}, "freed_bytes": before - self._allocated()}

    # ------------------------------------------------------------------ the op

    @contextmanager
    def gating(self, key: str, ready: Callable[[], None]) -> Iterator[None]:
        """Suspend a prepared grouped call at the CPU ready/commit boundary exactly once."""
        if key in self._gates:
            raise RuntimeError(f"a group preparation gate is already open for {key}")
        before = self.group_phase
        self.group_phase = "prepare"
        self._gates[key] = ready
        try:
            yield
        finally:
            self._gates.pop(key, None)
            self.group_phase = before

    def _ready(self, key: str) -> None:
        gate = self._gates.pop(key, None)
        if gate is not None:
            if self.cuda:
                try:
                    self.torch.cuda.synchronize(self.device)
                except Exception as exc:
                    self.unusable = (
                        f"{key}: device failed while completing group preparation: {exc}"
                    )
                    raise MemoryRefusal("device_fault", self.unusable, reusable=False) from None
            gate()
            self.group_phase = "execute"

    def preparation_bytes(self, error: BaseException | None) -> dict[str, int]:
        """The rank's room requirement at a control-socket preparation boundary."""
        free, total = self._driver() if self.cuda else (0, 0)
        grant = total if self.grant is None else self.grant
        needed = int(getattr(error, "needed", 0))
        available = int(getattr(error, "available", 0))
        oom = error is not None and is_oom(error)
        if oom:
            needed = self.oom_asked or _asked(str(error))
            available = self.allocatable()
        short = (
            max(needed - available, 1)
            if oom or getattr(error, "code", "") in (SHORTFALL, DEVICE_OOM)
            else 0
        )
        return {
            "needed": needed,
            "available": available,
            "free_bytes": free,
            "grant_bytes": grant,
            "free_wanted": free + short,
            "grant_wanted": max(grant, self._reserved() + self.overhead) + short,
        }

    def run(
        self,
        held: Held,
        method: str,
        fn: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Plan, run and measure one operation, replaying only a declared transaction."""
        try:
            if self.unusable:
                raise MemoryRefusal("device_fault", self.unusable, reusable=False)
            return self._run(held, method, fn, args, kwargs)
        except MemoryRefusal as exc:
            # A reusable child refusal may have crossed an un-restored parent body.
            exc.reusable = exc.reusable and not self.unusable
            self.refusal = exc
            raise

    def _run(
        self,
        held: Held,
        method: str,
        fn: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        if not self.cuda or held.hosted or any(frame.held is held for frame in self.stack):
            self._ready(held.key)
            return fn(*args, **kwargs)
        torch = self.torch
        leaves = list(_leaves((args, kwargs)))
        key = OpKey(
            held.kind,
            method,
            tuple((tuple(x.shape), str(x.dtype)) for x in leaves if isinstance(x, torch.Tensor)),
            tuple(x for x in leaves if isinstance(x, (bool, str, type(None)))),
            _mode(held.part.root),
        )
        held.compiled = _compiled(held.part.root)
        # A prepared group transaction executes once after commit even if a future
        # component classifier misses its sharded registration.
        retryable = held.key not in self._gates and self._retryable(held, leaves)
        self.clock += 1
        held.last_used = self.clock
        self.seq.append(held.key)
        self._pin_state(held)
        readonly = retryable or self._state_contract(held, leaves)
        for unit in held.part.units:
            unit.readonly = unit.readonly and readonly and not unit.mutable
            if not unit.readonly:
                # An unknown hook may write in place without changing a pointer. A
                # later declaration cannot validate an old host/checkpoint copy.
                unit.host.clear()
                unit.seen.clear()
        counts = self.operations.setdefault(f"{held.key}.{method}", {"calls": 0})
        counts["calls"] += 1
        counts["total_blocks"] = len(held.part.blocks)
        snapshot: Snapshot | None = None
        floor = False
        previous_room: int | None = None
        previous_asked = 0
        room_tried = False
        try:
            while True:
                plan = self._agreed(held, key, retryable, floor)
                counts["min_keep_blocks"] = min(counts.get("min_keep_blocks", plan.keep), plan.keep)
                counts["max_keep_blocks"] = max(counts.get("max_keep_blocks", plan.keep), plan.keep)
                frame: Frame | None = None
                entered = False
                asked, text = 0, ""
                try:
                    # Placement changes no authored state. Reclaim before snapshot allocation,
                    # and retain one baseline across retries rather than cloning it again.
                    with self._traffic(counts, "placement"):
                        self._apply(held, plan)
                    if previous_room is not None:
                        gained = self.allocatable() - previous_room
                        if gained < max(previous_asked, 1):
                            raise MemoryRefusal(
                                DEVICE_OOM,
                                f"{held.key}: no retry: placement reclaimed {gained} B; "
                                f"the failed allocation needs {previous_asked or 'unknown'} B",
                                previous_asked,
                                max(gained, 0),
                            )
                    if retryable and snapshot is None:
                        snapshot = self._arm(held, leaves)
                    self._sample()
                    frame = Frame(held, self._allocated())
                    self.stack.append(frame)
                    self._ready(held.key)
                    entered = True
                    with self._traffic(counts, "invoke"):
                        out = fn(*args, **kwargs)
                except BaseException as exc:
                    oom = is_oom(exc)
                    if oom:
                        self._sample()
                        asked = self.oom_asked or _asked(str(exc))
                        text = (str(exc).splitlines() or [type(exc).__name__])[0][:300]
                        # Failed Python frames can retain the very activations needed for
                        # rollback. Keep the evidence, then release them before restoring.
                        exc.__traceback__ = None
                    if entered:
                        self._rollback(held, snapshot)
                    if not oom:
                        raise
                else:
                    self._sample()
                    assert frame is not None
                    self.working = max(self.working, frame.working)
                    record = self.records.setdefault(key, Record())
                    record.ok = max(record.ok, frame.working)
                    if record.fail > record.ok:
                        record.fail = 0
                    seen = self.plans.setdefault(f"{held.key.rsplit('/', 1)[-1]}.{method}", [])
                    shape = f"{dict(key.mode)} {plan.keep}/{len(held.part.blocks)} blocks"
                    if shape not in seen:
                        seen.append(shape)
                    return out
                finally:
                    self._sample()
                    if frame is not None:
                        self.stack.remove(frame)
                    self.oom_asked = 0
                floor, previous_room, asked_worker = self._recover(
                    held,
                    key,
                    plan,
                    frame or Frame(held, self._allocated()),
                    snapshot,
                    asked,
                    text,
                    entered=entered,
                    request_room=not room_tried,
                )
                room_tried |= asked_worker
                previous_asked = asked

        finally:
            # Tracebacks may keep this frame alive after terminal refusal. They must
            # not retain the baseline's host allocations or opaque author snapshot.
            if snapshot is not None:
                snapshot.tensors.clear()
                snapshot.rng.clear()
                snapshot.state = ()
                snapshot.custom = None

    def _rollback(self, held: Held, snapshot: Snapshot | None) -> None:
        """Every failed invocation rolls back, even when no replay will follow."""
        if snapshot is None:
            self.unusable = f"{held.key}: failed operation has no rollback contract"
            return
        try:
            if self.cuda:
                self.torch.cuda.synchronize(self.device)
            self._restore(snapshot)
            if self.cuda:
                self.torch.cuda.synchronize(self.device)
        except BaseException as exc:
            self.unusable = f"{held.key}: rollback failed: {type(exc).__name__}: {exc}"
            exc.__traceback__ = None
            raise MemoryRefusal("device_fault", self.unusable, reusable=False) from None

    def _recover(
        self,
        held: Held,
        key: OpKey,
        plan: Plan,
        frame: Frame,
        snapshot: Snapshot | None,
        asked: int,
        text: str,
        *,
        entered: bool = True,
        request_room: bool = True,
    ) -> tuple[bool, int, bool]:
        """Record the OOM and select a strictly smaller placement of the same operation.

        The numerical mode belongs to the caller and never changes during planning or replay.
        """
        torch = self.torch
        try:
            tiers.trim(torch)
            clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear is not None:
                clear()
        except Exception as exc:
            self.unusable = f"{held.key}: the device failed while recovering: {exc}"
            raise MemoryRefusal("device_fault", self.unusable, reusable=False) from None
        bound = frame.working + asked
        record = self.records.setdefault(key, Record())
        record.fail = max(record.fail, bound)
        room = self.allocatable()
        numbers = (
            f"{held.key}.{key.method} ran out of device memory ({text}); it needs at least "
            f"{bound} B of working memory, {room} B are allocatable, "
            f"{plan.keep}/{len(held.part.blocks)} blocks resident"
        )
        self.log.append(numbers)
        smaller = snapshot is not None and plan.keep > self._minimum_keep(held)
        asked_worker = False
        # A known floor still gets one measured Worker request after an actual OOM.
        # Preparation can use this too: authored code has not run and needs no rollback.
        if not smaller and request_room and (snapshot is not None or not entered):
            asked_worker = self._ask_room(room + max(asked, 1), [], measured=True, force=True)
        if not smaller and not asked_worker:
            why = (
                "its state cannot be put back"
                if snapshot is None
                else "no smaller placement or additional room is left"
            )
            if not entered:
                why = "operation preparation did not complete"
            raise MemoryRefusal(
                DEVICE_OOM,
                numbers + "; not retried: " + why,
                bound,
                room,
                reusable=not self.unusable,
            )
        # The next iteration proves actual headroom growth before authored code runs.
        return True, room, asked_worker

    # ------------------------------------------------------------------ planning

    def _estimate(self, key: OpKey) -> tuple[int | None, bool]:
        """Working bytes for the same authored mode, exact or scaled from known shapes."""
        record = self.records.get(key)
        if record is not None and record.ok >= 0:
            return max(record.ok, record.fail), True
        scaled: list[float] = []
        for other, row in self.records.items():
            ratio = key.ratio(other) if row.ok >= 0 and other.component == key.component else None
            if ratio is not None:
                scaled.append(row.ok * max(ratio, 1.0))
        floor = record.fail if record is not None else 0
        return (max(int(max(scaled)), floor) if scaled else floor or None), False

    def _agreed(
        self,
        held: Held,
        key: OpKey,
        retryable: bool,
        floor: bool,
    ) -> Plan:
        """A sharded op's plan, agreed by every GPU of the group before any enters the
        collective: all refuse together, after rank 0 has asked the Worker for room once."""
        if self.agree is None or not self.sharded(held.key):
            return self._plan(held, key, retryable, floor)
        refusal: MemoryRefusal | None = None
        for last in (False, True):
            try:
                plan, short = self._plan(held, key, retryable, floor), 0
            except MemoryRefusal as exc:
                refusal = exc
                short = max(exc.needed - exc.available, 1)
            worst = self.agree(short)
            if worst == 0:
                return plan
            if last:
                raise refusal or MemoryRefusal(
                    SHORTFALL, f"{held.key}: another GPU of the group is short by {worst} B"
                )
        raise AssertionError("unreachable")

    def _plan(
        self,
        held: Held,
        key: OpKey,
        retryable: bool,
        floor: bool,
    ) -> Plan:
        working, exact = self._estimate(key)
        if self.cuda and self._reserved() > self._allocated():
            resident = self._need(held.part, len(held.part.blocks))
            if resident + (working or 0) > self.allocatable():
                # The previous op's cache may be all that stands in the way: give it back so
                # the plan reads what can really be reserved.
                tiers.trim(self.torch)
        for deep in (False, True):
            # The second pass also takes the common units of the scope's other components:
            # the last room there is, before a refusal.
            self.deep = deep
            victims = [] if self.spare else self._victims(held)
            try:
                if floor or (not retryable and not exact):
                    plan = self._fit(held, working or 0, victims, floor=True)
                elif retryable and working is None:
                    plan = self._max_room(held, victims)
                else:
                    plan = self._fit(held, working or 0, victims, floor=False)
            except MemoryRefusal:
                if deep or self.spare:
                    raise
                continue
            finally:
                self.deep = False
            plan.deep = deep
            return plan
        raise AssertionError("unreachable")

    def _fit(
        self,
        held: Held,
        working: int,
        victims: list[Held],
        *,
        floor: bool,
    ) -> Plan:
        """Keep the most blocks that fit; the floor streams every legal block."""
        needed = available = 0
        for keep in [self._minimum_keep(held)] if floor else self._keeps(held):
            needed = self._need(held.part, keep) + working
            chosen = self._room_from(victims, needed)
            if chosen is None and self._ask_room(needed, victims, measured=floor or working > 0):
                chosen = self._room_from(victims, needed)
            if chosen is not None:
                return Plan(keep, list(victims) if floor else chosen)
            available = self.allocatable() + sum(self._evictable(v) for v in victims)
        if self.spare:
            raise MemoryRefusal(
                SPARED, f"{held.key}: declined, {needed} B do not fit {available} B"
            )
        raise MemoryRefusal(
            SHORTFALL,
            f"{held.key}: the smallest plan needs {needed} B (common, one streamed block and "
            f"measured working memory) and {available} B can be made allocatable",
            needed,
            available,
        )

    def _max_room(self, held: Held, victims: list[Held]) -> Plan:
        """A retryable op never measured gets all the room this device can make, with its
        weights resident when they fit whole."""
        whole = self._need(held.part, len(held.part.blocks))
        own = sum(self._evictable(v) for v in victims)
        if not self.spare and whole > self.allocatable() + own:
            self._ask_room(whole, victims, measured=False)
        keep = (
            len(held.part.blocks) if whole <= self.allocatable() + own else self._minimum_keep(held)
        )
        return Plan(keep, list(victims))

    def _minimum_keep(self, held: Held) -> int:
        if held.compiled:
            return len(held.part.blocks)
        return max(
            (
                i + 1
                for i, unit in enumerate(held.part.blocks)
                if unit.mutable or (self.grouped and not unit.readonly)
            ),
            default=0,
        )

    def _keeps(self, held: Held) -> list[int]:
        """Resident block counts to try, most first; a component the store only refills whole
        and that is on disk, or a compiled one, is all or nothing."""
        blocks = len(held.part.blocks)
        if held.compiled or (held.part.on(DISK) and not held.part.unit_loads):
            return [blocks]
        return list(range(blocks, self._minimum_keep(held) - 1, -1))

    def _need(self, part: tiers.Component, keep: int) -> int:
        """Net bytes the device must still find for `part` with `keep` blocks resident."""
        common = part.common.nbytes if part.common.tier != DEVICE else 0
        if part.on(DISK) and not part.unit_loads and part.store is not None:
            common += part.store.transient(part.name)
        load = sum(u.nbytes for u in part.blocks[:keep] if u.tier != DEVICE)
        drop = sum(u.nbytes for u in part.blocks[keep:] if u.tier == DEVICE)
        stream = max((u.nbytes for u in part.blocks[keep:]), default=0)
        return common + load - drop + stream

    def _room_from(self, victims: list[Held], need: int) -> list[Held] | None:
        """The fewest victims, in order, whose bytes make `need` allocatable; None if all can't."""
        room = self.allocatable()
        chosen: list[Held] = []
        for victim in victims:
            if room >= need:
                break
            chosen.append(victim)
            room += self._evictable(victim)
        return chosen if room >= need else None

    def _ask_room(
        self,
        need: int,
        victims: list[Held],
        *,
        measured: bool,
        force: bool = False,
    ) -> bool:
        """Ask the Worker, once an attempt unless the grant grew, for driver-free bytes other
        tenants hold. True when it was asked, so the caller measures again."""
        if self.room is None or (self.asked_room and not force) or self.spare:
            return False
        own = sum(self._evictable(v) for v in victims)
        slack = self._reserved() - self._allocated()
        before = self.grant
        self.room_asks += 1
        grant = self.room(
            max(need - own - slack, 0) + self.overhead,
            self._allocated() - own + need + self.overhead,
            measured,
        )
        if grant is not None:
            self.grant = grant
        self.asked_room = grant is None or (before is not None and grant <= before)
        self._cap()
        return True

    def _pinned(self) -> tuple[set[str], set[str]]:
        """Components pinned whole (an open op, or one the package reads itself while a scope
        declaring it is open), and components whose common unit is pinned (in an open scope)."""
        whole = {frame.held.key for frame in self.stack}
        common: set[str] = set()
        for construction, names in self.scopes:
            for name in names:
                key = f"{construction}/{name}"
                if not self.deep:
                    common.add(key)
                held = self.held.get(key)
                if held is not None and held.whole:
                    whole.add(key)
        return whole, common

    def _evictable(self, victim: Held) -> int:
        whole, common = self._pinned()
        victim.compiled = _compiled(victim.part.root)
        if victim.key in whole or victim.compiled:
            return 0
        units = victim.part.blocks if victim.key in common else victim.part.units
        return sum(unit.nbytes for unit in units if unit.tier == DEVICE and not unit.mutable)

    def _victims(self, held: Held) -> list[Held]:
        """Components that may give device bytes to `held`, furthest next use first: other
        constructions, then what this attempt is done with, then the rest; ties LRU."""
        construction = held.construction
        previous = self.last_seq.get(construction, [])
        position = len(self.seq)

        def next_use(victim: Held) -> float:
            if victim.construction != construction:
                return math.inf
            later = [i for i in range(position, len(previous)) if previous[i] == victim.key]
            return float(later[0]) if later else math.inf

        candidates = [v for v in self.held.values() if v is not held and self._evictable(v)]
        return sorted(
            candidates,
            key=lambda v: (v.construction == construction, -next_use(v), v.last_used),
        )

    # ------------------------------------------------------------------ moving bytes

    def _apply(self, held: Held, plan: Plan) -> None:
        self.deep = plan.deep
        try:
            for victim in plan.evict:
                self._offload(victim)
        finally:
            self.deep = False
        part = held.part
        for unit in part.blocks[plan.keep :]:
            if unit.tier == DEVICE:
                self._home(part, unit)
                self.dropped = True
        self._load_common(held)
        for unit in part.blocks[: plan.keep]:
            self._load(part, unit)
        self._cap()

    @contextmanager
    def _moving(self) -> Iterator[None]:
        """Count what a weight move changes on the allocator: the attempt's transition."""
        before = self._allocated()
        try:
            yield
        except RuntimeFailure as exc:
            if exc.code == "device_fault":
                self.unusable = str(exc)
            raise
        finally:
            self.transition += self._allocated() - before

    def _load_common(self, held: Held) -> None:
        part = held.part
        if part.common.tier == DEVICE:
            return
        if part.common.tier == DISK and not part.unit_loads:
            self.refilled += part.on(DISK)
            with self._moving():
                tiers.from_disk(self.torch, part, None)
            return
        self._load(part, part.common)

    def _load(self, part: tiers.Component, unit: tiers.Unit) -> None:
        if unit.tier == DEVICE:
            return
        with self._moving():
            if unit.tier == HOST:
                self.moved += tiers.to_device(self.torch, unit, self.device)
            else:
                tiers.from_disk(self.torch, part, unit)
                self.moved += unit.nbytes
                self.refilled += unit.nbytes
        self.loads += 1

    def _home(self, part: tiers.Component, unit: tiers.Unit) -> None:
        """Send a device unit to host memory when the host has room, else to the store."""
        self.offloads += 1
        needed = tiers.host_copy_bytes(unit)
        with self._moving(), self._host_copy(needed, protected=(unit,)) as admitted:
            if admitted:
                self.moved += tiers.to_host(self.torch, unit)
            elif (
                unit.readonly
                and not unit.mutable
                and part.unit_loads
                and part.store is not None
                and tiers.store_safe(unit)
            ):
                self.stored += unit.nbytes
                tiers.to_disk(self.torch, part, unit)
            else:
                raise self._host_refusal(needed)

    def _offload(self, held: Held, *, everything: bool = False) -> None:
        """Move a victim's unpinned device units off the device."""
        part = held.part
        whole, common = self._pinned()
        held.compiled = _compiled(held.part.root)
        if held.compiled or (held.key in whole and not everything):
            return
        units = part.blocks if held.key in common and not everything else part.units
        moving = [unit for unit in units if unit.tier == DEVICE and not unit.mutable]
        self.dropped |= bool(moving)
        if (
            len(moving) == len(part.units)
            and all(unit.readonly and tiers.store_safe(unit) for unit in moving)
            and part.store is not None
            and not part.unit_loads
        ):
            needed = sum(tiers.host_copy_bytes(unit) for unit in moving)
            with self._moving(), self._host_copy(needed, protected=tuple(moving)) as admitted:
                if admitted:
                    for unit in moving:
                        self.moved += tiers.to_host(self.torch, unit)
                        self.offloads += 1
                else:
                    self.stored += sum(unit.nbytes for unit in moving)
                    tiers.to_disk(self.torch, part, None)
            return
        for unit in moving:
            self._home(part, unit)

    @contextmanager
    def _host_copy(self, nbytes: int, *, protected: tuple[tiers.Unit, ...] = ()) -> Iterator[bool]:
        """Reserve before allocating and release the pending promise on every exit.

        Standalone planners use local observed capacity only. Managed executors supply
        the aggregate Worker callback. Group phases never send Worker IPC: without a
        pregrant they can only rebind cached bytes or take a legal store path.
        """
        if not nbytes:
            yield True
            return
        if self.grouped or self.group_phase:
            yield False
            return
        self._host_sequence += 1
        token = f"host-{self._host_sequence}"

        def reserve() -> bool:
            if self.host_room is not None:
                return self.host_room(token, nbytes, False)
            return proctree.available_host_bytes() >= nbytes + self._host_pending

        admitted = reserve()
        if not admitted:
            self._release_host(nbytes, protected=protected)
            admitted = reserve()
        if not admitted:
            yield False
            return
        self._host_pending += nbytes
        try:
            yield True
        finally:
            self._host_pending -= nbytes
            if self.host_room is not None:
                self.host_room(token, nbytes, True)

    def _host_refusal(self, nbytes: int) -> MemoryRefusal:
        refusal = MemoryRefusal(
            "host_shortfall",
            f"no host reservation for {nbytes} B and no legal store fallback",
            nbytes,
        )
        refusal.shortfall["resource"] = "ram"
        return refusal

    def _release_host(self, nbytes: int, *, protected: tuple[tiers.Unit, ...] = ()) -> None:
        running = {frame.held.key for frame in self.stack}
        idle = sorted(
            (held for held in self.held.values() if held.key not in running),
            key=lambda held: held.last_used,
        )
        for clean in (True, False):
            for held in idle:
                for unit in held.part.units:
                    if unit in protected:
                        continue
                    if nbytes <= 0:
                        return
                    if clean and unit.tier == DEVICE and unit.host:
                        nbytes -= sum(t.numel() * t.element_size() for t in unit.host.values())
                        unit.host.clear()
                    elif (
                        not clean
                        and unit.tier == HOST
                        and held.part.unit_loads
                        and not unit.mutable
                        and unit.readonly
                        and tiers.store_safe(unit)
                    ):
                        nbytes -= unit.nbytes
                        self.stored += unit.nbytes
                        tiers.to_disk(self.torch, held.part, unit)

    def _enter_unit(self, part: tiers.Component, unit: tiers.Unit) -> None:
        """A block's forward is about to run: stream it in if the plan left it out."""
        self._sample()
        if unit.tier == DEVICE:
            return
        self._load(part, unit)
        unit.streamed = True
        if self.stack:
            self.stack[-1].loaded += unit.nbytes

    def _leave_unit(self, part: tiers.Component, unit: tiers.Unit) -> None:
        self._sample()
        if not unit.streamed:
            return
        unit.streamed = False
        self._home(part, unit)
        if self.stack:
            self.stack[-1].loaded -= unit.nbytes

    # ------------------------------------------------------------------ retry safety

    def _pin_state(self, held: Held) -> None:
        """Checkpoint residency never owns explicitly mutable tensor state."""
        provider = getattr(held.part.root, "_cozy_retry_tensors", None)
        if provider is None:
            return
        state = tuple(provider())
        identities = {id(tensor) for tensor in state}
        storages = {
            tensor.untyped_storage().data_ptr()
            for tensor in state
            if not tensor.is_meta and tensor.numel()
        }
        for unit in held.part.units:
            if any(
                id(tensor) in identities
                or (
                    not tensor.is_meta
                    and tensor.numel()
                    and tensor.untyped_storage().data_ptr() in storages
                )
                for *_, tensor in unit.slots()
            ):
                unit.mutable = True

    def _retryable(self, held: Held, leaves: list[object]) -> bool:
        """Only a positive author contract admits replay; unknown operations still run."""
        return (
            not self.torch.is_grad_enabled()
            and not held.compiled
            and not self.sharded(held.key)
            and self._state_contract(held, leaves)
        )

    def _state_contract(self, held: Held, leaves: list[object]) -> bool:
        """The current body and hooks are covered, even where replay is disabled."""
        torch = self.torch
        root = held.part.root
        hooks = torch.nn.modules.module
        if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
            return False
        custom = getattr(root, "_cozy_retry_state", None)
        forward = tiers.original(getattr(root, "forward", None))
        if custom is None and not (is_declared_pure(root) or is_declared_pure(forward)):
            return False

        def identity(value: object) -> tuple[int, int]:
            value = tiers.original(value)
            return id(getattr(value, "__self__", None)), id(getattr(value, "__func__", value))

        claimed_hooks = {identity(fn) for fn in getattr(root, "_cozy_retry_hooks", ())}
        claimed_forwards = {identity(fn) for fn in getattr(root, "_cozy_retry_forwards", ())}
        for module in root.modules():
            registry = vars(module).get("_diffusers_hook")
            if registry is not None and any(
                getattr(hook, "_is_stateful", True) and identity(hook) not in claimed_hooks
                for hook in getattr(registry, "hooks", {}).values()
            ):
                return False
            # The module contract covers class implementations, but not an unclassified
            # replacement installed afterward. Our adapters retain the actual callable.
            replaced = tiers.original(vars(module).get("forward"))
            if replaced is not None and not (
                is_declared_pure(replaced)
                or identity(replaced) in claimed_forwards
                or (
                    getattr(replaced, "__self__", None) is module
                    and getattr(replaced, "__func__", None)
                    is getattr(type(module), "forward", None)
                )
            ):
                return False
            if not all(tiers.ours(fn) or identity(fn) in claimed_hooks for fn in _hook_fns(module)):
                return False
        plain = (
            *_SCALAR,
            torch.Tensor,
            torch.Generator,
            torch.dtype,
            torch.device,
            torch.nn.Module,
        )
        return custom is not None or all(isinstance(leaf, plain) for leaf in leaves)

    def _arm(self, held: Held, leaves: list[object]) -> Snapshot:
        torch = self.torch
        root = held.part.root
        resolve = getattr(root, "_cozy_retry_tensors", lambda: ())
        state = tuple(resolve())
        storages: dict[tuple[str, int], Tensor] = {}
        for tensor in state:
            if tensor.is_meta or tensor.layout != torch.strided:
                raise TypeError("retry tensor state must have materialized strided storage")
            storage = tensor.untyped_storage()
            storages.setdefault((str(tensor.device), storage.data_ptr()), tensor)
        needed = sum(tensor.untyped_storage().nbytes() for tensor in storages.values())
        tensors: list[tuple[Tensor, Tensor]] = []
        with self._host_copy(needed) as admitted:
            if not admitted:
                raise self._host_refusal(needed)
            try:
                if self.cuda and state:
                    torch.cuda.synchronize(self.device)
                with torch.inference_mode(False), torch.no_grad():
                    for tensor in storages.values():
                        storage = tensor.untyped_storage()
                        view = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                            storage, 0, (storage.nbytes(),), (1,)
                        )
                        tensors.append((view, view.to(device="cpu", copy=True)))
                rng: list[tuple[object, object]] = [(None, torch.get_rng_state())]
                if self.cuda:
                    rng.append((self.device, torch.cuda.get_rng_state(self.device)))
                generators = [x for x in leaves if isinstance(x, torch.Generator)]
                generators += list(getattr(root, "_cozy_retry_generators", lambda: ())())
                unique = {id(generator): generator for generator in generators}
                rng += [(generator, generator.get_state()) for generator in unique.values()]
                custom = getattr(root, "_cozy_retry_state", None)
                return Snapshot(
                    tensors,
                    rng,
                    state,
                    tuple(_geometry(tensor) for tensor in state),
                    resolve,
                    (custom[1], custom[0]()) if custom else None,
                )
            except BaseException:
                tensors.clear()  # A retained failure traceback must not keep backing copies.
                raise

    def _restore(self, snapshot: Snapshot) -> None:
        torch = self.torch
        current = tuple(snapshot.resolve())
        if len(current) != len(snapshot.state) or any(
            tensor is not original or _geometry(tensor) != geometry
            for tensor, original, geometry in zip(
                current, snapshot.state, snapshot.geometry, strict=True
            )
        ):
            raise RuntimeError("declared retry tensor state was rebound or resized")
        with torch.no_grad():
            for original, saved in snapshot.tensors:
                if original.untyped_storage().nbytes() != saved.numel():
                    raise RuntimeError("declared retry tensor backing storage was resized")
                original.copy_(saved)
        if snapshot.custom is not None:
            restore, state = snapshot.custom
            restore(state)
        # Restore RNG last: an author's restore callback must not advance the baseline.
        for owner, state in snapshot.rng:
            if owner is None:
                torch.set_rng_state(state)
            elif isinstance(owner, torch.Generator):
                owner.set_state(state)
            else:
                torch.cuda.set_rng_state(state, owner)


def _compiled(root: Module) -> bool:
    return any(
        getattr(module, "_compiled_call_impl", None) is not None
        or getattr(module, "_orig_mod", None) is not None
        for module in root.modules()
    )


def _geometry(tensor: Tensor) -> tuple[object, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _hook_fns(module: Module) -> Iterator[object]:
    yield from module._forward_hooks.values()
    yield from module._forward_pre_hooks.values()


def _mode(root: Module) -> tuple[tuple[str, object], ...]:
    """Measurement identity only: Runtime never changes authored decomposition settings."""
    names: tuple[str, ...] = ("use_tiling", "use_slicing", "tile_overlap_factor")
    names += tuple(
        "tile_" + space + "_" + setting
        for space in ("sample", "latent")
        for setting in (
            "min_size",
            "min_height",
            "min_width",
            "min_num_frames",
            "stride_height",
            "stride_width",
            "stride_num_frames",
        )
    )
    return tuple((name, getattr(root, name)) for name in names if hasattr(root, name))


def hosting_plan(
    *,
    placeable: tuple[str, ...],
    sizes: Mapping[str, int],
    sharded: set[str],
    scopes: Mapping[str, tuple[str, ...]],
    grants: tuple[int, ...],
) -> dict[str, int]:
    """Choose shared component homes from the same authorized vector on every rank.

    These are candidate homes, not a whole-footprint fit proof: local preparation still
    accounts for overhead, live inputs and working memory. Ulysses distributes work, not
    weights, so each follower keeps the full sharded component bytes beside a hosted root.
    """
    if len(grants) < 2 or sum(sizes.values()) <= grants[0]:
        return {}
    shared = max(
        (sum(sizes.get(name, 0) for name in names if name in sharded) for names in scopes.values()),
        default=0,
    )
    room = {rank: max(grant - shared, 0) for rank, grant in enumerate(grants) if rank}
    hosted: dict[str, int] = {}
    for name in sorted(placeable, key=lambda key: (-sizes.get(key, 0), key)):
        rank = max(room, key=lambda rank: (room[rank], -rank))
        if name in sizes and name not in sharded and sizes[name] <= room[rank]:
            hosted[name] = rank
            room[rank] -= sizes[name]
    return hosted


# --------------------------------------------------------------------------- the machine


@dataclass(frozen=True, slots=True)
class Demand:
    """A queue forecast, never permission to execute or create a GPU context.

    Artifact and representation identify the exact reusable bytes. `next_use` is the
    scheduler's queue position, not a wall-clock deadline; load cost is measured seconds.
    """

    tenant: str
    artifact: str
    representation: str
    devices: tuple[int, ...]
    next_use: int
    load_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(sorted(set(self.devices))))
        if self.next_use < 0 or any(device < 0 for device in self.devices):
            raise ValueError("queue position and device ordinals must be nonnegative")
        if self.load_seconds is not None and (
            not math.isfinite(self.load_seconds) or self.load_seconds < 0
        ):
            raise ValueError("load cost must be finite and nonnegative")


class Tenant(Protocol):
    """One executor as the Worker's memory policy sees it: a process over some devices."""

    slot: str

    @property
    def ordinals(self) -> tuple[int, ...]: ...

    @property
    def held(self) -> dict[int, int]:
        """Bytes it holds per device: non-allocator overhead plus allocator reserve."""
        ...

    @property
    def overhead(self) -> dict[int, int]: ...

    @property
    def last_used(self) -> int: ...

    def state(self) -> str:
        """`active` (it holds its devices), `idle`, or `dead`."""
        ...

    def vacate(self, ordinals: tuple[int, ...]) -> bool:
        """Its weights leave these devices; `held` is updated from the reply."""
        ...

    def end(self, why: str) -> bool:
        """Kill and reap its process; its devices' bytes return to the driver."""
        ...


class Machine:
    """Worker memory policy. Callers serialize quotes, reservations and mutations.

    Queue hints rank otherwise legal victims; scheduler admission remains authoritative.
    Driver samples are not an atomic hardware reservation: the Worker holds its device
    leases while applying a quote and checks actual room again after every reclaim.
    """

    def __init__(
        self,
        budget: Mapping[int, int],
        driver: Callable[[int], tuple[int, int] | None],
        note: Callable[[str], None],
        *,
        host_available: Callable[[], int] | None = None,
    ) -> None:
        self.budget = dict(budget)
        self.driver = driver
        self.note = note
        self.queue: tuple[Demand, ...] = ()
        from threading import Lock

        self._host_lock = Lock()
        self.host_available = host_available
        self.host_reserved: dict[str, int] = {}

    def update_queue(self, demands: Iterable[Demand]) -> None:
        """Replace the forecast atomically; stale hints cannot authorize any operation."""
        self.queue = tuple(sorted(demands, key=lambda d: (d.next_use, d.tenant, d.artifact)))

    def prefetch_frontier(self) -> tuple[Demand, ...]:
        """The earliest cohort on each device, deduplicated by exact representation.

        This is only a store-read candidate list. The caller must separately establish
        eligibility and reserve host bytes; it must not create an executor from this hint.
        """
        seen: set[tuple[str, str, tuple[int, ...]]] = set()
        first: dict[int, int] = {}
        unknown = False
        frontier: list[Demand] = []
        for demand in self.queue:
            if not demand.devices:
                if unknown:
                    continue
                unknown = True
            elif any(
                first.setdefault(device, demand.next_use) < demand.next_use
                for device in demand.devices
            ):
                # A group waiting on one busy device still precedes later work on all
                # its other devices; do not let a later single-device hint pass it.
                for device in demand.devices:
                    first.setdefault(device, demand.next_use)
                continue
            key = (demand.artifact, demand.representation, demand.devices)
            if key not in seen:
                seen.add(key)
                frontier.append(demand)
        return tuple(frontier)

    def reserve_host(self, owner: str, nbytes: int) -> bool:
        """Reserve incremental copy/cache bytes against observed room and outstanding claims.

        Use one owner token per copy. Release its promise when allocation completes;
        cgroup-aware MemAvailable then reflects the actual backing. Unknown availability
        refuses new storage.
        """
        if nbytes < 0:
            raise ValueError("host reservation must be nonnegative")
        if not nbytes:
            return True
        if self.host_available is None:
            return False
        with self._host_lock:
            available = self.host_available()
            reserved = sum(self.host_reserved.values())
            if available < 0 or nbytes > available - reserved:
                return False
            self.host_reserved[owner] = self.host_reserved.get(owner, 0) + nbytes
            return True

    def release_host(self, owner: str) -> None:
        """The allocation completed or failed: stop charging its pending promise."""
        with self._host_lock:
            self.host_reserved.pop(owner, None)

    def release_host_prefix(self, prefix: str) -> None:
        """Drop a confirmed-reaped generation's claims atomically, preserving other owners."""
        if not prefix:
            raise ValueError("host claim cleanup requires an explicit owner namespace")
        with self._host_lock:
            for token in tuple(self.host_reserved):
                if token.startswith(prefix):
                    del self.host_reserved[token]

    def quote(self, me: Tenant, tenants: Iterable[Tenant]) -> dict[int, int | None]:
        """Quote a complete device vector from one copy of every tenant's reported facts.

        Missing facts for a live neighbour remain unknown, never a zero charge. The
        caller must include tenants from every overlapping group, including active ones.
        """
        rows = {t.slot: (t.ordinals, dict(t.held)) for t in tenants}
        if me.slot not in rows:
            rows[me.slot] = (me.ordinals, dict(me.held))
        result: dict[int, int | None] = {}
        for ordinal in me.ordinals:
            measured = self.driver(ordinal)
            unknown = any(
                ordinal in devices and ordinal not in held
                for slot, (devices, held) in rows.items()
                if slot != me.slot
            )
            if measured is None or (unknown and (ordinal in self.budget or -1 in self.budget)):
                result[ordinal] = None
                continue
            free, total = measured
            if unknown:
                # Without a quota, free physical bytes are usable even when a legacy
                # neighbour cannot attribute its occupancy. Its missing facts stay missing.
                result[ordinal] = min(total, free + rows[me.slot][1].get(ordinal, 0))
                continue
            others = sum(
                held[ordinal]
                for slot, (devices, held) in rows.items()
                if slot != me.slot and ordinal in devices
            )
            mine = rows[me.slot][1].get(ordinal, 0)
            foreign = max(total - free - others - mine, 0)
            limit = min(self.budget.get(ordinal, self.budget.get(-1, total)), total - foreign)
            result[ordinal] = limit - others
        return result

    def admit(
        self, me: Tenant, tenants: Sequence[Tenant], peak: int | Mapping[int, int] | None
    ) -> dict[int, int | None]:
        """Make room for measured demand only. An unseen call discovers need inside its ops.

        Check the complete vector before moving anything so an impossible follower does
        not cause pointless eviction on the leader. Unknown peaks are never whole-device
        claims. Dead processes can still be reaped without sacrificing warm neighbours.
        """
        for tenant in tenants:
            if (
                tenant.slot != me.slot
                and tenant.state() == "dead"
                and set(tenant.ordinals) & set(me.ordinals)
            ):
                tenant.end(f"reclaimed before {me.slot!r} runs")
        peaks = dict(peak) if isinstance(peak, Mapping) else dict.fromkeys(me.ordinals, peak)
        return self.room_vector(
            me,
            {ordinal: (0, value, True) for ordinal, value in peaks.items() if value is not None},
            tenants,
        )

    def room_vector(
        self,
        me: Tenant,
        needs: Mapping[int, tuple[int, int | None, bool]],
        tenants: Sequence[Tenant],
    ) -> dict[int, int | None]:
        """Preflight the complete rank request before any potentially useless reclamation.

        The Worker holds device leases through the vector operation. These are memory
        feasibility checks, never execution grants or promises about future allocations.
        """
        if any(
            ordinal not in me.ordinals or free < 0 or (grant is not None and grant < 0)
            for ordinal, (free, grant, _) in needs.items()
        ):
            raise ValueError("room vector must name this tenant's devices and nonnegative bytes")
        quotes = self.quote(me, tenants)
        if any(
            not self._possible(me, ordinal, free, grant, tenants, measured, quotes)
            for ordinal, (free, grant, measured) in needs.items()
        ):
            self.note(f"{me.slot!r}: measured device demand cannot fit; preserving idle tenants")
            return quotes
        for ordinal, (free, grant, measured) in needs.items():
            self.room(me, ordinal, free, grant, tenants, measured=measured)
        return self.quote(me, tenants)

    def _possible(
        self,
        me: Tenant,
        ordinal: int,
        free_wanted: int,
        grant_wanted: int | None,
        tenants: Sequence[Tenant],
        measured: bool,
        quotes: Mapping[int, int | None],
        driver: tuple[int, int] | None = None,
    ) -> bool:
        """An optimistic reclaim bound: failure proves eviction cannot satisfy the request."""
        reading = self.driver(ordinal) if driver is None else driver
        grant = quotes.get(ordinal)
        if reading is None or grant is None:
            return False
        reclaim = 0
        unknown = False
        for tenant in tenants:
            if (
                tenant.slot == me.slot
                or ordinal not in tenant.ordinals
                or tenant.state() == "active"
            ):
                continue
            held = tenant.held.get(ordinal)
            if held is None:
                unknown = True
                continue
            reclaim += (
                held
                if measured or tenant.state() == "dead"
                else max(held - tenant.overhead.get(ordinal, held), 0)
            )
        if unknown:
            # A legacy idle rank may hold some of the observed occupied bytes. Its
            # missing report cannot prove reclamation futile; try once and remeasure.
            pinned = sum(
                t.held.get(ordinal, 0)
                for t in tenants
                if t.slot == me.slot or t.state() == "active"
            )
            limit = min(self.budget.get(ordinal, self.budget.get(-1, reading[1])), reading[1])
            return free_wanted <= reading[1] - pinned and (
                grant_wanted is None or grant_wanted <= limit
            )
        return reading[0] + reclaim >= free_wanted and (
            grant_wanted is None or grant + reclaim >= grant_wanted
        )

    def room(
        self,
        me: Tenant,
        ordinal: int,
        free_wanted: int | None,
        grant_wanted: int | None,
        tenants: Sequence[Tenant],
        *,
        measured: bool,
    ) -> int | None:
        """Reclaim only enough for a concrete request, never an active tenant.

        `free_wanted=None` is an explicit legacy-executor request to vacate idle weights;
        new executors never use it for unseen demand. Recheck feasibility and actual facts
        after each action; each tenant/action is attempted at most once.
        """
        tried: set[str] = set()
        while True:
            reading = self.driver(ordinal)
            quotes = self.quote(me, tenants)
            grant = quotes.get(ordinal)
            if reading is None:
                return None
            if free_wanted is not None:
                if grant is None:
                    return None
                if reading[0] >= free_wanted and (grant_wanted is None or grant >= grant_wanted):
                    return grant
                if not self._possible(
                    me, ordinal, free_wanted, grant_wanted, tenants, measured, quotes, reading
                ):
                    self.note(f"GPU {ordinal}: requested room cannot fit; preserving idle tenants")
                    return grant
            victim = self._victim(me, ordinal, tenants, tried, measured and free_wanted is not None)
            if victim is None:
                self.note(f"GPU {ordinal}: no eligible reclaim remains for {me.slot!r}")
                return grant
            tenant, act = victim
            tried.add(f"{tenant.slot}:{act}")
            if act == "vacate":
                done = tenant.vacate((ordinal,))
            else:
                done = tenant.end(f"its process gave GPU {ordinal} to {me.slot!r}")
            verb = "vacated" if act == "vacate" else "ended"
            result = verb if done else f"could not {act}"
            self.note(f"GPU {ordinal}: {result} {tenant.slot!r} for {me.slot!r}")

    def _victim(
        self, me: Tenant, ordinal: int, tenants: Sequence[Tenant], tried: set[str], measured: bool
    ) -> tuple[Tenant, str] | None:
        present = [
            t
            for t in tenants
            if t.slot != me.slot and ordinal in t.ordinals and t.state() != "active"
        ]
        for act, pool in (
            ("dead", [t for t in present if t.state() == "dead"]),
            (
                "vacate",
                [
                    t
                    for t in present
                    if ordinal not in t.held or t.held[ordinal] > t.overhead.get(ordinal, 0)
                ],
            ),
            ("process", [t for t in present if t.held.get(ordinal, 0)] if measured else []),
        ):
            pool = [t for t in pool if f"{t.slot}:{act}" not in tried]
            if pool:
                return min(pool, key=lambda t: self._reuse(t, ordinal)), act
        return None

    def _reuse(self, tenant: Tenant, ordinal: int) -> tuple[float, float, int, str]:
        """Farthest next use first; among ties, discard the cheapest measured reload.

        Missing demand means no predicted reuse. Missing cost is not a free reload.
        Last use breaks prediction ties, so an empty queue retains a deterministic policy.
        """
        demands = [d for d in self.queue if d.tenant == tenant.slot and ordinal in d.devices]
        next_use = min((d.next_use for d in demands), default=math.inf)
        costs = [d.load_seconds for d in demands if d.next_use == next_use]
        cost = max((math.inf if c is None else c for c in costs), default=math.inf)
        return -next_use, cost, tenant.last_used, tenant.slot
