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
from cozy_runtime.internal import proctree, tiers
from cozy_runtime.internal.tiers import DEVICE, DISK, HOST, Device, Module, Tensor, Torch

SHORTFALL = "device_shortfall"
SPARED = "device_spared"
_ASKED = re.compile(r"Tried to allocate ([0-9.]+) ?([KMGT]?i?B)")
_SCALAR = (bool, int, float, str, type(None))


class MemoryRefusal(RuntimeFailure):
    """A typed memory outcome carrying the numbers the terminal reports."""

    def __init__(self, code: str, message: str, needed: int = 0, available: int = 0) -> None:
        super().__init__(message, code=code)
        self.needed, self.available = int(needed), int(available)
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
    if name == "OutOfMemoryError":
        return True
    if name == "AcceleratorError":
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


def _plain(value: object) -> bool:
    """A scalar, or a builtin list, tuple or dict holding only plain values: state a
    snapshot copies. Subclasses (a frozen config) are someone else's contract."""
    kind = type(value)
    if kind in _SCALAR:
        return True
    if isinstance(value, (list, tuple)) and kind in (list, tuple):
        return all(_plain(item) for item in value)
    if isinstance(value, dict) and kind is dict:
        return all(type(key) in _SCALAR and _plain(item) for key, item in value.items())
    return False


def _copy(value: object) -> object:
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    return value


def _put_back(owner: Module, name: str, saved: object) -> None:
    """Restore a changed attribute, in place for a container another object may share."""
    current = vars(owner).get(name)
    if type(current) is type(saved) and current == saved:
        return
    copied = _copy(saved)
    if type(current) is dict and isinstance(current, dict) and isinstance(copied, dict):
        current.clear()
        current.update(copied)
    elif type(current) is list and isinstance(current, list) and isinstance(copied, list):
        current[:] = copied
    else:
        setattr(owner, name, copied)


# --------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class OpKey:
    component: str
    method: str
    tensors: tuple[tuple[tuple[int, ...], str], ...]
    flags: tuple[bool | str | None, ...] = ()

    def ratio(self, other: OpKey) -> float | None:
        """How much larger this op's inputs are than `other`'s, argument by argument."""
        if (
            other.method != self.method
            or other.flags != self.flags
            or len(other.tensors) != len(self.tensors)
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
    vae: bool = False
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
    rung: int
    keep: int
    evict: list[Held]
    need: int
    working: int
    #: its victims include the scope's other components' common units
    deep: bool = False


@dataclass(eq=False)
class Snapshot:
    tensors: list[tuple[Tensor, Tensor]]
    rng: list[tuple[object, object]]
    attrs: list[tuple[Module, str, object]]
    custom: tuple[Callable[[object], None], object] | None


class Planner:
    """The executor's memory policy. One per process; the process owns one device."""

    def __init__(
        self,
        torch: Torch,
        device: Device,
        *,
        room: Callable[[int, int, bool], int | None] | None = None,
        sharded: Callable[[str], bool] = lambda key: False,
        agree: Callable[[int], int] | None = None,
    ) -> None:
        self.torch = torch
        self.device = device
        self.cuda = device.type == "cuda"
        #: asks the Worker for (driver-free bytes, total bytes to hold, measured need): the grant
        self.room = room
        self.sharded = sharded
        #: the group's largest shortfall, given this rank's (a collective)
        self.agree = agree
        self.grant: int | None = None
        #: this process's non-allocator device bytes: context, modules, communicators
        self.overhead = 0
        self.held: dict[str, Held] = {}
        self.records: dict[tuple[OpKey, int], Record] = {}
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
        #: allocator bytes the attempt's weight moves added (negative: gave back)
        self.transition = 0
        self.log: list[str] = []
        self.plans: dict[str, list[str]] = {}
        self.oom_asked = 0
        #: the typed refusal this attempt ended on, for the reply's shortfall numbers
        self.refusal: MemoryRefusal | None = None
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
        """Bytes an op may still allocate: inside the grant, and never the driver's headroom
        (one `overhead` kept free on the device). Negative when the process holds more than
        its grant, and the plan must give bytes back."""
        if not self.cuda:
            return 1 << 62
        free, total = self._driver()
        reserved = self._reserved()
        granted = (total if self.grant is None else self.grant) - self.overhead - reserved
        return min(granted, free - self.overhead) + reserved - self._allocated()

    def _cap(self) -> None:
        """The allocator stops at the grant, so an overrun is a clean OOM, not a driver fault."""
        if not self.cuda:
            return
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
            vae=callable(getattr(root, "enable_tiling", None)),
            compiled=getattr(root, "_compiled_call_impl", None) is not None,
            whole=bool(getattr(root, "_cozy_read_whole", False)),
        )
        self.held[key] = held
        if self.cuda:
            tiers.install(part, self._enter_unit, self._leave_unit)
            tiers.wrap(root, functools.partial(self.run, held))
            tiers.intercept_apply(self.torch, part, lambda: None)
            root.__dict__["_cozy_op"] = functools.partial(self.op, held)
        return held

    def op(self, held: Held, method: str) -> Callable[..., object]:
        """A method the package spreads, run as a planned op. It never tiles: another GPU may
        run the same call, and its output must not depend on which one did."""
        if method not in held.ops:
            bound = getattr(held.part.root, method)
            run = functools.partial(self.run, held, tiles=False)
            held.ops[method] = tiers.op(method, bound, run)
        return held.ops[method]

    def forget(self, construction: str) -> None:
        for key in [key for key, held in self.held.items() if held.construction == construction]:
            tiers.release(self.held.pop(key).part)

    def host(self, key: str) -> None:
        """Another GPU of the group keeps this component: it never occupies this one."""
        held = self.held[key]
        held.hosted = True
        if held.part.store is not None:
            tiers.to_disk(self.torch, held.part, None)

    # ------------------------------------------------------------------ attempts and scopes

    def begin(self, grant: int | None) -> None:
        if grant is not None:
            self.grant = grant
        self.asked_room = False
        self.peak = self.peak_allocated = self.working = 0
        self.loads = self.offloads = self.moved = self.transition = 0
        self.stored = self.refilled = 0
        self.log = []
        self.plans = {}
        self.refusal = None
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
            "attempt_activation_peak_bytes": self.working,
            "attempt_absolute_peak_bytes": self.peak_allocated,
            "allocator_delta_bytes": self.transition,
            "overhead_bytes": self.overhead,
            "grant_bytes": self.grant if self.grant is not None else -1,
            "log": list(self.log[-8:]),
            "plans": {op: list(seen) for op, seen in self.plans.items()},
            "poisoned": "",
        }

    def facts(self) -> dict[str, int]:
        return {
            "overhead_bytes": self.overhead,
            "reserved_bytes": self._reserved(),
            "allocated_bytes": self._allocated(),
            "peak_bytes": self.peak + self.overhead,
            "peak_allocated_bytes": self.peak_allocated,
            "working_bytes": self.working,
            "host_bytes": sum(
                tensor.numel() * tensor.element_size()
                for held in self.held.values()
                for unit in held.part.units
                for tensor in unit.host.values()
            ),
        }

    def enter(self, construction: str, method: str, components: tuple[str, ...]) -> None:
        """A declared method begins. Each declared component comes onto the device whole when
        it fits beside the rest of the scope, since a package may read its tensors outside
        any forward; otherwise its common unit does, so `.device` reads true, and its blocks
        arrive with its ops. One the package reads itself (`read_whole`) must fit whole."""
        self.scopes.append((construction, frozenset(components)))
        if not self.cuda:
            return
        scope = {f"{construction}/{name}" for name in components}
        for name in components:
            held = self.held.get(f"{construction}/{name}")
            if held is None or held.hosted:
                continue
            # A package may have replaced a forward since (a cache hook): it is an op again.
            tiers.wrap(held.part.root, functools.partial(self.run, held))
            part = held.part
            victims = [] if self.spare else [v for v in self._victims(held) if v.key not in scope]
            whole = part.nbytes - part.on(DEVICE)
            if part.on(DISK) and part.store is not None:
                whole += part.store.transient(part.name)
            common = part.common.nbytes if part.common.tier != DEVICE else 0
            chosen = self._room_from(victims, whole) if whole else []
            if chosen is None and (held.whole or not common):
                chosen = self._room_from(victims, whole) if self._ask_room(
                    whole, victims, measured=True
                ) else None
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

    def _refuse_scope(
        self, held: Held, method: str, need: int, victims: list[Held]
    ) -> NoReturn:
        self.scopes.pop()
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
                freed[held.key] = moved
        if self.cuda:
            tiers.trim(self.torch)
        return {"freed": freed, "held": {}, "freed_bytes": before - self._allocated()}

    # ------------------------------------------------------------------ the op

    def run(
        self,
        held: Held,
        method: str,
        fn: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        tiles: bool = True,
    ) -> object:
        """Plan, run, measure; after an OOM put the op's state back and run it again at the
        floor of the same rung, then of the next, when its state can be put back."""
        try:
            return self._run(held, method, fn, args, kwargs, tiles)
        except MemoryRefusal as exc:
            self.refusal = exc
            raise

    def _run(
        self,
        held: Held,
        method: str,
        fn: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        tiles: bool,
    ) -> object:
        if not self.cuda or held.hosted or any(frame.held is held for frame in self.stack):
            return fn(*args, **kwargs)
        torch = self.torch
        leaves = list(_leaves((args, kwargs)))
        key = OpKey(
            held.kind,
            method,
            tuple((tuple(x.shape), str(x.dtype)) for x in leaves if isinstance(x, torch.Tensor)),
            tuple(x for x in leaves if isinstance(x, (bool, str, type(None)))),
        )
        retryable = self._retryable(held, leaves)
        self.clock += 1
        held.last_used = self.clock
        self.seq.append(held.key)
        floor: int | None = None
        while True:
            plan = self._agreed(held, key, leaves, retryable, floor, tiles)
            snapshot = self._arm(held, leaves) if retryable else None
            frame: Frame | None = None
            asked, text = 0, ""
            try:
                self._apply(held, plan)
                self._sample()
                frame = Frame(held, self._allocated())
                self.stack.append(frame)
                with self._rung(held, plan.rung):
                    out = fn(*args, **kwargs)
            except BaseException as exc:
                if not is_oom(exc):
                    raise
                asked = self.oom_asked or _asked(str(exc))
                text = (str(exc).splitlines() or [type(exc).__name__])[0][:300]
                exc.__traceback__ = None
            else:
                self._sample()
                assert frame is not None
                self.working = max(self.working, frame.working)
                record = self.records.setdefault((key, plan.rung), Record())
                record.ok = max(record.ok, frame.working)
                if record.fail > record.ok:
                    record.fail = 0
                seen = self.plans.setdefault(f"{held.key.rsplit('/', 1)[-1]}.{method}", [])
                shape = f"{plan.rung or 'untiled'} {plan.keep}/{len(held.part.blocks)} blocks"
                if shape not in seen:
                    seen.append(shape)
                return out
            finally:
                self._sample()
                if frame is not None:
                    self.stack.remove(frame)
                self.oom_asked = 0
            floor = self._recover(
                held, key, plan, frame or Frame(held, self._allocated()), snapshot, asked, text
            )

    def _recover(
        self,
        held: Held,
        key: OpKey,
        plan: Plan,
        frame: Frame,
        snapshot: Snapshot | None,
        asked: int,
        text: str,
    ) -> int:
        """Free, then either put the op's state back and name the rung whose floor to retry
        at, or fail typed with the numbers."""
        torch = self.torch
        try:
            tiers.trim(torch)
            clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear is not None:
                clear()
        except Exception as exc:
            raise MemoryRefusal(
                "device_fault", f"{held.key}: the device failed while recovering: {exc}"
            ) from None
        bound = frame.working + asked
        record = self.records.setdefault((key, plan.rung), Record())
        record.fail = max(record.fail, bound)
        room = self.allocatable()
        numbers = (
            f"{held.key}.{key.method} ran out of device memory ({text}); it needs at least "
            f"{bound} B of working memory, {room} B are allocatable, "
            f"rung {plan.rung or 'untiled'}, {plan.keep}/{len(held.part.blocks)} blocks resident"
        )
        leaked = self._allocated() - frame.base
        if leaked > 0:
            numbers += f"; {leaked} B stayed allocated after it"
        self.log.append(numbers)
        rungs = self._rungs(held, key, True)[0]
        at = rungs.index(plan.rung) + (0 if plan.keep else 1)
        if snapshot is None:
            raise MemoryRefusal(
                DEVICE_OOM, numbers + "; not retried: its state cannot be put back", bound, room
            )
        if at >= len(rungs):
            raise MemoryRefusal(
                DEVICE_OOM, numbers + "; not retried: no smaller plan is left", bound, room
            )
        self._restore(snapshot)
        return at

    # ------------------------------------------------------------------ planning

    def _estimate(self, key: OpKey, rung: int, area: int) -> tuple[int | None, bool]:
        """Working bytes for `key` at `rung`: measured, else scaled from the same architecture's
        measured ops by the growth of each input and by the area it works on at once, else
        unknown."""
        record = self.records.get((key, rung))
        if record is not None and record.ok >= 0:
            return max(record.ok, record.fail), True
        pixels = rung * rung if rung else area
        scaled: list[float] = []
        for (other, other_rung), row in self.records.items():
            ratio = key.ratio(other) if row.ok >= 0 and other.component == key.component else None
            if ratio is None:
                continue
            # Working memory follows the area a VAE works on at once, both ways; it never
            # shrinks with a smaller input.
            spread = pixels / (other_rung * other_rung if other_rung else area) if area else 1.0
            scaled.append(row.ok * max(ratio, 1.0) * spread)
        floor = record.fail if record is not None else 0
        if not scaled:
            return (floor or None), False
        return max(int(max(scaled)), floor), False

    def _rungs(self, held: Held, key: OpKey, tiles: bool) -> tuple[list[int], int]:
        """0 is the untiled op; the rest are tile edges in pixels, halving, for a VAE."""
        ratio = _spatial_ratio(held.part.root) if held.vae and tiles else None
        if ratio is None or not key.tensors or len(key.tensors[0][0]) < 4:
            return [0], 0
        shape = key.tensors[0][0]
        edge = max(shape[-2:])
        pixels = edge if shape[1] == 3 else edge * ratio
        edges = [pixels >> step for step in (1, 2, 3) if pixels >> step >= 8 * ratio]
        return [0, *edges], pixels * pixels

    def _agreed(
        self,
        held: Held,
        key: OpKey,
        leaves: list[object],
        retryable: bool,
        floor: int | None,
        tiles: bool,
    ) -> Plan:
        """A sharded op's plan, agreed by every GPU of the group before any enters the
        collective: all refuse together, after rank 0 has asked the Worker for room once."""
        if self.agree is None or not self.sharded(held.key):
            return self._plan(held, key, leaves, retryable, floor, tiles)
        refusal: MemoryRefusal | None = None
        for last in (False, True):
            try:
                plan, short = self._plan(held, key, leaves, retryable, floor, tiles), 0
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
        leaves: list[object],
        retryable: bool,
        floor: int | None,
        tiles: bool,
    ) -> Plan:
        rungs, area = self._rungs(held, key, tiles)
        estimates = [(rung, *self._estimate(key, rung, area)) for rung in rungs]
        snapshot = (
            sum(x.numel() * x.element_size() for x in leaves if isinstance(x, self.torch.Tensor))
            if retryable
            else 0
        )
        exact = any(known for _, _, known in estimates)
        for deep in (False, True):
            # The second pass also takes the common units of the scope's other components:
            # the last room there is, before a refusal.
            self.deep = deep
            victims = [] if self.spare else self._victims(held)
            try:
                if floor is not None:
                    plan = self._fit(held, estimates[floor:], victims, snapshot, floor=True)
                elif retryable and all(working is None for _, working, _ in estimates):
                    plan = self._max_room(held, rungs[0], victims)
                elif not retryable and not exact:
                    plan = self._fit(held, estimates[-1:], victims, 0, floor=True)
                else:
                    plan = self._fit(held, estimates, victims, snapshot, floor=False)
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
        estimates: list[tuple[int, int | None, bool]],
        victims: list[Held],
        snapshot: int,
        *,
        floor: bool,
    ) -> Plan:
        """The first rung, and the most resident blocks, that fit after making room. A floor
        plan streams every block and clears every victim: the most room there is."""
        part = held.part
        last = (0, 0)
        for rung, working, _ in estimates:
            if working is None and not floor:
                continue
            working = (working or 0) + snapshot
            for keep in [0] if floor else self._keeps(held):
                need = self._need(part, keep) + working
                chosen = self._room_from(victims, need)
                if chosen is None and self._ask_room(need, victims, measured=working > 0):
                    chosen = self._room_from(victims, need)
                if chosen is not None:
                    return Plan(rung, keep, list(victims) if floor else chosen, need, working)
                last = (need, self.allocatable() + sum(self._evictable(v) for v in victims))
        needed, available = last
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

    def _max_room(self, held: Held, rung: int, victims: list[Held]) -> Plan:
        """A retryable op never measured gets all the room this device can make, with its
        weights resident when they fit whole."""
        whole = self._need(held.part, len(held.part.blocks))
        own = sum(self._evictable(v) for v in victims)
        if not self.spare and whole > self.allocatable() + own:
            self._ask_room(whole, victims, measured=False)
        keep = len(held.part.blocks) if whole <= self.allocatable() + own else 0
        return Plan(rung, keep, list(victims), self._need(held.part, keep), 0)

    def _keeps(self, held: Held) -> list[int]:
        """Resident block counts to try, most first; a component the store only refills whole
        and that is on disk, or a compiled one, is all or nothing."""
        blocks = len(held.part.blocks)
        if held.compiled or (held.part.on(DISK) and not held.part.unit_loads):
            return [blocks]
        return list(range(blocks, -1, -1))

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

    def _ask_room(self, need: int, victims: list[Held], *, measured: bool) -> bool:
        """Ask the Worker, once an attempt unless the grant grew, for driver-free bytes other
        tenants hold. True when it was asked, so the caller measures again."""
        if self.room is None or self.asked_room or self.spare:
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
        if victim.key in whole:
            return 0
        units = victim.part.blocks if victim.key in common else victim.part.units
        return sum(unit.nbytes for unit in units if unit.tier == DEVICE)

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
        with self._moving():
            if unit.host or not part.unit_loads or self._host_room(unit.nbytes):
                self.moved += tiers.to_host(self.torch, unit)
            else:
                self.stored += unit.nbytes
                tiers.to_disk(self.torch, part, unit)

    def _offload(self, held: Held, *, everything: bool = False) -> None:
        """Move a victim's unpinned device units off the device."""
        part = held.part
        whole, common = self._pinned()
        if held.key in whole and not everything:
            return
        units = part.blocks if held.key in common and not everything else part.units
        moving = [unit for unit in units if unit.tier == DEVICE]
        if (
            len(moving) == len(part.units)
            and part.store is not None
            and not part.unit_loads
            and not self._host_room(sum(unit.nbytes for unit in moving))
        ):
            self.stored += sum(unit.nbytes for unit in moving)
            with self._moving():
                tiers.to_disk(self.torch, part, None)
            return
        for unit in moving:
            self._home(part, unit)

    def _host_room(self, nbytes: int) -> bool:
        """Host memory takes a copy when what is available covers it (MemAvailable already
        nets out the kernel's reserve); otherwise clean caches, then idle host units, go back
        to the store. Stall pressure is no gate: a desktop reports it with gigabytes free."""
        available = proctree.available_host_bytes()
        if available < 0 or available >= nbytes:
            return True
        self._release_host(nbytes - available)
        return proctree.available_host_bytes() >= nbytes

    def _release_host(self, nbytes: int) -> None:
        running = {frame.held.key for frame in self.stack}
        idle = sorted(
            (held for held in self.held.values() if held.key not in running),
            key=lambda held: held.last_used,
        )
        for clean in (True, False):
            for held in idle:
                for unit in held.part.units:
                    if nbytes <= 0:
                        return
                    if clean and unit.tier == DEVICE and unit.host:
                        nbytes -= sum(t.numel() * t.element_size() for t in unit.host.values())
                        unit.host.clear()
                    elif not clean and unit.tier == HOST and held.part.unit_loads:
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

    # ------------------------------------------------------------------ activation rungs

    @contextmanager
    def _rung(self, held: Held, rung: int) -> Iterator[None]:
        """A tile edge for one VAE op, set on the module and restored after it."""
        if not rung:
            yield
            return
        root = held.part.root
        ratio = _spatial_ratio(root) or 1
        edits: dict[str, object] = {"use_tiling": True}
        if hasattr(root, "tile_sample_min_size"):
            edits |= {"tile_sample_min_size": rung, "tile_latent_min_size": rung // ratio}
        for axis in ("height", "width"):
            if hasattr(root, f"tile_sample_min_{axis}"):
                minimum = getattr(root, f"tile_sample_min_{axis}") or rung
                stride = getattr(root, f"tile_sample_stride_{axis}", minimum) or minimum
                edits[f"tile_sample_min_{axis}"] = rung
                edits[f"tile_sample_stride_{axis}"] = max(rung * stride // minimum, ratio)
        saved = {name: getattr(root, name, None) for name in edits}
        for name, value in edits.items():
            setattr(root, name, value)
        try:
            yield
        finally:
            for name, value in saved.items():
                setattr(root, name, value)

    # ------------------------------------------------------------------ retry safety

    def _retryable(self, held: Held, leaves: list[object]) -> bool:
        """An op can run again only if a snapshot can put back all it may change: no
        gradients, collectives or compiled graph; no hook, forward override or stateful
        diffusers hook but the Runtime's own or ones marked pure; plain arguments."""
        torch = self.torch
        if torch.is_grad_enabled() or held.compiled or self.sharded(held.key):
            return False
        hooks = torch.nn.modules.module
        if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
            return False
        for module in held.part.root.modules():
            registry = vars(module).get("_diffusers_hook")
            if registry is not None and any(
                getattr(hook, "_is_stateful", True)
                for hook in getattr(registry, "hooks", {}).values()
            ):
                return False
            forward = vars(module).get("forward")
            if forward is not None and not tiers.ours(forward):
                return False
            if not all(tiers.ours(fn) for fn in _hook_fns(module)):
                return False
        plain = (
            *_SCALAR,
            torch.Tensor,
            torch.Generator,
            torch.dtype,
            torch.device,
            torch.nn.Module,
        )
        return all(isinstance(leaf, plain) for leaf in leaves)

    def _arm(self, held: Held, leaves: list[object]) -> Snapshot:
        torch = self.torch
        root = held.part.root
        tensors = [(x, x.clone()) for x in leaves if isinstance(x, torch.Tensor)]
        rng: list[tuple[object, object]] = [
            (None, torch.get_rng_state()),
            (self.device, torch.cuda.get_rng_state(self.device)),
        ]
        attrs: list[tuple[Module, str, object]] = []
        generators = [x for x in leaves if isinstance(x, torch.Generator)]
        for module in root.modules():
            for name, value in vars(module).items():
                if isinstance(value, torch.Generator):
                    generators.append(value)
                elif not name.startswith("__") and _plain(value):
                    attrs.append((module, name, _copy(value)))
        rng += [(generator, generator.get_state()) for generator in generators]
        custom = getattr(root, "_cozy_retry_state", None)
        return Snapshot(tensors, rng, attrs, (custom[1], custom[0]()) if custom else None)

    def _restore(self, snapshot: Snapshot) -> None:
        torch = self.torch
        with torch.no_grad():
            for original, copy in snapshot.tensors:
                original.copy_(copy)
        for owner, state in snapshot.rng:
            if owner is None:
                torch.set_rng_state(state)
            elif isinstance(owner, torch.Generator):
                owner.set_state(state)
            else:
                torch.cuda.set_rng_state(state, owner)
        for module, name, value in snapshot.attrs:
            _put_back(module, name, value)
        if snapshot.custom is not None:
            restore, state = snapshot.custom
            restore(state)


def _hook_fns(module: Module) -> Iterator[object]:
    yield from module._forward_hooks.values()
    yield from module._forward_pre_hooks.values()


def _spatial_ratio(vae: Module) -> int | None:
    ratio = getattr(vae, "spatial_compression_ratio", None)
    if isinstance(ratio, int) and ratio > 0:
        return ratio
    channels = getattr(getattr(vae, "config", None), "block_out_channels", None)
    return 2 ** (len(channels) - 1) if channels else None


# --------------------------------------------------------------------------- the machine


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
    """The Worker's memory policy: grants and room across every tenant of every device."""

    def __init__(
        self,
        budget: Mapping[int, int],
        driver: Callable[[int], tuple[int, int] | None],
        note: Callable[[str], None],
    ) -> None:
        #: bytes per GPU ordinal; -1 applies to every GPU without its own entry
        self.budget = dict(budget)
        self.driver = driver
        self.note = note

    def grant(self, me: Tenant, ordinal: int, tenants: Iterable[Tenant]) -> int | None:
        """The bytes `me` may hold on `ordinal`: the budget, or what the driver has beside
        other programs, less what the other tenants hold."""
        measured = self.driver(ordinal)
        if measured is None:
            return None
        free, total = measured
        others = sum(t.held.get(ordinal, 0) for t in tenants if t.slot != me.slot and ordinal in t.ordinals)
        foreign = max(total - free - others - me.held.get(ordinal, 0), 0)
        limit = min(self.budget.get(ordinal, self.budget.get(-1, total)), total - foreign)
        return limit - others

    def admit(
        self, me: Tenant, tenants: Sequence[Tenant], peak: int | None
    ) -> dict[int, int | None]:
        """Before a call: dead tenants on its devices end, and its grant reaches the most it
        held during a call of the same shape, `peak` (memory its package holds between ops
        included). A shape never measured gets every idle tenant's weights off its devices:
        the call then measures what it really takes."""
        for tenant in tenants:
            if (
                tenant.slot != me.slot
                and tenant.state() == "dead"
                and set(tenant.ordinals) & set(me.ordinals)
            ):
                tenant.end(f"reclaimed before {me.slot!r} runs")
        for ordinal in me.ordinals:
            if peak is None:
                self.room(me, ordinal, None, None, tenants, measured=False)
            else:
                self.room(me, ordinal, 0, peak, tenants, measured=True)
        return {ordinal: self.grant(me, ordinal, tenants) for ordinal in me.ordinals}

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
        """Free `ordinal` for `me` until the driver has `free_wanted` bytes free (None: until
        nothing idle is left) and its grant reaches `grant_wanted`: dead tenants first, then
        idle tenants' weights, least recently used first, then (for a measured need) idle
        processes. Never an active tenant."""
        tried: set[str] = set()
        while True:
            measured_now = self.driver(ordinal)
            if measured_now is None:
                return None
            grant = self.grant(me, ordinal, tenants)
            if (
                free_wanted is not None
                and measured_now[0] >= free_wanted
                and (grant_wanted is None or grant is None or grant >= grant_wanted)
            ):
                return grant
            victim = self._victim(me, ordinal, tenants, tried, measured)
            if victim is None and free_wanted is None:
                return grant
            if victim is None:
                self.note(
                    f"GPU {ordinal}: nothing idle is left to give {me.slot!r} room "
                    f"({measured_now[0]} B free, {free_wanted} B wanted; grant {grant} B)"
                )
                return grant
            tenant, act = victim
            tried.add(f"{tenant.slot}:{act}")
            if act == "vacate":
                done = tenant.vacate((ordinal,))
                self.note(f"GPU {ordinal}: vacated {tenant.slot!r} for {me.slot!r}")
            else:
                done = tenant.end(f"its process gave GPU {ordinal} to {me.slot!r}")
                self.note(f"GPU {ordinal}: ended {tenant.slot!r} for {me.slot!r} ({act})")
            if not done and act == "vacate":
                tenant.end(f"it could not vacate GPU {ordinal} for {me.slot!r}")

    def _victim(
        self, me: Tenant, ordinal: int, tenants: Sequence[Tenant], tried: set[str], measured: bool
    ) -> tuple[Tenant, str] | None:
        present = [
            t for t in tenants if t.slot != me.slot and ordinal in t.ordinals and t.state() != "active"
        ]
        for act, pool in (
            ("dead", [t for t in present if t.state() == "dead"]),
            (
                "vacate",
                [t for t in present if t.held.get(ordinal, 0) > t.overhead.get(ordinal, 0)],
            ),
            ("process", [t for t in present if t.held.get(ordinal, 0)] if measured else []),
        ):
            pool = [t for t in pool if f"{t.slot}:{act}" not in tried]
            if pool:
                return min(pool, key=lambda t: t.last_used), act
        return None
