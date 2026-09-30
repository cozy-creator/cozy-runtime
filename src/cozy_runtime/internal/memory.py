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
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, Lock
from typing import TYPE_CHECKING, NoReturn, Protocol

from cozy_runtime.author._errors import DEVICE_OOM, RuntimeFailure
from cozy_runtime.author._errors import is_device_oom as is_oom
from cozy_runtime.author._executor_requests import RankRoom
from cozy_runtime.author._retry import is_declared_pure

if TYPE_CHECKING:
    from cozy_runtime.internal.parallel.group import OpReady
from cozy_runtime.internal import proctree, tiers
from cozy_runtime.internal.tiers import DEVICE, DISK, HOST, Device, Module, Tensor, Torch

SHORTFALL = "device_shortfall"
SPARED = "device_spared"
DEVICE_CAPACITY_REFUSALS = frozenset((SHORTFALL, DEVICE_OOM))
CAPACITY_REFUSALS = DEVICE_CAPACITY_REFUSALS | {"host_shortfall"}
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
    flags: tuple[bool | int | float | str | None, ...] = ()
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
    #: richest placement that failed; a smaller successful replay cannot retire its proof
    fail_keep: int = 0
    #: allocatable plus own reserve at failure; greater capacity permits a fresh proof
    fail_capacity: int = 0


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
    readonly_lost_reason: str = ""
    forward_only: bool = False
    scope_anchor: bool = False
    conversion_epoch: int = 0
    ops: dict[str, Callable[..., object]] = field(default_factory=dict)
    overlap_key: OpKey | None = None
    overlap_order: tuple[tiers.Unit, ...] = ()
    overlap_epoch: int = 0
    overlap_contract: tuple[object, ...] = ()


@dataclass(eq=False)
class Frame:
    held: Held
    base: int
    working: int = 0
    loaded: int = 0
    overlap_key: OpKey | None = None
    overlap_expected: tuple[tiers.Unit, ...] = ()
    overlap_visits: list[tiers.Unit] = field(default_factory=list)
    overlap_enabled: bool = False
    overlap_learnable: bool = False
    overlap_contract: tuple[object, ...] = ()
    overlap_obligation: int = 0
    overlap_pending: tiers.HostTransfer | None = None
    overlap_consumers: dict[tiers.Unit, tiers.HostTransfer] = field(default_factory=dict)
    overlap_drain: set[tiers.Unit] = field(default_factory=set)
    overlap_depth: dict[tiers.Unit, int] = field(default_factory=dict)
    overlap_entry: object = None
    overlap_joined: bool = False


@dataclass(slots=True)
class Plan:
    keep: int
    evict: list[Held]
    #: its victims include the scope's other components' common units
    #: concrete net room target; None means a measured recovery needs every movable byte
    needed: int | None = None


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
        process_bytes: Callable[[], int | None] | None = None,
    ) -> None:
        self.torch = torch
        self.device = device
        self.cuda = device.type == "cuda"
        #: asks the Worker for (driver-free bytes, total bytes to hold, measured need): the grant
        self.room = room
        self.host_room = host_room
        self._host_sequence = 0
        self._host_pending = 0
        self.staging_unbacked: dict[str, int] = {}
        self.construction_staging_credit: int | None = None
        self._group_host_credit: int | None = None
        self._last_host_credit = 0
        self.sharded = sharded
        self.grouped = grouped
        # Executor-lifetime strong custody: poisoned transfers survive failed calls.
        self._host_transfers: set[tiers.HostTransfer] = set()
        self._overlap_stream: object | None = None
        self._overlap_oom_room: int | None = None
        self._overlap_oom_unknown = False
        self.overlap: dict[str, int] = {}
        self.process_bytes = process_bytes
        self.driver_process_bytes = self.peak_driver_process_bytes = -1
        self.driver_observations = self.peak_driver_budget_excess_bytes = 0
        self.grant: int | None = None
        #: High-water consumed non-allocator bytes: context, modules, communicators.
        #: This is not a separate reservation for unmeasured future growth.
        self.overhead = 0
        self.held: dict[str, Held] = {}
        self.records: dict[OpKey, Record] = {}
        self.stack: list[Frame] = []
        self.scopes: list[tuple[str, frozenset[str]]] = []
        self.seq: list[str] = []
        self.last_seq: dict[str, list[str]] = {}
        self.clock = 0
        self.spare = False
        self.room_requests: set[tuple[int, int, bool, int | None]] = set()
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
        self.anchor_validation_calls = self.anchor_validation_ns = 0
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

    def observe_process(self) -> bool:
        """Sample this rank's driver footprint; keep observed overhead as a high-water mark.

        The allocator cannot constrain external CUDA allocations before they are observed.
        This is a sampled soft target, not a hardware process-memory fence. Never derive
        overhead across a concurrent allocator change or replace an unreadable fact by zero.
        """
        if not self.cuda or self.process_bytes is None:
            return False
        before = self._reserved()
        used = self.process_bytes()
        after = self._reserved()
        if used is None or used < 0:
            self.driver_process_bytes = -1
            return False
        self.driver_process_bytes = used
        self.peak_driver_process_bytes = max(self.peak_driver_process_bytes, used)
        self.driver_observations += 1
        if self.grant is not None:
            self.peak_driver_budget_excess_bytes = max(
                self.peak_driver_budget_excess_bytes, used - self.grant
            )
        if before != after or used < after:
            return False
        overhead = used - after
        if overhead <= self.overhead:
            return False
        self.log.append(f"observed process overhead grew from {self.overhead} B to {overhead} B")
        self.overhead = overhead
        return True

    def checkpoint(self) -> tuple[int, int, int]:
        """Driver free, allocator reserve and room requests now: the start of a measurement."""
        if not self.cuda:
            return 0, 0, self.room_asks
        return self._driver()[0], self._reserved(), self.room_asks

    def measure_overhead(self, free_before: int, reserved_before: int, asks: int = 0) -> None:
        """Non-allocator bytes this process took since `free_before` was read: called only at
        controlled points (init, warm, a communicator join), and discarded when a room
        request moved other tenants' bytes in between."""
        if self.process_bytes is not None:
            self.observe_process()
            if self.driver_process_bytes >= 0:
                self._cap()
                return
        if not self.cuda or free_before <= 0 or self.room_asks != asks:
            return
        free, _ = self._driver()
        grown = free_before - free - (self._reserved() - reserved_before)
        self.overhead += max(grown, 0)

    def _allocated(self) -> int:
        return self._allocator_scalar("allocated_bytes", "current", "memory_allocated")

    def _reserved(self) -> int:
        return self._allocator_scalar("reserved_bytes", "current", "memory_reserved")

    def _allocator_scalar(self, metric: str, field: str, fallback: str) -> int:
        """Read fresh native facts without flattening/sorting every allocator statistic.

        No cache: in particular, observe_process still reads reserve independently
        before and after the driver query. Older Torch uses its existing scalar API.
        """
        if not self.cuda:
            return 0
        cuda = self.torch.cuda
        nested = getattr(cuda, "memory_stats_as_nested_dict", None)
        if callable(nested):
            row = nested(self.device).get(metric, {})
            row = row.get("all", {}) if isinstance(row, Mapping) else {}
            return int(row.get(field, 0)) if isinstance(row, Mapping) else 0
        return int(getattr(cuda, fallback)(self.device))

    def _driver(self) -> tuple[int, int]:
        free, total = self.torch.cuda.mem_get_info(self.device)
        return int(free), int(total)

    def allocatable(self) -> int:
        """Bytes the allocator may still reserve, inside both its grant and driver free.
        Driver free already excludes consumed overhead; only the total grant charges it.
        Its cache is no credit: holes
        between live blocks cannot be released, and the cap counts them. Negative when the
        process holds more than its grant, and the plan must give bytes back."""
        if not self.cuda:
            return 1 << 62
        free, total = self._driver()
        granted = (total if self.grant is None else self.grant) - self.overhead - self._reserved()
        return min(granted, free)

    def _cap(self) -> None:
        """Cap allocator reserve after sampled process overhead; external growth is observed."""
        if not self.cuda:
            return
        if self.dropped:
            tiers.trim(self.torch)
            self.dropped = False
        self.observe_process()
        free, total = self._driver()
        limit = min(
            (total if self.grant is None else self.grant) - self.overhead,
            free + self._reserved(),
        )
        self.torch.cuda.set_per_process_memory_fraction(
            min(max(limit, 0) / total, 1.0), self.device
        )

    def _sample(self) -> None:
        if not self.cuda:
            return
        cuda = self.torch.cuda
        if self.observe_process():
            self._cap()
        peak = self._allocator_scalar("allocated_bytes", "peak", "max_memory_allocated")
        self.peak = max(
            self.peak, self._allocator_scalar("reserved_bytes", "peak", "max_memory_reserved")
        )
        self.peak_allocated = max(self.peak_allocated, peak)
        for frame in self.stack:
            pending = sum(
                transfer.source_bytes
                for transfer in self._host_transfers
                if transfer.state == "prepared"
            )
            frame.working = max(frame.working, peak - frame.base - frame.loaded - pending)
        cuda.reset_peak_memory_stats(self.device)

    # ------------------------------------------------------------------ components

    def adopt(
        self,
        construction: str,
        name: str,
        root: Module,
        store: tiers.Store | None,
        tier: str,
        *,
        forward_only: bool = False,
    ) -> Held:
        """Take a constructed component under management: units, hooks, op wrappers."""
        key = f"{construction}/{name}"
        tokens = {token for module in root.modules() for token in tiers._tokens(module)}
        for other in self.held.values():
            if other.key != key and any(
                token in tokens
                for module in other.part.root.modules()
                for token in tiers._tokens(module)
            ):
                raise MemoryRefusal(
                    "component_alias",
                    f"{key} shares a module, tensor or storage with "
                    f"{other.key}; register shared weights as one managed component",
                )
        part = tiers.partition(name, root, tier, store)
        held = Held(
            key,
            construction,
            part,
            kind=f"{type(root).__module__}.{type(root).__qualname__}:{name}:{part.nbytes}",
            compiled=_compiled(root),
            whole=bool(getattr(root, "_cozy_read_whole", False)),
            forward_only=forward_only,
        )
        self.held[key] = held
        root.__dict__["_cozy_select_decode_mode"] = functools.partial(self.select_decode_mode, held)
        root.__dict__["_cozy_optional_cache"] = functools.partial(self.optional_cache, held)
        self._pin_state(held)
        if (
            held.forward_only
            and not held.whole
            and not held.compiled
            and part.unit_loads
            and self._state_contract(held, [])
            and all(
                unit.readonly and not unit.mutable and (unit.tier != DISK or tiers.store_safe(unit))
                for unit in part.units
            )
        ):
            tiers.split_common(part)
            held.scope_anchor = part.anchor is not None
        if self.cuda:
            tiers.install(part, self._enter_unit, self._leave_unit)
            tiers.wrap(root, functools.partial(self.run, held))
            tiers.intercept_apply(
                self.torch,
                part,
                lambda: self._converted(held),
                host_copy=lambda size: self._host_copy(size, protected=part.units),
            )
            root.__dict__["_cozy_op"] = functools.partial(self.op, held)
        return held

    def _converted(self, held: Held) -> None:
        """A real package dtype conversion invalidates old working-memory observations."""
        self._overlap_retire(held=held, clear=True)
        self._validate_component(held)
        for unit in held.part.units:
            tiers.load_demand(self.torch, unit)
        self.records = {
            key: value for key, value in self.records.items() if key.component != held.kind
        }
        held.conversion_epoch += 1

    def op(self, held: Held, method: str) -> Callable[..., object]:
        """A method the package spreads, run in its unchanged authored numerical mode."""
        if method not in held.ops:
            bound = getattr(held.part.root, method)
            run = functools.partial(self.run, held)
            held.ops[method] = tiers.op(method, bound, run)
        return held.ops[method]

    def startup_budget(self, room: Callable[[int, int, bool], int | None]) -> None:
        """Context/warmup footprint admission before any model can be loaded."""
        held = self._reserved() + self.overhead
        if self.grant is not None and held > self.grant:
            tiers.trim(self.torch)
            held = self._reserved() + self.overhead
        if self.grant is not None and held > self.grant:
            room(0, held, True)
        self._cap()
        if self.grant is not None and held > self.grant:
            raise MemoryRefusal(
                SHORTFALL, "device startup exceeds its authorized footprint", held, self.grant
            )

    def group_room(
        self,
        statuses: Sequence[OpReady],
        progress: dict[int, tuple[int, int]],
        world: int,
        probe: Callable[[], tuple[int, ...]],
        request: Callable[[tuple[RankRoom, ...]], tuple[int, ...] | None],
        apply: Callable[[tuple[int, ...]], None],
    ) -> bool:
        """Retry only after all failed ranks meet absolute minima, with monotone progress.

        Both probes use the same CUDA source; Worker observations never masquerade as gain.
        The caller transports probes/room requests and applies the returned grant vector.
        """
        if not self.cuda:
            return False
        before = probe()
        failed = [status for status in statuses if not status.ok]
        if not failed:
            return False
        needs = tuple(
            RankRoom(
                rank=status.rank,
                free_bytes=status.free_wanted,
                grant_bytes=status.grant_wanted,
                measured=True,
            )
            for status in failed
        )
        if any(
            need.rank in progress
            and need.free_bytes <= progress[need.rank][0]
            and need.grant_bytes <= progress[need.rank][1]
            for need in needs
        ):
            return False
        self.room_asks += 1
        grants = request(needs)
        if grants is None or len(grants) != world:
            return False
        apply(grants)
        after = probe()
        if not all(
            (before[need.rank] >= 0 and after[need.rank] >= need.free_bytes)
            and grants[need.rank] >= need.grant_bytes
            for need in needs
        ):
            return False
        for need in needs:
            old_free, old_grant = progress.get(need.rank, (0, 0))
            progress[need.rank] = (
                max(old_free, after[need.rank]),
                max(old_grant, grants[need.rank]),
            )
        return True

    @staticmethod
    def spread_decline(exc: Exception) -> str | None:
        """A reusable refusal before the tensor answer can run on another rank."""
        if isinstance(exc, MemoryRefusal) and exc.reusable:
            return f"{exc.code}: {exc.message}"[:300]
        return None

    @staticmethod
    def group_retry(
        statuses: Sequence[OpReady],
        *,
        host_tried: bool,
        host_available: bool,
        device_available: bool,
    ) -> tuple[bool, bool, bool]:
        """Rank reusability and eligible host/device phases, in existing recovery order."""
        failed = [status for status in statuses if not status.ok]
        if any(not status.reusable for status in failed):
            return False, False, False
        host = bool(
            failed
            and not host_tried
            and host_available
            and any(status.host_needed for status in statuses)
            and all(status.code in CAPACITY_REFUSALS for status in failed)
        )
        device = bool(
            failed
            and device_available
            and all(status.code in DEVICE_CAPACITY_REFUSALS for status in failed)
        )
        return True, host, device

    def optional_cache[T](
        self,
        held: Held,
        nbytes: int,
        run: Callable[[bool], T],
        *,
        requested: bool | None = None,
    ) -> T:
        """Choose once before sampling; a cache can change floating-point results.

        Explicit modes are authored semantics. Automatic selection uses current room.
        Managed operations may recover within that mode, but a refusal must never replay
        the whole request with a different mode or repeat its step/output callbacks.
        """
        if nbytes < 0 or self.held.get(held.key) is not held:
            raise ValueError("optional cache requires owned component and nonnegative bytes")
        if requested is None:
            if nbytes > self.allocatable() and self._reserved() > self._allocated():
                tiers.trim(self.torch)
            if nbytes > self.allocatable():
                self._ask_room(nbytes, [], measured=False)
            enabled = nbytes <= self.allocatable()
        else:
            enabled = requested
        mode = "enabled" if enabled else "disabled"
        self.log.append(f"{held.key}: selected request cache mode {mode} ({nbytes} B)")
        return run(enabled)

    def select_decode_mode(
        self, held: Held, candidates: tuple[int, ...], probe: Callable[[int], None]
    ) -> int:
        """Choose a request's mode with finite real probes before any authored generation.

        The adapter supplies legal geometry and applies each mode only for its probe.
        Once returned, the request freezes this choice; ordinary retry never changes it.
        """
        if (
            not candidates
            or any(type(side) is not int or side < 0 for side in candidates)
            or len(set(candidates)) != len(candidates)
            or any(side == 0 for side in candidates[1:])
            or list(side for side in candidates if side)
            != sorted((side for side in candidates if side), reverse=True)
        ):
            raise ValueError(
                "decode modes must be unique descending tile sides, optionally 0 first"
            )
        if self.held.get(held.key) is not held or not self._state_contract(held, []):
            raise MemoryRefusal(
                "decode_probe_contract", "decode mode selection requires a managed retry contract"
            )
        torch = self.torch
        cpu_rng = torch.get_rng_state()
        device_rng = torch.cuda.get_rng_state(self.device) if self.cuda else None
        refusal: MemoryRefusal | None = None
        for side in candidates:
            try:
                probe(side)
            except MemoryRefusal as exc:
                if not exc.reusable or self.unusable or exc.code not in (SHORTFALL, DEVICE_OOM):
                    raise
                # A failed probe's traceback owns its disposable device inputs. Keep
                # the typed evidence, not a frame cycle waiting for Python's cyclic GC.
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
                refusal = exc
            else:
                refusal = None
                self.refusal = None
                self.log.append(f"{held.key}: selected request decode tile side {side}")
                return side
            finally:
                torch.set_rng_state(cpu_rng)
                if device_rng is not None:
                    torch.cuda.set_rng_state(device_rng, self.device)
        assert refusal is not None
        raise refusal

    def forget(self, construction: str) -> None:
        self._overlap_retire(clear=True)
        for key in [key for key, held in self.held.items() if held.construction == construction]:
            tiers.release(self.held.pop(key).part)

    def host(self, key: str) -> None:
        """Another GPU of the group keeps this component: it never occupies this one."""
        held = self.held[key]
        self._overlap_retire(held=held, clear=True)
        if held.part.store is not None:
            tiers.to_disk(self.torch, held.part, None)
        held.hosted = True

    # ------------------------------------------------------------------ attempts and scopes

    def begin(self, grant: int | None) -> None:
        if self.unusable:
            raise MemoryRefusal("device_fault", self.unusable, reusable=False)
        if grant is not None:
            self.grant = grant
        self._overlap_retire(clear=True)
        self.overlap = {}
        self.room_requests.clear()
        self.peak = self.peak_allocated = self.working = 0
        self.peak_driver_process_bytes = -1
        self.driver_observations = self.peak_driver_budget_excess_bytes = 0
        self.loads = self.offloads = self.moved = self.transition = 0
        self.stored = self.refilled = 0
        self.counters = self._counters()
        self.log = []
        self.plans = {}
        self.operations = {}
        self.anchor_validation_calls = self.anchor_validation_ns = 0
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
            "budget_enforcement": "allocator cap with sampled process overhead; soft target",
            "driver_process_bytes": self.driver_process_bytes,
            "peak_driver_process_bytes": self.peak_driver_process_bytes,
            "peak_driver_budget_excess_bytes": self.peak_driver_budget_excess_bytes,
            "driver_observations": self.driver_observations,
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
            "host_overlap": dict(self.overlap),
            "anchor_validation": {
                "calls": self.anchor_validation_calls,
                "ns": self.anchor_validation_ns,
            },
            "scope_backing": {
                h.key: {
                    "metadata_anchor_bytes": (h.part.anchor or h.part.common).nbytes,
                    "operation_common_bytes": (
                        h.part.common.nbytes
                        if h.part.anchor is not None and h.part.anchor is not h.part.common
                        else 0
                    ),
                    "operation_common_device_bytes": (
                        h.part.common.nbytes
                        if h.part.anchor is not None
                        and h.part.anchor is not h.part.common
                        and h.part.common.tier == DEVICE
                        else 0
                    ),
                    "anchor_only_scope": h.scope_anchor and not h.whole and not h.compiled,
                    "scope_minimum_bytes": sum(
                        unit.nbytes
                        for unit in (
                            h.part.units
                            if h.whole or h.compiled
                            else tuple(
                                set(self._scope_common(h))
                                | {unit for unit in h.part.units if unit.mutable}
                            )
                        )
                    ),
                }
                for h in mine
            },
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
        if self.observe_process():
            self._cap()
        reserved, allocated = self._reserved(), self._allocated()
        return {
            "overhead_bytes": self.overhead,
            "reserved_bytes": reserved,
            "allocated_bytes": allocated,
            "reserve_room_bytes": self.allocatable() if self.cuda else 0,
            # Visible cache is not a planning credit: fragmented cached blocks cannot
            # necessarily satisfy the next allocation or be returned to the driver.
            "reusable_cache_bytes": max(reserved - allocated, 0),
            "peak_bytes": self.peak + self.overhead,
            "driver_process_bytes": self.driver_process_bytes,
            "peak_driver_process_bytes": self.peak_driver_process_bytes,
            "peak_driver_budget_excess_bytes": self.peak_driver_budget_excess_bytes,
            "driver_observations": self.driver_observations,
            "peak_allocated_bytes": self.peak_allocated,
            "working_bytes": self.working,
            "host_bytes": sum(self._host_backings().values()),
            "host_resident_bytes": sum(self._host_backings(cached=False).values()),
            "host_cached_bytes": sum(self._host_backings(resident=False).values()),
            "disk_weight_bytes": sum(h.part.on(DISK) for h in self.held.values() if not h.hosted),
        }

    def _validate_component(self, held: Held) -> None:
        if held.part.anchor is None:
            return
        began = time.perf_counter_ns()
        try:
            tiers.validate_anchor(held.part)
        except RuntimeFailure as exc:
            self.unusable = f"{held.key}: {exc}"
            raise MemoryRefusal(exc.code, self.unusable, reusable=False) from None
        finally:
            self.anchor_validation_calls += 1
            self.anchor_validation_ns += time.perf_counter_ns() - began

    def _validate_active(self, held: Held) -> None:
        active = {held.key}
        active.update(
            f"{construction}/{name}" for construction, names in self.scopes for name in names
        )
        for key in active:
            if (component := self.held.get(key)) is not None:
                self._refresh_constraints(component)

    def _refresh_constraints(self, held: Held) -> None:
        """Refresh mutable/whole/compiled and alias facts at quote and movement boundaries."""
        self._validate_component(held)
        held.compiled = _compiled(held.part.root)
        held.whole = held.whole or bool(getattr(held.part.root, "_cozy_read_whole", False))
        self._pin_state(held)
        held.scope_anchor = (
            held.part.anchor is not None
            and held.forward_only
            and self._state_contract(held, [])
            and all(unit.readonly and not unit.mutable for unit in held.part.common_units)
        )

    def _scope_common(self, held: Held) -> tuple[tiers.Unit, ...]:
        """Scope metadata is real backing; unknown effects retain every common unit."""
        part = held.part
        if held.scope_anchor and part.anchor is not None:
            return (part.anchor,)
        return part.common_units

    def enter(self, construction: str, method: str, components: tuple[str, ...]) -> None:
        """Pin every declared component's minimum before any optional operation residency.

        Common units keep `.device` valid; blocks arrive with managed operations. Components
        explicitly read outside operations (`read_whole`) must fit whole.
        """
        if self.unusable:
            raise MemoryRefusal("device_fault", self.unusable, reusable=False)
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
        self._overlap_retire(pending_only=True)
        for frame in self.stack:
            frame.overlap_enabled = frame.overlap_learnable = False
        self._cap()
        scoped = [
            held
            for name in components
            if (held := self.held.get(f"{construction}/{name}")) is not None and not held.hosted
        ]
        # Refresh every sibling before any transfer can quote it as an optional victim.
        for held in scoped:
            # A package may have replaced a forward since (a cache hook): it is an op again.
            tiers.wrap(held.part.root, functools.partial(self.run, held))
            self._refresh_constraints(held)
        for held in scoped:
            part = held.part
            whole = tiers.load_sequence(self.torch, part.units, whole=not part.unit_loads).peak
            if part.on(DISK) and part.store is not None:
                whole += part.store.transient(part.name)
            whole_only = (
                held.whole or held.compiled or (bool(part.on(DISK)) and not part.unit_loads)
            )
            scope_units = self._scope_common(held)
            common = tiers.load_sequence(self.torch, scope_units).peak
            required = whole if whole_only else common
            if required > self.allocatable() and self._reserved() > self._allocated():
                # Cached segments can put signed reserve room below zero even when
                # this scope needs no new common bytes. Reclaim and remeasure before
                # refusing; live allocations still count against the unchanged grant.
                tiers.trim(self.torch)
            # Scope pins already protect every sibling's common bytes and read_whole
            # components. Their optional resident blocks remain cache and can make room
            # for later minima; greedily filling an earlier sibling cannot starve a VAE.
            victims = [] if self.spare else self._victims(held)
            chosen = self._room_from(victims, required)
            if chosen is None and self._ask_room(required, victims, measured=True):
                chosen = self._room_from(victims, required)
            if chosen is None:
                self._refuse_scope(held, method, required, victims)
            for victim in chosen:
                self._offload(victim, target=required)
            units = part.units if whole_only else scope_units
            if whole_only and part.on(DISK) == part.nbytes and part.store is not None:
                self._place_scope(held, method, functools.partial(self._refill_whole, held))
            else:
                for unit in units:
                    if unit.tier != DEVICE:
                        self._place_scope(held, method, functools.partial(self._load, part, unit))
            self._cap()

    def _place_scope(
        self,
        held: Held,
        method: str,
        place: Callable[[], None],
    ) -> None:
        """Recover an atomic transfer before the authored scope body starts.

        A group keeps its existing all-rank preparation/recovery authority. A spare
        operation never acquires another tenant's room. Neither replays its placement here.
        """
        if self.grouped or self.spare:
            place()
            return

        counts = self.operations.setdefault(f"{held.key}.scope:{method}", {"calls": 0})
        counts["calls"] += 1
        before = self.allocatable()
        while True:
            self.oom_asked = 0
            try:
                with self._traffic(counts, "placement"):
                    place()
            except BaseException as exc:
                if not is_oom(exc):
                    raise
                asked = max(self.oom_asked, _asked(str(exc)), 1)
                self.oom_asked = 0
                text = (str(exc).splitlines() or [type(exc).__name__])[0][:300]
                exc.__traceback__ = None
                counts["placement_ooms"] = counts.get("placement_ooms", 0) + 1
            else:
                return

            # Compare one atomic transfer with itself, not an entire partially staged
            # component. A previous successful unit may have reused reserved cache pages.
            # Also do not credit a failed allocation growing R, then trim undoing that
            # same growth: the recovery must improve on the pre-attempt high-water room.
            before = max(before, self.allocatable())
            # Tier moves either committed or restored their original bindings. No
            # application body or RNG draw is replayed; release unused allocator pages
            # and ask the existing Machine policy for a measured extra allocation.
            try:
                tiers.trim(self.torch)
                self._cap()
            except Exception as exc:
                self.unusable = f"{held.key}: scope placement recovery failed: {type(exc).__name__}"
                self.refusal = MemoryRefusal("device_fault", self.unusable, reusable=False)
                raise self.refusal from None
            target = max(before, 0) + asked
            if self.allocatable() < target:
                self._ask_room(target, [], measured=True)
            after = self.allocatable()
            if after < target:
                self.refusal = MemoryRefusal(
                    DEVICE_OOM,
                    f"{held.key} scope {method}: no placement retry: {text}; "
                    f"needed {target - before} B of extra room, gained {after - before} B",
                    target - before,
                    max(after - before, 0),
                )
                raise self.refusal from None
            self._refresh_constraints(held)
            counts["placement_retries"] = counts.get("placement_retries", 0) + 1
            counts["placement_gain_bytes"] = counts.get("placement_gain_bytes", 0) + after - before
            self.log.append(
                f"{held.key} scope {method}: retrying placement after gaining {after - before} B"
            )
            before = after

    def restore_resident(self, construction: str, names: tuple[str, ...]) -> dict[str, object]:
        """A revision-1 Worker's explicit restore, without evicting another component."""
        restored: dict[str, int] = {}
        held_back: dict[str, str] = {}
        for held in self.held.values():
            part = held.part
            if (
                held.construction != construction
                or held.hosted
                or (names and part.name not in names)
            ):
                continue
            demand = tiers.load_sequence(self.torch, part.units, whole=not part.unit_loads)
            if demand.resident <= 0:
                continue
            if demand.peak > self.allocatable():
                held_back[part.name] = "insufficient device room"
                continue
            self._stage_whole(held, [])
            restored[part.name] = demand.resident
        return {"restored": restored, "held": held_back, "restored_bytes": sum(restored.values())}

    def _stage_whole(self, held: Held, victims: list[Held]) -> None:
        for victim in victims:
            self._offload(victim)
        part = held.part
        if part.on(DISK) == part.nbytes and part.store is not None:
            self._refill_whole(held)
        else:
            self._load_common(held)
            for unit in part.blocks:
                self._load(part, unit)
        self._cap()

    def _refill_whole(self, held: Held) -> None:
        self._overlap_retire(pending_only=True)
        with self._moving():
            tiers.from_disk(self.torch, held.part, None)
        self.loads += 1
        self.refilled += held.part.nbytes

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
        self._overlap_retire(pending_only=True, clear=True)
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
        """Release movable cache; refusal to copy a unit does not damage an idle tenant."""
        freed: dict[str, int] = {}
        retained: dict[str, int] = {}
        before = self._allocated()
        for held in self.held.values():
            moved = held.part.on(DEVICE)
            if moved:
                try:
                    self._offload(held, everything=True)
                except MemoryRefusal as exc:
                    if not exc.reusable:
                        raise
                released = moved - held.part.on(DEVICE)
                if released:
                    freed[held.key] = released
                if remaining := held.part.on(DEVICE):
                    retained[held.key] = remaining
        if self.cuda:
            tiers.trim(self.torch)
        return {"freed": freed, "held": retained, "freed_bytes": before - self._allocated()}

    def reclaim_host(self, nbytes: int) -> int:
        """Drop only reproducible idle backing; return managed bytes released.

        This is not RSS credit: other references or the allocator can retain those
        pages. The Worker must remeasure aggregate headroom before granting storage.
        """
        if nbytes <= 0 or self.stack or self.scopes or self.group_phase:
            return 0
        staging_released = 0
        for store in self._staging_stores():
            release = getattr(store, "release_staging", None)
            if release is not None:
                try:
                    staging_released += int(release())
                except RuntimeFailure as exc:
                    if exc.code == "device_fault":
                        self.unusable = str(exc)
                    raise
                if staging_released >= nbytes:
                    return staging_released
        nbytes -= staging_released
        before = self._host_backings()
        for held in sorted(self.held.values(), key=lambda held: held.last_used):
            self._pin_state(held)
            if held.compiled or _compiled(held.part.root):
                continue
            part = held.part
            for unit in part.units:
                if unit.readonly and not unit.mutable and unit.tier == DEVICE:
                    unit.host.clear()
                    unit.seen.clear()
            if self._released_host_bytes(before) >= nbytes:
                break
            if part.store is None:
                continue
            if part.unit_loads:
                for unit in part.units:
                    if (
                        unit.tier == HOST
                        and unit.readonly
                        and not unit.mutable
                        and tiers.store_safe(unit)
                    ):
                        tiers.to_disk(self.torch, part, unit)
                        self.stored += unit.nbytes
                    if self._released_host_bytes(before) >= nbytes:
                        break
            elif any(unit.tier == HOST for unit in part.units) and all(
                unit.tier in (HOST, DISK)
                and unit.readonly
                and not unit.mutable
                and tiers.store_safe(unit)
                for unit in part.units
            ):
                self.stored += sum(unit.nbytes for unit in part.units if unit.tier == HOST)
                tiers.to_disk(self.torch, part, None)
            if self._released_host_bytes(before) >= nbytes:
                break
        return staging_released + self._released_host_bytes(before)

    def _staging_stores(self) -> tuple[object, ...]:
        return tuple(
            {
                id(h.part.store): h.part.store
                for h in self.held.values()
                if h.part.store is not None
            }.values()
        )

    def _staging_memory(self) -> tuple[int, int]:
        allocated = pinned = 0
        for store in self._staging_stores():
            observe = getattr(store, "staging_memory", None)
            if observe is not None:
                facts = observe()
                allocated += int(facts["allocated_bytes"])
                pinned += int(facts["pinned_bytes"])
        return allocated, pinned

    @staticmethod
    def _unit_host_backings(
        unit: tiers.Unit, *, cached: bool = True, resident: bool = True
    ) -> dict[int, int]:
        found: dict[int, int] = {}
        tensors = list(unit.host.values()) if cached else []
        if resident and unit.tier == HOST:
            tensors.extend(tensor for *_, tensor in unit.slots())
        for tensor in tensors:
            if tensor.device.type == "cpu":
                storage = tensor.untyped_storage()
                found[storage.data_ptr()] = storage.nbytes()
        return found

    def _host_backings(self, *, cached: bool = True, resident: bool = True) -> dict[int, int]:
        found: dict[int, int] = {}
        for held in self.held.values():
            for unit in held.part.units:
                found.update(self._unit_host_backings(unit, cached=cached, resident=resident))
        return found

    def host_reclaim_status(self) -> dict[str, object]:
        """Explain current gates without changing backing or claiming RSS ownership.

        Byte totals deduplicate storage within each reason and component. Shared
        backing can appear under multiple reasons; those subtotals are not additive.
        """
        blocked = (
            "active_stack"
            if self.stack
            else "active_scope"
            if self.scopes
            else "group_phase"
            if self.group_phase
            else ""
        )
        reasons: dict[str, dict[int, int]] = {}
        components: dict[str, object] = {}
        for held in self.held.values():
            part = held.part
            compiled = held.compiled or _compiled(part.root)
            whole_ready = all(
                unit.tier in (HOST, DISK)
                and unit.readonly
                and not unit.mutable
                and tiers.store_safe(unit)
                for unit in part.units
            )
            local: dict[str, dict[int, int]] = {}
            for unit in part.units:
                backing = self._unit_host_backings(unit)
                if not backing:
                    continue
                if blocked:
                    reason = blocked
                elif compiled:
                    reason = "compiled_component"
                elif unit.mutable:
                    reason = "mutable_state"
                elif not unit.readonly:
                    reason = "readonly_authority_lost"
                elif unit.tier == DEVICE:
                    reason = "eligible_cached_copy"
                elif part.store is None:
                    reason = "no_store"
                elif not tiers.store_safe(unit):
                    reason = "unsafe_storage_geometry"
                elif not part.unit_loads and not whole_ready:
                    reason = "whole_component_not_droppable"
                else:
                    reason = "eligible_host_weight"
                local.setdefault(reason, {}).update(backing)
                reasons.setdefault(reason, {}).update(backing)
            if local:
                components[held.key] = {
                    "bytes_by_reason": {
                        reason: sum(backing.values()) for reason, backing in local.items()
                    },
                    "readonly_lost_reason": held.readonly_lost_reason,
                    "unit_loads": part.unit_loads,
                    "has_store": part.store is not None,
                    "compiled": compiled,
                }
        return {
            "bytes_by_reason": {
                reason: sum(backing.values()) for reason, backing in reasons.items()
            },
            "components": components,
        }

    def host_memory(self) -> dict[str, int]:
        """Deduplicated managed CPU storage, separate from the process's total RSS."""
        staging, pinned = self._staging_memory()
        return {
            "managed_host_bytes": sum(self._host_backings().values()) + staging,
            "staging_host_bytes": staging,
            "staging_pinned_bytes": pinned,
            "staging_pageable_bytes": staging - pinned,
            "resident_host_bytes": sum(self._host_backings(cached=False).values()),
            "cached_host_bytes": sum(self._host_backings(resident=False).values()),
        }

    def _released_host_bytes(self, before: dict[int, int]) -> int:
        after = self._host_backings()
        return sum(size for identity, size in before.items() if identity not in after)

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
        capacity = oom or getattr(error, "code", "") in (SHORTFALL, DEVICE_OOM)
        # Each constraint prices the same additional allocation independently. Raising
        # both current readings by the shared deficit overprices the nonbinding one.
        required = max(needed, available + 1) if capacity else 0
        return {
            "needed": needed,
            "available": available,
            "free_bytes": free,
            "grant_bytes": grant,
            "free_wanted": required if capacity else 0,
            "grant_wanted": self._reserved() + self.overhead + required if capacity else 0,
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
        active = next((frame for frame in reversed(self.stack) if frame.held is held), None)
        if active is not None:
            forward = tiers.original(getattr(held.part.root, "forward", None))
            transparent = (
                not active.overlap_joined
                and getattr(active.overlap_entry, "__self__", None) is held.part.root
                and getattr(active.overlap_entry, "__func__", None)
                is self.torch.nn.Module._call_impl
                and getattr(fn, "__self__", None) is held.part.root
                and getattr(fn, "__func__", fn) is getattr(forward, "__func__", forward)
            )
            if not transparent:
                self._overlap_retire(pending_only=True)
                for parent in self.stack:
                    parent.overlap_enabled = parent.overlap_learnable = False
            previous = active.overlap_joined
            active.overlap_joined = True
            try:
                self._ready(held.key)
                return fn(*args, **kwargs)
            finally:
                active.overlap_joined = previous
        if not self.cuda or held.hosted:
            self._ready(held.key)
            return fn(*args, **kwargs)
        # A nested callee may fill/reclaim. Retire only unbound hints here; a parent
        # consumer retains its own custody until its posthook or failed-call cleanup.
        self._overlap_retire(pending_only=True)
        for parent in self.stack:
            parent.overlap_enabled = parent.overlap_learnable = False
        self._validate_active(held)
        torch = self.torch
        leaves = list(_leaves((args, kwargs)))
        key = OpKey(
            held.kind,
            method,
            tuple((tuple(x.shape), str(x.dtype)) for x in leaves if isinstance(x, torch.Tensor)),
            tuple(x for x in leaves if isinstance(x, _SCALAR)),
            _mode(held.part.root)
            + (
                (("weight_conversion", f"{held.key}:{held.conversion_epoch}"),)
                if held.conversion_epoch
                else ()
            ),
        )
        held.compiled = _compiled(held.part.root)
        # A prepared group transaction executes once after commit even if a future
        # component classifier misses its sharded registration.
        retryable = held.key not in self._gates and self._retryable(held, leaves)
        self.clock += 1
        held.last_used = self.clock
        self.seq.append(held.key)
        self._pin_state(held)
        contract_reason = "" if retryable else self._state_contract_reason(held, leaves)
        readonly = not contract_reason
        if contract_reason and not held.readonly_lost_reason:
            held.readonly_lost_reason = contract_reason
        for unit in held.part.units:
            unit.readonly = unit.readonly and readonly and not unit.mutable
            if not unit.readonly:
                # An unknown hook may write in place without changing a pointer. A
                # later declaration cannot validate an old host/checkpoint copy.
                unit.host.clear()
                unit.seen.clear()
        held.scope_anchor = (
            held.part.anchor is not None
            and held.forward_only
            and readonly
            and all(unit.readonly and not unit.mutable for unit in held.part.common_units)
        )
        counts = self.operations.setdefault(f"{held.key}.{method}", {"calls": 0})
        counts["calls"] += 1
        counts["total_blocks"] = len(held.part.blocks)
        snapshot: Snapshot | None = None
        floor = False
        previous_room: int | None = None
        previous_shortfall = 0
        room_tried = False
        replayed = False
        try:
            while True:
                plan = self._plan(held, key, retryable, floor)
                if previous_room is not None:
                    self._replay_room(held, plan, previous_room + max(previous_shortfall, 1))
                counts["min_keep_blocks"] = min(counts.get("min_keep_blocks", plan.keep), plan.keep)
                counts["max_keep_blocks"] = max(counts.get("max_keep_blocks", plan.keep), plan.keep)
                frame: Frame | None = None
                entered = False
                asked, shortfall, text = 0, 0, ""
                try:
                    # Placement changes no authored state. Reclaim before snapshot allocation,
                    # and retain one baseline across retries rather than cloning it again.
                    with self._traffic(counts, "placement"):
                        self._apply(held, plan)
                    if previous_room is not None:
                        gained = self.allocatable() - previous_room
                        if gained < max(previous_shortfall, 1) and not room_tried:
                            room_tried = self._ask_room(
                                previous_room + max(previous_shortfall, 1),
                                [],
                                measured=True,
                                force=True,
                            )
                            gained = self.allocatable() - previous_room
                        if gained < max(previous_shortfall, 1):
                            raise MemoryRefusal(
                                DEVICE_OOM,
                                f"{held.key}: no retry: placement reclaimed {gained} B; "
                                f"the failed allocation shortfall needs {previous_shortfall} B",
                                previous_shortfall,
                                max(gained, 0),
                            )
                    if retryable and snapshot is None:
                        snapshot = self._arm(held, leaves)
                    self._sample()
                    frame = Frame(held, self._allocated())
                    frame.overlap_entry = tiers.original(fn)
                    self._overlap_frame(frame, key, retryable and not replayed)
                    self.stack.append(frame)
                    self._ready(held.key)
                    entered = True
                    with self._traffic(counts, "invoke"):
                        out = fn(*args, **kwargs)
                    self._validate_active(held)
                except BaseException as exc:
                    self._overlap_retire(held=held, clear=True)
                    replayed = True
                    oom = is_oom(exc)
                    if oom:
                        self._sample()
                        asked = self.oom_asked or _asked(str(exc))
                        # Capture the exact remaining reserve room before failed frames
                        # or cache cleanup release bytes. Only the uncovered part of the
                        # allocation must be gained; existing room is not needed twice.
                        shortfall = max(asked - max(self.allocatable(), 0), 1)
                        counts["oom_allocation_bytes"] = max(
                            counts.get("oom_allocation_bytes", 0), asked
                        )
                        counts["oom_shortfall_bytes"] = max(
                            counts.get("oom_shortfall_bytes", 0), shortfall
                        )
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
                    self._overlap_learn(frame, key)
                    if record.fail and (
                        frame.working >= record.fail or plan.keep >= record.fail_keep
                    ):
                        record.fail = 0
                        record.fail_keep = 0
                        record.fail_capacity = 0
                    seen = self.plans.setdefault(f"{held.key.rsplit('/', 1)[-1]}.{method}", [])
                    shape = f"{dict(key.mode)} {plan.keep}/{len(held.part.blocks)} blocks"
                    if shape not in seen:
                        seen.append(shape)
                    return out
                finally:
                    self._sample()
                    if frame is not None:
                        self._overlap_retire(held=held)
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
                    shortfall=shortfall,
                )
                room_tried |= asked_worker
                previous_shortfall = shortfall

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
        shortfall: int | None = None,
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
        shortfall = max(asked, 1) if shortfall is None else shortfall
        bound = frame.working + asked
        record = self.records.setdefault(key, Record())
        record.fail = max(record.fail, bound)
        record.fail_keep = max(record.fail_keep, plan.keep)
        room = self.allocatable()
        record.fail_capacity = max(record.fail_capacity, room + self._reserved())
        numbers = (
            f"{held.key}.{key.method} ran out of device memory ({text}); it needs at least "
            f"{bound} B of working memory, {room} B are allocatable, "
            f"{plan.keep}/{len(held.part.blocks)} blocks resident"
        )
        self.log.append(numbers)
        reclaim = not self.spare and any(self._evictable(v) for v in self._victims(held))
        smaller = (snapshot is not None or not entered) and (
            plan.keep > self._minimum_keep(held) or reclaim
        )
        asked_worker = False
        # A known floor still gets one measured Worker request after an actual OOM.
        # Preparation can use this too: authored code has not run and needs no rollback.
        if not smaller and request_room and (snapshot is not None or not entered):
            asked_worker = self._ask_room(room + shortfall, [], measured=True, force=True)
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
                room + shortfall,
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
        scaled: list[tuple[float, float]] = []
        for other, row in self.records.items():
            ratio = key.ratio(other) if row.ok >= 0 and other.component == key.component else None
            if ratio is not None:
                tiled = bool(dict(key.mode).get("use_tiling"))
                scaled.append((abs(ratio - 1), row.ok * (1.0 if tiled else max(ratio, 1.0))))
        floor = record.fail if record is not None else 0
        return (max(int(min(scaled)[1]), floor) if scaled else floor or None), False

    def _plan(
        self,
        held: Held,
        key: OpKey,
        retryable: bool,
        floor: bool,
    ) -> Plan:
        if self.observe_process():
            self._cap()
        working, exact = self._estimate(key)
        record = self.records.get(key)
        measured = exact or not working or bool(record and record.fail >= (working or 0))
        ceiling = (
            record.fail_keep - 1
            if record is not None
            and record.fail_keep
            and self.allocatable() + self._reserved() <= record.fail_capacity
            else None
        )
        if self.cuda and self._reserved() > self._allocated():
            resident = self._need(held.part, len(held.part.blocks))
            if resident + (working or 0) > self.allocatable():
                # The previous op's cache may be all that stands in the way: give it back so
                # the plan reads what can really be reserved.
                tiers.trim(self.torch)
        victims = [] if self.spare else self._victims(held)
        try:
            if floor or (not retryable and not exact):
                return self._fit(held, working or 0, victims, floor=True, measured=measured)
            if retryable and working is None:
                return self._max_room(held, victims)
            return self._fit(
                held, working or 0, victims, floor=False, ceiling=ceiling, measured=measured
            )
        except MemoryRefusal as exc:
            if exact or self.spare or exc.code != SHORTFALL:
                raise
            # An inexact shape forecast is not a refusal proof. Metadata pins stay
            # intact while the legal floor gets the same ordinary admission/retry path.
            record = self.records.get(key)
            failure_floor = record.fail if record is not None else 0
            plan = self._fit(held, failure_floor, victims, floor=True)
            self.log.append(f"{held.key}.{key.method}: forecast exceeded room; attempting floor")
            return plan

    def _replay_room(self, held: Held, plan: Plan, target: int) -> None:
        """Price the failed allocation's full gain against legally reclaimable sibling cache.

        Working-memory samples are lower bounds. Even when that bound fits the same
        placement, an actual replay must make the additional room the failure requested.
        """
        part = held.part
        stream = max(
            (tiers.load_demand(self.torch, unit).peak for unit in part.blocks[plan.keep :]),
            default=0,
        )
        placement = self._need(part, plan.keep) - stream
        needed = max(plan.needed or 0, target + placement)
        victims = [] if self.spare else self._victims(held)
        chosen = self._room_from(victims, needed)
        plan.evict = victims if chosen is None else chosen
        plan.needed = needed

    def _fit(
        self,
        held: Held,
        working: int,
        victims: list[Held],
        *,
        floor: bool,
        ceiling: int | None = None,
        measured: bool = True,
    ) -> Plan:
        """Keep the most blocks that fit; the floor streams every legal block."""
        needed = self._need(held.part, self._minimum_keep(held)) + working

        def quote() -> dict[Held, int] | None:
            # Ordinary byte enumeration runs no authored code or transfers. Whole-store
            # callbacks, pending Torch casts and group host-credit checks stay uncached.
            part = held.part
            if (
                self.grouped
                or self.group_phase
                or (part.on(DISK) and not part.unit_loads)
                or any(unit.tier == DISK and unit.pending for unit in part.units)
            ):
                return None
            return {victim: self._evictable(victim) for victim in victims}

        credits = quote()
        available = self.allocatable() + sum(
            self._evictable(v) if credits is None else credits[v] for v in victims
        )
        keeps = [self._minimum_keep(held)] if floor else self._keeps(held)
        if ceiling is not None:
            keeps = [keep for keep in keeps if keep <= ceiling]
        for keep in keeps:
            needed = self._need(held.part, keep) + working
            chosen = self._room_from(victims, needed, credits=credits)
            minimum = keep == self._minimum_keep(held)
            if chosen is None and self._ask_room(
                needed, victims, measured=measured and minimum, credits=credits
            ):
                credits = quote()
                chosen = self._room_from(victims, needed, credits=credits)
            if chosen is not None:
                return Plan(keep, chosen, needed=needed)
            available = self.allocatable() + sum(
                self._evictable(v) if credits is None else credits[v] for v in victims
            )
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
            if whole > self.allocatable() + own:
                self._ask_room(
                    self._need(held.part, self._minimum_keep(held)), victims, measured=True
                )
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
                if unit.mutable
                or (
                    self.grouped
                    and (
                        (not unit.readonly and self._group_host_credit is None)
                        or not self._movable(held.part, unit)
                    )
                )
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
        """Net settled placement plus its largest real load/conversion overlap."""
        if part.on(DISK) and not part.unit_loads and part.store is not None:
            return tiers.load_sequence(
                self.torch, part.units, whole=True
            ).peak + part.store.transient(part.name)
        placement = tiers.load_sequence(self.torch, (*part.common_units, *part.blocks[:keep]))
        drop = sum(unit.nbytes for unit in part.blocks[keep:] if unit.tier == DEVICE)
        stream = max(
            (tiers.load_demand(self.torch, unit).peak for unit in part.blocks[keep:]), default=0
        )
        return max(placement.peak, placement.resident + stream) - drop

    def _room_from(
        self, victims: list[Held], need: int, *, credits: Mapping[Held, int] | None = None
    ) -> list[Held] | None:
        """The fewest victims, in order, whose bytes make `need` allocatable; None if all can't."""
        room = self.allocatable()
        chosen: list[Held] = []
        for victim in victims:
            if room >= need:
                break
            chosen.append(victim)
            room += self._evictable(victim) if credits is None else credits[victim]
        return chosen if room >= need else None

    def _ask_room(
        self,
        need: int,
        victims: list[Held],
        *,
        measured: bool,
        force: bool = False,
        credits: Mapping[Held, int] | None = None,
    ) -> bool:
        """Ask once per concrete demand and grant; an infeasible rich plan cannot consume
        a later minimum request. True when asked, so the caller measures again."""
        if self.room is None or self.spare:
            return False
        own = sum(self._evictable(v) if credits is None else credits[v] for v in victims)
        before = self.grant
        # Physical free excludes already-consumed context bytes. The total grant must
        # still cover them and every reserved segment, including unreleasable cache holes.
        request = (
            max(need - own, 0),
            max(self._reserved() - own + need + self.overhead, 0),
            measured,
            before,
        )
        if request in self.room_requests and not force:
            return False
        self.room_requests.add(request)
        self.room_asks += 1
        grant = self.room(*request[:3])
        if grant is not None:
            self.grant = grant
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
                common.add(key)
                held = self.held.get(key)
                if held is not None and held.whole:
                    whole.add(key)
        return whole, common

    def _evictable(self, victim: Held) -> int:
        self._refresh_constraints(victim)
        whole, common = self._pinned()
        if victim.key in whole or victim.compiled:
            return 0
        protected = set(self._scope_common(victim)) if victim.key in common else set()
        units = tuple(unit for unit in victim.part.units if unit not in protected)
        whole_store = units == victim.part.units and self._whole_store(victim.part)
        return sum(
            unit.nbytes
            for unit in units
            if unit.tier == DEVICE
            and not unit.mutable
            and (whole_store or self._movable(victim.part, unit))
        )

    def _movable(self, part: tiers.Component, unit: tiers.Unit) -> bool:
        """Count only homes legal under the current rank's host-admission contract."""
        return (
            not (self.grouped or self.group_phase)
            or unit.tier == HOST
            or (
                self._group_host_credit is not None
                and tiers.host_copy_bytes(unit) <= self._group_host_credit
            )
            or tiers.host_copy_bytes(unit) == 0
            or (
                unit.readonly
                and not unit.mutable
                and part.unit_loads
                and part.store is not None
                and tiers.store_safe(unit)
            )
        )

    @staticmethod
    def _whole_store(part: tiers.Component) -> bool:
        return (
            part.store is not None
            and not part.unit_loads
            and all(
                unit.tier in (DEVICE, HOST)
                and unit.readonly
                and not unit.mutable
                and tiers.store_safe(unit)
                for unit in part.units
            )
        )

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

        # Re-evaluate replacement/compile state once per candidate pass, not once per
        # block-count alternative considered by _fit.
        for victim in self.held.values():
            victim.compiled = _compiled(victim.part.root)
        candidates = [v for v in self.held.values() if v is not held and self._evictable(v)]
        return sorted(
            candidates,
            key=lambda v: (v.construction == construction, -next_use(v), v.last_used),
        )

    # ------------------------------------------------------------------ moving bytes

    def _apply(self, held: Held, plan: Plan) -> None:
        self._cap()
        for victim in plan.evict:
            if plan.needed is not None and self.allocatable() >= plan.needed:
                break
            self._offload(victim, target=plan.needed)
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
        self._overlap_retire(pending_only=True)
        part = held.part
        if all(unit.tier == DEVICE for unit in part.common_units):
            return
        if any(unit.tier == DISK for unit in part.common_units) and not part.unit_loads:
            self.refilled += part.on(DISK)
            with self._moving():
                tiers.from_disk(self.torch, part, None)
            return
        for unit in part.common_units:
            self._load(part, unit)

    def _load(self, part: tiers.Component, unit: tiers.Unit) -> None:
        if unit.tier == DEVICE:
            return
        self._overlap_retire(pending_only=True)
        tiers.load_demand(self.torch, unit)  # Refuse unbounded pending transforms before a fill.
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

    def _offload(self, held: Held, *, everything: bool = False, target: int | None = None) -> None:
        """Move a victim's unpinned device units off the device."""
        self._overlap_retire(pending_only=True)
        self._refresh_constraints(held)
        part = held.part
        whole, common = self._pinned()
        if held.compiled or (held.key in whole and not everything):
            return
        protected = (
            set(self._scope_common(held)) if held.key in common and not everything else set()
        )
        units = tuple(unit for unit in part.units if unit not in protected)
        moving = [unit for unit in units if unit.tier == DEVICE and not unit.mutable]
        if units == part.units and self._whole_store(part):
            needed = sum(tiers.host_copy_bytes(unit) for unit in moving)
            with self._moving(), self._host_copy(needed, protected=tuple(moving)) as admitted:
                if admitted:
                    for unit in moving:
                        self.moved += tiers.to_host(self.torch, unit)
                        self.offloads += 1
                else:
                    self.stored += sum(unit.nbytes for unit in moving)
                    tiers.to_disk(self.torch, part, None)
            self.dropped |= bool(moving)
            return
        for unit in moving:
            if target is not None and self.allocatable() >= target:
                break
            if self._movable(part, unit):
                self._home(part, unit)
        self.dropped |= any(unit.tier != DEVICE for unit in moving)

    def host_needed(self) -> int:
        """Fresh backing for at most one copy of every movable unit in this operation.

        Partitioned units own disjoint storage; cross-component aliases are rejected at
        adoption. Existing host backings cost nothing, and ranks own separate address spaces.
        """
        return sum(self.staging_unbacked.values()) + sum(
            tiers.load_demand(self.torch, unit).resident
            if unit.tier == DISK
            else tiers.host_copy_bytes(unit)
            for held in self.held.values()
            for unit in held.part.units
            if unit.tier != HOST and not unit.mutable
        )

    @contextmanager
    def host_credit(self, nbytes: int | None) -> Iterator[None]:
        """One prepared operation's prepaid, consumable credit; never a process allowance."""
        if nbytes is not None and (type(nbytes) is not int or nbytes < 0):
            raise ValueError("group host credit must be nonnegative bytes")
        before = self._group_host_credit
        self._group_host_credit = nbytes
        self._last_host_credit = nbytes or 0
        try:
            yield
        finally:
            self._last_host_credit = self._group_host_credit or 0
            self._group_host_credit = before

    def host_remaining(self) -> int:
        return (
            self._last_host_credit if self._group_host_credit is None else self._group_host_credit
        )

    @contextmanager
    def _host_copy(
        self, nbytes: int, *, protected: tuple[tiers.Unit, ...] = (), retain_on_fault: bool = False
    ) -> Iterator[bool]:
        """Reserve before allocating and release the pending promise on every exit.

        Standalone planners use local observed capacity only. Managed executors supply
        the aggregate Worker callback. Group phases never send Worker IPC: without a
        pregrant they can only rebind cached bytes or take a legal store path.
        """
        if not nbytes:
            yield True
            return
        self._overlap_retire(pending_only=True)
        if self.grouped or self.group_phase:
            credit = self._group_host_credit
            admitted = credit is not None and nbytes <= credit
            if admitted:
                assert credit is not None
                # Credit was reserved by the Worker for all ranks before this round.
                # Consume once even if a copy raises: partial allocations may remain live.
                self._group_host_credit = credit - nbytes
            yield admitted
            return
        self._host_sequence += 1
        token = f"host-{self._host_sequence}"

        def reserve() -> bool:
            if self.host_room is not None:
                return self.host_room(token, nbytes, False)
            return proctree.available_host_bytes() >= nbytes + self._host_pending

        admitted = reserve()
        if not admitted:
            if self._release_host(nbytes, protected=protected):
                # The active requester cannot be an idle Worker victim. Return its
                # own allocator-free pages before asking for observed headroom again.
                proctree.trim_host_allocator()
            admitted = reserve()
        if not admitted:
            yield False
            return
        self._host_pending += nbytes
        try:
            yield True
        finally:
            if not (retain_on_fault and self.unusable):
                self._host_pending -= nbytes
                if self.host_room is not None:
                    self.host_room(token, nbytes, True)

    @staticmethod
    def staging_vector(
        demands: tuple[tuple[int, int, int], ...], reserve: Callable[[int], bool]
    ) -> tuple[int, ...]:
        """Lower slot count, never slot quantum; old readers require their full geometry."""
        if not demands or any(q <= 0 or low <= 0 or high < low for q, high, low in demands):
            raise ValueError("invalid staging demand vector")
        previous: tuple[int, ...] = ()
        for cap in range(max(high for _, high, _ in demands), 0, -1):
            sizes = tuple(q * max(low, min(cap, high)) for q, high, low in demands)
            if sizes != previous and reserve(sum(sizes)):
                return sizes
            previous = sizes
        return ()

    def reserve_local_staging(
        self, demands: tuple[tuple[int, int, int], ...]
    ) -> tuple[tuple[int, ...], int]:
        """Legacy Worker fallback: one leader-observed vector, not a global promise."""

        def reserve(nbytes: int) -> bool:
            if proctree.available_host_bytes() < nbytes + self._host_pending:
                return False
            self._host_pending += nbytes
            return True

        credits = self.staging_vector(demands, reserve)
        return credits, sum(credits)

    def release_local_staging(self, nbytes: int) -> None:
        """The caller proves all rank fills drained/reaped before ending this promise."""
        if nbytes < 0 or nbytes > self._host_pending:
            raise ValueError("invalid local staging settlement")
        self._host_pending -= nbytes

    @contextmanager
    def staging_host(self, quantum: int, slots: int, *, owner: str = "") -> Iterator[int]:
        """Fund one retained ring allocation; its backing is reused, never credit-refunded."""
        if quantum <= 0 or slots <= 0:
            raise ValueError("staging geometry must be positive")
        if self.unusable:
            raise MemoryRefusal("device_fault", self.unusable, reusable=False)
        self.staging_unbacked[owner] = quantum
        construction = self.construction_staging_credit
        if self.grouped and construction is not None:
            admitted = min(slots, construction // quantum)
            if not admitted:
                raise self._host_refusal(quantum)
            self.construction_staging_credit = construction - admitted * quantum
            try:
                yield admitted
            except RuntimeFailure as exc:
                if exc.code == "device_fault":
                    self.unusable = str(exc)
                raise
            else:
                self.staging_unbacked.pop(owner, None)
            return
        # An operation's group vector carries the minimum ring plus independent backing
        # copies. Use only that minimum; never consume their promised bytes opportunistically.
        counts = (1,) if self.grouped or self.group_phase else range(slots, 0, -1)
        for admitted in counts:
            with self._host_copy(quantum * admitted, retain_on_fault=True) as ok:
                if not ok:
                    continue
                try:
                    yield admitted
                except RuntimeFailure as exc:
                    if exc.code == "device_fault":
                        self.unusable = str(exc)
                    raise
                else:
                    self.staging_unbacked.pop(owner, None)
                return
        raise self._host_refusal(quantum)

    def _host_refusal(self, nbytes: int) -> MemoryRefusal:
        refusal = MemoryRefusal(
            "host_shortfall",
            f"no host reservation for {nbytes} B and no legal store fallback",
            nbytes,
        )
        refusal.shortfall["resource"] = "ram"
        return refusal

    def _release_host(self, nbytes: int, *, protected: tuple[tiers.Unit, ...] = ()) -> bool:
        """Drop eligible local backing; whether references changed is not RSS credit."""
        released = False
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
                        return released
                    if clean and unit.tier == DEVICE and unit.host:
                        nbytes -= sum(t.numel() * t.element_size() for t in unit.host.values())
                        unit.host.clear()
                        released = True
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
                        released = True
        return released

    def _enter_unit(self, part: tiers.Component, unit: tiers.Unit) -> None:
        """At actual demand, publish a proven copy or use the ordinary load path."""
        self._sample()
        frame = next((f for f in reversed(self.stack) if f.held.part is part), None)
        if frame is not None:
            depth = frame.overlap_depth.get(unit, 0)
            frame.overlap_depth[unit] = depth + 1
            if depth:
                self._overlap_retire(pending_only=True)
                frame.overlap_enabled = frame.overlap_learnable = False
                return
        pending = frame.overlap_pending if frame is not None else None
        hit = False
        if pending is not None:
            assert frame is not None
            if pending.unit is unit:
                started = time.perf_counter_ns()
                with self._moving():
                    hit = pending.publish()
                self._overlap_count("copy_wait_ns", time.perf_counter_ns() - started)
                frame.overlap_pending = None
                if hit:
                    frame.overlap_consumers[unit] = pending
                    self.loads += 1
                    self.moved += pending.source_bytes
                    self._overlap_count("hits")
                else:
                    self._overlap_count("wasted_bytes", pending.submitted_bytes)
                    self.torch.cuda.reset_peak_memory_stats(self.device)
            else:
                self._overlap_retire(pending_only=True)
        if hit or unit.tier != DEVICE:
            if not hit:
                self._load(part, unit)
            unit.streamed = True
            if frame is not None:
                frame.loaded += unit.nbytes
        if frame is not None:
            frame.overlap_visits.append(unit)
            self._overlap_prepare(frame, unit)

    def _leave_unit(self, part: tiers.Component, unit: tiers.Unit) -> None:
        self._sample()
        frame = next((f for f in reversed(self.stack) if f.held.part is part), None)
        if frame is not None:
            depth = frame.overlap_depth.get(unit, 0)
            if depth > 1:
                frame.overlap_depth[unit] = depth - 1
                return
            frame.overlap_depth.pop(unit, None)
            transfer = frame.overlap_consumers.pop(unit, None)
            if transfer is not None:
                with self._moving():
                    transfer.finish()
            elif unit in frame.overlap_drain:
                # Authored purity does not promise exclusive use of the current stream.
                try:
                    self.torch.cuda.synchronize(self.device)
                except BaseException as exc:
                    self.unusable = f"HOST overlap consumer drain failed: {exc}"
                    if frame.overlap_pending is not None:
                        frame.overlap_pending._poison(exc, "consumer drain")
                    raise MemoryRefusal("device_fault", self.unusable, reusable=False) from exc
            frame.overlap_drain.discard(unit)
        if not unit.streamed:
            return
        unit.streamed = False
        self._home(part, unit)
        if frame is not None:
            frame.loaded -= unit.nbytes

    def _overlap_count(self, name: str, amount: int = 1) -> None:
        self.overlap[name] = self.overlap.get(name, 0) + amount

    def _overlap_observed(self, transfer: tiers.HostTransfer) -> None:
        self._overlap_count("cpu_staged_bytes", transfer.staged_bytes)
        self._overlap_count("h2d_submitted_bytes", transfer.submitted_bytes)
        if transfer.state != "prepared":
            self._overlap_count("wasted_bytes", transfer.submitted_bytes)

    @staticmethod
    def _overlap_contract(held: Held) -> tuple[object, ...]:
        # Once per operation boundary, never once per keep alternative/block. The
        # positive pure/retry declaration supplies immutable custody inside the body;
        # tensor version snapshots alone cannot detect inference/.data writers.
        return tuple(
            (
                id(module),
                id(
                    getattr(
                        tiers.original(getattr(module, "forward", None)),
                        "__func__",
                        tiers.original(getattr(module, "forward", None)),
                    )
                ),
                tuple(id(fn) for fn in _hook_fns(module)),
            )
            for module in held.part.root.modules()
        ) + (id(getattr(held.part.root, "_cozy_retry_state", None)),)

    def _overlap_frame(self, frame: Frame, key: OpKey, retryable: bool) -> None:
        held = frame.held
        frame.overlap_key = key
        eligible = (
            self.cuda
            and not self.stack
            and not self.grouped
            and not self.group_phase
            and not self.spare
            and retryable
            and held.forward_only
            and not held.whole
            and not held.compiled
            and not held.hosted
            and not self.sharded(held.key)
            and not self.torch.is_grad_enabled()
            and all(u.readonly and not u.mutable and not u.pending for u in held.part.units)
        )
        if not eligible:
            held.overlap_key, held.overlap_order = None, ()
            return
        frame.overlap_learnable = True
        frame.overlap_contract = self._overlap_contract(held)
        record = self.records.get(key)
        if (
            held.overlap_contract == frame.overlap_contract
            and record is not None
            and record.ok > 0
            and held.overlap_key == key
            and held.overlap_epoch == held.conversion_epoch
        ):
            frame.overlap_enabled = True
            frame.overlap_expected = held.overlap_order
            frame.overlap_obligation = max(record.ok, record.fail)

    def _overlap_learn(self, frame: Frame, key: OpKey) -> None:
        held, visits = frame.held, frame.overlap_visits
        # A repeated/nested block has ambiguous consumer lifetime in this experiment.
        if (
            frame.overlap_learnable
            and visits
            and len(set(visits)) == len(visits)
            and frame.overlap_contract == self._overlap_contract(held)
        ) and all(u.readonly and not u.mutable and not u.pending for u in held.part.units):
            held.overlap_key, held.overlap_order = key, tuple(visits)
            held.overlap_epoch = held.conversion_epoch
            held.overlap_contract = frame.overlap_contract
        else:
            held.overlap_key, held.overlap_order = None, ()

    def _overlap_retire(
        self, *, held: Held | None = None, pending_only: bool = False, clear: bool = False
    ) -> None:
        for frame in self.stack:
            if held is not None and frame.held is not held:
                continue
            if clear:
                frame.overlap_enabled = frame.overlap_learnable = False
            transfer = frame.overlap_pending
            if transfer is not None:
                self._sample()
                with self._moving():
                    transfer.abort()
                self._overlap_count("wasted_bytes", transfer.source_bytes)
                frame.overlap_pending = None
                self.torch.cuda.reset_peak_memory_stats(self.device)
            if not pending_only:
                for unit, transfer in tuple(frame.overlap_consumers.items()):
                    with self._moving():
                        transfer.finish()
                    # Explicit conversion/unload can close custody inside a body.
                    # Its eventual posthook still drains any later submissions.
                    frame.overlap_drain.add(unit)
                frame.overlap_consumers.clear()
        if clear:
            for each in self.held.values():
                if held is None or each is held:
                    each.overlap_key, each.overlap_order = None, ()

    def _overlap_prepare(self, frame: Frame, current: tiers.Unit) -> None:
        if not frame.overlap_enabled:
            return
        order, seen = frame.overlap_expected, frame.overlap_visits
        if tuple(seen) != order[: len(seen)]:
            frame.overlap_enabled = False
            self._overlap_count("decline_order")
            return
        if len(seen) >= len(order):
            return
        unit = order[len(seen)]
        if (
            unit.tier != HOST
            or unit.pending
            or unit.mutable
            or not unit.readonly
            or frame.overlap_pending is not None
            or self.unusable
        ):
            self._overlap_count("decline_source")
            return
        acquire = getattr(frame.held.part.store, "borrow_staging", None)
        if acquire is None:
            self._overlap_count("decline_staging")
            return
        self.observe_process()
        self._cap()
        room = self.allocatable()
        # Keep the entire measured working obligation available. This conservative
        # first experiment does not infer live activation reuse or evict a resident.
        available = room - frame.overlap_obligation
        if (
            self._overlap_oom_unknown
            or available < unit.nbytes
            or (self._overlap_oom_room is not None and room < self._overlap_oom_room)
        ):
            self._overlap_count("decline_capacity")
            return
        epoch = frame.held.conversion_epoch

        def valid() -> bool:
            return (
                not self.unusable
                and frame in self.stack
                and frame.overlap_enabled
                and frame.held.conversion_epoch == epoch
                and unit.readonly
                and not unit.mutable
                and not unit.pending
                and not self.torch.is_grad_enabled()
            )

        self._sample()
        started = time.perf_counter_ns()
        old_asked = self.oom_asked
        self.oom_asked = 0
        try:
            if self._overlap_stream is None:
                self._overlap_stream = self.torch.cuda.Stream(device=self.device)
            with self._moving():
                transfer = tiers.HostTransfer.prepare(
                    self.torch,
                    unit,
                    self.device,
                    self._overlap_stream,
                    acquire=acquire,
                    device_capacity=available,
                    valid=valid,
                    owned=self._host_transfers,
                    observed=self._overlap_observed,
                )
            if transfer is None:
                self._overlap_count("decline_mechanism")
                return
            frame.overlap_pending = transfer
            frame.overlap_drain.add(current)
            self._overlap_count("prepared_bytes", transfer.source_bytes)
        except BaseException as exc:
            if not is_oom(exc) or self.unusable:
                raise
            asked = self.oom_asked or _asked(str(exc))
            # Unknown allocation size supplies no reproducible gain proof.
            if asked > 0:
                self._overlap_oom_room = max(room, self.allocatable()) + asked
            else:
                self._overlap_oom_unknown = True
            self._overlap_count("optional_oom")
        finally:
            self.oom_asked = old_asked
            self._overlap_count("prepare_ns", time.perf_counter_ns() - started)
            # No authored operation ran here. Transfer-only allocation peaks remain
            # in global allocator evidence, but never become activation measurements.
            self.peak_allocated = max(
                self.peak_allocated, int(self.torch.cuda.max_memory_allocated(self.device))
            )
            self.peak = max(self.peak, int(self.torch.cuda.max_memory_reserved(self.device)))
            self.torch.cuda.reset_peak_memory_stats(self.device)

    # ------------------------------------------------------------------ retry safety

    def _pin_state(self, held: Held) -> None:
        """Checkpoint residency never owns explicitly mutable tensor state."""
        provider = getattr(held.part.root, "_cozy_retry_tensors", None)
        if provider is None:
            return
        state = tuple(provider())
        if not state:
            return
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
        return not self._state_contract_reason(held, leaves)

    def _state_contract_reason(self, held: Held, leaves: list[object]) -> str:
        """The first unproven contract, retained when an operation loses readonly authority."""
        torch = self.torch
        root = held.part.root
        hooks = torch.nn.modules.module
        if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
            return "global_forward_hook"
        custom = getattr(root, "_cozy_retry_state", None)
        forward = tiers.original(getattr(root, "forward", None))
        if custom is None and not (is_declared_pure(root) or is_declared_pure(forward)):
            return "undeclared_root"

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
                return "stateful_diffusers_hook"
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
                return "unclaimed_forward_replacement"
            if not all(tiers.ours(fn) or identity(fn) in claimed_hooks for fn in _hook_fns(module)):
                return "unclaimed_forward_hook"
        plain = (
            *_SCALAR,
            torch.Tensor,
            torch.Generator,
            torch.dtype,
            torch.device,
            torch.nn.Module,
        )
        covered = (
            custom is not None and not getattr(root, "_cozy_retry_plain_args", False)
        ) or all(isinstance(leaf, plain) for leaf in leaves)
        return "" if covered else "unsupported_argument_state"

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


class HostTenant(Protocol):
    """Host victim identity; reclaim_host is an optional negotiated operation."""

    slot: str

    @property
    def last_used(self) -> int: ...

    def state(self) -> str:
        """`active`, `idle`, or `dead`."""
        ...


class Tenant(HostTenant, Protocol):
    """One executor as the Worker's device policy sees it: a process over some devices."""

    @property
    def ordinals(self) -> tuple[int, ...]: ...

    @property
    def held(self) -> dict[int, int]:
        """Bytes it holds per device: non-allocator overhead plus allocator reserve."""
        ...

    @property
    def overhead(self) -> dict[int, int]: ...

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
        host_pressure: Callable[[], proctree.HostPressure | None] | None = None,
        store_ready: Callable[[], None] = lambda: None,
    ) -> None:
        self.budget = dict(budget)
        self.driver = driver
        self.note = note
        self.queue: tuple[Demand, ...] = ()
        self._host_lock = Lock()
        #: Serializes host-reclaim RPCs; the reservation lock never spans an RPC.
        self._host_reclaim_lock = Lock()
        self.host_available = host_available
        self.host_pressure = host_pressure
        self.host_reserved: dict[str, int] = {}
        self.optional_host_available: Callable[[], int] | None = None
        self._store_condition = Condition()
        self._store_demands: set[str] = set()
        self._speculative_reader = False
        self.store_ready = store_ready

    def _store_ready(self) -> None:
        # Optional scheduling cannot fail a foreground ACK or generation cleanup.
        try:
            self.store_ready()
        except Exception:
            pass

    def begin_store_read(self, token: str, valid: Callable[[], bool] = lambda: True) -> bool:
        """Publish demand BEFORE waiting: its ACK follows native speculative reader drain."""
        ready = False
        waited = False
        began = time.perf_counter()
        with self._store_condition:
            if not valid():
                return False
            self._store_demands.add(token)
            self._store_condition.notify_all()
            while self._speculative_reader and token in self._store_demands:
                waited = True
                self._store_condition.wait()
            if not valid():
                before = bool(self._store_demands)
                self._store_demands.discard(token)
                ready = before and not self._store_demands
                self._store_condition.notify_all()
            admitted = token in self._store_demands
        if ready:
            self._store_ready()
        if waited and admitted:
            try:
                self.note(
                    f"foreground store demand waited {(time.perf_counter() - began) * 1000:.1f}ms for speculative reader drain"
                )
            except Exception:
                pass
        return admitted

    def end_store_read(self, token: str) -> None:
        with self._store_condition:
            before = bool(self._store_demands)
            self._store_demands.discard(token)
            ready = before and not self._store_demands
            self._store_condition.notify_all()
        if ready:
            self._store_ready()

    def release_store_prefix(self, prefix: str) -> None:
        """Only confirmed executor retirement can abandon that generation's demand tokens."""
        if not prefix:
            raise ValueError("store read cleanup requires an explicit generation namespace")
        with self._store_condition:
            before = bool(self._store_demands)
            self._store_demands = {t for t in self._store_demands if not t.startswith(prefix)}
            ready = before and not self._store_demands
            self._store_condition.notify_all()
        if ready:
            self._store_ready()

    @contextmanager
    def store_demand(self, token: str, retain: Callable[[], str] = lambda: "") -> Iterator[None]:
        token = token + ":" + uuid.uuid4().hex
        if not self.begin_store_read(token):
            raise RuntimeFailure("store demand generation was retired", code="executor_replaced")
        try:
            yield
        finally:
            with self._store_condition:
                # A still-bound failed generation may own native readers. Rebind its
                # phase protection atomically with the same lock as confirmed-reap cleanup.
                before = bool(self._store_demands)
                owner = retain()
                if owner:
                    self._store_demands.add(owner + token)
                self._store_demands.discard(token)
                ready = before and not self._store_demands
                self._store_condition.notify_all()
            if ready:
                self._store_ready()

    @contextmanager
    def optional_store_read(self) -> Iterator[bool]:
        """One optional native reader, atomically excluded by waiting or active demand."""
        with self._store_condition:
            admitted = not self._store_demands and not self._speculative_reader
            if admitted:
                self._speculative_reader = True
        try:
            yield admitted
        finally:
            if admitted:
                with self._store_condition:
                    self._speculative_reader = False
                    self._store_condition.notify_all()

    def optional_store_wanted(self) -> bool:
        with self._store_condition:
            return not self._store_demands

    def prefetch_allowance(self, slots_bytes: int) -> int:
        """Optional cache work spends only observed unreclaimed room, never demand credit."""
        if self.optional_host_available is None:
            return 0
        room = self.optional_host_available()
        if room < 0:
            return 0
        with self._host_lock:
            return max(0, room - sum(self.host_reserved.values()) - slots_bytes)

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

    def reserve_host(
        self,
        owner: str,
        nbytes: int,
        *,
        requester: str = "",
        tenants: Sequence[HostTenant] = (),
        optional: bool = False,
    ) -> bool:
        """Reserve against fresh aggregate room, reclaiming safe idle cache if needed.

        Victims release progressively, without holding the reservation lock across RPC.
        A peer's claimed release never grants credit: only observed headroom does.
        """
        if nbytes < 0:
            raise ValueError("host reservation must be nonnegative")
        if not nbytes:
            return True
        if self.host_available is None:
            return False

        def reserve() -> bool:
            assert self.host_available is not None
            with self._host_lock:
                available = self.host_available()
                if optional:
                    if self.optional_host_available is None:
                        return False
                    available = min(available, self.optional_host_available())
                reserved = sum(self.host_reserved.values())
                if available < 0 or nbytes > available - reserved:
                    return False
                self.host_reserved[owner] = self.host_reserved.get(owner, 0) + nbytes
                return True

        if reserve():
            return True
        if optional:
            return False  # Speculation never reclaims a tenant's useful backing.
        # Normal reservations may race with one another, but only one operation at a
        # time may ask an idle executor to release host backing.  The reclaim lock is
        # held across the RPC; the short reservation lock is not.
        with self._host_reclaim_lock:
            if reserve():
                return True
            for tenant in sorted(tenants, key=lambda tenant: self._reuse(tenant, None)):
                if tenant.slot == requester or tenant.state() != "idle":
                    continue
                reclaim = getattr(tenant, "reclaim_host", None)
                if reclaim is None:
                    continue
                remaining = math.inf
                while tenant.state() == "idle":
                    result = reclaim(nbytes)
                    if reserve():
                        return True
                    if result is None:
                        break
                    released, held = result
                    # Logical progress permits another bounded step even when usage is
                    # still above high. RSS alone cannot prove more cache was dropped.
                    if released <= 0 or held <= 0 or held >= remaining:
                        break
                    remaining = held
        return False

    def relieve_host_pressure(self, tenants: Sequence[HostTenant]) -> dict[str, int | str] | None:
        """Release idle managed host backing while an exact cgroup debt remains.

        This maintenance pass is intentionally independent of ordinary admission.  It
        never touches active tenants, and it stops when the pressure fact, executor
        generation, or managed-held measurement stops advancing.
        """
        if self.host_pressure is None or not self._host_reclaim_lock.acquire(blocking=False):
            return None
        try:
            before = self.host_pressure()
            debt = before.debt_bytes if before is not None else -1
            if debt <= 0:
                return None
            initial_debt = debt
            initial_scope = before.scope if before is not None else ""
            reclaimed = attempted = 0
            after = before
            generation_used: object | None = None
            for tenant in sorted(tenants, key=lambda item: self._reuse(item, None)):
                if tenant.state() != "idle":
                    continue
                reclaim = getattr(tenant, "reclaim_host", None)
                if reclaim is None:
                    continue
                generation = getattr(tenant, "reclaim_generation", None)
                if generation is None:
                    continue
                held_before: int | None = None
                while debt > 0 and tenant.state() == "idle":
                    if getattr(tenant, "reclaim_generation", None) != generation:
                        break
                    attempted += 1
                    result = reclaim(debt)
                    after = self.host_pressure()
                    debt = after.debt_bytes if after is not None else -1
                    if result is None:
                        break
                    released, held = result
                    if type(released) is not int or type(held) is not int:
                        break
                    if (
                        released <= 0
                        or held < 0
                        or (held_before is not None and held >= held_before)
                    ):
                        break
                    reclaimed += released
                    held_before = held
                    generation_used = generation
                    if held == 0 or debt <= 0:
                        break
                if debt <= 0:
                    break
            if after is None or debt < 0:
                return {
                    "initial_debt_bytes": initial_debt,
                    "reclaimed_managed_bytes": reclaimed,
                    "attempts": attempted,
                    "debt_bytes": -1,
                    "pressure_scope": "",
                    "initial_pressure_scope": initial_scope,
                    "generation": str(generation_used) if generation_used is not None else "",
                }
            return {
                "initial_debt_bytes": initial_debt,
                "reclaimed_managed_bytes": reclaimed,
                "attempts": attempted,
                "debt_bytes": debt,
                "pressure_scope": after.scope,
                "initial_pressure_scope": initial_scope,
                "generation": str(generation_used) if generation_used is not None else "",
            }
        finally:
            self._host_reclaim_lock.release()

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

    def grants(
        self,
        me: Tenant,
        tenants: Sequence[Tenant],
        lane_id: str,
        entries: tuple[str, ...],
        *,
        refuse: Callable[[str, str], Exception],
    ) -> tuple[int, ...]:
        """Authorize load/start ranks, reclaiming only against their measured minimum."""
        self.room_vector(me, {o: (0, me.overhead.get(o, 0), True) for o in me.ordinals}, tenants)
        quotes = self.quote(me, tenants)
        granted = [quotes[ordinal] for ordinal in me.ordinals]
        if any(grant is None for grant in granted):
            raise refuse(
                "device_capacity_unreadable",
                f"{lane_id}: device capacity or a live tenant's rank footprint is unknown "
                f"for {entries!r}",
            )
        negative = [
            (ordinal, grant)
            for ordinal, grant in zip(me.ordinals, granted, strict=True)
            if grant is not None and grant < 0
        ]
        if negative:
            raise refuse(
                SHORTFALL, f"{lane_id}: other tenants exceed the device budget: {negative!r}"
            )
        return tuple(grant for grant in granted if grant is not None)

    @staticmethod
    def executor_placement(capabilities: Iterable[str]) -> str:
        """Only an executor advertising the planner may stage within an operation."""
        return "component_staged" if "plan" in capabilities else "all_resident"

    def attempt_grants(
        self,
        me: Tenant,
        tenants: Sequence[Tenant],
        peak: int | Mapping[int, int] | None,
        authorize: Callable[[], tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Apply admission policy before a fresh quote; transport/ledger updates stay outside."""
        self.admit(me, tenants, peak)
        return authorize()

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
            if measured is None:
                result[ordinal] = None
                continue
            free, total = measured
            if unknown:
                # PID namespaces can hide a legacy neighbour's driver identity. Charge
                # all observed occupancy outside our reported footprint once, including
                # foreign use, rather than guessing a per-process value or refusing the
                # lane. This is a conservative quota, not attribution to any tenant.
                mine = rows[me.slot][1].get(ordinal, 0)
                room = min(total, free + mine)
                if ordinal in self.budget or -1 in self.budget:
                    occupied_elsewhere = max(total - free - mine, 0)
                    room = min(
                        room,
                        self.budget.get(ordinal, self.budget.get(-1, total)) - occupied_elsewhere,
                    )
                result[ordinal] = room
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
        """Release idle weight caches; process retirement requires a current minimum.

        Historical peaks are hints, never authority to end an idle process. The executor
        measures its own context and operation minimum under the returned grant.
        """
        for tenant in tenants:
            if (
                tenant.slot != me.slot
                and tenant.state() == "dead"
                and set(tenant.ordinals) & set(me.ordinals)
            ):
                tenant.end(f"reclaimed before {me.slot!r} runs")
        if peak is None:
            # With no activation measurement, idle weights are cache, not a claim against
            # first-use work. Contexts survive unless the current measured minimum needs them.
            for ordinal in me.ordinals:
                self.room(me, ordinal, None, None, tenants, measured=False)
        # The current context is a minimum, unlike the historical high-water mark.
        self.room_vector(me, {o: (0, me.overhead.get(o, 0), True) for o in me.ordinals}, tenants)
        peaks = dict(peak) if isinstance(peak, Mapping) else dict.fromkeys(me.ordinals, peak)
        return self.room_vector(
            me,
            {ordinal: (0, value, False) for ordinal, value in peaks.items() if value is not None},
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

        `free_wanted=None` releases idle weight caches at unseen admission. Recheck actual
        facts after each action; each tenant/action is attempted at most once.
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
                victim = (
                    min(pool, key=lambda t: (t.last_used, t.slot))
                    if act == "process"
                    else min(pool, key=lambda t: self._reuse(t, ordinal))
                )
                return victim, act
        return None

    def _reuse(self, tenant: HostTenant, ordinal: int | None) -> tuple[float, float, int, str]:
        """Farthest next use first; among ties, discard the cheapest measured reload.

        Missing demand means no predicted reuse. Missing cost is not a free reload.
        Last use breaks prediction ties, so an empty queue retains a deterministic policy.
        """
        demands = [
            d
            for d in self.queue
            if d.tenant == tenant.slot and (ordinal is None or ordinal in d.devices)
        ]
        next_use = min((d.next_use for d in demands), default=math.inf)
        costs = [d.load_seconds for d in demands if d.next_use == next_use]
        cost = max((math.inf if c is None else c for c in costs), default=math.inf)
        return -next_use, cost, tenant.last_used, tenant.slot
