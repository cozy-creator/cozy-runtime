"""Executor weights on the TensorFS weight plane (weight-plane.md §3-4).

Construction runs on meta and torch never allocates a weight. The plane holds every weight
byte, one weight set per component, in regions: `common` and each innermost no-split block
(`paging.partition`). A scope is a stage: at `admit` the policy (`weight_policy.stage`) picks
each component's resident set, window and ring inside the plane budget; the resident set is
wanted and a cursor streams the rest through its byte ring in the blocks' execution order.
Each block runs inside exactly one acquire/release; a resident region binds at its stable
address, a streamed one at its place in the ring, from cached views. A torch OOM at a block
boundary lowers the budget (unmap only) and runs the block again; capacity never poisons the
executor.
"""

from __future__ import annotations

import mmap
import os
import re
import time
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import msgspec
import tensorfs

from cozy_runtime.author._activity import aside
from cozy_runtime.author._errors import RuntimeFailure, is_device_oom, is_library_refusal
from cozy_runtime.author._loader import Census, TensorLike, TensorSpec, _members
from cozy_runtime.author._stage_memo import inexact
from cozy_runtime.internal import (
    accel,
    budget_cell,
    canonical,
    image_vae,
    plane,
    proctree,
    weight_policy,
)
from cozy_runtime.internal.derive import placing
from cozy_runtime.internal.encoding import (
    TORCH_DTYPES,
    Capabilities,
    DeviceFacts,
    LeafProvider,
    Selection,
    device_line,
    implementation_digest,
    launch_providers,
    measure_device,
    measure_runtime,
)
from cozy_runtime.internal.executor_replies import PlaneFacts, Streamed
from cozy_runtime.internal.fill import (
    Checkpoint,
    PlanRow,
    ReadLeases,
    _refuse,
    dtype_name,
    servability,
    store_refusals,
    tensorfs_requirement_dtype,
)
from cozy_runtime.internal.paging import partition
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.probe import QualificationResult, observed_capability_records, qualify
from cozy_runtime.internal.resolution import ResolvedModelPlan, TensorResolution, script_plan

if TYPE_CHECKING:
    from tensorfs.plane import Cursor, Lease, Source, Ticket, WeightSet
    from torch import Tensor
    from torch import device as Device
    from torch import dtype as Dtype
    from torch.nn import Module

#: a module's parameter or buffer table (`Module._parameters`, `Module._buffers`)
type Table = dict[str, Tensor | None]
#: where a stand-in's bytes bind: its plane, weight set, region and the table entry it holds
type Origin = tuple[Weights, Component, Region, Table, str]

#: The largest region at the sub-block grain (proving-cpu.md C2: SDXL runs at 1 GiB).
REGION_LIMIT = 16 << 20
#: The smallest image VAE decode tile side, in pixels.
MIN_TILE = 128
_REQUESTED = re.compile(r"Tried to allocate ([0-9.]+) (B|KiB|MiB|GiB)")
_UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
#: The TensorFS planning window a weight set's read plan is cut into.
READ_WINDOW = 4 << 20


class ResidencyRefusal(RuntimeFailure):
    """A stage cannot run in what the device has: capacity, never broken state. Carries the
    structured shortfall with both numbers."""

    def __init__(self, code: str, detail: str, shortfall: dict[str, Any] | None = None) -> None:
        super().__init__(detail, code=code)
        self.detail = detail
        self.shortfall = shortfall


# ---------------------------------------------------------------------------- the model


@dataclass(slots=True)
class Weight:
    """One checkpoint tensor: its stored parts in its region and where it binds."""

    row: PlanRow
    route: str
    provider: Any
    #: role -> (byte offset from the region's first byte, TensorFS dtype, shape)
    parts: dict[str, tuple[int, str, tuple[int, ...]]]
    #: (tensor table, name, parameter?) for every module attribute that names this tensor
    sites: list[tuple[Table, str, bool]] = field(default_factory=list)
    #: an `encoded_gemm` weight's installed leaf and the module it replaced (on meta)
    leaf: Any = None
    replaced: Any = None
    #: leaf roles the provider transforms from the stored bytes (recomputed at bind)
    derived_roles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Views:
    """A region's typed views at one address: the writes that bind them (destination, the
    view, and the object written: a Parameter over it, or the view itself), and each
    weight's stored parts by role (what decode and derived leaf roles read)."""

    writes: list[tuple[Table, str, Tensor, Tensor]]
    parts: list[dict[str, Tensor]]


@dataclass(slots=True)
class Region:
    index: int
    path: str
    module: Module
    weights: list[Weight]
    #: byte offset of the region in the weight set's layout, and its 2 MiB-aligned span
    offset: int = 0
    span: int = 0
    #: device address (its home, or a slot of the current cursor) -> typed views there
    views: dict[int, Views] = field(default_factory=dict)
    #: meta placeholders: the writes that unbind it
    unbound: list[tuple[Table, str, Tensor]] = field(default_factory=list)
    #: the address the destinations point at now; None is meta
    bound: int | None = None
    #: decoded tensors and derived leaf roles kept alive while bound
    derived: list[Tensor] = field(default_factory=list)
    #: whether its forward writes into its inputs in place, learned on its first run: such a
    #: block can never run again, so its ops recover from out-of-memory one by one
    mutates: bool | None = None

    @property
    def transforms(self) -> bool:
        return any(w.route == "decoded_float" or w.derived_roles for w in self.weights)

    @property
    def decoded(self) -> int:
        """Device bytes its decoded weights take beside the stored ones while it is bound (a
        float copy of each: fp8 on a card without it)."""
        return sum(
            _nbytes(w.row.shape, w.row.dtype) for w in self.weights if w.route == "decoded_float"
        )


@dataclass(slots=True)
class Component:
    """One weight set: a constructed component's regions in the plane."""

    name: str
    #: the weight set's name in the plane, unique in this process
    key: str
    root: Module
    ws: WeightSet
    device: int
    #: every region, by plane index
    regions: list[Region]
    #: the region outside every block, if any, and the blocks in address order
    common: Region | None
    blocks: list[Region]
    layout: weight_policy.Layout
    #: regions of at most `REGION_LIMIT` bytes: the grain a budget below the block floor needs
    fine: bool = False
    #: block region indexes in execution order, once a pass has been observed
    order: list[int] = field(default_factory=list)
    observed: bool = False
    seen: list[int] = field(default_factory=list)
    #: block passes in the running stage, and in its last one: >1 is a cyclic (denoise) stage
    passes: int = 0
    last_passes: int = 0
    residence: weight_policy.Residence | None = None
    cursor: Cursor | None = None
    #: the common region's lease for the running stage
    lease: Lease | None = None
    #: region indexes wanted resident for the running stage
    resident: tuple[int, ...] = ()
    #: block region index -> its position in `blocks` (the policy's block number)
    numbers: dict[int, int] = field(default_factory=dict)
    #: a CPU executor's weight set, mapped from its pinned tier (the bytes it binds)
    mapping: Any = None
    #: the pinned tier's fill in flight (a weight set closes only after it)
    filling: Ticket | None = None

    def __post_init__(self) -> None:
        self.numbers = {region.index: i for i, region in enumerate(self.blocks)}

    @property
    def planned(self) -> weight_policy.Layout:
        """Its layout as a stage's budget counts it: each region with the float copies its
        decoded weights take while bound, which are torch's bytes, not the plane's."""
        decoded = sum(region.decoded for region in self.regions)
        if not decoded:
            return self.layout
        layout = self.layout
        grow = 1 + decoded / max(layout.total, 1)
        return replace(
            layout,
            common=layout.common + (self.common.decoded if self.common is not None else 0),
            blocks=tuple(region.span + region.decoded for region in self.blocks),
            fine=None
            if layout.fine is None
            else (int(layout.fine[0] * grow), int(layout.fine[1] * grow)),
        )


# ------------------------------------------------------------------------ the process


class Weights:
    """This executor process's plane, budgets and every registered weight set (one per rank).

    Every component of every construction shares one plane. Least recently used across them
    is the plane's priority order, so a construction switch moves nothing until a stage wants
    the room, and an evicted component's bytes stay in the pinned tier.
    """

    def __init__(self, torch: ModuleType, device: Device, kind: str) -> None:
        self.torch = torch
        self.device = device
        self.kind = kind
        self.index = int(device.index or 0)
        #: no device in the plane: a CPU executor, whose pinned tier IS its device memory
        self.host_only = device.type != "cuda"
        self.plane = plane.open_plane(None if self.host_only else self.index)
        #: the Worker's budgets; -1: none given, each stage derives one from the driver
        self.budget = -1
        self.pinned = -1
        #: an older Worker's load ceiling, bounding driver-derived budgets; -1: none
        self.ceiling = -1
        #: what the plane holds now (budgets start at 0 in the plane)
        self.applied = 0
        self.pinned_applied = 0
        self.clock = 0
        self.components: dict[str, Component] = {}
        self.activation: dict[str, int] = {}
        #: methods whose stage fits itself to the room it finds (an image VAE's decode mode):
        #: their activations never size a grant
        self.adaptive: set[str] = set()
        self.oom_retries = 0
        self.modes: dict[str, str] = {}
        #: the stages open now, outermost first. A stage another model slot enters inside one
        #: (H3's turbo LoRA inside its DiT's) is placed with it, inside the one budget.
        self.open: list[tuple[WeightResidency, str, tuple[str, ...]]] = []
        self.residencies: weakref.WeakSet[WeightResidency] = weakref.WeakSet()
        self._recovery: type | None = None
        #: one GPU of a group: a block's forward holds collectives the other GPUs are already
        #: in, so it can never run again; its ops recover one by one instead
        self.grouped = False
        #: the GPUs of its group: each pins its own weights, so each takes this share of a
        #: host budget the Worker states for the whole group
        self.world = 1
        #: blocks in their forward now (their leases pin their own weights)
        self.running = 0
        #: a read outside a module's forward is being mapped (`reading`)
        self.mapping = False
        #: a group's stage whose activations are not measured yet runs its first pass at the
        #: floor, and is placed again at its second with what that pass measured
        self.growing = False
        #: the Worker's mid-call budget (`budget_cell`): read at every block boundary
        self.cell: budget_cell.Cell | None = None
        #: asks the Worker for `bytes` free on this GPU from its other tenants (`DeviceRoom`);
        #: None in a process without a Worker
        self.room: Callable[[int], None] | None = None
        #: the CUDA context and library workspaces outside torch and the plane, measured once
        self.context_bytes = -1

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def stream(self) -> int:
        return int(self.torch.cuda.current_stream(self.device).cuda_stream)

    # -- budgets

    def set_budget(self, vram: int, pinned: int = -1) -> int:
        """The Worker's budgets. Lowering unmaps at once; returns the device bytes freed. A
        negative `vram` is no grant: each stage derives its own from the driver."""
        if pinned >= 0:
            self.pinned = pinned
            self._pin(pinned)
        if vram < 0:
            self.budget = -1
            return 0
        self.budget = vram
        freed = self._apply(vram)
        if vram == 0:
            accel.release_cached(self.torch, self.kind)
        return freed

    def sizing(self) -> int:
        """The largest measured activations a grant must leave room for (adaptive stages
        make their own room)."""
        return max(
            (v for m, v in self.activation.items() if m not in self.adaptive),
            default=max(self.activation.values(), default=0),
        )

    def _apply(self, nbytes: int) -> int:
        nbytes = max(nbytes, 0)
        if nbytes == self.applied or self.host_only:
            return 0
        if nbytes < self.applied:
            self.quiesce()
        trim = self.plane.set_vram_budget(self.index, nbytes)
        self.applied = nbytes
        self._evicted(trim.get("evicted", ()))
        return int(trim.get("freed", 0))

    def _pin(self, nbytes: int) -> None:
        if nbytes != self.pinned_applied:
            self.plane.set_pinned_budget(nbytes)
            self.pinned_applied = nbytes

    def quiesce(self) -> None:
        """Wait for every kernel and copy this process queued. Only leases tell the plane
        what a kernel reads, and bound weights are read past them (a stage's resident set,
        code outside block forwards): unmapping under a queued kernel faults the device's
        MMU, which on a GPU that drives a display can hang the whole machine."""
        if not self.host_only:
            self.torch.cuda.synchronize(self.device)

    def vacate(self) -> int:
        """Unmap every unleased weight (an older Worker's `Vacate`) without fixing a budget:
        the next stage derives its own and wants its weights back from the pinned tier."""
        freed = self._apply(0)
        accel.release_cached(self.torch, self.kind)
        return freed

    def stage_budget(self, method: str) -> int:
        """The Worker's budget, else what the driver says (`driver_budget`) up to an older
        Worker's ceiling. The Worker's grant leaves room for the attempt's largest
        activations; a stage whose own measured ones are smaller takes the difference (the
        same device bytes, re-split per stage)."""
        if self.budget >= 0:
            own = self.activation.get(method)
            if own is None:
                return self.budget
            return self.budget + max(self.sizing() - own, 0)
        driver = self.driver_budget(method)
        return min(driver, self.ceiling) if self.ceiling >= 0 else driver

    def driver_budget(self, method: str) -> int:
        """What the driver has free beside this process's weights, less the activation growth
        this method measured (activations first)."""
        memory = accel.allocation(self.torch, self.kind)
        committed = max(plane.stats(self.plane).device(self.index).committed, 0)
        # torch reuses its own cached segments first; only growth beyond them needs the driver
        slack = max(memory["reserved_bytes"] - memory["allocated_bytes"], 0)
        growth = max(self.activation.get(method, 0) - slack, 0)
        # the bound weights' decoded copies are the weights' own, in torch's bytes
        held = committed + self.decoded()
        return weight_policy.plane_budget(memory["driver_free_bytes"] + held, growth)

    def decoded(self) -> int:
        """Device bytes the bound regions' decoded copies hold now."""
        return sum(
            int(t.numel() * t.element_size())
            for component in self.components.values()
            for region in component.regions
            for t in region.derived
        )

    def shrink(self, need: int, floor: int) -> int:
        """OOM at a block boundary: give torch `need` more bytes by unmapping plane bytes. The
        cut is from what is mapped, which a budget larger than the model never reaches, and
        it holds until the Worker's next grant."""
        mapped = max(plane.stats(self.plane).device(self.index).committed, 0)
        target = max(min(self.applied, mapped) - need - weight_policy.MARGIN, floor)
        if target >= self.applied:
            return 0
        self.oom_retries += 1
        if self.budget >= 0:
            self.budget = min(self.budget, target)
        return self._apply(target)

    def opened(self) -> dict[str, tuple[WeightResidency, str, Component]]:
        """Every open stage's components by weight set: their slot's residency and method."""
        return {
            row.components[name].key: (row, method, row.components[name])
            for row, method, local in self.open
            for name in local
        }

    def floor(self) -> int:
        """The plane bytes the open stages cannot run below, each component at its grain
        (mid-stage none can refine: its leases hold the coarse regions)."""
        return sum(
            weight_policy.floor(replace(component.planned, fine=None))
            for _, _, component in self.opened().values()
        )

    @contextmanager
    def recovering(self, watch: frozenset[int] = frozenset()) -> Iterator[Watch]:
        """Run the caller's torch ops with out-of-memory recovery: an op that cannot allocate
        unmaps weights and runs again. With `watch` (storage addresses), the yielded record
        notes whether any op wrote into one of them."""
        seen = Watch(watch)
        if self.host_only:
            yield seen
            return
        if self._recovery is None:
            self._recovery = _recovery_type(self.torch)
        with self._recovery(self, seen):
            yield seen

    def make_room(self, need: int, operands: Sequence[object], op: str = "") -> None:
        """An op outside every block ran out of device memory: unmap `need` bytes of weights,
        idle ones first. An operand that is itself a weight must still be mapped after."""
        read = [
            (region, region.bound)
            for pointer in _pointers(operands)
            for component in self.components.values()
            for region in component.regions
            if region.bound is not None and region.bound <= pointer < region.bound + region.span
        ]
        accel.release_cached(self.torch, self.kind)
        before = self.allocatable()
        if self.running:
            # Inside a block (one GPU of a group): nothing is re-placed under its lease. Other
            # weights leave, least recently used first; their blocks map them again as they run.
            self.shrink(need, self.floor())
            self.drain()
        elif self.open:
            self.open[-1][0]._make_room(need, op=op)
        else:
            self.shrink(need, 0)
        if any(region.bound != address for region, address in read):
            raise ResidencyRefusal(
                "device_shortfall",
                f"an allocation of {need} B needed the room of the weights the same op reads",
            )
        self.progressed(before, need, op)

    def allocatable(self) -> int:
        """Device bytes torch can allocate now: the driver's free plus its own idle cache."""
        memory = accel.allocation(self.torch, self.kind)
        return memory["driver_free_bytes"] + max(
            memory["reserved_bytes"] - memory["allocated_bytes"], 0
        )

    def progressed(self, before: int, need: int, op: str = "") -> None:
        """An out-of-memory retry runs again only when memory came back: first this process's
        own unmapping, else the Worker's (its other tenants on this GPU give theirs). The same
        failure retried at the same free bytes would churn maps on a full card."""
        after = self.allocatable()
        if after <= before and self.room is not None:
            self.room(need + weight_policy.MARGIN)
            after = self.allocatable()
        if after <= before:
            raise ResidencyRefusal(
                "device_shortfall",
                f"{op or 'an allocation'} of {need} B: nothing on this GPU could give device "
                f"memory back ({after} B allocatable before and after)",
            )

    @contextmanager
    def reading(self, standins: Sequence[Tensor], op: str) -> Iterator[Read]:
        """A weight read outside its own module's forward (a fused kernel handed a sibling's
        scale, a package reading `.weight`): its region is mapped and leased for the op, as
        its forward would, and leaves with it. Only inside its component's stage: anywhere
        else the read is refused, and no kernel is ever handed its empty pointer."""
        opened = self.opened()
        held: list[tuple[Component, Region, Lease]] = []
        if self.mapping:  # a read from inside the mapping of another: never a second one
            raise _unbound_read(op)
        self.mapping = True
        try:
            for standin in standins:
                origin: Origin | None = getattr(standin, "origin", None)
                if origin is None or origin[0] is not self:
                    raise _unbound_read(op)
                _, component, region, _, _ = origin
                row = opened.get(component.key)
                if row is None or row[2] is not component:
                    raise _unbound_read(op)
                if region.bound is None and all(region is not other for _, other, _ in held):
                    held.append((component, region, row[0]._lease(component, region, op)))
            for _, region, lease in held:  # a re-plan for a later lease unbound earlier ones
                bind(self.torch, region, lease)
            real: dict[int, Tensor] = {}
            for standin in standins:
                _, _, _, table, name = getattr(standin, "origin")  # noqa: B009
                value = table[name]
                if value is None or type(value) is type(standin):
                    raise _unbound_read(op)
                real[id(standin)] = value.to(dtype=standin.dtype, device=standin.device)
            self.mapping = False
            yield Read(
                real, tuple((r.bound, r.bound + r.span) for _, r, _ in held if r.bound is not None)
            )
        finally:
            self.mapping = False
            for component, region, lease in reversed(held):
                if region.index not in component.resident:
                    unbind(region)
                lease.release()

    def drain(self) -> None:
        """Read the plane's events: a region evicted from this device goes back to meta before
        any kernel could read unmapped pages (a pinned-tier eviction leaves it bound)."""
        self._evicted(
            (event.ws, event.region)
            for event in plane.events(self.plane)
            if event.kind == "evicted"
            and event.device == self.index
            and event.ws is not None
            and event.region is not None
        )

    def _evicted(self, pairs: Iterable[tuple[str, int]]) -> None:
        for name, index in pairs:
            component = self.components.get(str(name))
            if component is not None and 0 <= int(index) < len(component.regions):
                unbind(component.regions[int(index)])

    def measure_context(self) -> None:
        """What the driver charges this process beyond torch's segments and the plane's maps:
        the CUDA context and library workspaces the Worker must leave room for."""
        row = accel.process_memory(os.getpid(), self.kind)
        if row.state == "present":
            reserved = accel.allocation(self.torch, self.kind)["reserved_bytes"]
            mapped = max(plane.stats(self.plane).device(self.index).committed, 0)
            self.context_bytes = max(row.bytes - reserved - mapped, self.context_bytes, 0)

    def resident_bytes(self, component: Component) -> int:
        return plane.stats(self.plane).ready(component.key, self.index)

    # -- the pinned tier

    def host_share(self) -> int:
        """This process's pinned tier at most: its weights, inside the machine's pinned total
        (`weight_policy.pinned_total`) less what others pin already. -1: the host is
        unreadable."""
        total = sum(c.ws.nbytes for c in self.components.values())
        try:
            memory = proctree.host_memory()
        except proctree.ProcessTreeUnsupported:
            return -1
        if memory.available < 0:
            return -1
        others = max(memory.shmem - max(plane.stats(self.plane).host.used, 0), 0)
        allowed = weight_policy.pinned_total(memory.available, memory.shmem)
        return min(total, max(allowed - others, 0) // self.world)

    def hold_host(self) -> None:
        """Keep the pinned tier inside what the host has now, whoever set its budget: the
        limit may have moved and other memory grown since. Read at every stage and every
        pass; only ever lowers (the plane punches its least recently used regions, which
        then come from the page cache or disk)."""
        if self.host_only:
            return  # a CPU executor computes on its tier
        share = self.host_share()
        if 0 <= share < self.pinned_applied:
            self._pin(share)

    def register(self, component: Component) -> None:
        """Track a new weight set and start filling the pinned tier with it. Without a Worker
        budget the tier holds what this process registered, inside its share of the host; it
        is optional and shrinkable (page cache and disk remain beneath it)."""
        self.components[component.key] = component
        if self.host_only:
            self._bind_host(component)
            return
        if self.pinned < 0:
            self._pin(max(self.host_share(), 0))
        self._fill(component)

    def _bind_host(self, component: Component) -> None:
        """A CPU executor computes on its pinned tier: every region pinned there (never evicted
        implicitly) and bound in place, the pages shared with every executor of the layout."""
        if self.pinned < 0:
            self._pin(sum(c.ws.nbytes for c in self.components.values()))
        try:
            ticket = self.plane.want(
                component.ws, plane.PINNED, None, priority=self.tick(), pin=True
            )
        except (plane.error("Shortfall"), plane.error("BelowFloor")) as exc:
            raise ResidencyRefusal(
                "device_shortfall", f"{component.key} does not fit host memory: {exc}"
            ) from None
        ticket.wait()
        fd = os.dup(component.ws.host_fd)
        try:
            # Private: the pages are the tier's until written, and nothing here writes them.
            mapped = mmap.mmap(
                fd, component.ws.nbytes, mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE
            )
        finally:
            os.close(fd)
        component.mapping = self.torch.frombuffer(mapped, dtype=self.torch.uint8)
        for region in component.regions:
            bind(self.torch, region, _HostSpan(component.mapping, region))

    def prefetch(self, construction: str) -> int:
        """Fill the pinned tier with `construction`'s regions, now; returns the bytes asked."""
        return sum(
            self._fill(component)
            for component in self.components.values()
            if component.key.startswith(construction + "/")
        )

    def _fill(self, component: Component) -> int:
        """Want the component pinned, as many regions as the tier holds: the tier is optional,
        so one larger than it pins a prefix and the rest reads from disk."""
        room, regions = self.pinned_applied, []
        for region in component.regions:
            if region.span > room:
                break
            room -= region.span
            regions.append(region.index)
        if not regions:
            return 0
        try:
            component.filling = self.plane.want(
                component.ws, plane.PINNED, regions, priority=self.tick()
            )
        except (plane.error("Shortfall"), plane.error("BelowFloor")):
            return 0
        return sum(component.regions[i].span for i in regions)

    def forget(self, construction: str) -> None:
        """Close `construction`'s weight sets and give their device handles back to the
        driver (the plane keeps released handles for reuse until a budget is set)."""
        for key in [key for key in self.components if key.startswith(construction + "/")]:
            self.close(self.components.pop(key))
        if not self.host_only:
            self.plane.set_vram_budget(self.index, self.applied)

    def close(self, component: Component) -> None:
        self.quiesce()
        close_cursor(component)
        if component.lease is not None:
            component.lease.release()
            component.lease = None
        for region in component.regions:
            unbind(region)
            region.views.clear()
        component.mapping = None
        if component.filling is not None:
            component.filling.wait()  # a filling tier refuses to close
            component.filling = None
        try:
            component.ws.close()
        except plane.error("LeaseViolation"):
            # device copies still read its pinned tier (the plane's own stream): a rare
            # close (refine, unload) waits for the device, then closes
            self.torch.cuda.synchronize(self.device)
            self.plane.set_vram_budget(self.index, self.applied)
            component.ws.close()

    # -- facts

    def facts(self) -> PlaneFacts:
        held = plane.stats(self.plane)
        device = held.device(self.index)
        resident: dict[str, int] = {}
        for component in self.components.values():
            resident[component.name] = resident.get(component.name, 0) + held.ready(
                component.key, self.index
            )
        stages = [row for row in held.stages if row.device == self.index]
        copied = device.host_copy_bytes + device.disk_copy_bytes
        copy_ns = device.host_copy_ns + device.disk_copy_ns
        return PlaneFacts(
            budget_bytes=device.budget,
            committed_bytes=device.committed,
            leased_bytes=device.leased_bytes + device.held_bytes,
            pinned_budget_bytes=held.host.budget,
            pinned_bytes=held.host.used,
            context_bytes=self.context_bytes,
            resident=resident,
            streamed={
                component.name: Streamed(
                    blocks=len(component.layout.blocks),
                    resident_blocks=len(component.residence.resident),
                    window=component.residence.window,
                )
                for component in self.components.values()
                if component.residence is not None
            },
            h2d_bytes=copied,
            h2d_gbps=round(copied / copy_ns, 3) if copy_ns else 0.0,
            disk_copy_bytes=device.disk_copy_bytes,
            mapped_copy_bytes=device.mapped_copy_bytes,
            disk_read_bytes=held.host.disk_read_bytes,
            late=sum(row.late for row in stages),
            stall_ns=sum(row.stall_ns for row in stages),
            misses=device.misses,
            evictions=device.evictions,
            oom_retries=self.oom_retries,
            modes=dict(self.modes),
        )


# ---------------------------------------------------------------------------- binding


@dataclass(frozen=True, slots=True)
class _HostSpan:
    """A region of a CPU executor's mapped tier, in the shape `bind` takes a lease in."""

    base: Tensor
    region: Region
    ring_offset: None = None

    @property
    def ptr(self) -> int:
        return int(self.base.data_ptr()) + self.region.offset

    def view(self) -> Tensor:
        return self.base[self.region.offset : self.region.offset + self.region.span]


def bind(torch: ModuleType, region: Region, lease: Lease | _HostSpan) -> None:
    """Point the region's destinations at the lease's bytes: its home while resident, its place
    in the stage's ring while streamed. Views are cached by address (ring placements repeat
    with the window); decoded tensors and derived leaf roles are recomputed whenever the bytes
    moved or were streamed in again."""
    address = int(lease.ptr)
    views = region.views.get(address)
    if views is None:
        if len(region.views) > weight_policy.MAX_WINDOW:
            del region.views[next(a for a in region.views if a != region.bound)]
        views = region.views[address] = _views(torch, region, torch.from_dlpack(lease.view()))
    moved = region.bound != address
    if moved:
        for i, (table, name, tensor, value) in enumerate(views.writes):
            want = getattr(table.get(name), "dtype", tensor.dtype)
            if want != tensor.dtype:
                # The component was converted (a package's `module.to(dtype)`): honour it.
                converted = tensor.to(want)
                parameter = value is not tensor
                table[name] = (
                    torch.nn.Parameter(converted, requires_grad=False) if parameter else converted
                )
                continue
            if value is not tensor and (
                value.dtype != tensor.dtype or value.data_ptr() != tensor.data_ptr()
            ):
                # a conversion rewrote the cached Parameter's data: bind a fresh one
                value = torch.nn.Parameter(tensor, requires_grad=False)
                views.writes[i] = (table, name, tensor, value)
            table[name] = value
    if region.transforms and (moved or lease.ring_offset is not None):
        region.derived = _transform(torch, region, views.parts)
    region.bound = address


_UNBOUND: dict[int, type] = {}


def unbound(torch: ModuleType, tensor: Tensor | None, device: Device) -> Tensor:
    """The stand-in a weight holds while no plane bytes are bound to it: its shape, dtype and
    execution device, so code can ask where a component lives (a package placing its inputs
    beside an encoder's embedding), while any computation on it raises."""
    assert tensor is not None  # a registered weight's destination
    kind = _UNBOUND.get(id(torch))
    if kind is None:
        kind = _UNBOUND[id(torch)] = _unbound_type(torch)
    held: Tensor = kind(tuple(tensor.shape), tensor.dtype, device)
    if isinstance(tensor, torch.nn.Parameter):
        held = torch.nn.Parameter(held, requires_grad=False)
    return held


def _retyped(standin: Tensor, dtype: Dtype) -> Tensor:
    held = standin.to(dtype)
    kind: Any = type(standin)
    return kind.parameter(held) if getattr(standin, "_is_param", False) else held


def _unbound_type(torch: ModuleType) -> type:
    aten = torch.ops.aten

    base: type = torch.Tensor

    class Unbound(base):  # type: ignore[misc]
        #: where its bytes bind
        origin: Origin | None = None

        @staticmethod
        def parameter(held: Tensor) -> Tensor:
            out: Tensor = torch.nn.Parameter(held, requires_grad=False)
            return out

        @staticmethod
        def __new__(cls, shape: tuple[int, ...], dtype: Dtype, device: Device) -> Any:
            # A zero-byte storage, as packages know a parked weight (H3's numerical checks
            # skip one): its shape and device are facts, its bytes are not here.
            return torch.Tensor._make_wrapper_subclass(
                cls, shape, dtype=dtype, device=device, storage_size=0
            )

        @classmethod
        @contextmanager
        def reading(cls, standins: Sequence[Tensor], op: str) -> Iterator[Read]:
            """Real bytes behind `standins` for one op (`Weights.reading`), or a refusal."""
            origins = (getattr(standin, "origin", None) for standin in standins)
            origin: Origin | None = next((found for found in origins if found is not None), None)
            if origin is None:
                raise _unbound_read(op)
            with origin[0].reading(standins, op) as read:
                yield read

        @classmethod
        def __torch_dispatch__(
            cls, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
        ) -> Any:
            if func in (aten.detach.default, aten.alias.default):
                return args[0]
            kwargs = kwargs or {}
            if func is aten._to_copy.default or func.overloadpacket is aten.to:
                # A conversion (`module.to(dtype)`, a retyped stand-in) is still a stand-in:
                # no bytes move until something computes with it.
                held = args[0]
                names = [argument.name for argument in func._schema.arguments]
                given = {**dict(zip(names, args, strict=False)), **kwargs}
                other = given.get("other")
                dtype = given.get("dtype") or getattr(other, "dtype", held.dtype)
                device = given.get("device") or getattr(other, "device", held.device)
                copy = cls(tuple(held.shape), dtype, torch.device(device))
                copy.origin = held.origin
                return copy
            operands = (*args, *kwargs.values())
            standins = [v for v in _tensor_operands(operands) if isinstance(v, cls)]
            if func._schema.is_mutable and any(
                isinstance(v, cls) for v in _tensor_operands(tuple(_written(func, args, kwargs)))
            ):
                raise _unbound_read(str(func))  # a write would be lost with the mapping
            with cls.reading(standins, str(func)) as read:
                return read.kept(
                    func(
                        *(read.operand(v) for v in args),
                        **{k: read.operand(v) for k, v in kwargs.items()},
                    )
                )

    return Unbound


def _unbound_read(op: str) -> ResidencyRefusal:
    return ResidencyRefusal(
        "weight_unbound",
        f"{op}: a weight is read while no plane bytes are bound to it (outside its stage)",
    )


@dataclass(frozen=True, slots=True)
class Read:
    """Weights mapped for one op that reads them outside their own module's forward."""

    #: stand-in id -> the bound tensor it stands for
    real: dict[int, Tensor]
    #: device address ranges the op's leases hold
    spans: tuple[tuple[int, int], ...]

    def operand(self, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return type(value)(self.real.get(id(item), item) for item in value)
        return self.real.get(id(value), value)

    def kept(self, out: object) -> object:
        """The op's result, without a view into bytes that leave with the op's leases."""
        if isinstance(out, (tuple, list)):
            return type(out)(self.kept(item) for item in out)
        if not getattr(out, "is_cuda", False):
            return out
        tensor = cast("Tensor", out)
        address = int(tensor.data_ptr()) if tensor.numel() else 0
        return tensor.clone() if any(a <= address < b for a, b in self.spans) else out


def _recovery_type(torch: ModuleType) -> type:
    mode: type = torch.utils._python_dispatch.TorchDispatchMode

    class Recovery(mode):  # type: ignore[misc]
        """Every torch op outside a block: one that runs out of device memory runs again
        after weights were unmapped for it (retry the op, never the run)."""

        def __init__(self, weights: Weights, seen: Watch) -> None:
            super().__init__()
            self.weights = weights
            self.seen = seen

        def __torch_dispatch__(
            self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
        ) -> Any:
            kwargs = kwargs or {}
            seen = self.seen
            if seen.storages and not seen.mutated and func._schema.is_mutable:
                seen.mutated = any(
                    _storage(value) in seen.storages for value in _written(func, args, kwargs)
                )
            refused = False
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    need, refused = _room_for(exc, refused)
                # outside `except`, so the failed op's frames are gone
                self.weights.make_room(need, (*args, *kwargs.values()), str(func))

    return Recovery


@dataclass(slots=True)
class Watch:
    """Storages to watch for in-place writes, and whether an op wrote into one."""

    storages: frozenset[int]
    mutated: bool = False


def _written(func: Any, args: tuple[object, ...], kwargs: Mapping[str, object]) -> Iterator[object]:
    """The arguments an op writes into (`Tensor(a!)` in its schema, `out=` included)."""
    for index, argument in enumerate(func._schema.arguments):
        if argument.alias_info is not None and argument.alias_info.is_write:
            yield args[index] if index < len(args) else kwargs.get(argument.name)


def _storage(value: object) -> int:
    pointer = getattr(value, "untyped_storage", None)
    return int(pointer().data_ptr()) if pointer is not None else -1


def _tensor_operands(operands: Sequence[object]) -> Iterator[object]:
    for value in operands:
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if hasattr(item, "untyped_storage"):
                yield item


def _pointers(operands: Sequence[object]) -> Iterator[int]:
    """Device addresses of an op's tensor operands (one level deep)."""
    for value in operands:
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if getattr(item, "is_cuda", False) and type(item).__name__ != "Unbound":
                yield int(item.data_ptr())


def unbind(region: Region) -> None:
    """Put the stand-ins back, keeping a dtype the component was converted to meanwhile."""
    for table, name, standin in region.unbound:
        dtype = getattr(table.get(name), "dtype", standin.dtype)
        table[name] = standin if dtype == standin.dtype else _retyped(standin, dtype)
    region.derived = []
    region.bound = None


def close_cursor(component: Component) -> None:
    """Close the stage's cursor after dropping every view into its ring (one keeps the cursor
    open)."""
    cursor = component.cursor
    if cursor is None:
        return
    ring = range(cursor.ring_ptr, cursor.ring_ptr + cursor.ring_nbytes)
    for region in component.regions:
        if region.bound is not None and region.bound in ring:
            unbind(region)
        for address in [a for a in region.views if a in ring]:
            del region.views[address]
    component.cursor = None
    try:
        cursor.close()
    except plane.error("Poisoned"):
        raise
    except plane.error("PlaneError"):
        # A view into the ring outlives the stage (a failed block's frames hold one): the
        # dropped cursor closes itself when the last of them goes.
        pass


def _views(torch: ModuleType, region: Region, base: Tensor) -> Views:
    writes: list[tuple[Table, str, Tensor, Tensor]] = []
    parts: list[dict[str, Tensor]] = []
    for weight in region.weights:
        typed = {
            role: base[offset : offset + _nbytes(shape, dtype)]
            .view(getattr(torch, TORCH_DTYPES[dtype]))
            .view(shape)
            for role, (offset, dtype, shape) in weight.parts.items()
        }
        parts.append(typed)
        if weight.route == "verbatim":
            (tensor,) = typed.values()
            for table, name, parameter in weight.sites:
                value = torch.nn.Parameter(tensor, requires_grad=False) if parameter else tensor
                writes.append((table, name, tensor, value))
        elif weight.route == "encoded_gemm":
            writes.extend(
                (weight.leaf._buffers, role, tensor, tensor)
                for role, tensor in typed.items()
                if role not in weight.derived_roles
            )
    return Views(writes, parts)


def _transform(torch: ModuleType, region: Region, parts: list[dict[str, Tensor]]) -> list[Tensor]:
    """Decode `decoded_float` weights into fresh device tensors and recompute derived leaf
    roles, on the current stream. Returns what must stay alive while the region is bound."""
    kept: list[Tensor] = []
    for weight, stored in zip(region.weights, parts, strict=True):
        if weight.route == "decoded_float":
            row = weight.row
            device = next(iter(stored.values())).device
            destination = torch.empty(
                row.shape, dtype=getattr(torch, TORCH_DTYPES[row.dtype]), device=device
            )
            weight.provider.decode(torch, stored, destination)
            for table, name, parameter in weight.sites:
                table[name] = (
                    torch.nn.Parameter(destination, requires_grad=False)
                    if parameter
                    else destination
                )
            kept.append(destination)
        elif weight.derived_roles:
            fresh = weight.provider.leaf(torch, stored, weight.replaced, weight.leaf.out_dtype)
            for role in weight.derived_roles:
                weight.leaf._buffers[role] = fresh._buffers[role]
                kept.append(fresh._buffers[role])
    return kept


def _nbytes(shape: Sequence[int], dtype: str) -> int:
    count = 1
    for extent in shape:
        count *= int(extent)
    return count * int(tensorfs.DTYPES[dtype])


# ---------------------------------------------------------------------------- the backend


class PlaneBackend:
    """The loader's `Backend` on the plane (§1.1's one fill path): `fit` judges, `materialize`
    keeps weights on meta and derived buffers real, `fill` validates each destination against
    its stored row, and `commit` installs encoded leaves and registers every component's
    regions with the plane. No byte moves here; the first stage wants them."""

    def __init__(
        self,
        checkpoints: Mapping[str, Checkpoint],
        rows: Sequence[PlanRow],
        *,
        plan: ResolvedModelPlan,
        weights: Weights,
        construction: str,
        qualified: tuple[Capabilities, QualificationResult],
        device_facts: DeviceFacts,
        custody: str = "canonical",
        tiers: HostTiers | None = None,
    ) -> None:
        self.weights = weights
        #: the worker's pinned host tiers, kept across executors: adopted when it holds one
        #: for exactly this layout, offered when this process filled a new one
        self.tiers = tiers
        self.torch = weights.torch
        self.construction = construction
        self.checkpoints = dict(checkpoints)
        self.plan = {row.key: row for row in rows}
        self.custody = custody
        self.device = weights.device
        self.kind = weights.kind
        #: Verified at `expect`, then handed to the plane: one source per manifest, shared by
        #: its components' weight sets, which re-read evicted regions from the store.
        self.leases = ReadLeases(self.checkpoints)
        self.sources: dict[str, Source] = {}
        self.state = "open"
        self.expected: dict[str, frozenset[str]] = {}
        #: component -> stored tensors the code does not build: named in its read plan,
        #: selected out of it, so never read or mapped
        self.skipped: dict[str, list[str]] = {}
        self.enqueued: dict[str, TensorLike] = {}
        self.roots: dict[str, Module] = {}
        self.components: dict[str, Component] = {}
        self.derived: dict[str, dict[tuple[Module, str], Tensor]] = {}
        self.stats: dict[str, Any] = {}
        self.device_facts = device_facts
        self.providers = launch_providers()
        self.capabilities, self.probe = qualified
        self.execution = plan
        self.resolved: dict[str, TensorResolution] = plan.by_key()
        planned, held = set(self.resolved), set(self.plan)
        if planned != held:
            raise _refuse(
                "plan_divergence",
                f"the resolved plan [{plan.digest()[:19]}…] names {len(planned)} destinations "
                f"and this checkpoint set supplies {len(held)}: missing from the plan "
                f"{sorted(held - planned)[:3]}, absent from the checkpoint "
                f"{sorted(planned - held)[:3]}",
                sorted(planned ^ held)[:6],
            )
        self.servability = servability(rows, plan)
        self.selected: dict[str, Selection] = {
            key: self._select(key, row) for key, row in self.plan.items()
        }
        self.encoded_leaves = "refuse"
        #: contract key -> (installed leaf, replaced module, derived roles)
        self.leaves: dict[str, tuple[Any, Any, tuple[str, ...]]] = {}

    @classmethod
    def for_script(
        cls,
        checkpoints: Mapping[str, Checkpoint],
        rows: Sequence[PlanRow],
        *,
        weights: Weights,
        construction: str,
        release: str,
        objective: str = "latency",
        encoded_leaves: str = "accept",
        tiers: HostTiers | None = None,
    ) -> PlaneBackend:
        """A backend for a driver script with no model class: measure, qualify, resolve a
        `script_plan` over these rows, then construct, through the executor's own authority."""
        torch = weights.torch
        facts = measure_device(torch, weights.index)
        providers = launch_providers()
        qualified = qualify(
            torch,
            providers,
            facts,
            sorted({dtype_name(row.dtype) for row in rows}),
            release=release,
        )
        plan = script_plan(
            rows=rows,
            components=sorted({row.component for row in rows}),
            release=release,
            store="script",
            snapshot=next(iter(checkpoints.values())).manifest_id,
            snapshots={name: book.manifest_id for name, book in checkpoints.items()},
            providers=providers,
            capabilities=qualified[0],
            device=facts,
            runtime=measure_runtime(torch, release),
            facts=PlanFacts(),
            objective=objective,
            encoded_leaves=encoded_leaves,
            dtype_name=dtype_name,
        )
        return cls(
            checkpoints,
            rows,
            plan=plan,
            weights=weights,
            construction=construction,
            qualified=qualified,
            device_facts=facts,
            tiers=tiers,
        )

    def _select(self, key: str, row: PlanRow) -> Selection:
        """The plan's provider for one destination, matched by digest against this build and
        this device's capability records. Anything else is `plan_divergence`, never a
        re-selection."""
        resolved = self.resolved.get(key)
        if resolved is None:
            raise _refuse("plan_divergence", f"{key}: the plan names no such destination", [key])
        actual = (row.encoded.encoding, tuple(row.shape), dtype_name(row.dtype))
        planned = (resolved.encoding, tuple(resolved.geometry), resolved.output_dtype)
        if actual != planned:
            raise _refuse(
                "plan_divergence",
                f"{key}: the plan priced encoding/geometry/dtype {planned} and the checkpoint "
                f"holds {actual}",
                [key],
            )
        provider = next(
            (
                c
                for c in self.providers.get(row.encoded.encoding, ())
                if implementation_digest(c) == resolved.implementation_digest
            ),
            None,
        )
        if provider is None or str(provider.route) != resolved.route:
            raise _refuse(
                "plan_divergence",
                f"{key}: the plan executes {resolved.implementation} "
                f"[{resolved.implementation_digest[:19]}…, {resolved.route}] and this build "
                + (
                    "carries no implementation with that digest"
                    if provider is None
                    else f"has it on route {provider.route!r}"
                ),
                [key],
            )
        records = self.capabilities.qualified(
            row.encoded.encoding,
            self.device_facts,
            resolved.output_dtype,
            {resolved.implementation_digest: provider},
        )
        if not records:
            raise _refuse(
                "plan_divergence",
                f"{key}: {device_line(self.device_facts)} holds no capability record for "
                f"{resolved.implementation}",
                [key],
            )
        return Selection(provider, records[0], row.encoded, resolved.reason)

    # -- the Backend protocol

    def expect(self, sets: Mapping[str, Sequence[str]]) -> None:
        """Completeness sets, and the verified read leases: a corrupt artifact refuses here,
        before construction."""
        self.expected = {name: frozenset(keys) for name, keys in sets.items()}
        for name in sets:
            self.leases.acquire(name)
        self.stats["leases"] = self.leases.document()

    def fit(self, walked: Census, *, encoded_leaves: str) -> Mapping[str, object] | None:
        """`tensorfs.fit` over the serving census, per checkpoint the census reads."""
        by_checkpoint: dict[str, tuple[Checkpoint, list[str]]] = {}
        for component in walked.components:
            checkpoint = self.checkpoints.get(component)
            if checkpoint is None:
                raise _refuse(
                    "destination_absent",
                    f"the construction built component {component!r} and the binding names "
                    f"no checkpoint for it (bound: {', '.join(sorted(self.checkpoints))})",
                    [component],
                )
            by_checkpoint.setdefault(checkpoint.manifest_id, (checkpoint, []))[1].append(component)
        observations = canonical.write(observed_capability_records(self.probe))
        verdict: dict[str, object] | None = None
        ignored: list[str] = []
        warnings: list[str] = []
        routes = {"verbatim": 0, "decoded_float": 0, "encoded_gemm": 0}
        for checkpoint, constructed in by_checkpoint.values():
            members = set(constructed)
            requirements = [
                tensorfs.TensorRequirement(
                    component=d.component,
                    key=d.key[len(d.component) + 1 :],
                    shape=list(d.spec.shape),
                    logical_dtype=tensorfs_requirement_dtype(d.spec.dtype),
                )
                for d in walked.destinations
                if d.component in members
            ]
            with store_refusals(f"fitting {checkpoint.manifest_id[:23]}"):
                fit = dict(
                    tensorfs.fit(
                        requirements,
                        checkpoint.header_bytes,
                        custody=self.custody,
                        encoded_leaves=encoded_leaves == "accept",
                        device=self.device_facts.predicate,
                        observations=observations,
                    )
                )
            if not fit.get("ok"):
                return fit
            verdict = fit
            extras = [str(k) for k in fit.get("ignored", ())]  # type: ignore[attr-defined]
            ignored.extend(extras)
            if extras:
                # One line per checkpoint. A TensorFS before 0.3.90 lists the keys only.
                warnings.append(str(fit.get("warning") or self._skipped_line(extras)))
            counted: Mapping[str, int] = fit.get("routes", {})  # type: ignore[assignment]
            for route in routes:
                routes[route] += int(counted.get(route, 0))
        if verdict is None:
            return None
        for full in ignored:
            component, _, key = full.partition("/")
            self.skipped.setdefault(component, []).append(key)
            if component in self.expected:
                self.expected[component] -= {f"{component}.{key}"}
        if len(by_checkpoint) > 1:
            verdict = {**verdict, "ignored": ignored, "routes": routes}
        return {**verdict, "warnings": warnings}

    def _skipped_line(self, extras: Sequence[str]) -> str:
        """TensorFS 0.3.90's warning for the stored tensors fit skipped: how many, how many
        bytes, and the first five."""
        rows = (self.plan.get(full.replace("/", ".", 1)) for full in extras)
        nbytes = sum(row.stored_nbytes for row in rows if row is not None)
        more = f", and {len(extras) - 5} more" if len(extras) > 5 else ""
        return (
            f"{len(extras)} stored tensor(s) the code does not build were skipped, not "
            f"loaded: {nbytes} B ({', '.join(extras[:5])}{more})"
        )

    def materialize(
        self, obj: object, walked: Census, *, encoded_leaves: str = "refuse"
    ) -> Mapping[str, TensorLike]:
        """Weights stay on meta. Derived (non-persistent) buffers, which no checkpoint
        supplies, move to the device with the values their construction computed."""
        self.encoded_leaves = encoded_leaves
        native = sorted(k for k, t in self.resolved.items() if t.route == "encoded_gemm")
        if native and encoded_leaves != "accept":
            raise _refuse(
                "leaf_unconsented",
                f"{len(native)} of this artifact's tensors ({', '.join(native[:3])}…) serve "
                f"through a native encoded leaf on {device_line(self.device_facts)}, and this "
                f"package declares encoded_leaves={encoded_leaves!r}: the leaf replaces the "
                'linear op, so the package must state `encoded_leaves="accept"`',
                native[:6],
            )
        roots = dict(_members(obj))
        with placing():
            for name in walked.components:
                member = roots.get(name)
                if not isinstance(member, self.torch.nn.Module):
                    continue
                self.roots[name] = member
                self.derived[name] = _derived_values(member)
                _restore_derived(self.derived[name], self.device)
        return walked.live

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None:
        """Validate one destination against its stored row. Nothing moves."""
        row = self.plan.get(key)
        if row is None:
            raise _refuse(
                "destination_absent",
                f"{key!r} is not a destination this checkpoint supplies",
                [key],
            )
        shape = tuple(destination.shape)
        if shape != tuple(row.shape) or (spec.nbytes and spec.nbytes != row.nbytes):
            raise _refuse(
                "length_mismatch",
                f"{key}: the destination is {shape} and the checkpoint supplies {row.shape} "
                f"({row.nbytes} B)",
                [key],
            )
        if dtype_name(destination.dtype) != dtype_name(row.dtype):
            raise _refuse(
                "dtype_mismatch",
                f"{key}: the checkpoint's logical dtype is {row.dtype} and the destination is "
                f"{dtype_name(destination.dtype)}",
                [key],
            )
        self.enqueued[key] = destination
        self.state = "staging"

    def commit(self) -> None:
        """Install encoded leaves on meta, then register every component with the plane."""
        missing = sorted(
            key
            for name, keys in self.expected.items()
            if name in self.roots
            for key in keys
            if key not in self.enqueued
        )
        if missing:
            raise _refuse(
                "incomplete_fill",
                f"{len(missing)} destination(s) never reached the fill plane "
                f"({', '.join(missing[:4])}): a DestinationSet is complete or nothing commits",
                missing[:6],
            )
        started = time.perf_counter()
        order = list(self.enqueued)
        self.leaves = {
            key: self._install_leaf(key) for key in order if self._route(key) == "encoded_gemm"
        }
        for name, root in self.roots.items():
            keys = [key for key in order if self.plan[key].component == name]
            if keys:
                component = self._register(name, root, keys)
                self.components[name] = component
                self.weights.register(component)
        self.stats["register_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self.state = "ready"

    def poison(self, keys: Sequence[str]) -> None:
        """Abort: forget every registered weight set and give the leases back."""
        self.close()
        self.state = "poisoned"

    def close(self) -> None:
        self.weights.forget(self.construction)
        self.components.clear()
        self.sources.clear()
        self.leases.release()

    # -- construction internals

    def _route(self, key: str) -> str:
        return str(self.selected[key].provider.route)

    def _install_leaf(self, key: str) -> tuple[Any, Any, tuple[str, ...]]:
        """Replace the linear op that owns `key` with its encoded leaf, on meta. Returns the
        leaf, the replaced module, and the roles the provider derives from stored bytes."""
        torch = self.torch
        row = self.plan[key]
        component, _, local = key.partition(".")
        owner_path, _, attribute = local.removesuffix(".weight").rpartition(".")
        root = self.roots.get(component)
        owner = root.get_submodule(owner_path) if root is not None and owner_path else root
        replaced = getattr(owner, attribute, None) if attribute else None
        weight = getattr(replaced, "weight", None)
        if (
            not local.endswith(".weight")
            or len(row.shape) != 2
            or weight is None
            or tuple(weight.shape) != tuple(row.shape)
        ):
            raise _refuse(
                "leaf_schema",
                f"{key}: the native route replaces a rank-2 linear op named by its `.weight`; "
                "this destination has no such module",
                [key],
            )
        provider = self.selected[key].provider
        assert isinstance(provider, LeafProvider)
        parts = {
            part.role: torch.empty(
                part.shape, dtype=getattr(torch, TORCH_DTYPES[part.dtype]), device="meta"
            )
            for part in row.encoded.parts
        }
        leaf = provider.leaf(torch, parts, replaced, getattr(torch, TORCH_DTYPES[row.dtype]))
        for hooks in (
            "_forward_pre_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_forward_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_hooks_always_called",
        ):
            setattr(leaf, hooks, getattr(replaced, hooks))
        with placing():
            for name in leaf._non_persistent_buffers_set:
                held = leaf._buffers.get(name)
                if held is not None:
                    leaf._buffers[name] = torch.empty_like(held, device=self.device)
        setattr(owner, attribute, leaf)
        kept = leaf.roles()
        derived = tuple(sorted(role for role in parts if kept.get(role) is not parts[role]))
        return leaf, replaced, derived

    def refine(self, name: str) -> Component:
        """Register `name` again at the sub-block grain (regions of at most `REGION_LIMIT`
        bytes): the only way a budget below its block floor can run it."""
        old = self.components[name]
        self.weights.close(old)
        keys = [key for key in self.enqueued if self.plan[key].component == name]
        component = self._register(name, old.root, keys, fine=True)
        self.components[name] = component
        self.weights.register(component)
        return component

    def _register(
        self, name: str, root: Module, keys: list[str], *, fine: bool = False
    ) -> Component:
        """Partition the component into regions (common, then each block or sub-block unit),
        register them as one weight set, and bind every stored part to its destinations."""
        try:
            layout = partition(root, REGION_LIMIT if fine else None)
        except ValueError:
            layout = None  # storage shared across units: one region, not streamable
        units = [unit.path for unit in layout.blocks] if layout is not None else []
        members: list[list[str]] = [[] for _ in range(len(units) + 1)]
        for key in keys:
            path = key[len(name) + 1 :].rpartition(".")[0]
            members[_unit_of(path, units)].append(key)
        kept = [(index, group) for index, group in enumerate(members) if group]
        checkpoint = self.checkpoints[name]
        skipped = self.skipped.get(name, ())
        plan = checkpoint.read_plan(
            [(name, self.plan[key].name) for key in keys] + [(name, key) for key in skipped],
            READ_WINDOW,
            (name,),
        )
        if skipped:
            # The plan names every stored tensor; only what the code builds is read or mapped.
            plan = plan.select(
                [
                    self.plan[key].role_what(part.role)
                    for key in keys
                    for part in self.plan[key].encoded.parts
                ]
            )
        weights = self.weights
        key = f"{self.construction}/{name}"
        ids = [[f"{name}/{self.plan[k].name}" for k in group] for _, group in kept]
        digest = canonical.digest({"manifest": checkpoint.manifest_id, "regions": ids})
        tiers = None if fine else self.tiers
        with store_refusals(f"registering {name}"):
            source = self.sources.get(checkpoint.manifest_id)
            if source is None:
                source = self.sources[checkpoint.manifest_id] = plane.source(
                    weights.plane, checkpoint.store, self.leases.take(name)
                )
            held = tiers.ask(key, digest) if tiers is not None else None
            try:
                ws = weights.plane.register(key, source, plan, ids, host_fd=held)
            except plane.error("Invalid"):
                if held is None:
                    raise
                ws = None  # the held tier is not this layout after all
            finally:
                if held is not None:
                    os.close(held)  # the plane reopened its own description
            if ws is None:
                held, ws = None, weights.plane.register(key, source, plan, ids)
        if tiers is not None:
            if held is None:
                tiers.offer(key, digest, ws.host_fd)
            tiers.share(key, digest, ws.host_fd)
        spans = msgspec.convert(ws.regions, list[_Span])
        placed = {row.what: row for row in msgspec.convert(ws.parts, list[_Placed])}
        sites = _sites(root, name)
        regions: list[Region] = []
        for (index, group), span in zip(kept, spans, strict=True):
            region = Region(
                index=len(regions),
                path=units[index - 1] if index else "",
                module=root.get_submodule(units[index - 1]) if index else root,
                weights=[],
                offset=span.offset,
                span=span.span,
            )
            for key in group:
                row = self.plan[key]
                weight = Weight(
                    row=row,
                    route=self._route(key),
                    provider=self.selected[key].provider,
                    parts={
                        part.role: (
                            placed[row.role_what(part.role)].offset - span.offset,
                            part.dtype,
                            tuple(part.shape),
                        )
                        for part in row.encoded.parts
                    },
                    sites=sites.get(key, []),
                )
                if key in self.leaves:
                    weight.leaf, weight.replaced, weight.derived_roles = self.leaves[key]
                    region.unbound.extend(
                        (weight.leaf._buffers, role, unbound(self.torch, tensor, weights.device))
                        for role, tensor in weight.leaf.roles().items()
                    )
                region.unbound.extend(
                    (table, attr, unbound(self.torch, table[attr], weights.device))
                    for table, attr, _ in weight.sites
                )
                region.weights.append(weight)
            unbind(region)
            regions.append(region)
        common = regions[0] if kept[0][0] == 0 else None
        blocks = [region for region in regions if region is not common]
        component = Component(
            name=name,
            key=f"{self.construction}/{name}",
            root=root,
            ws=ws,
            device=weights.index,
            regions=regions,
            common=common,
            blocks=blocks,
            layout=weight_policy.Layout(
                common=common.span if common else 0,
                blocks=tuple(region.span for region in blocks),
                fine=None if fine else _fine(root),
            ),
            fine=fine,
        )
        for region in regions:
            for table, attr, standin in region.unbound:
                setattr(standin, "origin", (weights, component, region, table, attr))  # noqa: B010
        return component


@dataclass(frozen=True, slots=True)
class HostTiers:
    """Where a weight set's pinned tier comes from and goes to: `ask` returns a memfd already
    holding exactly (weight set, layout) or None (the worker's, or a group's rank 0's);
    `offer` hands the worker one this process filled; `share` hands every registered one on
    (rank 0 to its followers, so K ranks pin one copy)."""

    ask: Callable[[str, str], int | None]
    offer: Callable[[str, str, int], None]
    share: Callable[[str, str, int], None] = lambda name, layout, memfd: None


class _Span(msgspec.Struct, frozen=True):
    index: int
    offset: int
    nbytes: int
    span: int


class _Placed(msgspec.Struct, frozen=True):
    what: str
    region: int
    offset: int
    nbytes: int


def _fine(root: Module) -> tuple[int, int] | None:
    """(common, largest region) spans at the sub-block grain, for the refusal floor."""
    try:
        layout = partition(root, REGION_LIMIT)
    except ValueError:
        return None
    if layout is None:
        return None
    return _span(layout.common.nbytes), _span(max(unit.nbytes for unit in layout.blocks))


def _span(nbytes: int) -> int:
    """A region's footprint in the plane: its bytes rounded up to 2 MiB."""
    return -(-nbytes // (2 << 20)) * (2 << 20)


def _unit_of(path: str, blocks: Sequence[str]) -> int:
    """The region a module path belongs to: 1 + its block's index, or 0 (common)."""
    for index, block in enumerate(blocks):
        if path == block or path.startswith(block + "."):
            return index + 1
    return 0


def _sites(root: Module, name: str) -> dict[str, list[tuple[Table, str, bool]]]:
    """Every module attribute naming each checkpoint tensor, tied names included: a tensor
    shared by two modules binds at both."""
    by_tensor: dict[int, list[tuple[Table, str, bool]]] = {}
    keys: dict[str, int] = {}
    for path, module in root.named_modules(remove_duplicate=False):
        prefix = f"{name}.{path}." if path else f"{name}."
        tables = ((module._parameters, True), (module._buffers, False))
        for table, parameter in tables:
            for attr, tensor in table.items():
                if tensor is None or (not parameter and attr in module._non_persistent_buffers_set):
                    continue
                keys[prefix + attr] = id(tensor)
                sites = by_tensor.setdefault(id(tensor), [])
                if not any(held is table and held_attr == attr for held, held_attr, _ in sites):
                    sites.append((table, attr, parameter))
    return {key: by_tensor[tensor] for key, tensor in keys.items()}


def _derived_values(root: Module) -> dict[tuple[Module, str], Tensor]:
    """Non-persistent buffers with their constructed values: no checkpoint supplies them."""
    held: dict[tuple[Module, str], Tensor] = {}
    for _, sub in root.named_modules():
        for local in getattr(sub, "_non_persistent_buffers_set", ()):
            value = sub._buffers.get(local)
            if value is not None and str(value.device) != "meta":
                held[(sub, local)] = value.detach()
    return held


def _restore_derived(held: Mapping[tuple[Module, str], Tensor], device: Device) -> None:
    for (sub, local), value in held.items():
        sub._buffers[local] = value.to(device, copy=True)


# -------------------------------------------------------------------------- the stages


class StageExit(msgspec.Struct, frozen=True, kw_only=True):
    """One stage's measured facts at scope exit, for the scheduler's cost book (W4)."""

    method: str
    components: tuple[str, ...]
    passes: dict[str, int]
    wall_ns: int
    #: torch's peak above its bytes at stage entry: activations and decode scratch
    growth_bytes: int
    #: the plane's streaming facts for this stage: late copies, the GPU wait on them, bytes
    late: int
    stall_ns: int
    streamed_bytes: int
    resident_blocks: dict[str, int]
    window: dict[str, int]


class WeightResidency:
    """The author `Residency` on the plane: `admit`/`release` per component-use scope (a
    stage), one wrapper per block. Installed on a constructed generation as
    `_cozy_residency`; the package cannot observe it."""

    def __init__(
        self,
        weights: Weights,
        components: Mapping[str, Component],
        scopes: Mapping[str, tuple[str, ...]] | None = None,
        refine: Callable[[str], Component] | None = None,
    ) -> None:
        self.weights = weights
        self.components = dict(components)
        #: re-registers a component at the sub-block grain (the backend's `refine`)
        self.refine = refine
        #: component -> (module, had its own `forward`, that forward) per hooked region
        self.hooks: dict[str, list[tuple[Module, bool, Callable[..., object]]]] = {}
        #: method -> the components it declares (`component_use`)
        self.scopes = dict(scopes or {})
        self.hosted: frozenset[str] = frozenset()
        self.active: tuple[str, tuple[str, ...]] | None = None
        self.spare = False
        self.poisoned = ""
        self.last_shortfall: dict[str, Any] | None = None
        self.refusals: list[str] = []
        #: this attempt's activation peak per method, and per batch item with the batch it ran
        self.peaks: dict[str, int] = {}
        self.per_item: dict[str, int] = {}
        self.batch: dict[str, int] = {}
        self.baseline = -1
        self.derived_at_open = 0
        #: The scheduler's turn (W4, `stage/1`): called before a stage plans, returns the
        #: plane budget the Worker granted for it, or None to keep the current one; and its
        #: sink for each stage's measured facts at exit.
        self.turn: Callable[[str, tuple[str, ...]], int | None] | None = None
        self.exit: Callable[[StageExit], None] | None = None
        self.entered_ns = 0
        self.stage_mark = (0, 0, 0)
        #: stages of each method in this attempt, and the most any attempt has run: a method
        #: entered again and again (a denoise step per stage) is cyclic like a multi-pass one
        self.entries: dict[str, int] = {}
        self.repeats: dict[str, int] = {}
        #: a repeating method's stage left open between its entries (its resident set bound,
        #: its common leases and cursors live, the cursor already copying the next pass), and
        #: the Worker's budget it was planned in
        self.parked: tuple[str, tuple[str, ...], int] | None = None
        self._shortfall_error = (plane.error("Shortfall"), plane.error("BelowFloor"))
        self._budget_error = plane.error("BudgetExceeded")
        self._capacity = (*self._shortfall_error, self._budget_error)
        self._poisoned_error = plane.error("Poisoned")
        self._quiet = weights.torch.utils._python_dispatch._disable_current_modes
        weights.residencies.add(self)
        for component in self.components.values():
            self._hook(component)
            vae: Any = component.root
            if not weights.host_only and _image_vae(vae):
                vae.decode = self._decode(component.name, vae.decode)

    def _hook(self, component: Component) -> None:
        """One wrapper per block region: its module's forward acquires and releases it. A CPU
        executor has none: its weights are bound whole at registration, at either grain."""
        if self.weights.host_only:
            return
        rows = self.hooks[component.name] = []
        for region in component.blocks:
            module = region.module
            rows.append((module, "forward" in module.__dict__, module.forward))
            module.forward = self._block(component, region, module.forward)

    def _unhook(self, name: str) -> None:
        for module, had, forward in self.hooks.pop(name, []):
            if had:
                module.forward = forward
            else:
                del module.forward

    # -- the Residency protocol

    def open_attempt(self, *_: object) -> None:
        """A new attempt: its activation peaks start empty. (Older Workers also send a
        placement rung and headroom; the plane has no rungs, so they are ignored.)"""
        self.unpark()
        self.peaks.clear()
        self.entries.clear()
        self.last_shortfall = None
        self.derived_at_open = self._derived()

    def admit(self, method: str, components: tuple[str, ...]) -> None:
        self._require_clean(method)
        try:
            with self._quiet():  # the plane's own binds are not the package's ops
                self._admit(method, components)
        except self._poisoned_error as exc:
            raise self._poison(exc) from exc

    def _admit(self, method: str, components: tuple[str, ...]) -> None:
        if self.active is not None:
            self.refusals.append(f"fault scope_overlap: {method} entered inside {self.active[0]}")
            self.release(*self.active)
        local = tuple(
            name for name in components if name in self.components and name not in self.hosted
        )
        weights = self.weights
        if self.turn is not None:
            granted = self.turn(method, components)
            if granted is not None:
                weights.set_budget(granted)
        self.entries[method] = self.entries.get(method, 0) + 1
        self.repeats[method] = max(self.repeats.get(method, 0), self.entries[method])
        weights.hold_host()
        if weights.host_only:  # everything is bound in place already; no device to measure
            self.active = (method, local)
            self.entered_ns = time.perf_counter_ns()
            return
        if self.parked == (method, local, weights.budget):
            # The same stage again: its plan, leases and cursors carry on.
            self.parked = None
            self.active = (method, local)
            weights.open.append((self, method, local))
            for name in local:
                self.components[name].passes = 0
            self._entered(method)
            return
        # A new plan: no stage left open by an earlier step survives it, in any model slot.
        for row in list(weights.residencies):
            row.unpark()
        self.active = (method, local)
        weights.open.append((self, method, local))
        try:
            budget = weights.stage_budget(method)
            if method not in weights.activation:
                # Activations first, and these are not measured yet: the open stages' own
                # weights are all the plane keeps; every other byte stays torch's until it is.
                # One GPU of a group keeps only a streaming window for the first pass (its
                # blocks cannot run again after an out-of-memory), then grows by what it measured.
                own = [component.planned for _, _, component in weights.opened().values()]
                weights.growing = weights.grouped
                budget = min(
                    budget,
                    # a window of two blocks: one more than the floor, so nothing refines
                    sum(o.common + 2 * o.largest if weights.grouped else o.total for o in own),
                )
            budget = self._place(budget)
            stream = weights.stream()
            for name in local:
                component = self.components[name]
                component.passes = 0
                if component.common is not None:
                    try:
                        component.lease = weights.plane.acquire(
                            component.ws, weights.index, component.common.index, stream
                        )
                    except self._capacity as exc:
                        raise self._shortfall(method, 0, budget, str(exc)) from None
                    bind(weights.torch, component.common, component.lease)
        except BaseException:
            with suppress(Exception):  # the original failure is the one to report
                self._close(local)
            self._left()
            raise
        self._entered(method)

    def _place(self, budget: int, resume: Mapping[str, int] = {}) -> int:
        """`_plan` the open stages in `budget`. When the driver itself has less (another
        process, or torch's idle cache, holds it), plan again inside what it has now, never
        above `budget` and never below a two-block streaming window. When binding runs out
        of device memory (it decodes and converts weights into torch's bytes, and torch may
        hold more than the budget knew), plan again in what is left. Returns the budget used."""
        weights = self.weights
        method = weights.open[-1][1]
        weights.quiesce()  # a plan unmaps: other components' sets, this one's stale regions
        pooled = False
        while True:
            oom: tuple[int, str] | None = None
            try:
                self._plan(budget, resume)
                return budget
            except self._budget_error as exc:
                if pooled or not _device_pool(exc):
                    raise self._shortfall(method, 0, budget, str(exc)) from None
                pooled = True
            except Exception as exc:
                if not is_device_oom(exc):
                    raise
                oom = (_requested(exc), str(exc).splitlines()[0][:300])
            # outside `except`, so the failed plan's frames (and their tensors) are gone
            if oom is not None:
                budget = self._after_bind(budget, method, *oom)
                continue
            accel.release_cached(weights.torch, weights.kind)
            window = sum(
                c.planned.common + 2 * c.planned.largest for _, _, c in weights.opened().values()
            )
            budget = min(budget, max(weights.driver_budget(method), window))

    def _after_bind(self, budget: int, method: str, need: int, failed: str) -> int:
        """The budget to plan again in after binding ran out of device memory: what the open
        stages' weights and their decoded copies can have now (mapped, decoded and free), or
        the failure by its own name when that is no less than before."""
        weights = self.weights
        accel.release_cached(weights.torch, weights.kind)
        mapped = max(plane.stats(weights.plane).device(weights.index).committed, 0)
        room = mapped + weights.decoded() + weights.allocatable() - need - weight_policy.MARGIN
        lowered = max(min(room, budget - weight_policy.MARGIN), weights.floor())
        if lowered >= budget:
            raise self._shortfall(
                method,
                need,
                room,
                f"binding its weights ran out of device memory at the stage's floor: {failed}",
            )
        weights.oom_retries += 1
        return lowered

    def _left(self) -> None:
        if self.active is not None:
            with suppress(ValueError):
                self.weights.open.remove((self, *self.active))
        self.active = None
        if not self.weights.open:
            self.weights.growing = False

    def _asked(self, resume: Mapping[str, int] = {}) -> None:
        """The Worker lowered (or raised) this executor's budget mid-call: apply it at this
        block boundary, never below the open stages' floor, and say what was applied."""
        weights = self.weights
        assert weights.cell is not None
        wanted = weights.cell.poll()
        if wanted is None:
            return
        opened = [component for _, _, component in weights.opened().values()]
        floor = weights.floor()
        for component in opened:
            close_cursor(component)
        weights.set_budget(max(wanted, floor) if wanted >= 0 else wanted)
        weights.drain()
        if opened and weights.open:
            self._place(max(weights.applied, floor), resume)
        weights.cell.acknowledge(weights.applied)

    def _grow(self) -> None:
        """A group's unmeasured stage begins its second pass: place the open stages again
        with the activations the first pass measured, inside what the driver has now."""
        weights = self.weights
        weights.growing = False
        for row, method, _ in weights.open:
            if row.baseline >= 0:
                peak = accel.peak_allocated(weights.torch, weights.kind) - row.baseline
                weights.activation[method] = max(weights.activation.get(method, 0), peak, 0)
        method = weights.open[-1][1]
        budget = min(weights.stage_budget(method), weights.driver_budget(method))
        if budget <= weights.applied:
            return  # no room to grow into: the floor's plan carries on
        for _, _, component in weights.opened().values():
            close_cursor(component)
        self._place(budget)

    def _entered(self, method: str) -> None:
        weights = self.weights
        cudnn = weights.torch.backends.cudnn
        if cudnn.benchmark:
            # Autotuning times its candidates, and streaming copies share the GPU with them:
            # the algorithm, and so the output bits, would follow the copies. Heuristics
            # choose by shape alone.
            cudnn.benchmark = False
            weights.modes["cudnn.benchmark"] = "off (the package had turned it on)"
        for name in self.active[1] if self.active is not None else ():
            if self.components[name].fine:  # the run's record of how tight its memory was
                weights.modes[f"{name}.grain"] = "sub-block"
        self.baseline = accel.allocated(weights.torch, weights.kind)
        accel.reset_peak(weights.torch, weights.kind)
        self.entered_ns = time.perf_counter_ns()
        if self.exit is not None:
            self.stage_mark = _stage_totals(weights, method)

    def release(self, method: str, components: tuple[str, ...]) -> None:
        try:
            with self._quiet():
                self._release(method, components)
        except self._poisoned_error as exc:
            raise self._poison(exc) from exc

    def _release(self, method: str, components: tuple[str, ...]) -> None:
        if self.active is None:
            self.refusals.append(f"fault scope_over_release: {method}")
            return
        weights = self.weights
        method, local = self.active
        try:
            if self.repeats.get(method, 0) > 1 and not weights.host_only:
                # A repeating stage (a denoise step per entry) stays open for the next entry.
                for name in local:
                    self.components[name].last_passes = self.components[name].passes
                self.parked = (method, local, weights.budget)
            else:
                self._close(local)
            weights.drain()
        finally:
            self._left()
        peak = 0
        if self.baseline >= 0:
            peak = max(accel.peak_allocated(weights.torch, weights.kind) - self.baseline, 0)
            self.peaks[method] = max(self.peaks.get(method, 0), peak)
            weights.activation[method] = max(weights.activation.get(method, 0), peak)
            item = peak // max(self.batch.pop(method, 1), 1)
            self.per_item[method] = max(self.per_item.get(method, 0), item)
        self.baseline = -1
        if self.exit is not None:
            late, stall, streamed = (
                now - then
                for now, then in zip(_stage_totals(weights, method), self.stage_mark, strict=True)
            )
            placed = {name: self.components[name] for name in local}
            self.exit(
                StageExit(
                    method=method,
                    components=local,
                    passes={name: c.last_passes for name, c in placed.items()},
                    wall_ns=time.perf_counter_ns() - self.entered_ns,
                    growth_bytes=peak,
                    late=late,
                    stall_ns=stall,
                    streamed_bytes=streamed,
                    resident_blocks={
                        name: len(c.residence.resident) if c.residence else 0
                        for name, c in placed.items()
                    },
                    window={
                        name: c.residence.window if c.residence else 0 for name, c in placed.items()
                    },
                )
            )

    # -- the executor's and the package's calls

    def host(self, name: str) -> None:
        """`name` lives on another GPU of the group: never held here."""
        self.unpark()
        component = self.components.get(name)
        if component is not None:
            self._drop([component])
        self.hosted = self.hosted | {name}

    @contextmanager
    def sparing(self) -> Iterator[None]:
        """An optional call: taken only if it fits beside what this GPU holds."""
        self.spare = True
        try:
            yield
        finally:
            self.spare = False

    def observe_remote(self, method: str, working_bytes: int) -> None:
        self.peaks[method] = max(self.peaks.get(method, 0), int(working_bytes))

    def shed(self) -> None:
        """Unmap every component outside the active scope (a hosted call's OOM retry)."""
        self.unpark()
        held = set(self.active[1]) if self.active is not None else set()
        self._drop([c for name, c in self.components.items() if name not in held])

    def vacate(self) -> dict[str, Any]:
        """Unmap this construction's idle weights. They stay in the pinned tier; the next
        stage wants them back at link speed."""
        self.unpark()
        idle = [c for c in self.components.values() if c.lease is None and c.cursor is None]
        freed = sum(self.weights.resident_bytes(c) for c in idle)
        self._drop(idle)
        return {"freed": {}, "held": {}, "freed_bytes": freed}

    def batch_fits(self, method: str, batch: int) -> bool:
        """`batch` items of `method` at once, if their measured activations fit what is free
        plus the weights the plane may unmap beyond the stage's own floor. Recorded."""
        per_item = self.per_item.get(method)
        if per_item is None:
            fits = True
        else:
            free, evictable = self.room(self.scopes.get(method, ()))
            fits = weight_policy.batch_fits(free, evictable, per_item * batch)
        if fits:
            self.batch[method] = batch
        self.weights.modes[f"{method}.batch"] = str(batch) if fits else "1"
        return fits

    def room(self, components: Sequence[str]) -> tuple[int, int]:
        """(allocatable device bytes now, plane bytes a stage of `components` could unmap):
        everything mapped but what is leased or held, and what `components` keep resident
        (theirs is the stage's own set, just wanted)."""
        weights = self.weights
        memory = accel.allocation(weights.torch, weights.kind)
        free = memory["driver_free_bytes"] + max(
            memory["reserved_bytes"] - memory["allocated_bytes"], 0
        )
        device = plane.stats(weights.plane).device(weights.index)
        held = {self.components[name].key for name in components if name in self.components}
        held.update(weights.opened())
        own = sum(
            weights.components[key].regions[index].span
            for key in held
            for index in weights.components[key].resident
        )
        movable = device.committed - device.leased_bytes - device.held_bytes - own
        return free, max(movable, 0)

    def document(self) -> dict[str, Any]:
        weights = self.weights
        return {
            "resident": {
                name: weights.resident_bytes(component)
                for name, component in self.components.items()
            },
            "evicted": {},
            "hosted": sorted(self.hosted),
            "active": (
                {"method": self.active[0], "components": list(self.active[1])}
                if self.active
                else None
            ),
            # what a grant must leave for activations: an adaptive stage (an image VAE's
            # decode) fits itself to the room it finds, so its peak never sizes one
            "attempt_activation_peak_bytes": max(
                (v for m, v in self.peaks.items() if m not in weights.adaptive),
                default=max(self.peaks.values(), default=0),
            ),
            "attempt_activation_peaks": dict(self.peaks),
            # torch bytes this attempt's bindings moved: decoded resident weights and derived
            # leaf roles kept or dropped, the ledger's declared transition
            "allocator_delta_bytes": self._derived() - self.derived_at_open,
            "poisoned": self.poisoned,
            "refusals": self.refusals[-8:],
            # the plane's counters (cumulative for this process): what streamed, how late
            "plane": msgspec.to_builtins(weights.facts()),
        }

    # -- the stage

    def _plan(self, budget: int, resume: Mapping[str, int] = {}) -> None:
        """Place every open stage's components inside `budget`: each resident set wanted, the
        rest streamed by a cursor in the blocks' execution order (from `resume[weight set]`
        when a block boundary re-plans mid-pass). Before a component's order is observed its
        blocks are acquired on demand. A component whose block floor does not fit moves to
        the sub-block grain first."""
        weights = self.weights
        opened = weights.opened()
        components = {key: component for key, (_, _, component) in opened.items()}
        method = weights.open[-1][1]
        arriving: list[Ticket] = []

        def stage(
            layouts: Mapping[str, weight_policy.Layout],
        ) -> dict[str, weight_policy.Residence]:
            return weight_policy.stage(
                layouts,
                budget,
                orders={
                    key: [c.numbers[index] for index in c.order]
                    for key, c in components.items()
                    if c.observed
                },
                costs=_costs(weights, {key: (c, opened[key][1]) for key, c in components.items()}),
                cyclic={
                    key
                    for key, (row, tag, c) in opened.items()
                    if c.last_passes != 1 or row.repeats.get(tag, 0) > 1
                },
            )

        # Every component keeps its grain while the budget holds each one's floor at it: a
        # larger component never takes the room a smaller one needs to stay whole (H3's
        # turbo LoRA beside its DiT, run 2947). Only when those floors do not fit does a
        # component that can still refine count at its finer floor, and move to it.
        layouts = {key: replace(c.planned, fine=None) for key, c in components.items()}
        try:
            try:
                placed = stage(layouts)
            except weight_policy.BelowFloor:
                layouts = {
                    key: c.planned if opened[key][0]._refinable(c) else layouts[key]
                    for key, c in components.items()
                }
                placed = stage(layouts)
        except weight_policy.BelowFloor as exc:
            if exc.component in opened and opened[exc.component][0]._refine(exc.component):
                return self._plan(budget, resume)
            raise self._shortfall(method, exc.needed, exc.available, str(exc)) from None
        # A budget that only fits one block at a time streams with no overlap: regions of at
        # most 16 MiB let it hold a window of several. (Window 1 chosen when two fit stays.)
        for key, residence in placed.items():
            layout = layouts[key]
            if (
                residence.streamed
                and residence.window == 1
                and layout.common + 2 * layout.largest > budget
                and opened[key][0]._refine(key)
            ):
                return self._plan(budget, resume)
        need = sum(r.vram_bytes(layouts[key]) for key, r in placed.items())
        if self.spare and need > budget:
            raise ResidencyRefusal("device_spared", f"{method}() declined: needs {need} B")
        weights._apply(budget)
        clock = weights.tick()
        for key, residence in placed.items():
            component = components[key]
            close_cursor(component)
            component.residence = residence
            before = set(component.resident)
            component.resident = (
                *((component.common.index,) if component.common is not None else ()),
                *(component.blocks[i].index for i in residence.resident),
            )
            streams = bool(residence.streamed) and component.observed
            if stale := sorted(
                (before - set(component.resident))
                | (
                    {r.index for r in component.blocks} - set(component.resident)
                    if streams
                    else set()
                )
            ):
                # Only the planned set may be home to the cursor, so the ring has the room
                # the budget promised it: what left the set, and blocks demand-filled
                # earlier, leave the device (the pinned tier keeps them). A region a dropped
                # cursor still holds (a failed block's frames keep its view) goes with it.
                with suppress(plane.error("LeaseViolation")):
                    weights.plane.drop(component.ws, weights.index, stale)
                for index in stale:
                    unbind(component.regions[index])
            try:
                arriving.append(
                    weights.plane.want(
                        component.ws,
                        weights.index,
                        list(component.resident),
                        priority=plane.PREFIX + clock,
                    )
                )
            except self._shortfall_error as exc:
                raise self._shortfall(method, need, budget, str(exc)) from None
        # Every eviction so far is past: its regions leave their bindings now, before the
        # resident sets bind (a region evicted earlier and wanted again binds after this).
        weights.drain()
        # The plan's bytes reach the device before they bind: the stage cannot compute without
        # them, and that wait is loading, in the record by its own name and off the stage's
        # clock. A fill that failed is reported where its region is used, as before.
        with aside("loading weights"), suppress(plane.error("PlaneError")):
            for ticket in arriving:
                ticket.wait()
        stream = weights.stream()
        for key, residence in placed.items():
            component = components[key]
            # The resident set binds for the whole stage, ordered after its fills on the
            # compute stream: code that reads a resident weight outside its block's forward
            # sees real bytes, exactly as when the component was simply on the device.
            for index in component.resident:
                lease = weights.plane.acquire(component.ws, weights.index, index, stream)
                try:
                    bind(weights.torch, component.regions[index], lease)
                finally:
                    lease.release()
            if residence.streamed and component.observed:
                order = component.order
                start = order.index(resume[key]) if key in resume else 0
                try:
                    component.cursor = weights.plane.stream(
                        component.ws,
                        weights.index,
                        order[start:] + order[:start],
                        residence.window,
                        repeat=0,
                        priority=plane.STREAMED + clock,
                        tag=opened[key][1],
                        # the plane's ring holds the stored bytes; their decoded copies
                        # are counted in the budget beside it
                        ring_bytes=weight_policy.ring_bytes(
                            [component.layout.blocks[i] for i in residence.streamed],
                            residence.window,
                        ),
                    )
                except self._shortfall_error as exc:
                    raise self._shortfall(method, need, budget, str(exc)) from None

    def _derived(self) -> int:
        return sum(
            int(t.numel() * t.element_size())
            for c in self.components.values()
            for r in c.regions
            for t in r.derived
        )

    def _refinable(self, component: Component) -> bool:
        return not (
            component.fine
            or component.layout.fine is None
            or self.refine is None
            or any(c.lease is not None or c.cursor is not None for c in self.components.values())
        )

    def _refine(self, key: str) -> bool:
        """Move the weight set `key` to the sub-block grain, if it still can: at a stage's
        start, never inside one (its slot's leases pin the coarse regions)."""
        name = next((n for n, c in self.components.items() if c.key == key), "")
        component = self.components.get(name)
        if component is None or self.refine is None or not self._refinable(component):
            return False
        self._unhook(name)
        component = self.components[name] = self.refine(name)
        self._hook(component)
        return True

    def _block(
        self, component: Component, region: Region, forward: Callable[..., Any]
    ) -> Callable[..., Any]:
        """The one hook per block: acquire, bind, run, release. A torch OOM releases, lowers
        the plane budget and runs the block again, while its inputs are untouched."""
        weights = self.weights
        torch = weights.torch

        def run(*args: Any, **kwargs: Any) -> Any:
            try:
                if region.mutates is False and not weights.grouped:
                    with self._quiet():  # the block retries as a unit, not op by op
                        return attempt(*args, **kwargs)
                # Its first run, a block that writes into its inputs, or one GPU of a group:
                # its ops recover one by one, and the first run learns whether it writes.
                inputs = (*args, *kwargs.values())
                watch = (
                    frozenset(_storage(v) for v in _tensor_operands(inputs))
                    if region.mutates is None
                    else frozenset()
                )
                with weights.recovering(watch) as seen:
                    out = attempt(*args, **kwargs)
                if region.mutates is None and not weights.host_only:
                    region.mutates = seen.mutated
                return out
            except self._poisoned_error as exc:
                raise self._poison(exc) from exc

        def attempt(*args: Any, **kwargs: Any) -> Any:
            if weights.cell is not None and not weights.running:
                self._asked({component.key: region.index})
            passes = component.passes
            self._observe(component, region.index)
            if component.passes > passes:
                weights.hold_host()
            if weights.growing and component.passes == 2 > passes:
                self._grow()
            versions = _versions(args, kwargs)
            # a library's refusal is answered once: by its op where ops recover one by one
            refused = weights.grouped or region.mutates is not False
            while True:
                stream = weights.stream()
                try:
                    lease = self._acquire(component, region.index, stream)
                except self._capacity:
                    self._make_room(
                        region.span,
                        {component.key: region.index},
                        f"{component.name}.{region.path}",
                    )
                    continue
                need = 0
                try:
                    try:
                        bind(torch, region, lease)
                        weights.running += 1
                        try:
                            return forward(*args, **kwargs)
                        finally:
                            weights.running -= 1
                    except Exception as exc:
                        need, refused = _room_for(exc, refused)
                        need = need or region.span
                finally:
                    # Outside its lease only a stage-resident region stays bound: any other
                    # read finds the stand-in (`Weights.reading`), never a stale address.
                    if region.index not in component.resident:
                        unbind(region)
                    lease.release()
                if weights.grouped or region.mutates is not False:
                    raise self._shortfall(
                        self.active[0] if self.active else "",
                        need,
                        0,
                        f"{component.name}.{region.path} ran out of memory outside torch's "
                        "own ops, and it shares its call with its group or writes into its "
                        "inputs, so it cannot run again",
                    )
                if _versions(args, kwargs) != versions:
                    raise self._shortfall(
                        self.active[0] if self.active else "",
                        need,
                        0,
                        f"{component.name}.{region.path} ran out of memory after changing its "
                        "inputs in place, so it cannot run again",
                    )
                self._make_room(
                    need, {component.key: region.index}, f"{component.name}.{region.path}"
                )

        return run

    def _acquire(self, component: Component, index: int, stream: int) -> Lease:
        """The block's lease: from the stage's cursor (a departure from its order skips ahead
        or demand-fills, never raises), else resident access. A demand fill ranks above every
        earlier streamed block and below the stage's prefix, so the least recently used
        block is the one it can displace."""
        weights = self.weights
        if component.cursor is not None:
            lease: Lease = component.cursor.acquire(index, stream)
        else:
            priority = None if index in component.resident else plane.STREAMED + weights.tick()
            lease = weights.plane.acquire(
                component.ws, weights.index, index, stream, priority=priority
            )
        weights.drain()  # what this fill displaced is unbound at once, in every weight set
        return lease

    def _lease(self, component: Component, region: Region, op: str) -> Lease:
        """A region's lease for an op that reads it outside its forward (`Weights.reading`)."""
        weights = self.weights
        while True:
            try:
                return self._acquire(component, region.index, weights.stream())
            except self._capacity:
                if weights.running:
                    raise self._shortfall(
                        weights.open[-1][1],
                        region.span,
                        0,
                        f"{op} reads {component.name}.{region.path} inside another block's "
                        "forward, whose lease no weight can leave under",
                    ) from None
                weights.open[-1][0]._make_room(
                    region.span,
                    {component.key: region.index},
                    f"{op} ({component.name}.{region.path})",
                )

    def _make_room(self, need: int, resume: Mapping[str, int] = {}, op: str = "") -> None:
        """Activations first: unmap `need` bytes of weights, idle ones first, then re-place the
        open stages (a block that runs again resumes its component's cursor: `resume`)."""
        weights = self.weights
        opened = [component for _, _, component in weights.opened().values()]
        method = weights.open[-1][1] if weights.open else ""
        before = weights.allocatable()
        for component in opened:
            close_cursor(component)
        weights.shrink(need, weights.floor())
        if opened:
            self._place(weights.applied, resume)
        weights.progressed(before, need, op or f"{method}()")

    def _observe(self, component: Component, index: int) -> None:
        """Count passes and learn the blocks' execution order from a stage's first pass."""
        if component.observed:
            if index == component.order[0]:
                component.passes += 1
            return
        if not component.seen:
            component.passes += 1
        elif index == component.seen[0]:
            component.passes += 1
            unseen = [r.index for r in component.blocks if r.index not in component.seen]
            component.order = [*component.seen, *unseen]
            component.observed = True
            return
        if index not in component.seen:
            component.seen.append(index)

    def _decode(self, name: str, decode: Callable[..., Any]) -> Callable[..., Any]:
        """An image VAE's decode mode, decided before the call from what is free plus what the
        plane may unmap: untiled when its measured activations fit, else tiled with the tile
        clamped to what is free. A size never measured at or above runs untiled (its blocks
        retry on OOM), sized by the largest size measured below it: a fixed cost makes small
        decodes' per-pixel peaks overstate large ones.
        A decode that cannot run untiled even with every idle weight unmapped runs again tiled,
        and a tile that does not fit runs again at half the side, down to the smallest tile.
        The mode is recorded in the run's outputs."""
        root: Any = self.components[name].root  # a diffusers image VAE, read as it is
        vae = image_vae.geometry(root)
        assert vae is not None
        weights = self.weights
        #: pixels -> the untiled peak measured at that size
        peaks: dict[int, int] = {}
        #: pixels -> the tile side that last fit
        sides: dict[int, int] = {}
        scale = vae.scale

        def run(z: Tensor, *args: object, **kwargs: object) -> object:
            if self.active is not None:
                weights.adaptive.add(self.active[0])
            height, width = int(z.shape[-2]) * scale, int(z.shape[-1]) * scale
            pixels = int(z.shape[0]) * height * width
            free, evictable = self.room((name,))
            size = min((n for n in peaks if n >= pixels), default=0)
            known = size or max(peaks, default=0)
            per_pixel = peaks[known] // known if known else 0
            mode = (
                weight_policy.decode_mode(free, evictable, per_pixel * pixels)
                if size
                else "untiled"
            )
            # Activations first: idle weights leave before a decode needs their room (all of
            # them unless a measured untiled peak says less), since a decode run short of
            # memory takes slower convolutions.
            need = per_pixel * pixels if mode == "untiled" and known else free + evictable
            free = self._room_for_decode(name, free, evictable, min(need, free + evictable))
            # a tile under half the image, so tiling always lowers the peak
            largest = _tile(min(height, width) // 2)
            side = min(_side(free // max(per_pixel, 1)), sides.get(pixels, largest), largest)
            baseline = accel.allocated(weights.torch, weights.kind)
            accel.reset_peak(weights.torch, weights.kind)
            while True:
                if mode == "tiled":
                    vae.tile(side)
                else:
                    root.disable_tiling()
                try:
                    out = decode(z, *args, **kwargs)
                    break
                except Exception as exc:
                    capacity = is_device_oom(exc) or (
                        isinstance(exc, ResidencyRefusal) and exc.code == "device_shortfall"
                    )
                    if not capacity or (mode == "tiled" and side <= MIN_TILE):
                        raise
                # outside `except`, so the failed call's frames are gone
                accel.release_cached(weights.torch, weights.kind)
                if mode == "tiled":
                    side = _tile(side // 2)
                    continue
                # it needs more than all it had: the next decode of this size starts tiled
                peaks[pixels] = max(peaks.get(pixels, 0), free + evictable + 1)
                mode = "tiled"
                free, evictable = self.room((name,))
                self._room_for_decode(name, free, evictable, free + evictable)
            if mode == "tiled":
                sides[pixels] = side
                inexact()  # a tiled decode's arithmetic differs from an untiled one
            else:
                peak = accel.peak_allocated(weights.torch, weights.kind) - baseline
                peaks[pixels] = max(peaks.get(pixels, 0), peak)
            weights.modes[f"{name}.decode"] = f"tiled {side}" if mode == "tiled" else mode
            # The decode's transient activations go back to the driver: kept in torch's
            # cache they shrink the next grant, and the weights the decode unmapped come
            # back into exactly that room.
            accel.release_cached(weights.torch, weights.kind)
            return out

        return run

    def unpark(self) -> None:
        """Close a repeating stage left open: another stage, an attempt's end, or a budget
        change comes first."""
        if self.parked is not None:
            local = self.parked[1]
            self.parked = None
            self._close(local)

    def _close(self, local: Sequence[str]) -> None:
        """End the stage for `local`: cursors closed, common leases released, and every region
        at idle priority, still mapped and bound (least recently used goes first)."""
        weights = self.weights
        idle = weights.tick()
        failure: BaseException | None = None
        for name in local:
            component = self.components[name]
            component.last_passes = component.passes
            try:
                close_cursor(component)
            except BaseException as exc:  # release the rest, then raise the first
                failure = failure or exc
            if component.lease is not None:
                component.lease.release()
                component.lease = None
            if not weights.host_only:
                weights.plane.prioritise(component.ws, weights.index, None, priority=idle)
        if failure is not None:
            raise failure

    def _room_for_decode(self, name: str, free: int, evictable: int, need: int) -> int:
        """Activations first: idle weights leave before a decode needs their room, since a
        decode run short of memory takes slower convolutions. Never what the open stages hold
        (it may still be filling) and never their floor: a component that streams has little
        resident, and its next block needs its ring. Returns what is free after."""
        if need <= free or evictable <= 0:
            return free
        weights = self.weights
        mapped = plane.stats(weights.plane).device(weights.index).committed
        weights.shrink(need - free, max(mapped - evictable, weights.floor()))
        return self.room((name,))[0]

    def _drop(self, components: Sequence[Component]) -> None:
        weights = self.weights
        if weights.host_only:
            return
        weights.quiesce()
        for component in components:
            close_cursor(component)
            weights.plane.drop(component.ws, weights.index, None)
            for region in component.regions:
                unbind(region)

    def _shortfall(self, method: str, needed: int, available: int, detail: str) -> ResidencyRefusal:
        shortfall = {
            "resource": "vram",
            "scope": "component_use",
            "needed_bytes": int(needed),
            "available_bytes": int(available),
            "evidence_class": "measured",
            "request_shape": method,
        }
        self.last_shortfall = shortfall
        self.refusals.append(f"device_shortfall: {method}: {detail}"[:400])
        return ResidencyRefusal("device_shortfall", f"{method}(): {detail}", shortfall)

    def _poison(self, exc: BaseException) -> ResidencyRefusal:
        """The plane latched poison (a lease lost, a driver mapping in doubt): this process is
        replaced; it never serves another stage."""
        self.poisoned = self.poisoned or f"weight plane: {exc}"[:400]
        return ResidencyRefusal("residency_poisoned", self.poisoned)

    def _require_clean(self, what: str) -> None:
        if self.poisoned:
            raise ResidencyRefusal(
                "residency_poisoned", f"{what}() against a poisoned weight plane: {self.poisoned}"
            )


def _costs(
    weights: Weights, components: Mapping[str, tuple[Component, str]]
) -> dict[str, weight_policy.Costs]:
    """Each component's measured streaming costs for its stage's tag: GPU seconds per use of
    each block from the plane's per-region timing, and the achieved pinned-to-device rate."""
    held = plane.stats(weights.plane)
    device = held.device(weights.index)
    # the pinned tier's rate (a ring streams from it), not blended with the page cache's
    tier_ns = device.host_copy_ns - device.mapped_copy_ns
    link = (device.host_copy_bytes - device.mapped_copy_bytes) * 1e9 / tier_ns if tier_ns else 0.0
    out: dict[str, weight_policy.Costs] = {}
    for key, (component, tag) in components.items():
        compute = {
            component.numbers[row.region]: row.compute_ns / row.uses / 1e9
            for row in held.regions
            if row.ws == key
            and row.tag == tag
            and row.device == weights.index
            and row.uses
            and row.region in component.numbers
        }
        out[key] = weight_policy.Costs(compute=compute, link_bps=link)
    return out


def _stage_totals(weights: Weights, tag: str) -> tuple[int, int, int]:
    """The plane's cumulative (late, stall ns, streamed bytes) for stages tagged `tag`."""
    rows = [
        row
        for row in plane.stats(weights.plane).stages
        if row.tag == tag and row.device == weights.index
    ]
    return (
        sum(row.late for row in rows),
        sum(row.stall_ns for row in rows),
        sum(row.streamed_bytes for row in rows),
    )


def _device_pool(exc: BaseException) -> bool:
    """A `BudgetExceeded` from the driver itself (no memory left to map), not the budget.
    The plane names the pool only in its message today (W1 R23)."""
    pool = getattr(exc, "pool", None)
    if pool is not None:
        return bool(pool == "device")
    return str(exc).startswith("budget exceeded in device")


def _versions(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[int, ...]:
    """In-place mutation counters of the block's tensor inputs (one level deep). Inference
    tensors keep none; a block under `inference_mode` re-runs unchecked (retry the op)."""
    out: list[int] = []
    for value in (*args, *kwargs.values()):
        for item in value if isinstance(value, (tuple, list)) else (value,):
            inference = getattr(item, "is_inference", None)
            if inference is not None and not inference():
                out.append(int(item._version))
    return tuple(out)


def _requested(exc: BaseException) -> int:
    """The allocation torch could not make, from its own message."""
    found = _REQUESTED.search(str(exc))
    return int(float(found.group(1)) * _UNITS[found.group(2)]) if found else 0


def _room_for(exc: BaseException, refused: bool) -> tuple[int, bool]:
    """The room to make before a failed op runs again, and whether a library's refusal has now
    been answered: out of device memory, what torch asked for; a cuBLAS or cuDNN status, the
    libraries' own room, once (`refused`: the same op already had it). Anything else, and a
    second refusal, is the op's own failure and raises."""
    if is_device_oom(exc):
        return _requested(exc), refused
    if refused or not is_library_refusal(exc):
        raise exc
    return weight_policy.LIBRARY_ROOM, True


def _image_vae(root: Module) -> bool:
    """An image VAE whose decode the plane sizes: one frame, tiling it can switch."""
    return image_vae.geometry(root) is not None


def _tile(side: int) -> int:
    """`side` as a tile's: a multiple of 64, at least the smallest tile."""
    return max(side // 64 * 64, MIN_TILE)


def _side(pixels: int) -> int:
    """The largest tile side whose square is at most `pixels`."""
    return _tile(int(max(pixels, 0) ** 0.5))
