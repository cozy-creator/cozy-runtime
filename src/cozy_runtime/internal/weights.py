"""Runtime model weights on its in-process TensorFS plane (weight-plane.md §3-4).

TensorD makes stored weight data available and prepares/retains shared CPU buffers through
TensorFS. Runtime constructs models, starts stored parameters as meta placeholders, and
uses the plane to allocate, copy, map and unmap GPU regions. Decoded/derived Torch buffers
remain executor allocations. TensorD coordinates budgets and can retain exported GPU
allocation handles; Runtime attaches those allocations and binds its own tensor views.

The plane registers one weight set per component in regions: `common` and each innermost
no-split block (`paging.partition`). A scope is a stage: at `admit` the policy
(`weight_policy.stage`) picks each component's resident set, window and ring inside the
plane budget; the resident set is
wanted and a cursor streams the rest through its byte ring in the blocks' execution order.
Each block runs inside exactly one acquire/release; a resident region binds at its stable
address, a streamed one at its place in the ring, from cached views. Every op that runs out
of device memory runs again once room is made (`memory.Memory`); a block that changes nothing
outside itself runs with no dispatch layer and, out of memory, once again with one (`Effects`);
capacity never poisons the executor.
"""

from __future__ import annotations

import functools
import mmap
import os
import time
import weakref
import zlib
from collections import Counter, deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import msgspec
import tensorfs

from cozy_runtime.author._activity import aside
from cozy_runtime.author._errors import RuntimeFailure, is_device_oom
from cozy_runtime.author._executor_requests import DeviceTierAnswer, SealedPlan, SharedRegion
from cozy_runtime.author._loader import Census, TensorLike, TensorSpec, _members
from cozy_runtime.author._model import PLACED, RELEASERS
from cozy_runtime.internal import (
    accel,
    bands,
    budget_cell,
    canonical,
    lora_bake,
    module_state,
    plane,
    proctree,
    spill,
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
from cozy_runtime.internal.encoding.leaves import sites as leaf_sites
from cozy_runtime.internal.executor_replies import PlaneFacts, Streamed
from cozy_runtime.internal.fetch_plan import Fetches, Group, Ref, region_objects, stage_components
from cozy_runtime.internal.fill import (
    Checkpoint,
    PlanRow,
    _refuse,
    dtype_name,
    servability,
    store_refusals,
    tensorfs_requirement_dtype,
)
from cozy_runtime.internal.memory import (
    Memory,
    MemoryExhausted,
    Rung,
    reap,
    shortage,
)
from cozy_runtime.internal.paging import partition
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.probe import QualificationResult, observed_capability_records
from cozy_runtime.internal.qualification import qualified
from cozy_runtime.internal.resolution import ResolvedModelPlan, TensorResolution, script_plan
from cozy_runtime.internal.unified_plane import Lease as UnifiedLease
from cozy_runtime.internal.unified_plane import UnifiedPlane

if TYPE_CHECKING:
    from tensorfs import ReadPlan
    from tensorfs.plane import Cursor, Lease, Plane, Source, Ticket, WeightSet
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
#: The TensorFS planning window a weight set's read plan is cut into.
READ_WINDOW = 4 << 20
#: The stage `WeightResidency.map_ahead` opens: no package method has this name.
MAPPING = "(map)"


class ResidencyRefusal(RuntimeFailure):
    """A weight read or a stage the plane refuses for its state, not its memory (that is
    `memory.MemoryExhausted`)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail, code=code)
        self.detail = detail


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
    #: views over a unified plane's region tensor keep its memory: they go with the binding
    owned: bool = False


@dataclass(slots=True)
class Bake:
    """One LoRA'd linear of a region: the index of its stored base weight and everything that
    adds an update to it. A prepared graph's PEFT layer, whose factors are the region's own
    weights (indexes of each adapter's A and B, and the scale PEFT applies); and a model's
    consumer hooks, which supply their own (`LoRASource`). All of it is baked in one rounding,
    and each skips its update (`_cozy_baked`) while the bytes hold its delta."""

    weight: int
    layer: Any = None
    factors: list[tuple[int, int, float]] = field(default_factory=list)
    hooks: list[LoRASource] = field(default_factory=list)


class _Stale(Exception):
    """A region's resident bytes hold a delta the forward about to read does not want."""


class _Regrained(Exception):
    """A block's component moved to another grain under it: its weight set is closed."""


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
    #: what its forward changes outside itself, learned on its first run (`Effects`)
    effects: Effects | None = None
    #: its LoRA'd linears; whether to bake them when its bytes are placed; and whether its
    #: resident bytes hold their deltas now (cleared when the plane evicts it)
    bakes: list[Bake] = field(default_factory=list)
    bake: bool = False
    baked: bool = False
    #: what each bake added (`_bake_key`), while `baked`
    baked_with: list[tuple[object, ...]] = field(default_factory=list)
    #: while `baked`: the digest naming those deltas to other executors (`_variant`); "" when
    #: it cannot be named, and the bytes then stay this process's own
    variant: str = ""
    #: the variant of baked bytes mapped from the machine's custody, until the first forward
    #: that reads them says whether they are what it would bake
    attached: str = ""
    #: the weight set whose plane bytes a tensor views (`Weights.content`)
    locate: Callable[[Tensor], str | None] | None = None
    #: a bake failed part way: the resident bytes must refill before any use
    dirty: bool = False
    #: its forward ran out of device memory with no dispatch layer: its ops recover one by
    #: one from then on, so room is made for the op that needs it, never the block again
    tight: bool = False

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
    #: its bytes are the machine's sealed layout: adopted read-only, never filled here
    sealed: bool
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
    #: the cursor's schedule: `order` from the block it started at
    schedule: list[int] = field(default_factory=list)
    #: the common region's lease for the running stage
    lease: Lease | None = None
    #: the stage's leases on its resident blocks, taken once where the plan binds them: a
    #: block that finds its own here runs with it (no acquire per call); any unmapping lets
    #: them go first (`let_go`)
    holds: dict[int, Lease] = field(default_factory=dict)
    hold_stream: int = 0
    #: gave its weights back to another component's activations for the rest of the stage:
    #: none of its regions is leased or planned; its common maps again when it next runs
    spared: bool = False
    #: region indexes wanted resident for the running stage
    resident: tuple[int, ...] = ()
    #: block region index -> its position in `blocks` (the policy's block number)
    numbers: dict[int, int] = field(default_factory=dict)
    #: a CPU executor's weight set, mapped from its pinned tier (the bytes it binds)
    mapping: Any = None
    #: the pinned tier's fills in flight, one per region (a weight set closes only after them)
    filling: list[Ticket] = field(default_factory=list)
    #: region indexes mapped from GPU memory the machine keeps (`weights.attach/1`)
    shared: set[int] = field(default_factory=set)
    #: the machine's lease sockets for those regions: open while they are mapped, closed (by
    #: this process or its death) once they are let go, which ends the lease
    leases: list[int] = field(default_factory=list)
    #: a layout the machine stages in part: its plan, and the regions staged so far (the rest
    #: copy from the object files; the stream's regions are asked for as stages plan them)
    staging: tuple[SealedPlan, HostTiers] | None = None
    staged: set[int] = field(default_factory=set)
    #: its checkpoint, each region's objects (by plane index), and the regions whose objects
    #: are known here: a region is read only once they landed (`WeightResidency._arrive`)
    manifest: str = ""
    objects: list[list[Ref]] = field(default_factory=list)
    landed: set[int] = field(default_factory=set)
    #: cursors a view into their ring kept open past their stage (`close_cursor`, `reap`)
    orphans: list[Cursor] = field(default_factory=list)

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


class Reservation:
    """Device bytes a live owner keeps for speed and gives back under pressure. They live in
    torch's cache, which the ledger counts already: a reservation is the owner's admission
    token and shed handle, and `granted` is a fact, never added to the ledger."""

    def __init__(self, weights: Weights, owner: str, shed: Callable[[int], int]) -> None:
        self.weights, self.owner, self.shed = weights, owner, shed
        self.granted = 0

    def resize(self, nbytes: int) -> bool:
        """Whether the owner may hold `nbytes` now. Never waits or raises: False pauses the
        owner, which tries again when its own work frees room."""
        if nbytes > self.granted and not self.weights.grant(nbytes - self.granted):
            return False
        self.granted = nbytes
        return True

    def close(self) -> None:
        self.weights.reservations.pop(self, None)
        self.granted = 0


class Weights:
    """This executor process's plane, budgets and every registered weight set (one per rank).

    TensorD supplies budgets; this object carries out local buffer/device operations through
    TensorFS. Every component of every construction shares one plane. Least recently used
    across them is the plane's priority order, so a construction switch moves nothing until
    a stage wants the room, and an evicted component's bytes stay in the pinned tier.
    """

    def __init__(self, torch: ModuleType, device: Device, kind: str) -> None:
        self.torch = torch
        self.device = device
        self.kind = kind
        self.index = int(device.index or 0)
        #: no device in the plane: a CPU executor, whose pinned tier IS its device memory
        self.host_only = device.type == "cpu"
        #: one pool (Apple GPUs): the page cache is the host tier and regions are torch tensors,
        #: through a plane that speaks the TensorFS plane's contract
        self.unified = accel.memory_class(kind) == "unified"
        self.pool = UnifiedPlane(torch, device) if self.unified else None
        self.plane: Plane = (
            cast("Plane", self.pool)
            if self.pool is not None
            else plane.open_plane(None if self.host_only else self.index)
        )
        #: TensorD's budgets; -1: none given, each stage derives one from the driver
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
        #: each stage method's last plan inputs (`driver_budget`), in the facts
        self.plan_inputs: dict[str, dict[str, int]] = {}
        #: per stage method, the most room torch had (`allocatable`) at an entry in which it
        #: dropped its whole cache more than once: under a limit torch refuses reserved plus
        #: one request, so a peak that fits can still squeeze. An entry of that method is
        #: given more room first (`WeightResidency._clear`). Measured here or seeded.
        self.squeezed: dict[str, int] = {}
        #: those this request's own entries measured: what the owner keeps for its shape
        self.squeezes: dict[str, int] = {}
        #: components given back to make torch room (`page_out`), and their bytes
        self.paged_out = [0, 0]
        #: methods whose stage fits itself to the room it finds (it runs its ops in `split`
        #: pieces), and the room their smallest piece needs: all they ask a grant or a plan
        #: for, and what a model's weights kept whole must leave them
        self.adaptive: dict[str, int] = {}
        self.oom_retries = 0
        self.modes: dict[str, str] = {}
        #: `spill.TOTALS` as the attempt's first spill began: its mode counts from there
        self.spilled_from = list(spill.TOTALS)
        #: the stages open now, outermost first. A stage another model slot enters inside one
        #: (H3's turbo LoRA inside its DiT's) is placed with it, inside the one budget.
        self.open: list[tuple[WeightResidency, str, tuple[str, ...]]] = []
        self.residencies: weakref.WeakSet[WeightResidency] = weakref.WeakSet()
        #: one GPU of a group: a block's forward holds collectives the other GPUs are already
        #: in, so it can never run again; its ops recover one by one instead
        self.grouped = False
        #: the GPUs of its group: each pins its own weights, so each takes this share of a
        #: host budget the Worker states for the whole group
        self.world = 1
        #: a group's GPU 1 and up: the owner learns what it holds only from rank 0's replies
        self.follower = False
        #: the components of the blocks in their forward now (their leases pin their own
        #: weights), innermost last; and the component whose block ran last
        self.running: list[str] = []
        self.last = ""
        #: how deep reads outside a module's forward are being mapped (`reading`): a LoRA bake
        #: in a region such a read binds reads its overlay's factors, and a placement that read
        #: makes can bind and bake more, so reads nest; the regions they are mapping
        self.mapping = 0
        self.mapped: set[int] = set()
        self.mapping_ops: list[str] = []
        #: the plane is placing or giving room: an allocator refusal inside is its own to answer
        self.placing = False
        #: a group's stage whose activations are not measured yet runs its first pass at the
        #: floor, and is placed again at its second with what that pass measured
        self.growing = False
        #: the Worker's mid-call budget (`budget_cell`): read at every block boundary
        self.cell: budget_cell.Cell | None = None
        #: asks the Worker for `bytes` free on this GPU from its other tenants (`DeviceRoom`);
        #: None in a process without a Worker
        self.room: Callable[[int], None] | None = None
        #: device bytes live owners keep for speed and shed under pressure (`reserve`)
        self.reservations: dict[Reservation, None] = {}
        #: the CUDA context and library workspaces outside torch and the plane, measured once
        self.context_bytes = -1
        #: the machine's custody of GPU weights (`weights.attach/1`); None: every region private
        self.sharing: DeviceTiers | None = None
        #: the call's fetch channel: readers wait on their objects there; None: all are here
        self.fetch: Fetches | None = None
        #: the machine keeps baked variants beside stored bytes (its answers say so)
        self.variants = False
        #: how LoRA'd linears are served: baked into their weights (owner, 2026-10-04: the
        #: default for every LoRA on every lane), or their update computed every forward (a
        #: request's `lora_mode: per_step`, `set_lora`)
        self.lora: lora_bake.Strategy = "bake"
        #: the owner's cap on this process's device bytes (context, torch's reserved bytes and
        #: the plane's maps); -1: none
        self.cap = -1
        self.total = (
            0
            if self.host_only
            else accel.allocation(torch, kind)["driver_total_bytes"]
            if self.unified
            else int(torch.cuda.get_device_properties(device).total_memory)
        )
        #: physical free bytes this process leaves on its GPU: a display's own floor, raised to
        #: the owner's (`owner_floor`, -1: none given)
        self.display_floor = (
            0
            if self.host_only or self.unified  # a unified budget leaves the system its share
            else weight_policy.free_floor(self.total, accel.drives_display(self.index))
        )
        self.owner_floor = -1
        self.floor_free = self.display_floor
        #: ops run in bands of rows when their operands live in host memory or they run out
        self.bands = None if self.host_only else bands.Bands(self)
        #: the one owner of this GPU's bytes: its ladder and its refusal
        self.memory = Memory(self)
        if not self.host_only:
            self.measure_context()
        if kind == "cuda":
            _observe_refusals(torch, self)

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def stream(self) -> int:
        if self.kind != "cuda":
            return 0  # one command queue
        return int(self.torch._C._cuda_getCurrentRawStream(self.index))

    def allocation(self) -> dict[str, int]:
        """`accel.allocation`, with a unified plane's regions taken out of torch's figures:
        they are torch tensors there, and the plane counts them as its own (`committed`)."""
        memory = accel.allocation(self.torch, self.kind)
        if self.pool is not None:
            for name in ("allocated_bytes", "reserved_bytes"):
                memory[name] = max(memory[name] - self.pool.committed, 0)
        return memory

    def synchronize(self) -> None:
        if self.kind == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif not self.host_only:
            accel.synchronize(self.torch, self.kind)

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

    def take_cap(self) -> None:
        """A process cap the owner wrote into the cell (a room grant raising it into what the
        other tenants gave, the floor watchdog lowering it), applied now: a group's follower
        hears its cap only there, and an op waiting for room reads no block boundary (run
        11275: refused under a 718 MB cap on an empty 8 GB GPU). Plane budgets wait for one."""
        cell = self.cell
        wanted = cell.pending() if cell is not None else None
        if cell is not None and wanted is not None and wanted <= budget_cell.CAP:
            cell.poll()
            self.set_cap(budget_cell.CAP - wanted)
            cell.acknowledge(self.applied)

    def set_cap(self, cap: int) -> None:
        """The owner's process cap; -1 lifts it. Every plane budget stays inside it, and torch's
        allocator is held to what it leaves beside the context and the plane's maps: an
        activation over its estimate is an out-of-memory the plane recovers from by unmapping,
        never a full card. TensorD 0.10.0 grants a follower its GPU's free bytes plus what it
        held at its last reply (run 11399: 670 MB while it held 7.07 GiB), which counts its
        bytes twice; only a card below its floor (the watchdog) holds one under its own count."""
        if cap >= 0 and self.follower:
            memory, held = self._held()
            cap = max(cap, held + memory["driver_free_bytes"] - self.floor_free)
        self.cap = cap
        self.limit_torch()

    def limit_torch(self) -> None:
        """Hold torch's allocator to what the driver has free above the floor now, and to what
        the cap leaves beside the context and the plane. A cap is no promise the card still
        has the room: the driver keeps its own memory (fault buffers, page tables) from what
        is free, and with none left its library faulted (run 5422: SIGSEGV in libcuda after
        NV_ERR_NO_MEMORY, the laptop's free at 10 MB). Torch then runs out of memory at the
        limit and the plane makes room, before the driver does. On unified memory only an
        owner's cap limits torch: past the budget the system pages, which is slower, not a
        failure."""
        if self.host_only:
            return
        if self.unified:
            accel.hold_pool(self.torch, self.cap)
            return
        memory = self.allocation()
        reserved = memory["reserved_bytes"]
        allowed = reserved + memory["driver_free_bytes"] - self.floor_free
        if self.cap >= 0:
            committed = max(plane.stats(self.plane).device(self.index).committed, 0)
            allowed = min(allowed, self.cap - max(self.context_bytes, 0) - committed)
        if reserved > allowed:
            # A lowered limit stops new segments; the idle ones torch keeps go back now.
            accel.release_cached(self.torch, self.kind)
        fraction = min(max(allowed, 0) / max(self.total, 1), 1.0)
        self.torch.cuda.set_per_process_memory_fraction(fraction, self.device)

    def set_floor(self, floor: int) -> None:
        """The owner's physical free floor on this GPU; -1 leaves only a display's own."""
        self.owner_floor = floor
        self.floor_free = max(self.display_floor, floor)

    def seed_activations(self, measured: Mapping[str, int]) -> None:
        """Activation growth the owner measured for these methods in earlier processes: this
        process plans its first stages with room for them instead of keeping every weight
        and recovering from the out-of-memory that follows. Only ever raises an estimate."""
        for method, nbytes in measured.items():
            if nbytes > self.activation.get(method, 0):
                self.activation[method] = int(nbytes)

    def seed_squeezed(self, measured: Mapping[str, int]) -> None:
        """Rooms the owner saw squeeze these methods at this request's shape in earlier
        processes (`squeezed`), as the request begins. Only ever raises one."""
        self.squeezes = {}
        for method, nbytes in measured.items():
            self.squeezed[method] = max(self.squeezed.get(method, 0), int(nbytes))

    def retries(self) -> int:
        """Times torch dropped its whole cache to place one allocation, since the process began.
        Read at every block boundary of a stage: the allocator's own counters, without the
        flattened copy `torch.cuda.memory_stats` makes of them."""
        if self.kind != "cuda":
            return 0
        return int(self.torch._C._cuda_memoryStats(self.index).get("num_alloc_retries", 0))

    def page_out(self, keep: Collection[str] | None, need: int = -1) -> bool:
        """Give torch the device bytes of one component no stage uses now (none in `keep`): the
        one with the fewest bytes to copy back, any that holds baked bytes (a bake to restore)
        after the rest; with a `need`, only that many, its last blocks first. Its own bytes
        leave the plane and its budget. Bytes mapped from the machine's custody are let go and
        the machine is asked for them (`DeviceRoom`: it drops a holding nobody maps, and this
        process's cap rises into that room). False: nothing to give. `keep` None: every
        component an open or parked stage uses."""
        keep = _in_use(self) if keep is None else keep
        asks = self.room is not None and self.cap >= 0
        idle = [
            (any(region.baked for region in c.regions), self.resident_bytes(c), key)
            for key, c in self.components.items()
            if key not in keep and (asks or not c.shared)
        ]
        idle = [row for row in idle if row[1] > 0]
        if not idle:
            return False
        component = self.components[min(idle)[2]]
        regions = component.regions
        if need >= 0 and not component.shared:
            # Its last blocks first: the stage that next reads it begins with its first.
            regions, given = [], 0
            for region in reversed(
                [component.regions[i] for i in component.order] or component.blocks
            ):
                if given >= need:
                    break
                if region.bound is not None:
                    regions.append(region)
                    given += region.span
            regions = regions or component.regions
        self.quiesce()
        close_cursor(component)
        before = max(plane.stats(self.plane).device(self.index).committed, 0)
        try:
            shared = self._release_shared(component, sorted(component.shared))
            own = int(self.plane.drop(component.ws, self.index, [r.index for r in regions]))
        except plane.error("LeaseViolation"):
            self.drain()  # something reads it after all: what did leave is unbound
            return False
        for region in regions:
            unbind(region)
            _unbaked(region)
        if own:
            # Dropped bytes stay the plane's idle chunks until its budget leaves them no room.
            # From what was committed before (the drop's own trim may have released some), and
            # never above the budget a spare lowered; torch's limit follows either way.
            self._apply(min(self.applied, max(before - own, 0)))
            self.limit_torch()
        if shared:
            free = self.allocation()["driver_free_bytes"] - self.floor_free
            self.memory.ask(max(free, 0) + shared - weight_policy.MARGIN)
        self.paged_out[0] += 1
        self.paged_out[1] += own + shared
        return True

    def room_before(self, method: str) -> None:
        """Under a cap, before a stage plans: when other processes on the GPU leave less free
        than this stage's measured activations need of what the cap grants, ask the owner to
        make room first (`DeviceRoom`: idle tenants leave). Activations cannot shrink once
        the stage runs; a stage never measured asks nothing. A process over its cap gives
        back its own cached segments and idle maps first, then asks for the rest."""
        if self.room is None or self.cap < 0 or self.host_only:
            return
        memory, held = self._held()
        if held > self.cap:
            accel.release_cached(self.torch, self.kind)
            memory, held = self._held()
            if held > self.cap:
                for residency in list(self.residencies):
                    residency.unpark()
                self.shrink(held - self.cap)
                self.drain()
                memory, held = self._held()
        slack = max(memory["reserved_bytes"] - memory["allocated_bytes"], 0)
        # an adaptive stage fits itself to the room it finds: it asks for its smallest piece
        need = self.adaptive.get(method, self.activation.get(method, 0)) - slack
        if held > self.cap:
            # Still over: the owner raises the cap into the room it makes.
            self.memory.ask(max(need, 0) + weight_policy.MARGIN)
            return
        granted = min(need + weight_policy.MARGIN, self.cap - held)
        if need > 0 and memory["driver_free_bytes"] - self.floor_free < granted:
            self.memory.ask(granted)

    def _held(self) -> tuple[Mapping[str, int], int]:
        """Torch's allocation figures, and what this process holds on the device: its context,
        torch's reserved segments and the plane's maps."""
        memory = self.allocation()
        committed = max(plane.stats(self.plane).device(self.index).committed, 0)
        return memory, max(self.context_bytes, 0) + memory["reserved_bytes"] + committed

    def headroom(self, memory: Mapping[str, int]) -> int:
        """Device bytes this process may still take: the driver's free above the floor and,
        under a cap, the cap less what the process holds (context, torch, the plane's maps)."""
        free = memory["driver_free_bytes"] - self.floor_free
        if self.cap >= 0:
            committed = max(plane.stats(self.plane).device(self.index).committed, 0)
            held = max(self.context_bytes, 0) + memory["reserved_bytes"] + committed
            free = min(free, self.cap - held)
        return max(free, 0)

    def sizing(self) -> int:
        """The largest measured activations a grant must leave room for: an adaptive stage
        makes its own room, all but its smallest piece's."""
        sized = (v for m, v in self.activation.items() if m not in self.adaptive)
        return max((*sized, *self.adaptive.values()), default=0) or max(
            self.activation.values(), default=0
        )

    def _apply(self, nbytes: int, keep_holds: bool = False) -> int:
        """The plane's budget; lowered, the stages' holds go first unless `keep_holds`."""
        nbytes = max(nbytes, 0)
        if nbytes == self.applied or self.host_only:
            return 0
        if nbytes < self.applied:
            self.quiesce()
            for component in () if keep_holds else self.components.values():
                let_go(component)
        trim = self.plane.set_vram_budget(self.index, nbytes)
        self.applied = nbytes
        self._evicted(trim.get("evicted", ()))
        self.limit_torch()
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
        self.synchronize()

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
            granted = self.budget if own is None else self.budget + max(self.sizing() - own, 0)
            if self.cap < 0 and not self.floor_free:
                return granted
            return min(granted, self.driver_budget(method))
        driver = self.driver_budget(method)
        return min(driver, self.ceiling) if self.ceiling >= 0 else driver

    def driver_budget(self, method: str) -> int:
        """What the driver has free beside this process's weights, inside its cap and above its
        floor, less the activation growth this method measured (activations first)."""
        memory = self.allocation()
        committed = max(plane.stats(self.plane).device(self.index).committed, 0)
        # torch reuses its own cached segments first; only growth beyond them needs the driver
        slack = max(memory["reserved_bytes"] - memory["allocated_bytes"], 0)
        growth = max(self.activation.get(method, 0) - slack, 0)
        # the bound weights' decoded copies are the weights' own, in torch's bytes
        decoded = self.decoded()
        held = committed + decoded
        available = self.headroom(memory) + held
        # Activations first, but never below the open stages' floor while that fits: an
        # estimate that left a stage no room for its own weights would refuse it for size, and
        # it recovers from its out-of-memory by unmapping. An adaptive stage keeps its own
        # weights and its model's components placed whole as far as its smallest piece still
        # fits beside them: pieces cut no finer than that (SeedVR2's DiT, one latent frame).
        keep = self.window()
        if method in self.adaptive and self.open:
            residency, _, local = self.open[-1]
            least = self.adaptive[method] + weight_policy.MARGIN
            keep = max(keep, min(residency.kept(local), available - least))
        budget = max(weight_policy.plane_budget(available, growth), min(keep, available))
        self.plan_inputs[method] = {
            "driver_free": memory["driver_free_bytes"],
            "reserved": memory["reserved_bytes"],
            "allocated": memory["allocated_bytes"],
            "committed": committed,
            "decoded": decoded,
            "activation": self.activation.get(method, 0),
            "growth": growth,
            "cap": self.cap,
            "context": self.context_bytes,
            "budget": budget,
        }
        return budget

    def window(self) -> int:
        """What the open stages stream in at least: each component's common weights and two
        of its largest regions (one in use, the next arriving), at its current grain."""
        return sum(
            c.planned.common + 2 * c.planned.largest
            for _, _, c in self.opened().values()
            if not c.spared
        )

    def decoded(self) -> int:
        """Device bytes the bound regions' decoded copies hold now."""
        return sum(
            int(t.numel() * t.element_size())
            for component in self.components.values()
            for region in component.regions
            for t in region.derived
        )

    def shrink(self, need: int, keep: int = 0, keep_holds: bool = False) -> int:
        """Out of memory: give torch `need` more bytes by unmapping plane bytes, least recently
        used first, as far as what is in use (leased, held by a cursor, a ring) and never below
        `keep`: never a planning floor, which counts regions no kernel reads now. The cut is
        from what is mapped, which a budget larger than the model never reaches; the plane
        holds it until the Worker's next grant, and a stage planned later never below the open
        stages' floor. `keep_holds`: only weights no stage holds (memory rung 1)."""
        target = max(min(self.applied, self._committed()) - need - weight_policy.MARGIN, keep)
        if target >= self.applied:
            return 0
        self.oom_retries += 1
        freed = self._apply(target, keep_holds)
        # What the plane could not unmap is in use: the budget holds it, so the next map has
        # exactly the room torch leaves (`grow`), not a deficit.
        self._apply(max(self.applied, self._committed()), keep_holds)
        if self.budget >= 0:
            self.budget = min(self.budget, max(self.applied, self.floor()))
        return freed

    def _committed(self) -> int:
        return max(plane.stats(self.plane).device(self.index).committed, 0)

    def opened(self) -> dict[str, tuple[WeightResidency, str, Component]]:
        """Every open stage's components by weight set: their slot's residency and method."""
        return {
            row.components[name].key: (row, method, row.components[name])
            for row, method, local in self.open
            for name in local
        }

    def floor(self) -> int:
        """The plane bytes the open stages cannot run below: each component at its finest
        grain, which it moves to at its next block boundary (`WeightResidency._refine`)."""
        return sum(
            weight_policy.floor(component.planned)
            for _, _, component in self.opened().values()
            if not component.spared
        )

    @contextmanager
    def recovering(self, journal: Journal | None = None) -> Iterator[Journal | None]:
        """Run the caller's torch ops with out-of-memory recovery: an op that cannot allocate
        runs again once room is made (a seeded op from its generator's state before it). With
        a `journal`, every write into a tensor that existed before is kept first (`Journal`).
        Inside this plane's recovery it adds no dispatch layer (`_noting`). Leaving the
        outermost, what is out of the device comes back while there is room: only an op in
        this mode brings it back by itself."""
        if self.host_only:
            yield journal
            return
        outermost = _innermost(self.torch) is None
        with _noting(self.torch, self, journal):
            yield journal
        if outermost and spill.OFF and not self.open:  # a call's, not a block's
            self.restore_spilled()

    def refused(self, need: int) -> None:
        """torch's allocator refused `need` bytes. A caller that catches that never reaches a
        block's recovery: cuDNN tries its next convolution algorithm, down to one that needs
        no workspace (4-7x slower for SeedVR2's VAE convolutions on an RTX 4060 Ti), and torch
        keeps that choice for the shape as long as the process lives. So the room is made
        before that next attempt, from weights no stage holds (rungs 0-1; the owner is not
        asked from inside torch's allocator). A refusal that reaches the caller is the op's."""
        if self.placing:
            return
        self.memory.asking = False
        try:
            self.memory.free_for(need, "an allocation", upto=Rung.IDLE)
        finally:
            self.memory.asking = True

    def trim_cache(self, method: str) -> None:
        """Before a stage plans: torch's idle cache beyond the stage's activations goes back to
        the driver when the stage's weights still to map do not fit beside it (run 5937: a
        decode's 5.5 GiB left a transformer a 0.5 GiB budget). Otherwise it stays, and the next
        decode reuses it: given back, every request mapped and unmapped a VAE decode's 5 GiB
        again (45 ms on an RTX 5090, seconds on a host whose driver maps slowly, runs 9466-9479)."""
        if self.host_only:
            return
        memory = self.allocation()
        idle = memory["reserved_bytes"] - memory["allocated_bytes"]
        if idle <= self.activation.get(method, 0) + weight_policy.MARGIN:
            return
        unmapped = sum(
            max(c.ws.nbytes - self.resident_bytes(c), 0) for _, _, c in self.opened().values()
        )
        if unmapped + weight_policy.MARGIN > self.headroom(memory):
            accel.release_cached(self.torch, self.kind)

    def spill(self, need: int, operands: Sequence[object], disk: bool = False) -> bool:
        """Memory rungs 4 and 5 (`disk`): this process's own tensors leave the device (`spill`),
        the op's operands last. Mapped to page-locked host memory while the host can give it
        (`lockable`); those no op touched since their stage began out to host memory where
        nothing can be mapped (unified memory, a host that locks no more), and with `disk` to
        disk once the host can give no more. Whether any moved."""
        if self.host_only:
            return False
        if "spilled" not in self.modes:  # a new attempt: its own moves from here
            self.spilled_from = list(spill.TOTALS)
        want = need + weight_policy.MARGIN
        room = self.lockable(want)
        with self.torch.utils._python_dispatch._disable_current_modes():
            moved = spill.spill(
                self.torch, self._device(), want, operands, room, not self.unified, disk
            )
        accel.release_cached(self.torch, self.kind)
        self.limit_torch()
        self.modes["spilled"] = spill.summary(self.spilled_from)
        return moved > 0

    def _device(self) -> Device:
        return cast("Device", self.torch.device(self.device.type, self.index))

    def pin(self, operands: Sequence[object]) -> list[Lease]:
        """Leases on the weight regions an op that ran out of memory reads at their homes: the
        room made for it never unmaps them, and one the allocator's own refusal unmapped maps
        again at its home (it never moves), where the op's tensors point."""
        pointers, held = list(_pointers(operands)), cast("list[Lease]", [])
        try:
            for component in self.components.values() if pointers else ():
                cursor = component.cursor
                ring = (
                    range(cursor.ring_ptr, cursor.ring_ptr + cursor.ring_nbytes) if cursor else ()
                )
                for region in component.regions:
                    homes = [region.bound] if region.bound is not None else list(region.views)
                    spans = [(a, a + region.span) for a in homes if a not in ring]
                    if not any(a <= p < b for a, b in spans for p in pointers):
                        continue
                    held.append(
                        self.memory.retry(
                            functools.partial(
                                self.plane.acquire,
                                component.ws,
                                self.index,
                                region.index,
                                self.stream(),
                            ),
                            f"{component.name}.{region.path}",
                        )
                    )
                    if region.bound != int(held[-1].ptr):
                        bind(self.torch, region, held[-1])
        except BaseException:
            release_all(held)
            raise
        return held

    def spare(self, operands: Sequence[object]) -> bool:
        """Memory rung 2: the open stages' components with no block in its forward and none of
        `operands` give their weights back (`WeightResidency._spare`), never the one whose block
        ran last: its own code between its blocks (or a sub-block grain's parts) may hold its
        weights, and a kernel would be handed them unmapped (runs 11440, 11536)."""
        if not self.open:
            return False
        pointers = list(_pointers(operands))
        reads = {
            component.key
            for component in self.components.values()
            for region in component.regions
            if region.bound is not None
            and any(region.bound <= p < region.bound + region.span for p in pointers)
        }
        keep, residency = {*self.running, *reads}, self.open[-1][0]
        return residency._spare({*keep, self.last})

    def restore_spilled(self) -> int:
        """Tensors an earlier op moved off the device come back while it has room beside the
        floor and the cap: left mapped, a cache or a decoder's state is read over the bus at
        every later step (runs 10143-10144: a drafted decode at 90 ms/token instead of 18; a
        prefill 3x slower). Called as a request begins, before its stages plan."""
        if self.host_only or not spill.OFF:
            return 0
        accel.release_cached(self.torch, self.kind)
        memory, held = self._held()
        room = memory["driver_free_bytes"] - self.floor_free - weight_policy.MARGIN
        if self.cap >= 0:
            room = min(room, self.cap - held - weight_policy.MARGIN)
        with self.torch.utils._python_dispatch._disable_current_modes():
            back = spill.restore(self.torch, self._device(), room)
        if back:
            self.limit_torch()
            self.modes["restored"] = f"{back} B from host"
        return back

    def lockable(self, want: int) -> int:
        """Host bytes this process may take for `want` (a move off the device, a banded op's
        result): what the host has beside a margin for everything else on it. A free locked
        block of its size is taken as it is; short of room, free blocks go back to the host
        first, then the pinned tier's weights, which the page cache and disk hold too.
        -1: unreadable."""
        if spill.reusable(want):
            return want
        room = self._lockable()
        if 0 <= room < want and self.kind == "cuda":
            spill.release(self.torch)
            room = self._lockable()
        if 0 <= room < want and self.pinned_applied > 0:
            self._pin(max(self.pinned_applied - (want - room), 0))
            room = self._lockable()
        return room

    def _lockable(self) -> int:
        """What the host has (its tightest cgroup's room, else `MemAvailable`) less its
        margin. -1: unreadable."""
        try:
            available = proctree.host_memory().available
        except proctree.ProcessTreeUnsupported:
            return -1
        return max(available - weight_policy.HOST_MARGIN, 0) if available >= 0 else -1

    def activation_room(self, whole: bool = False) -> tuple[int, int]:
        """What torch can allocate now, and the mapped weight bytes no open stage holds: the
        room an op's own tensors may have, idle weights unmapped first (`free_for`). `whole`:
        an op cut into pieces (`split`), and more pieces cost less than a model's weights read
        back every time its stages come round, so the open model's components placed whole
        are held too."""
        if self.open:
            residency, _, local = self.open[-1]
            if whole:
                local = (*local, *residency.whole())
            return residency.room(local)
        device = plane.stats(self.plane).device(self.index)
        movable = device.committed - device.leased_bytes - device.held_bytes
        return self.allocatable(), max(movable, 0)

    def free_for(self, need: int, whole: bool = False) -> None:
        """Activations first: idle weights unmap until torch can allocate `need` bytes, as far
        as `activation_room` counts them."""
        free, movable = self.activation_room(whole)
        if need > free and movable > 0:
            mapped = plane.stats(self.plane).device(self.index).committed
            self.shrink(need - free, max(mapped - movable, self.floor()))

    def allocatable(self) -> int:
        """Device bytes torch can allocate now: its headroom plus its own idle cache."""
        memory = self.allocation()
        return self.headroom(memory) + max(memory["reserved_bytes"] - memory["allocated_bytes"], 0)

    def reserve(self, owner: str, nbytes: int, shed: Callable[[int], int]) -> Reservation:
        """A hold on device bytes `owner` keeps for speed (an LM engine's slots), granted as
        far as there is room now (`Reservation.granted`). Under pressure `shed(need)` gives
        back what it can before any weight unmaps."""
        reservation = Reservation(self, owner, shed)
        self.reservations[reservation] = None
        reservation.resize(nbytes)
        return reservation

    def grant(self, nbytes: int) -> bool:
        """Whether `nbytes` more fit now beside the floor, components no stage uses paging out
        for them first. Never waits."""
        if self.host_only:
            return True
        want = nbytes + weight_policy.MARGIN
        if self.allocatable() >= want:
            return True
        return self.page_out(_in_use(self), want) and self.allocatable() >= want

    def shed(self, need: int) -> bool:
        """Reservations give back toward `need`, the youngest first. Whether any gave."""
        gave = False
        for reservation in reversed(list(self.reservations)):
            if self.allocatable() >= need + weight_policy.MARGIN:
                break
            gave = reservation.shed(need) > 0 or gave
        return gave

    @contextmanager
    def reading(self, standins: Sequence[Tensor], op: str) -> Iterator[Read]:
        """A weight read outside its own module's forward (a fused kernel handed a sibling's
        scale, a package reading `.weight`): its region is mapped and leased for the op, as
        its forward would, and leaves with it. Only inside its component's stage: anywhere
        else the read is refused, and no kernel is ever handed its empty pointer."""
        opened = self.opened()
        held: list[tuple[Component, Region, Lease]] = []
        mine = {id(o[2]) for s in standins if (o := getattr(s, "origin", None)) is not None}
        if mine & self.mapped:  # a mapping that maps its own weights again (run 2793)
            names = [
                f"{o[1].name}.{o[2].path}:{o[4]}"
                for s in standins
                if (o := getattr(s, "origin", None)) is not None and id(o[2]) in self.mapped
            ]
            raise _unbound_read(
                op,
                f"a weight read re-entered its own mapping ({', '.join(names)}; inside "
                f"{' > '.join(self.mapping_ops)})",
            )
        self.mapping += 1
        self.mapped |= mine
        self.mapping_ops.append(op)
        mapping = True
        try:
            bound_at_entry: set[int] = set()
            for standin in standins:
                origin: Origin | None = getattr(standin, "origin", None)
                if origin is None or origin[0] is not self:
                    raise _unbound_read(op, "the weight belongs to no plane of this process")
                _, component, region, _, name = origin
                row = opened.get(component.key)
                if row is None or row[2] is not component:
                    stages = sorted(c.name for _, _, c in opened.values())
                    raise _unbound_read(
                        op,
                        f"{component.name}.{region.path}:{name} is in no open stage "
                        f"(open: {', '.join(stages) or 'none'})",
                    )
                if region.bound is not None:
                    bound_at_entry.add(id(region))
                if region.bound is None and all(region is not other for _, other, _ in held):
                    held.append((component, region, row[0]._lease(component, region, op)))
            for _, region, lease in held:  # a re-plan for a later lease unbound earlier ones
                bind(self.torch, region, lease)
            real: dict[int, Tensor] = {}
            for standin in standins:
                _, component, region, table, name = getattr(standin, "origin")  # noqa: B009
                value = table[name]
                if value is None or type(value) is type(standin):
                    leased = any(region is other for _, other, _ in held)
                    raise _unbound_read(
                        op,
                        f"{component.name}.{region.path}:{name} lost its bytes before the op ran "
                        f"(region {region.index}, bound at entry {id(region) in bound_at_entry}, "
                        f"leased {leased}, {len(standins)} weight operands, "
                        f"{len(held)} regions leased)",
                    )
                real[id(standin)] = value.to(dtype=standin.dtype, device=standin.device)
            self.mapping, mapping = self.mapping - 1, False
            self.mapped -= mine
            self.mapping_ops.pop()
            yield Read(
                real, tuple((r.bound, r.bound + r.span) for _, r, _ in held if r.bound is not None)
            )
        finally:
            if mapping:
                self.mapping -= 1
                self.mapped -= mine
                self.mapping_ops.pop()
            for component, region, _ in held:
                if region.index not in component.resident:
                    unbind(region)
            release_all(lease for *_, lease in reversed(held))

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
                region = component.regions[int(index)]
                unbind(region)
                _unbaked(region)

    def measure_context(self) -> None:
        """What the driver charges this process beyond torch's segments and the plane's maps:
        the CUDA context and library workspaces TensorD must leave room for."""
        pid = os.getpid()
        device_id = self._uuid()
        row = accel.process_memories_by_device({pid}, self.kind, (device_id,))[device_id][pid]
        if row.state == "present":
            reserved = self.allocation()["reserved_bytes"]
            device = plane.stats(self.plane).device(self.index)
            # Shared regions (exported or attached) are the machine's to count, once: the
            # driver charges one to whichever process it picks, never this one's context.
            mapped = max(device.committed, 0) + max(device.shared_bytes, 0)
            self.context_bytes = max(row.bytes - reserved - mapped, self.context_bytes, 0)
            self.limit_torch()

    def adopt_context(self, measured: int) -> None:
        """The context this process's own driver row showed at device init (-1: none)."""
        if measured > self.context_bytes:
            self.context_bytes = measured
            self.limit_torch()

    def resident_bytes(self, component: Component) -> int:
        return plane.stats(self.plane).ready(component.key, self.index)

    # -- GPU allocations retained by TensorD through exported handles (`weights.attach/1`)

    def _shares(self) -> bool:
        return (
            self.sharing is not None
            and not self.host_only
            and bool(self.plane.device_sharing(self.index))
        )

    def _uuid(self) -> str:
        return dict(self.plane.devices).get(self.index, "")

    def content(self, tensor: Tensor) -> str | None:
        """The layout digest of the weight set whose plane bytes `tensor` views: what names
        its content to another executor. None for a decoded or computed tensor."""
        if not getattr(tensor, "is_cuda", False) or type(tensor).__name__ == "Unbound":
            return None
        pointer = int(tensor.data_ptr())
        for component in self.components.values():
            for region in component.regions:
                if (
                    region.bound is not None
                    and region.bound <= pointer < region.bound + region.span
                ):
                    return str(component.ws.digest)
        return None

    def attach(self, component: Component) -> int:
        """Use this executor's plane to map GPU allocations TensorD retains for the layout.
        Attaching is read-only and copies no payload; unavailable regions fill privately
        under the executor's budget. Returns the bytes attached."""
        if self.sharing is None or not self._shares():
            return 0
        attached = self._attach(component, "")
        if any(region.bake for region in component.regions):
            # Baked bytes are a variant of the layout: the machine names the one it holds, and
            # the first forward that reads a region says whether it is its own (`_bake`).
            attached += self._attach(component, "*")
        return attached

    def _attach(self, component: Component, variant: str) -> int:
        assert self.sharing is not None
        answer = self.sharing.ask(component.key, component.ws.digest, self._uuid(), variant)
        self.variants = self.variants or answer.variants
        if not answer.held:
            return 0
        attached, at = 0, 0
        try:
            for row in answer.regions:
                fds = answer.fds[at : at + len(row.chunks)]
                at += len(row.chunks)
                if not 0 <= row.region < len(component.regions):
                    continue
                region = component.regions[row.region]
                if region.bake != bool(variant) or row.region in component.shared:
                    continue  # stored bytes where a bake is due, or baked ones where none is
                graph = variant and not any(bake.hooks for bake in region.bakes)
                if graph and _variant(region, list(map(_bake_key, region.bakes))) != answer.variant:
                    continue  # only a prepared graph bakes here, always the same: known now
                try:
                    # The plane maps only when the holder's layout is this weight set's.
                    self.plane.attach_device(
                        component.ws,
                        self.index,
                        row.region,
                        list(zip(row.chunks, fds, strict=True)),
                        answer.layout,
                    )
                except plane.error("Poisoned"):
                    raise
                except plane.error("PlaneError"):
                    continue
                component.shared.add(row.region)
                attached += region.span
                if variant:
                    region.baked, region.baked_with, region.dirty = True, [], False
                    region.attached = answer.variant
        finally:
            for fd in answer.fds:
                os.close(fd)
            if attached:
                component.leases.append(answer.lease_fd)
            else:
                os.close(answer.lease_fd)
        return attached

    def share(self, components: Iterable[Component] | None = None) -> int:
        """Offer the machine every region on the GPU not shared yet (of `components`, else
        all), so it outlives this process and other executors attach it. Returns the bytes
        the machine kept."""
        if self.sharing is None or not self._shares():
            return 0
        self.quiesce()
        kept = 0
        offers = [
            (component, variant, indexes)
            for component in list(self.components.values() if components is None else components)
            for variant, indexes in _variants(component).items()
        ]
        # Stored bytes first: their answers say whether the machine keeps variants at all.
        for component, variant, indexes in sorted(offers, key=lambda offer: offer[1] != ""):
            if variant and not self.variants:
                continue
            exported = self.plane.export_device(component.ws, self.index, indexes)
            if not exported:
                continue
            regions = tuple(
                SharedRegion(region=row["region"], chunks=tuple(n for n, _ in row["chunks"]))
                for row in exported
            )
            fds = tuple(fd for row in exported for _, fd in row["chunks"])
            indexes = [row.region for row in regions]
            component.shared.update(indexes)
            try:
                answer = self.sharing.offer(
                    component.key, component.ws.digest, self._uuid(), regions, fds, variant
                )
            finally:
                for fd in fds:
                    os.close(fd)
            self.variants = self.variants or answer.variants
            if answer.ok and not answer.duplicate:
                if answer.lease_fd >= 0:
                    component.leases.append(answer.lease_fd)
                kept += sum(component.regions[i].span for i in indexes)
                continue
            # Not kept: ours goes (it is outside the budget now); a duplicate is replaced by
            # the copy the machine keeps, any other region fills privately when next used.
            self._release_shared(component, indexes)
            if answer.duplicate:
                self.attach(component)
        return kept

    def set_lora(self, strategy: lora_bake.Strategy) -> None:
        """Serve LoRA'd linears by `strategy` from their next placement on. A region whose
        resident bytes are the other form is dropped: it refills from the unmodified host
        copy and its next placement bakes, or not."""
        if strategy == self.lora:
            return
        self.lora = strategy
        for row in list(self.residencies):
            row.unpark()  # a repeating stage left open holds its leases
        self.quiesce()
        for component in self.components.values():
            dropped = []
            for region in component.regions:
                region.bake = _bakes_here(self, component, region)
                held = region.bound is not None or region.index in component.shared
                if region.bakes and region.baked != region.bake and held:
                    unbind(region)
                    dropped.append(region.index)
                if not (region.baked and region.bake):
                    region.baked, region.baked_with = False, []
            if dropped:
                self._release_shared(component, [i for i in dropped if i in component.shared])
                self.plane.drop(component.ws, self.index, dropped)

    def refill(self, component: Component, regions: Sequence[Region]) -> None:
        """Drop regions' resident bytes (they are unleased): their next placement copies the
        unmodified host bytes again. Bytes mapped from the machine's custody are let go."""
        for region in regions:
            unbind(region)
        self.quiesce()
        self._release_shared(component, [r.index for r in regions if r.index in component.shared])
        own = [r for r in regions if r.index not in component.shared]
        if own:
            self.plane.drop(component.ws, self.index, [r.index for r in own])
        for region in own:
            _unbaked(region)

    def revoke(self, layout: str, regions: Collection[int] = ()) -> int:
        """Stop using the machine's regions of `layout`, all of them or only `regions`: queued
        work finishes, then they are unmapped and released. A later use fills a private copy.
        Returns the bytes let go."""
        self.quiesce()
        return sum(
            self._release_shared(
                component, sorted(component.shared & set(regions) if regions else component.shared)
            )
            for component in self.components.values()
            if component.shared and component.ws.digest == layout
        )

    def _release_shared(self, component: Component, indexes: Sequence[int]) -> int:
        if not indexes:
            return 0
        freed = int(self.plane.drop(component.ws, self.index, list(indexes)))
        for index in indexes:
            unbind(component.regions[index])
            _unbaked(component.regions[index])
            component.shared.discard(index)
        if not component.shared:
            _end_leases(component)
        return freed

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
        if self.host_only or self._all_sealed():
            return  # a CPU executor computes on its tier; a sealed one is the machine's RAM
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
        if self._all_sealed():
            # The machine's RAM, already resident: its budget is what this process registers.
            self._pin(self.pinned if self.pinned >= 0 else self._total())
        else:
            share = self.host_share()
            if self.pinned < 0:
                self._pin(max(share, 0))
            else:  # the Worker's budget, inside what the host has now
                self._pin(self.pinned if share < 0 else min(self.pinned, share))
        self._fill(component)

    def _all_sealed(self) -> bool:
        return all(c.sealed for c in self.components.values())

    def _total(self) -> int:
        return sum(c.ws.nbytes for c in self.components.values())

    def _bind_host(self, component: Component) -> None:
        """A CPU executor computes on its pinned tier: every region pinned there (never evicted
        implicitly) and bound in place, the pages shared with every executor of the layout.
        The tier holds every set it registered: past the host, the machine's layout pages."""
        self._pin(self._total())
        if self.fetch is not None:
            self.fetch.wait(component.manifest, _ordered(component.objects))
        self.plane.want(component.ws, plane.PINNED, None, priority=self.tick(), pin=True).wait()
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

    def _fill(self, component: Component) -> int:
        """Want the component pinned, as many regions as the tier holds: the tier is optional,
        so one larger than it pins a prefix and the rest reads from disk."""
        room, regions = self.pinned_applied, []
        for region in component.regions:
            if region.index in component.shared:
                continue  # on the GPU already, kept by the machine
            if region.span > room:
                break
            room -= region.span
            regions.append(region.index)
        # One ticket per region: the tier is shrinkable, and a ticket over several reports the
        # first region the tier gave back instead of waiting for the others (run 4365).
        priority, asked = self.tick(), 0
        for index in regions:
            try:
                ticket = self.plane.want(component.ws, plane.PINNED, [index], priority=priority)
            except (plane.error("Shortfall"), plane.error("BelowFloor")):
                break
            component.filling.append(ticket)
            asked += component.regions[index].span
        return asked

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
        component.shared.clear()
        for ticket in component.filling:
            # A filling tier refuses to close. A region the tier gave back since, or whose
            # fill failed, is not filling: either way the set can close.
            with suppress(plane.error("PlaneError")):
                ticket.wait()
        component.filling.clear()
        try:
            component.ws.close()
        except plane.error("LeaseViolation"):
            # device copies still read its pinned tier (the plane's own stream): a rare
            # close (refine, unload) waits for the device, then closes
            self.synchronize()
            self.plane.set_vram_budget(self.index, self.applied)
            component.ws.close()
        _end_leases(component)  # after the close released every shared region

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
        copied = device.host_copy_bytes + device.bounce_copy_bytes + device.disk_copy_bytes
        reserved = (
            -1
            if self.host_only
            else self.allocation()["reserved_bytes"]
            if self.unified
            else int(self.torch.cuda.memory_reserved(self.device))
        )
        # This process's own count (per-process NVML only refines the context in it); an
        # unmeasured context counts nothing rather than hiding the rest.
        process = (
            max(self.context_bytes, 0) + reserved + max(device.committed, 0)
            if reserved >= 0
            else -1
        )
        copy_ns = device.host_copy_ns + device.bounce_copy_ns + device.disk_copy_ns

        def gbps(nbytes: int, ns: int) -> float:
            return round(nbytes / ns, 3) if ns else 0.0

        allocator = {} if self.kind != "cuda" else self.torch.cuda.memory_stats(self.device)
        return PlaneFacts(
            budget_bytes=device.budget,
            committed_bytes=device.committed,
            leased_bytes=device.leased_bytes + device.held_bytes,
            pinned_budget_bytes=held.host.budget,
            pinned_bytes=held.host.used,
            context_bytes=self.context_bytes,
            reserved_bytes=reserved,
            process_bytes=process,
            cap_bytes=self.cap,
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
            h2d_gbps=gbps(copied, copy_ns),
            disk_copy_bytes=device.disk_copy_bytes,
            mapped_copy_bytes=device.mapped_copy_bytes,
            bounce_copy_bytes=device.bounce_copy_bytes,
            host_gbps=gbps(device.host_copy_bytes, device.host_copy_ns),
            bounce_gbps=gbps(device.bounce_copy_bytes, device.bounce_copy_ns),
            disk_gbps=gbps(device.disk_copy_bytes, device.disk_copy_ns),
            register_failed=held.host.register_failed,
            plan_inputs={method: dict(row) for method, row in self.plan_inputs.items()},
            shared_bytes=device.shared_bytes,
            disk_read_bytes=held.host.disk_read_bytes,
            sealed_bytes=held.host.sealed_bytes,
            lora_baked=int(BAKED[0]),
            lora_bake_ms=round(_bake_ms(), 1),
            lora_attached=int(BAKED[2]),
            late=sum(row.late for row in stages),
            stall_ns=sum(row.stall_ns for row in stages),
            misses=device.misses,
            evictions=device.evictions,
            oom_retries=self.oom_retries,
            alloc_retries=int(allocator.get("num_alloc_retries", 0)),
            alloc_ooms=int(allocator.get("num_ooms", 0)),
            cache_releases=sum(count for count, _ in accel.RELEASED.values()),
            cache_released_bytes=sum(freed for _, freed in accel.RELEASED.values()),
            cache_released_by={who: (n, freed) for who, (n, freed) in accel.RELEASED.items()},
            squeezed=dict(self.squeezes),
            paged_out=self.paged_out[0],
            paged_out_bytes=self.paged_out[1],
            spilled=spill.TOTALS[0],
            spilled_bytes=spill.TOTALS[1],
            modes={**self.modes, **lora_bake.mode()},
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


def bind(
    torch: ModuleType,
    region: Region,
    lease: Lease | UnifiedLease | _HostSpan,
    *,
    refill: bool = False,
) -> bool:
    """Point the region's destinations at the lease's bytes: its home while resident, its place
    in the stage's ring while streamed. Views are cached by address (ring placements repeat
    with the window); decoded tensors and derived leaf roles are recomputed whenever the bytes
    moved or were streamed in again. Returns whether it baked resident bytes (`_bake`)."""
    address = int(lease.ptr)
    views = region.views.get(address)
    if views is None:
        if len(region.views) > weight_policy.MAX_WINDOW:
            del region.views[next(a for a in region.views if a != region.bound)]
        view = lease.view()  # DLPack, or a unified plane's own tensor
        base = view if isinstance(view, torch.Tensor) else torch.from_dlpack(view)
        owned = isinstance(lease, UnifiedLease)
        views = region.views[address] = _views(torch, region, base, owned)
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
    baked = bool(region.bakes) and _bake(
        torch, region, views.parts, lease.ring_offset is None, refill
    )
    if (moved or lease.ring_offset is not None) and region.transforms:
        region.derived = _transform(torch, region, views.parts)
    region.bound = address
    return baked


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
                raise _unbound_read(op, "no weight operand carries its plane origin")
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
                # a write would be lost with the mapping
                raise _unbound_read(str(func), "the op writes into the weight")
            with cls.reading(standins, str(func)) as read:
                return read.kept(
                    torch,
                    func(
                        *(read.operand(v) for v in args),
                        **{k: read.operand(v) for k, v in kwargs.items()},
                    ),
                )

    return Unbound


def _unbound_read(op: str, why: str = "") -> ResidencyRefusal:
    detail = f"{op}: a weight is read while no plane bytes are bound to it (outside its stage)"
    return ResidencyRefusal("weight_unbound", f"{detail}: {why}" if why else detail)


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

    def kept(self, torch: ModuleType, out: object) -> object:
        """The op's result, without a view into bytes that leave with the op's leases. The
        copy is made outside inference mode: it stands for a view of the stand-in, and torch
        refuses an inference tensor as the view of a normal one ('Cannot set version_counter
        for inference tensor', runs 11026 and 11034)."""
        if isinstance(out, (tuple, list)):
            return type(out)(self.kept(torch, item) for item in out)
        viewed = any(
            a <= int(tensor.data_ptr()) < b
            for tensor in spill.stored((out,))
            if (tensor.is_cuda or tensor.is_mps) and tensor.numel()
            for a, b in self.spans
        )
        if not viewed:
            return out
        with torch.inference_mode(False):
            return cast("Tensor", out).clone()


#: the CUDA planes of this process, which torch's one out-of-memory observer answers for
_REFUSALS: weakref.WeakSet[Weights] = weakref.WeakSet()
_observing = False


def _observe_refusals(torch: ModuleType, weights: Weights) -> None:
    global _observing
    _REFUSALS.add(weights)
    if _observing:
        return
    _observing = True

    def refused(device: int, need: int, _limit: int, _free: int) -> None:
        # Called inside torch's allocator, right before it raises: nothing may escape it.
        with suppress(Exception):
            for held in list(_REFUSALS):
                if held.index == device:
                    held.refused(need)

    torch._C._cuda_attach_out_of_memory_observer(refused)


_OPS: dict[int, type] = {}


class _Ops(Protocol):
    weights: Weights | None
    journals: list[Journal]


def _innermost(torch: ModuleType) -> _Ops | None:
    """The process's dispatch mode (`_ops_type`) when it is the innermost one."""
    mode = torch.utils._python_dispatch._get_current_dispatch_mode()
    return cast("_Ops", mode) if type(mode) is _OPS.get(id(torch)) else None


@contextmanager
def _noting(torch: ModuleType, weights: Weights | None, journal: Journal | None) -> Iterator[None]:
    """Torch ops in this context recover (with `weights`) and are noted in `journal`. Directly
    inside the process's dispatch mode (`_ops_type`), this joins it: every op still passes one
    Python layer. A group GPU's block op passed three (the invoke's recovery, the call's
    journal, the block's): +29 us per op, 0.16-0.5 s of CPU per H3 forward."""
    kind = _OPS.get(id(torch))
    if kind is None:
        kind = _OPS[id(torch)] = _ops_type(torch)
    mode = _innermost(torch)
    if mode is None or weights not in (None, mode.weights):
        with kind(weights), _noting(torch, None, journal):
            yield
        return
    if journal is not None:
        mode.journals.append(journal)
    try:
        yield
    finally:
        if journal is not None:
            mode.journals.remove(journal)


def _ops_type(torch: ModuleType) -> type:
    mode: type = torch.utils._python_dispatch.TorchDispatchMode
    seeded = torch.Tag.nondeterministic_seeded
    pointwise = torch.Tag.pointwise
    #: op -> whether it draws from a generator, writes in place, is pointwise
    traits: dict[Any, tuple[bool, bool, bool]] = {}

    class Ops(mode):  # type: ignore[misc]
        """Every torch op, in one dispatch layer. With `weights`, one that runs out of device
        memory runs again after weights were unmapped for it (retry the op, never the run): a
        seeded op from the generator state it started from, one that writes in place from a
        copy of what it writes into (never a pointwise op, which allocates nothing and so
        fails before it writes: a whole video's `div_` was copied, doubling it). Each open
        journal (a group call's, a block's) keeps what it writes into first and notes what it
        makes and the generators it draws from; a call's journal also marks device progress."""

        def __init__(self, weights: Weights | None) -> None:
            super().__init__()
            self.weights = weights
            self.journals: list[Journal] = []

        @classmethod
        def _should_skip_dynamo(cls) -> bool:
            return False  # nothing compiles here: no Dynamo wrapper around every op

        def __torch_dispatch__(
            self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
        ) -> Any:
            kwargs = kwargs or {}
            if spill.OFF:  # a storage out of the device comes back before an op touches it
                retry = self.weights.memory.retry if self.weights is not None else None
                spill.fault(torch, (*args, *kwargs.values()), retry)
            out = self._run(func, args, kwargs)
            if spill.PRESSED[0]:
                spill.touch(torch, (*args, *kwargs.values(), out))
            for journal in self.journals:
                journal.made(out)
                if journal.marking:
                    journal.ops += 1
                    if journal.ops % MARK_EVERY == 0 and torch.cuda.is_initialized():
                        mark = torch.cuda.Event()
                        mark.record()
                        journal.marks.append(mark)
            return out

        def _run(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            weights, journals = self.weights, self.journals
            banded = weights.bands if weights is not None else None
            if banded is not None:
                try:
                    out = banded.run(func, args, kwargs)
                except Exception as exc:
                    if not is_device_oom(exc):
                        raise
                    exc.__traceback__ = None
                    out = NotImplemented  # it recovers as any op does, below
                if out is not NotImplemented:
                    return out
            known = traits.get(func)
            if known is None:
                tags = func.tags
                known = traits[func] = (seeded in tags, func._schema.is_mutable, pointwise in tags)
            draws, writes, pure = known
            states: list[tuple[Any, Any]] = []
            if draws:
                states = [(g, g.get_state()) for g in _generators(torch, func, args, kwargs)]
                for journal in journals:
                    journal.draws(states)
            targets: list[Tensor] = []
            if writes:
                targets = list(_tensor_operands(tuple(_written(func, args, kwargs))))
            written: list[tuple[Tensor, Tensor]] | None = None
            seen, pins = -1, cast("list[Lease]", [])
            try:
                while True:
                    try:
                        if written is None:  # the copies are allocations too: they recover first
                            for journal in journals if targets else ():
                                journal.before_write(targets)
                            copied = weights is not None and not pure
                            written = [(t, spill.keep(t)) for t in targets] if copied else []
                        return func(*args, **kwargs)
                    except Exception as exc:
                        if weights is None or (short := shortage(exc)) is None:
                            raise
                        why = str(exc).splitlines()[0][:200] if str(exc) else ""
                        up = exc if is_device_oom(exc) and bands.raises() else None
                        exc.__traceback__ = None  # the failed op's frames free their tensors
                    # outside `except`: it runs again from what it read, its generators and
                    # the tensors it writes into as they were
                    for generator, state in states:
                        generator.set_state(state)
                    for target, first in written or ():
                        target.copy_(first)
                    operands, op, memory = (*args, *kwargs.values()), str(func), weights.memory
                    need = short[0] or sum(t.nbytes for t in spill.stored(operands))
                    pins = pins or weights.pin(operands)
                    if (now := memory.again(need, op, operands, seen, Rung.IDLE)) >= 0:
                        seen = now
                        continue
                    if up is not None:  # the call around it cuts itself (`Bands.attend`) first
                        raise up
                    if banded is not None:
                        try:  # still short: in bands of rows when it can be
                            out = banded.run(func, args, kwargs, short=True)
                        except Exception as exc:
                            if not is_device_oom(exc):
                                raise
                            exc.__traceback__ = None
                            out = NotImplemented
                        if out is not NotImplemented:
                            return out
                    if (seen := memory.again(need, op, operands, seen)) < 0:
                        shapes = ", ".join(
                            f"{t.dtype}{tuple(t.shape)}" for t in spill.stored(operands)
                        )
                        dtype = kwargs.get("dtype")
                        memory.exhausted(
                            need, f"{op}({shapes}{f' -> {dtype}' if dtype else ''})", why
                        )
            finally:
                release_all(pins)

    return Ops


@dataclass(frozen=True, slots=True)
class Effects:
    """What a block's forward changes outside itself, as its first run showed: whether it
    writes into tensors that existed before it (its inputs, its buffers, anything held
    elsewhere), the generators its seeded ops draw from, and whether the bands rewrote an op
    of it (`Bands.rewrites`: a convolution past its padding's zeros, an op cut in rows). A
    forward that changes nothing else runs with no dispatch layer and, out of memory, again
    from its generators' states. One that writes, that the bands rewrote, or that changes a
    plain attribute of a module in it (`state`: HiDiffusion's step counters) keeps its ops'
    recovery; one that does not write runs again from those attributes too."""

    writes: bool
    generators: tuple[Any, ...]
    banded: bool = False
    #: the (module, attribute) pairs its first run changed
    state: tuple[tuple[Module, str], ...] = ()


@dataclass(slots=True, eq=False)  # one is told from another by identity, never content
class Journal:
    """One observed run (a block's first, or a group GPU's whole call): the storages its ops
    made, the first content of every other storage they wrote into, and the generators its
    seeded ops drew from. `rewind` puts those back, so the run can start again from them."""

    made_here: set[int] = field(default_factory=set)
    kept: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    #: generator -> its state before this attempt first drew from it; every one drawn from
    first: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    drawn: dict[int, Any] = field(default_factory=dict)
    #: device progress of a journaled call (`journaling`): an event recorded every
    #: `MARK_EVERY` ops on a GPU, and how many of them the device has finished
    marking: bool = False
    ops: int = 0
    marks: deque[Any] = field(default_factory=deque)
    done: int = 0

    def progress(self) -> int:
        """Marks the device has passed: what a watcher reads, from its own thread, while the
        call runs. A GPU waiting on a collective that cannot complete passes none."""
        while self.marks and self.marks[0].query():
            self.marks.popleft()
            self.done += 1
        return self.done

    def made(self, out: object) -> None:
        for value in spill.stored((out,)):
            self.made_here.add(int(value.untyped_storage().data_ptr()))

    def before_write(self, written: Iterable[object]) -> None:
        for value in spill.stored(tuple(written)):
            storage = value.untyped_storage()
            key = int(storage.data_ptr())
            if key in self.made_here or key in self.kept:
                continue
            self.kept[key] = (storage, spill.keep(storage))

    def draws(self, states: Iterable[tuple[Any, Any]]) -> None:
        for generator, state in states:
            self.first.setdefault(id(generator), (generator, state))
            self.drawn.setdefault(id(generator), generator)

    def rewind(self) -> None:
        for storage, first in self.kept.values():
            storage.copy_(first)
        for generator, state in self.first.values():
            generator.set_state(state)
        self.kept.clear()
        self.made_here.clear()
        self.first.clear()

    def effects(self, banded: bool = False, state: tuple[tuple[Module, str], ...] = ()) -> Effects:
        return Effects(bool(self.kept), tuple(self.drawn.values()), banded, state)


def _class_forward(module: Module, *args: object, **kwargs: object) -> object:
    return type(module).forward(module, *args, **kwargs)


def _start(
    effects: Effects, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> Callable[[], None]:
    """What a block's run starts from, to put back before it runs again: the attributes its
    first run changed, its generators' states, the containers it was handed."""
    attrs = module_state.take(effects.state)
    kept = [(g, g.get_state()) for g in effects.generators]
    held = _containers(args, kwargs)

    def restore() -> None:
        module_state.put_back(attrs)
        for generator, state in kept:
            generator.set_state(state)
        _put_back(held)

    return restore


#: how often a journaled call marks its device progress, in ops (an event costs microseconds)
MARK_EVERY = 32


@contextmanager
def journaling(torch: ModuleType, journal: Journal) -> Iterator[Journal]:
    """Every torch op in this context writes into `journal`: the storages it makes, the first
    content of every other storage it writes into, the generator state a seeded op draws
    from, and the device progress its ops make (`Journal.progress`). One GPU of a group keeps
    one for a whole call, so the call can be rewound when the group ends it
    (`mirror.mirror_component`); its blocks keep none of their own. On any device."""
    journal.marking = True
    with _noting(torch, None, journal):
        yield journal


def _in_call(torch: ModuleType) -> bool:
    """Whether ops run in a group call's journal now (`journaling`)."""
    mode = _innermost(torch)
    return mode is not None and any(j.marking for j in mode.journals)


def _generators(
    torch: ModuleType, func: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> list[Any]:
    """The generators a seeded op draws from: its own, else its device's default."""
    found = []
    device = None
    for index, argument in enumerate(func._schema.arguments):
        value = args[index] if index < len(args) else kwargs.get(argument.name)
        if isinstance(value, torch.Generator):
            found.append(value)
        elif argument.name == "device" and value is not None:
            device = torch.device(value)
    if found:
        return found
    if device is None:
        tensor = next(_tensor_operands((*args, *kwargs.values())), None)
        device = getattr(tensor, "device", torch.device("cpu"))
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return [torch.cuda.default_generators[index]]
    if device.type == "mps":
        return [torch.mps._get_default_mps_generator()]
    return [torch.default_generator]


def _written(func: Any, args: tuple[object, ...], kwargs: Mapping[str, object]) -> Iterator[object]:
    """The arguments an op writes into (`Tensor(a!)` in its schema, `out=` included)."""
    for index, argument in enumerate(func._schema.arguments):
        if argument.alias_info is not None and argument.alias_info.is_write:
            yield args[index] if index < len(args) else kwargs.get(argument.name)


def _tensor_operands(operands: Sequence[object]) -> Iterator[Tensor]:
    """Tensor arguments/results of Torch ops, including one level of tensor sequences.
    The structural check keeps this module importable without importing Torch itself."""
    for value in operands:
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if hasattr(item, "untyped_storage"):
                yield cast("Tensor", item)


def _pointers(operands: Sequence[object]) -> Iterator[int]:
    """Device addresses of an op's tensor operands (one level deep)."""
    for tensor in spill.stored(operands):
        if tensor.is_cuda or tensor.is_mps:
            yield int(tensor.data_ptr())


def _unbaked(region: Region) -> None:
    """The region's bytes went: the next ones are the stored bytes, unmodified."""
    region.baked, region.baked_with, region.dirty, region.attached = False, [], False, ""


def unbind(region: Region) -> None:
    """Put the stand-ins back, keeping a dtype the component was converted to meanwhile."""
    for table, name, standin in region.unbound:
        dtype = getattr(table.get(name), "dtype", standin.dtype)
        table[name] = standin if dtype == standin.dtype else _retyped(standin, dtype)
    region.derived = []
    region.bound = None
    if any(views.owned for views in region.views.values()):
        region.views = {a: views for a, views in region.views.items() if not views.owned}


def _drop_copies(region: Region) -> None:
    """Unbind a region whose destinations a package's `module.to(dtype)` gave bytes of their
    own: torch's copies beside the plane's (SDXL's VAE upcast and cast back: 160 MiB kept
    across requests). Its next bind converts again."""
    bound = region.bound
    if bound is None:
        return
    views = region.views.get(bound)
    if views is not None and any(
        (held := table.get(name)) is not None and held.data_ptr() != tensor.data_ptr()
        for table, name, tensor, _ in views.writes
    ):
        del region.views[bound]  # a cached Parameter may hold a copy as its `.data`
        unbind(region)


def let_go(component: Component) -> None:
    """Release the stage's leases on the component's resident blocks; they stay bound, and
    each block leases its own again until the next plan holds them."""
    holds, component.holds = component.holds, {}
    release_all(holds.values())


def release_all(leases: Iterable[Lease]) -> None:
    """Release every lease, then raise the first failure: one a failure skipped would be
    collected unreleased, which poisons the plane (run 10996, a release under device OOM)."""
    failure: BaseException | None = None
    for lease in leases:
        try:
            lease.release()
        except BaseException as exc:
            failure = failure or exc
    if failure is not None:
        raise failure


def close_cursor(component: Component) -> None:
    """Let the stage's holds go (`let_go`) and close its cursor after dropping every view into
    its ring (one keeps the cursor open)."""
    let_go(component)
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
        # A view into the ring outlives the stage: a failed block's frames, in a cycle, hold
        # one (run 10562). The next stage short of room collects them first (memory's rung 1).
        component.orphans.append(cursor)


def _bake_key(bake: Bake) -> tuple[object, ...]:
    """What baking `bake` now would add: its graph, and each hook's current factors (None: the
    hook serves no update now)."""
    hooks = (hook.lora_factors() for hook in bake.hooks)
    return ("graph" if bake.layer is not None else None, *(g and (g[0], g[3]) for g in hooks))


def _bake(
    torch: ModuleType,
    region: Region,
    parts: list[dict[str, Tensor]],
    home: bool,
    refill: bool,
) -> bool:
    """Add each LoRA'd linear's delta into the region's bytes before any decode reads them;
    its update then skips. Resident bytes are baked the first time they are placed. A
    streamed copy is a fresh copy of the stored bytes at every acquire (each cursor position
    is its own copy), so it is baked as it arrives, the same way: one seed per weight makes
    every bake of a weight the same bytes, and outputs never depend on placement.

    Resident bytes holding a delta the coming forward does not want (a base forward of a DiT
    baked for its turbo overlay, or another overlay), or left half baked by a failure, raise
    `_Stale` where the caller can refill them (`refill`, a block's own lease); elsewhere the
    read sees the bytes as they are. Returns whether it baked resident bytes now."""
    keys = [_bake_key(bake) for bake in region.bakes]
    some = region.bake and any(key is not None for row in keys for key in row)
    if home and region.attached and some and _variant(region, keys) == region.attached:
        # Baked bytes from the machine's custody, this forward's own: its bake would be the
        # same variant. A placement ahead of its forward leaves the question to that forward.
        region.baked_with, region.variant, region.attached = keys, region.attached, ""
        BAKED[2] += len(region.bakes)
    if home and refill and _stale(region, keys):
        raise _Stale(region.index)
    bakes_now = some and (not home or not region.baked)
    if bakes_now and _factors_mapping(region, keys):
        # Placed inside the read that is mapping its overlay's factors (run 10549): they cannot
        # be read again from here. The region serves unbaked, its hooks adding the delta per call.
        for bake in region.bakes:
            if bake.layer is not None:
                bake.layer._cozy_baked = False
            for hook in bake.hooks:
                hook._cozy_baked = False
        return False
    now = home and some and not region.baked
    if now:
        region.dirty = True  # until every weight is baked
    if now or (some and not home):
        with _timed(torch, parts):
            BAKED[0] += _apply(torch, region, parts, keys)
    if now:
        region.baked, region.baked_with, region.dirty = True, keys, False
        region.variant = _variant(region, keys)
    baked = region.baked and keys == region.baked_with if home else some
    for bake, (graph, *hooks) in zip(region.bakes, keys, strict=True):
        if bake.layer is not None:
            bake.layer._cozy_baked = baked and graph is not None
        for hook, key in zip(bake.hooks, hooks, strict=True):
            hook._cozy_baked = baked and key is not None
    return now


def _factors_mapping(region: Region, keys: list[tuple[object, ...]]) -> bool:
    """Whether a factor some hook of the region would bake is a stand-in whose region a weight
    read in flight is mapping now."""
    for bake, row in zip(region.bakes, keys, strict=True):
        for hook, key in zip(bake.hooks, row[1:], strict=True):
            got = hook.lora_factors() if key is not None else None
            for factor in (got[1], got[2]) if got is not None else ():
                origin = getattr(factor, "origin", None)
                if origin is not None and id(origin[2]) in origin[0].mapped:
                    return True
    return False


def _stale(region: Region, keys: list[tuple[object, ...]]) -> bool:
    """Whether a forward that bakes `keys` finds the region's resident bytes holding a delta it
    does not want, or half of one. Baked bytes from the machine's custody are its own only when
    its bake would be their variant."""
    if region.attached:
        some = region.bake and any(key is not None for row in keys for key in row)
        return not (some and _variant(region, keys) == region.attached)
    return region.dirty or (region.baked and keys != region.baked_with)


@contextmanager
def _timed(torch: ModuleType, parts: list[dict[str, Tensor]]) -> Iterator[None]:
    """Count a bake's milliseconds into `BAKED`. On a GPU, events around it on its stream are
    read when facts are (`_bake_ms`): timing never waits for the queue."""
    if stored_device(parts[0]).type != "cuda":
        began = time.perf_counter()
        yield
        BAKED[1] += (time.perf_counter() - began) * 1000
        return
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    _TIMED.append((start, end))


def _bake_ms() -> float:
    """`BAKED`'s milliseconds, with every bake the GPU has finished since the last read."""
    while _TIMED and _TIMED[0][1].query():
        start, end = _TIMED.popleft()
        BAKED[1] += start.elapsed_time(end)
    return BAKED[1]


#: This process's bakes (one plane per process): weights baked, the GPU milliseconds the bakes
#: took (resident and streamed), and weights whose baked bytes came from the machine's custody.
BAKED: list[float] = [0, 0.0, 0]
_TIMED: deque[tuple[Any, Any]] = deque()


def _in_use(weights: Weights) -> set[str]:
    """The weight sets an open stage, or a repeating one left open, uses."""
    keys = set(weights.opened())
    for row in weights.residencies:
        if row.parked is not None:
            keys.update(row.components[name].key for name in row.parked[1])
    return keys


def _variant(region: Region, keys: list[tuple[object, ...]]) -> str:
    """The digest that names what baking `keys` into the region adds, the same in every
    executor: a prepared graph's factors are the layout's own weights, so their scales name
    it; a hook's are named by the weight set its factors are bytes of (`Weights.content`) and
    its scale. "" when a hook's factors are not plane bytes: such a bake is never shared."""
    rows: set[tuple[str, str]] = set()
    for bake, (graph, *hooks) in zip(region.bakes, keys, strict=True):
        if graph is not None:
            # an adapter's factors are `<site>.lora_A.<adapter>.weight`: its name and scale
            rows.update(
                ("graph:" + region.weights[a].row.name.rpartition(".lora_A.")[2], repr(scale))
                for a, _, scale in bake.factors
            )
        for hook, key in zip(bake.hooks, hooks, strict=True):
            got = hook.lora_factors() if key is not None else None
            if got is None:
                continue
            locate = region.locate
            content = {locate(got[1]), locate(got[2])} if locate is not None else {None}
            (named,) = content if len(content) == 1 else (None,)
            if named is None:
                return ""
            rows.add((named, repr(float(got[3]))))
    return canonical.digest([lora_bake.VARIANT, sorted(rows)]) if rows else ""


def _variants(component: Component) -> dict[str, list[int]]:
    """The component's regions the machine may keep, by what their bytes hold: "" for stored
    bytes, a variant for baked ones it can name. Bytes mapped from custody but not yet read
    are not this process's to offer."""
    found: dict[str, list[int]] = {}
    for region in component.regions:
        private = region.bake and not region.baked  # its next forward bakes into these bytes
        if region.dirty or region.attached or private or (region.baked and not region.variant):
            continue
        found.setdefault(region.variant if region.baked else "", []).append(region.index)
    return found


def _apply(
    torch: ModuleType,
    region: Region,
    parts: list[dict[str, Tensor]],
    keys: list[tuple[object, ...]],
) -> int:
    """Bake every LoRA'd weight of the region in place, a row chunk at a time: everything
    that updates a weight now (`keys`) summed, then rounded once. Returns the weights baked."""
    count = 0
    for bake, row in zip(region.bakes, keys, strict=True):
        downs, ups = [], []
        if bake.layer is not None:
            downs += [parts[a]["value"] for a, _, _ in bake.factors]
            ups += [parts[b]["value"] * scale for _, b, scale in bake.factors]
        for hook, key in zip(bake.hooks, row[1:], strict=True):
            got = hook.lora_factors() if key is not None else None
            if got is not None:
                downs.append(got[1])
                ups.append(got[2] * got[3])
        if not downs:
            continue
        count += 1
        kind = downs[0].dtype
        lora_bake.bake(
            torch,
            parts[bake.weight],
            torch.cat([d.to(kind) for d in downs]),
            torch.cat([u.to(kind) for u in ups], dim=1),
            seed=zlib.crc32(region.weights[bake.weight].row.name.encode()),
        )
    return count


def stored_device(stored: dict[str, Tensor]) -> Any:
    return next(iter(stored.values())).device


def _views(torch: ModuleType, region: Region, base: Tensor, owned: bool = False) -> Views:
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
    return Views(writes, parts, owned)


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
    regions with Runtime's TensorFS plane. Stages request GPU residency; CPU-only execution
    can fill/bind host buffers when those regions register."""

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
        #: shared CPU weight buffers: reuse TensorD's matching layout, or offer a buffer
        #: this executor filled through its TensorFS plane
        self.tiers = tiers
        self.torch = weights.torch
        self.construction = construction
        self.checkpoints = dict(checkpoints)
        self.plan = {row.key: row for row in rows}
        self.custody = custody
        self.device = weights.device
        self.kind = weights.kind
        #: the components in the order its stages first use them (`fetch_plan`): the order
        #: their weights are fetched in
        self.order: tuple[str, ...] = ()
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
        #: (component, provider) -> installed leaves, counted when they install (`gemm.selected`)
        self.gemm_sites: Counter[tuple[str, str]] = Counter()

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
        qualification = qualified(
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
            capabilities=qualification.capabilities,
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
            qualified=(qualification.capabilities, qualification.result),
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

    @property
    def bound(self) -> frozenset[str]:
        """The components the binding selects; the construction's others stay on meta."""
        return frozenset(self.checkpoints)

    def expect(self, sets: Mapping[str, Sequence[str]]) -> None:
        """Completeness sets: a destination the census never reaches refuses the commit."""
        self.expected = {name: frozenset(keys) for name, keys in sets.items()}

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
        self.gemm_sites = leaf_sites(self.leaves)
        prepared = {
            name: self._prepare_layout(name, root, keys)
            for name, root in self.roots.items()
            if (keys := [key for key in order if self.plan[key].component == name])
        }
        fetch = self.weights.fetch
        if fetch is not None:
            # The Load pass: only what the code builds, in the order its stages use it.
            first = [name for name in self.order if name in prepared]
            rest = sorted(set(prepared) - set(first), key=lambda n: prepared[n].plan.bytes)
            fetch.schedule(
                [
                    (prepared[n].checkpoint.manifest_id, _ordered(prepared[n].objects))
                    for n in first + rest
                ],
                key=str(id(self)),  # one model of a many-model load: its part of the pass
            )
        for name, layout in prepared.items():
            component = self._register(name, self.roots[name], layout.keys, prepared=layout)
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
        component = self._register(name, old.root, keys, refine=True)
        self.components[name] = component
        self.weights.register(component)
        return component

    def _source(
        self, name: str, checkpoint: Checkpoint, refs: Sequence[Ref], layout: _Prepared
    ) -> Source:
        """A weight set's own store source, leased over the objects it reads once they are
        here: an explicit Store's path (tests, tools). A managed executor reads no store; on
        unified memory it reads the machine's verified object files, whose pages are the host
        tier."""
        pool = self.weights.pool
        files = self.tiers.files if self.tiers is not None and pool is not None else None
        if checkpoint.store is None and files is None:
            raise _refuse(
                "store_unusable",
                f"{name}: the machine served no layout for this weight set, and this "
                "executor reads no store",
            )
        if self.weights.fetch is not None:
            self.weights.fetch.wait(checkpoint.manifest_id, refs)
        if checkpoint.store is None:
            assert files is not None and pool is not None
            handed = files(layout.sealed, _objects(layout.plan))
            if not handed:
                raise _refuse("store_unusable", f"{name}: the machine served no object files")
            try:
                return cast("Source", pool.files_source(handed, checkpoint.header))
            finally:
                for _, _, fd in handed:
                    os.close(fd)
        with store_refusals(f"leasing {name}"):
            lease = checkpoint.store.acquire(checkpoint.manifest_id, sorted(refs))
        return plane.source(self.weights.plane, checkpoint.store, lease)

    def _prepare_layout(
        self, name: str, root: Module, keys: list[str], *, fine: bool = False
    ) -> _Prepared:
        """Partition the component into regions (common, then each block or sub-block unit)
        and plan its read: everything its weight set's layout is made of. A component that
        declares no blocks is paged at the sub-block grain: as one region nothing below its
        whole size could hold it (run 5424, SeedVR2's 6.4 GB DiT on an 8 GB card)."""
        try:
            layout = None if fine else partition(root)
        except ValueError:
            layout = None  # its blocks share storage: the sub-block grain places them
        if layout is None:
            with suppress(ValueError):  # custom state-dict keys: one region, not streamable
                layout = partition(root, REGION_LIMIT)
            fine = layout is not None
        units = [unit.paths for unit in layout.blocks] if layout is not None else []
        members: list[list[str]] = [[] for _ in range(len(units) + 1)]
        for key in keys:
            path = key[len(name) + 1 :].rpartition(".")[0]
            members[_unit_of(path, units)].append(key)
        kept = [(index, group) for index, group in enumerate(members) if group]
        checkpoint = self.checkpoints[name]
        skipped = self.skipped.get(name, ())
        traversal = [(name, self.plan[key].name) for key in keys] + [(name, key) for key in skipped]
        plan = checkpoint.read_plan(traversal, READ_WINDOW, (name,))
        parts = (
            [
                self.plan[key].role_what(part.role)
                for key in keys
                for part in self.plan[key].encoded.parts
            ]
            if skipped
            else []
        )
        if skipped:
            # The plan names every stored tensor; only what the code builds is read or mapped.
            plan = plan.select(parts)
        key = f"{self.construction}/{name}"
        ids = [[f"{name}/{self.plan[k].name}" for k in group] for _, group in kept]
        digest = canonical.digest({"manifest": checkpoint.manifest_id, "regions": ids})
        sealed_plan = SealedPlan(
            manifest=checkpoint.manifest_id,
            name=key,
            layout=digest,
            window=READ_WINDOW,
            traversal=tuple(traversal),
            components=(name,),
            regions=tuple(tuple(group) for group in ids),
            parts=tuple(parts),
        )
        objects = region_objects(plan, ids)
        return _Prepared(
            keys, units, kept, checkpoint, plan, ids, digest, sealed_plan, fine, objects
        )

    def _register(
        self,
        name: str,
        root: Module,
        keys: list[str],
        *,
        refine: bool = False,
        prepared: _Prepared | None = None,
    ) -> Component:
        """Register the component's layout as one weight set and bind every stored part to
        its destinations; `refine`: again, at the sub-block grain."""
        layout = prepared or self._prepare_layout(name, root, keys, fine=refine)
        units, kept, checkpoint, plan = layout.units, layout.kept, layout.checkpoint, layout.plan
        ids, key = layout.ids, layout.sealed.name
        weights = self.weights
        # Standalone checkpoints can reread their explicit Store for a finer layout.
        # Managed checkpoints ask TensorD for the finer shared CPU layout instead.
        tiers = None if refine and checkpoint.store is not None else self.tiers
        manifest, refs = checkpoint.manifest_id, _ordered(layout.objects)
        here = weights.fetch is None or weights.fetch.present(manifest, refs)
        staged: list[int] = []
        with store_refusals(f"registering {name}"):
            held = tiers.seal(layout.sealed) if tiers is not None else None
            sealed = held is not None and plane.is_sealed(held)
            files: list[tuple[str, int, int]] = []
            if sealed and tiers is not None and tiers.files:
                if not here and tiers.stage is not None:
                    # An absent object has no file: the machine stages every region, in
                    # order, as its objects land, and each is used once Ready.
                    staged = tiers.stage(layout.sealed, list(range(len(ids))))
                if len(staged) < len(ids):
                    # The holes copy from the verified object files, once they are here.
                    if not here and weights.fetch is not None:
                        weights.fetch.wait(manifest, refs)
                        here = True
                    files = tiers.files(layout.sealed, _objects(plan))
            try:
                if sealed:
                    assert held is not None
                    # TensorD supplied the CPU buffer; this executor opens no Store or
                    # local read lease. TensorFS can read unstaged regions from supplied
                    # object descriptors (`files` is absent for older plane APIs).
                    handed: dict[str, Any] = {"files": files} if files else {}
                    ws = weights.plane.register_sealed(key, plan, ids, held, **handed)
                else:
                    ws = weights.plane.register(
                        key, self._source(name, checkpoint, refs, layout), plan, ids, host_fd=held
                    )
            except plane.error("Invalid"):
                if held is None:
                    raise
                ws = None  # the held tier is not this layout after all
            finally:
                if held is not None:
                    os.close(held)  # the plane reopened its own description
                for _, _, fd in files:
                    os.close(fd)  # the plane duplicated each
            if ws is None:
                held, sealed = None, False
                ws = weights.plane.register(
                    key, self._source(name, checkpoint, refs, layout), plan, ids
                )
        spans = msgspec.convert(ws.regions, list[_Span])
        placed = {row.what: row for row in msgspec.convert(ws.parts, list[_Placed])}
        sites = _sites(root, name)
        regions: list[Region] = []
        for (index, group), span in zip(kept, spans, strict=True):
            region = Region(
                index=len(regions),
                path=units[index - 1][0] if index else "",
                module=root.get_submodule(units[index - 1][0]) if index else root,
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
            sealed=sealed,
            common=common,
            blocks=blocks,
            layout=weight_policy.Layout(
                common=common.span if common else 0,
                blocks=tuple(region.span for region in blocks),
                fine=None if layout.fine else _fine(root),
            ),
            fine=layout.fine,
            staging=(layout.sealed, tiers)
            if sealed and tiers is not None and tiers.stage
            else None,
            staged=set(staged),
            manifest=manifest,
            objects=layout.objects,
            # a store source waited for every object before its lease
            landed=set(range(len(regions))) if here or not sealed else set(),
        )
        for region in regions:
            for table, attr, standin in region.unbound:
                setattr(standin, "origin", (weights, component, region, table, attr))  # noqa: B010
            region.bakes = _lora_layers(root, region)
            region.bake = _bakes_here(weights, component, region)
            region.locate = weights.content
        if any(region.bake for region in regions):  # its kernels build at load, in a child
            lora_bake.warm(weights.device.index or 0)
        if not refine:
            weights.attach(component)
        return component


@dataclass(frozen=True, slots=True)
class DeviceTiers:
    """Exchange exported GPU allocation handles with TensorD (`weights.attach/1`).

    `ask` returns retained regions/fds for (weight set, layout, GPU, variant), or `held`
    False. Runtime attaches/maps them locally. `offer` hands TensorD handles exported by
    this executor's plane after Runtime filled the regions; it transfers no model object.
    """

    ask: Callable[[str, str, str, str], DeviceTierAnswer]
    offer: Callable[
        [str, str, str, tuple[SharedRegion, ...], tuple[int, ...], str], DeviceTierAnswer
    ]


@dataclass(frozen=True, slots=True)
class HostTiers:
    """The machine's sealed CPU layouts (TensorD prepares and keeps them through TensorFS),
    distinct from stored files and GPU allocations."""

    #: TensorD's sealed CPU layout for a plan (`SealedTier`); None: it keeps none
    seal: Callable[[SealedPlan], int | None]
    #: stage more regions of a layout staged in part (`SealedStage`): those staged or staging
    stage: Callable[[SealedPlan, list[int]], list[int]] | None = None
    #: the verified object files behind a plan, `(sha256, length, fd)` per object of the list
    #: given (the plan's, in SHA-256 order; `ObjectFiles`)
    files: Callable[[SealedPlan, list[tuple[str, int]]], list[tuple[str, int, int]]] | None = None


@dataclass(frozen=True, slots=True)
class _Prepared:
    """One component's layout before it registers: its keys, block units, kept groups,
    checkpoint, read plan, region ids and digest, and the plan the machine fills from."""

    keys: list[str]
    units: list[tuple[str, ...]]
    kept: list[tuple[int, list[str]]]
    checkpoint: Checkpoint
    plan: ReadPlan
    ids: list[list[str]]
    digest: str
    sealed: SealedPlan
    #: regions of at most `REGION_LIMIT` bytes
    fine: bool
    #: each region's objects, in read order
    objects: list[list[Ref]]


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


def _lora_layers(root: Module, region: Region) -> list[Bake]:
    """The region's LoRA'd linears whose stored weight takes a bake: PEFT layers whose every
    active adapter's factors lie in the same region (so they are bound together), and, in a
    block, linears carrying a model's consumer hook (`LoRASource`), under a PEFT layer or not."""
    where = {weight.row.name: index for index, weight in enumerate(region.weights)}
    found: list[Bake] = []
    for index, weight in enumerate(region.weights):
        roles = {role: dtype for role, (_, dtype, _) in weight.parts.items()}
        site = weight.row.name.removesuffix(".weight")
        if site == weight.row.name or not lora_bake.bakeable(roles):
            continue
        bake = Bake(index)
        try:
            hooks: Iterable[object] = root.get_submodule(site)._forward_hooks.values()
        except AttributeError:  # a tensor named `.weight` on no module of its own
            hooks = ()
        if region.path:
            bake.hooks = [hook for hook in hooks if isinstance(hook, LoRASource)]
        target = site.removesuffix(".base_layer")
        if target != site:
            layer: Any = root.get_submodule(target)
            for name in getattr(layer, "active_adapters", ()):
                a = where.get(f"{target}.lora_A.{name}.weight")
                b = where.get(f"{target}.lora_B.{name}.weight")
                if a is None or b is None:
                    break
                bake.factors.append((a, b, float(layer.scaling[name])))
            else:
                if bake.factors:
                    bake.layer = layer
        if bake.layer is not None or bake.hooks:
            found.append(bake)
    return found


def _bakes_here(weights: Weights, component: Component, region: Region) -> bool:
    """A region bakes its LoRA'd linears unless the request said per step, on a GPU: a CPU
    executor computes on its CPU buffer, which other executors reuse as source bytes."""
    del component
    return bool(region.bakes) and not weights.host_only and weights.lora == "bake"


@runtime_checkable
class LoRASource(Protocol):
    """A model's forward hook on a linear that adds a LoRA update the model owns (H3's turbo
    overlay). The weight plane bakes `lora_factors()` (identity, down, up, scale; None when
    the hook adds nothing now) into the linear's bytes, and sets `_cozy_baked` while the
    bytes it reads hold it."""

    _cozy_baked: bool

    def lora_factors(self) -> tuple[object, Tensor, Tensor, float] | None: ...


def _fine(root: Module) -> tuple[int, int] | None:
    """(common, largest region) spans at the sub-block grain, for its floor."""
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


def _unit_of(path: str, units: Sequence[Sequence[str]]) -> int:
    """The region a module path belongs to: 1 + its unit's index, or 0 (common)."""
    for index, paths in enumerate(units):
        if any(path == unit or path.startswith(unit + ".") for unit in paths):
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
        #: method -> the components it declares (`component_use`), in definition order
        self.scopes = dict(scopes or {})
        #: methods in the order this process first entered them: the stage order it learned
        self.stages: list[str] = []
        self.hosted: frozenset[str] = frozenset()
        self.active: tuple[str, tuple[str, ...]] | None = None
        self.spare = False
        self.poisoned = ""
        #: a block that writes what it did not make failed this attempt with no journal: if
        #: the attempt fails, the process is replaced, never served again (state, not capacity)
        self.unjournaled = ""
        self.refusals: list[str] = []
        #: this attempt's activation peak per method, and per batch item with the batch it ran
        self.peaks: dict[str, int] = {}
        self.per_item: dict[str, int] = {}
        self.batch: dict[str, int] = {}
        self.baseline = -1
        #: torch's idle cache at the entry's start: segments an earlier stage reserved
        self.cached = 0
        #: torch's room and its cache drops so far, at the open entry's start
        self.pressure = (0, 0)
        self.derived_at_open = 0
        #: stages of each method in this attempt, and the most any attempt has run: a method
        #: entered again and again (a denoise step per stage) is cyclic like a multi-pass one
        self.entries: dict[str, int] = {}
        self.repeats: dict[str, int] = {}
        #: a repeating method's stage left open between its entries (its resident set bound,
        #: its common leases and cursors live, the cursor already copying the next pass), and
        #: TensorD's budget it was planned in
        self.parked: tuple[str, tuple[str, ...], int] | None = None
        self._poisoned_error = plane.error("Poisoned")
        self._quiet = weights.torch.utils._python_dispatch._disable_current_modes
        #: block modules run while `recording`, by component
        self.read: dict[str, set[Module]] | None = None
        self.attempts = 0
        #: an on-demand model's: asks for its weights as a stage begins (`Fetches.want`)
        self.wanted: Callable[[], None] | None = None
        self._modes = weights.torch._C._len_torch_dispatch_stack
        #: `held` scopes the open stage keeps (`hold_stage`): (held, layout) by component
        self.stage_holds: dict[str, tuple[bool, tuple[object, ...]]] = {}
        self._stage_stack = ExitStack()
        weights.residencies.add(self)
        for name, component in self.components.items():
            PLACED[component.root] = (self, name)
            self._hook(component)

    def _hook(self, component: Component) -> None:
        """One wrapper per block region: its module's forward acquires and releases it. A CPU
        executor has none: its weights are bound whole at registration, at either grain."""
        if self.weights.host_only:
            return
        rows = self.hooks[component.name] = []
        for region in component.blocks:
            module = region.module
            had = "forward" in module.__dict__
            rows.append((module, had, module.forward))
            # Read at each call: a package that swaps the module's class after load
            # (HiDiffusion, at its first 4 MP request) runs its own forward.
            forward = module.forward if had else functools.partial(_class_forward, module)
            module.forward = self._block(component, region, forward)

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
        self.unjournaled = ""
        self.derived_at_open = self._derived()
        self.attempts += 1

    def serve_lora(self, strategy: lora_bake.Strategy) -> None:
        """A request's LoRA serving (`lora_mode`): baked, or per step."""
        self.weights.set_lora(strategy)

    def admit(self, method: str, components: tuple[str, ...]) -> None:
        if self.wanted is not None:
            self.wanted()
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
        if method not in self.stages:
            self.stages.append(method)
        self.entries[method] = self.entries.get(method, 0) + 1
        self.repeats[method] = max(self.repeats.get(method, 0), self.entries[method])
        weights.hold_host()
        self._clear(method, local)
        weights.room_before(method)
        if weights.host_only:  # everything is bound in place already; no device to measure
            self.active = (method, local)
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
            weights.trim_cache(method)
            budget = weights.stage_budget(method)
            # One GPU of a group keeps only a streaming window for a stage's first pass (its
            # blocks cannot run again after an out-of-memory), then grows by what it measured.
            weights.growing = weights.grouped and method not in weights.activation
            if method not in weights.activation:
                # Activations first, and these are not measured yet: the open stages' own
                # weights are all the plane keeps; every other byte stays torch's until it is.
                own = [_paid(c, c.planned)[0] for _, _, c in weights.opened().values()]
                keep = sum(o.common + 2 * o.largest if weights.grouped else o.total for o in own)
                if weights.cap >= 0:
                    # Under an owner's cap other stages' weights stay mapped too (keep hot): an
                    # activation that then lacks room is an out-of-memory at the cap, and the
                    # plane unmaps the least recently used to run it (a group GPU's op by op).
                    # Without one, they leave. A group's first decode unmapped its DiT, which
                    # the next segment read back: 51 GiB, 1.3-2.0 s on 2x H100 (6719, 6772).
                    keep += max(plane.stats(weights.plane).device(weights.index).committed, 0)
                # a window of two blocks: one more than the floor, so nothing refines
                budget = min(budget, keep)
            budget = self._place(budget)
            stream = weights.stream()
            for name in local:
                component = self.components[name]
                component.passes = 0
                if component.common is not None:
                    component.lease = self._acquire_common(component, stream)
                    bind(weights.torch, component.common, component.lease)
        except BaseException:
            with suppress(Exception):  # the original failure is the one to report
                self._close(local)
            self._left()
            raise
        self._entered(method)

    def _clear(self, method: str, local: Sequence[str]) -> None:
        """Before an entry of `method`: while torch has no more room than an entry of it is
        known to squeeze in (`Weights.squeezed`), components no stage uses now give it the
        bytes it lacks, the cheapest to restore first. They fill again when their own stage
        opens, or once the attempt ends (`restore`)."""
        weights = self.weights
        bound = weights.squeezed.get(method)
        if bound is None or weights.host_only or weights.grouped or method in weights.adaptive:
            return
        keep = {self.components[name].key for name in local} | _in_use(weights)
        while (need := bound + weight_policy.MARGIN - weights.allocatable()) >= 0:
            if not weights.page_out(keep, need + weight_policy.MARGIN):
                break

    def _squeezing(self) -> None:
        """At an outer block boundary inside a stage: torch dropped its whole cache more than
        once since the entry or the last page-out, so the room it has squeezes this entry (one
        stage can be a whole denoise). Read at every block, so it acts within the step that
        squeezes (one read, 11 us on an RTX 4070). The room is kept for later entries and
        executors, and the idle component cheapest to restore leaves now: never one an open or
        repeating stage uses (`_in_use`), so never one `_spare` gave back."""
        weights = self.weights
        if self.active is None or self.baseline < 0 or weights.grouped or weights.mapping:
            return
        if self.active[0] in weights.adaptive:
            return
        room, dropped = self.pressure
        if weights.retries() - dropped <= 1:
            return
        method = self.active[0]
        weights.squeezed[method] = weights.squeezes[method] = max(
            weights.squeezed.get(method, 0), room
        )
        weights.page_out(_in_use(weights))
        self.pressure = (weights.allocatable(), weights.retries())

    def _acquire_common(self, component: Component, stream: Any, upto: Rung = Rung.DISK) -> Any:
        """The stage's common region's lease, room made for it by the ladder (`upto`)."""
        weights, common = self.weights, component.common
        assert common is not None
        self._arrive(component, common.index)
        return weights.memory.retry(
            lambda: weights.plane.acquire(component.ws, weights.index, common.index, stream),
            f"{component.name}.common",
            upto=upto,
        )

    def _place(self, budget: int, resume: Mapping[str, int] = {}) -> int:
        """`_plan` the open stages in `budget`. Short of memory anywhere in it (a plane refusal
        for its real deficit, a map the device refuses, binding out of device memory), rungs
        0-1 make room and the same plan runs again; else the plane's budget grows into room the
        device has; else the sets plan again in less, finer, down to their floor; there the
        rest of the ladder runs, and only when that gives nothing is it `memory_exhausted`.
        Returns the budget used."""
        weights = self.weights
        placing, weights.placing = weights.placing, True
        try:
            return self._placed(budget, resume)
        finally:
            weights.placing = placing

    def _placed(self, budget: int, resume: Mapping[str, int]) -> int:
        weights = self.weights
        memory, method = weights.memory, weights.open[-1][1]
        op = f"{method}()"
        weights.quiesce()  # a plan unmaps: other components' sets, this one's stale regions
        outside = 0
        while True:
            try:
                self._plan(budget, resume, outside)
                weights.limit_torch()
                memory.shrunk, memory.plan = False, -1
                return budget
            except weight_policy.BelowFloor as exc:
                (nbytes, own), floor = (max(exc.needed - exc.available, 1), True), True
            except Exception as exc:
                short = shortage(exc)
                if short is None:
                    raise
                (nbytes, own), floor = short, False
                exc.__traceback__ = None  # the failed plan's frames free their tensors
            nbytes = nbytes or weight_policy.MARGIN
            if self.spare:  # an optional call is taken only if it fits beside what is here
                raise ResidencyRefusal("device_spared", f"{op} declined: {nbytes} B short")
            if own and memory.collect():  # the plane's own bytes: a ring a view kept open
                continue
            if memory.free_for(nbytes, op, upto=Rung.IDLE) and memory.freed:
                continue
            unused = max(budget - weights._committed(), 0)
            if own and memory.free() >= unused + nbytes + weight_policy.MARGIN:
                budget += nbytes  # the plane's budget grows into room the device has
            elif not floor:
                outside += nbytes  # a ring a view keeps mapped (run 10562), a full device
            elif method == MAPPING or memory.again(nbytes, op, seen=memory.free()) < 0:
                memory.exhausted(nbytes, op)  # mapping ahead moves no tensor to host
            weights.oom_retries += 1

    def _left(self) -> None:
        if self.stage_holds:
            self.stage_holds = {}
            stack, self._stage_stack = self._stage_stack, ExitStack()
            stack.close()
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
        before, capped = weights.applied, wanted <= budget_cell.CAP
        if capped:  # a process cap: the plane re-placed inside it
            weights.set_cap(budget_cell.CAP - wanted)
            wanted = weights.stage_budget(weights.open[-1][1]) if weights.open else -1
        weights.set_budget(max(wanted, floor) if wanted >= 0 else wanted)
        if not capped and wanted >= 0 and weights.cap >= 0:
            # a plane ask moves the cap by what it moved the plane
            weights.set_cap(max(weights.cap + weights.applied - before, 0))
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
        self.cached = accel.reserved(weights.torch, weights.kind) - self.baseline
        self.pressure = (weights.allocatable(), weights.retries())
        accel.reset_peak(weights.torch, weights.kind)
        if len(weights.open) == 1:
            spill.began()

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
            if weights.sharing is not None:
                # Shared while resident: the next stage's plan would evict it otherwise.
                fresh = [c for c in map(self.components.__getitem__, local) if _unshared(c)]
                if fresh:
                    weights.share(fresh)
        finally:
            self._left()
        peak = 0
        if self.baseline >= 0:
            # What the stage took beyond its live bytes at entry: its tensors' peak, or the
            # segments torch reserved for them, whichever is more (a cap counts the segments).
            peak = max(
                accel.peak_allocated(weights.torch, weights.kind) - self.baseline,
                accel.peak_reserved(weights.torch, weights.kind) - self.baseline - self.cached,
                0,
            )
            self.peaks[method] = max(self.peaks.get(method, 0), peak)
            weights.activation[method] = max(weights.activation.get(method, 0), peak)
            room, dropped = self.pressure
            if weights.retries() - dropped > 1:
                # One drop clears another stage's cache; a second in the same entry is this
                # stage's own blocks against torch's limit: the room it had squeezes it.
                room = max(weights.squeezed.get(method, 0), room)
                weights.squeezed[method] = weights.squeezes[method] = room
            item = peak // max(self.batch.pop(method, 1), 1)
            self.per_item[method] = max(self.per_item.get(method, 0), item)
        self.baseline = -1
        if self.parked is None and not weights.host_only:
            # Host blocks no tensor holds any more are the host's again. Torch's idle device
            # cache stays: the next stage's plan trims what its weights need (`trim_cache`,
            # run 5908), and the machine asks for the rest when another tenant needs it.
            spill.release(weights.torch)

    def map_ahead(self) -> tuple[int, bool]:
        """Map every component's resident set now, as one stage of all of them would place
        it inside the budget, and leave it mapped at idle priority as after a call: nothing
        runs and nothing is measured. Mapped privately and asking the owner for nothing; a
        call shares it at its stage's exit. Returns the device bytes its components hold,
        mapped here or attached from the machine, and whether that is every weight."""
        weights = self.weights
        names = tuple(name for name in self.components if name not in self.hosted)
        if weights.host_only or not names:
            return 0, not names
        self.unpark()
        sharing, room = weights.sharing, weights.room
        weights.sharing = weights.room = None
        try:
            self.admit(MAPPING, names)
            self.release(MAPPING, names)
        except (ResidencyRefusal, MemoryExhausted):
            pass  # not even the floor fits: what was placed stays, the rest is host memory
        finally:
            weights.sharing, weights.room = sharing, room
            for table in (
                weights.activation,
                weights.plan_inputs,
                weights.squeezed,
                weights.squeezes,
                self.peaks,
                self.per_item,
                self.entries,
                self.repeats,
            ):
                table.pop(MAPPING, None)
        held, complete = 0, True
        for component in map(self.components.__getitem__, names):
            on = set(component.resident) | component.shared
            held += sum(region.span for region in component.regions if region.index in on)
            complete = complete and all(region.index in on for region in component.regions)
        return held, complete

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
        for release in RELEASERS:
            release()
        self.unpark()
        idle = [c for c in self.components.values() if c.lease is None and c.cursor is None]
        freed = sum(self.weights.resident_bytes(c) for c in idle)
        self._drop(idle)
        return {"freed": {}, "held": {}, "freed_bytes": freed}

    def batch_fits(self, method: str, batch: int) -> bool:
        """`batch` items of `method` at once, if their measured activations fit what is free
        plus the weights the plane may unmap beyond the stage's own floor. Recorded."""
        # Measured here, else what earlier executors measured for the method (a peak at any
        # batch, so never smaller than one item): unmeasured anywhere is yes, and the first
        # run measures.
        per_item = self.per_item.get(method, self.weights.activation.get(method))
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
        memory = weights.allocation()
        free = weights.headroom(memory) + max(
            memory["reserved_bytes"] - memory["allocated_bytes"], 0
        )
        device = plane.stats(weights.plane).device(weights.index)
        held = {self.components[name].key for name in components if name in self.components}
        held.update(weights.opened())
        own = 0  # their resident set the stage's leases do not count already (its holds do)
        for key in held:
            component = weights.components[key]
            common = component.common if component.lease is not None else None
            leased = {*component.holds, *([common.index] if common is not None else [])}
            own += sum(component.regions[i].span for i in component.resident if i not in leased)
        movable = device.committed - device.leased_bytes - device.held_bytes - own
        return free, max(movable, 0)

    @contextmanager
    def recording(self) -> Iterator[dict[str, set[Module]]]:
        """The block modules the body runs, by component."""
        read: dict[str, set[Module]] = {}
        self.read = read
        try:
            yield read
        finally:
            self.read = None

    @contextmanager
    def held(self, name: str, blocks: Collection[Module]) -> Iterator[bool]:
        """`name`'s `blocks` leased at their homes and bound there, their hooks off, beside the
        stage's common lease, for the body: kernels it launches may be captured into CUDA graphs
        over those addresses. A region's home is fixed for its weight set's life (an eviction
        unmaps it, a refill maps it there again), and a lease keeps it mapped. Yields False,
        having held nothing, when `name` is not in the open stage, a block bakes LoRA, or the
        blocks do not all fit at home beside what is leased."""
        weights = self.weights
        component = self.components.get(name)
        why = (
            "no such component"
            if component is None
            else "a CPU executor"
            if weights.host_only
            else "a GPU group"
            if weights.grouped
            else "outside its stage"
            if self.active is None
            or name not in self.active[1]
            or (component.common is not None and component.lease is None)
            else ""
        )
        regions = [] if component is None else [r for r in component.blocks if r.module in blocks]
        if not why and len(regions) != len(blocks):
            why = "its blocks are not the recorded ones (it moved to another grain)"
        if not why and any(region.bakes for region in regions):
            why = "it bakes LoRA"
        if why:
            self.note(f"{name}.held", f"declined: {why}")
            yield False
            return
        assert component is not None
        leases: list[Lease] = []
        with self._quiet():
            try:
                stream = weights.stream()
                for region in regions:
                    self._arrive(component, region.index)
                    leases.append(self._hold(component, region, stream))
                    weights.drain()
                for region, lease in zip(regions, leases, strict=True):
                    bind(weights.torch, region, lease, refill=True)
            except BaseException as exc:
                self._let_go(component, regions[: len(leases)], leases)
                if not isinstance(exc, MemoryExhausted) and shortage(exc) is None:
                    raise
                why = str(exc).splitlines()[0][:200]
        if why:
            self.note(f"{name}.held", f"declined: {why}")
            yield False
            return
        hooked = {r.index: row for r, row in zip(component.blocks, self.hooks[name], strict=True)}
        wrappers = [(hooked[region.index][0], region.module.forward) for region in regions]
        for region in regions:
            module, had, forward = hooked[region.index]
            if had:
                module.forward = forward
            else:
                del module.forward
        try:
            yield True
        finally:
            for module, wrapper in wrappers:
                module.forward = wrapper
            with self._quiet():
                self._let_go(component, regions, leases)

    def _hold(self, component: Component, region: Region, stream: int) -> Lease:
        """`region`'s home lease for `held`: an optional speed, so room for it comes only from
        what no stage holds (rungs 0-1: idle weights, the machine's custody, idle tenants)."""
        weights = self.weights
        return weights.memory.retry(
            lambda: weights.plane.acquire(
                component.ws,
                weights.index,
                region.index,
                stream,
                priority=plane.PREFIX + weights.tick(),
            ),
            f"{component.name}.{region.path} held",
            upto=Rung.IDLE,
        )

    def note(self, key: str, value: str) -> None:
        self.weights.modes[key] = value

    def allocatable(self) -> int:
        return self.weights.allocatable()

    def hold_stage(self, name: str, blocks: Collection[Module]) -> tuple[bool, tuple[object, ...]]:
        """`held` for the rest of the open stage, entered once per component: whether it holds,
        and the layout it holds them at."""
        found = self.stage_holds.get(name)
        if found is None:
            held = self._stage_stack.enter_context(self.held(name, blocks))
            found = self.stage_holds[name] = (held, self.layout(name, blocks) if held else ())
        return found

    def layout(self, name: str, blocks: Collection[Module]) -> tuple[object, ...]:
        component = self.components.get(name)
        if component is None:
            return ()
        regions = [r for r in component.blocks if r.module in blocks]
        if component.common is not None:
            regions.insert(0, component.common)
        return (
            component.key,
            id(component),
            *((r.index, r.bound, *(t.data_ptr() for t in r.derived)) for r in regions),
        )

    def _let_go(
        self, component: Component, regions: Sequence[Region], leases: Sequence[Lease]
    ) -> None:
        """End `held`'s leases; outside its lease only a stage-resident region stays bound."""
        for region in regions:
            if region.index not in component.resident:
                unbind(region)
        release_all(leases)
        self.weights.drain()

    def whole(self) -> list[str]:
        """Its components a stage last placed whole here: none of their blocks streams."""
        return [
            name
            for name, c in self.components.items()
            if name not in self.hosted and c.residence is not None and not c.residence.streamed
        ]

    def kept(self, local: Sequence[str]) -> int:
        """What an adaptive stage of `local` keeps on the device: those components whole, and
        the resident sets of the model's other components placed whole."""
        own = sum(self.components[name].planned.total for name in local)
        return own + sum(
            self.components[name].regions[index].span
            for name in self.whole()
            if name not in local
            for index in self.components[name].resident
        )

    def restore(self) -> None:
        """After an attempt: the components its stages placed whole and then gave to activations
        map again (keep hot), in the background, into the room the device has now, in the order
        the stages read them. Nothing is evicted for them; the next stage waits only for what has
        not arrived yet. A streamed component's prefix stays where its stage left it: the next
        request would evict it again (an 8 GB card copied 16 GB more per request)."""
        weights = self.weights
        if weights.host_only or weights.grouped or weights.open or self.parked is not None:
            return
        weights.drain()
        weights._apply(max(weights.applied, weights.stage_budget(MAPPING)))
        weights.plan_inputs.pop(MAPPING, None)
        whole = set(self.whole())
        gone = [
            (component, index)
            for name in stage_components(self.stages, self.scopes)
            if name in whole and (component := self.components.get(name)) is not None
            for index in component.resident
            if component.regions[index].bound is None
        ]
        for component, index in gone:
            try:  # the lowest priority: it takes only room nothing else holds
                weights.plane.want(component.ws, weights.index, [index], priority=0)
            except Exception as exc:
                if shortage(exc) is None:
                    raise
                break
        weights.limit_torch()

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
            # what a grant must leave for activations: an adaptive stage (ops cut into
            # pieces) fits itself to the room it finds, so only its smallest piece sizes one
            "attempt_activation_peak_bytes": max(
                (weights.adaptive.get(m, v) for m, v in self.peaks.items()), default=0
            )
            or max(self.peaks.values(), default=0),
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

    def _plan(self, budget: int, resume: Mapping[str, int] = {}, outside: int = 0) -> None:
        """Place every open stage's components inside `budget`: each resident set wanted, the
        rest streamed by a cursor in the blocks' execution order (from `resume[weight set]`
        when a block boundary re-plans mid-pass). Before a component's order is observed its
        blocks are acquired on demand. A component whose block floor does not fit moves to
        the sub-block grain first. `outside`: what the sets leave of `budget` (a ring a view
        still keeps mapped, a device that gives less). A plane refusal raises to `_placed`."""
        weights = self.weights
        opened = {key: row for key, row in weights.opened().items() if not row[2].spared}
        components = {key: component for key, (_, _, component) in opened.items()}
        method = weights.open[-1][1]
        arriving: list[Ticket] = []
        # The policy places only the blocks the budget pays for: a region attached from the
        # machine is on the device already and is never streamed (see `_paid`).
        kept = {key: _paid(c, c.planned)[1] for key, c in components.items()}
        at = {key: {block: p for p, block in enumerate(blocks)} for key, blocks in kept.items()}
        measured = _costs(weights, {key: (c, opened[key][1]) for key, c in components.items()})

        def stage(
            layouts: Mapping[str, weight_policy.Layout],
        ) -> dict[str, weight_policy.Residence]:
            room = budget - outside
            if weights.growing:  # a group's first pass: the budget past its window keeps hot
                room = min(room, sum(o.common + 2 * o.largest for o in layouts.values()))
            return weight_policy.stage(
                layouts,
                room,
                orders={
                    key: [at[key][n] for n in map(c.numbers.__getitem__, c.order) if n in at[key]]
                    for key, c in components.items()
                    if c.observed
                },
                costs={
                    key: replace(
                        cost,
                        compute={at[key][n]: v for n, v in cost.compute.items() if n in at[key]},
                    )
                    for key, cost in measured.items()
                },
                # A stage on its second pass is cyclic whatever its last run was (a denoise
                # after a one-pass warm-up): placed again, it keeps a resident set.
                cyclic={
                    key
                    for key, (row, tag, c) in opened.items()
                    if c.passes > 1 or c.last_passes != 1 or row.repeats.get(tag, 0) > 1
                },
            )

        # A component counts at its finest grain only while it can still move there.
        layouts = {
            key: _paid(c, c.planned if row._refinable(c) else replace(c.planned, fine=None))[0]
            for key, (row, _, c) in opened.items()
        }
        try:
            paid = stage(layouts)
        except weight_policy.BelowFloor as exc:
            if exc.component in opened and opened[exc.component][0]._refine(exc.component):
                return self._plan(budget, resume, outside)
            raise
        # A budget that only fits one block at a time streams with no overlap: regions of at
        # most 16 MiB let it hold a window of several. (Window 1 chosen when two fit stays.)
        for key, residence in paid.items():
            layout = layouts[key]
            if (
                residence.streamed
                and residence.window == 1
                and layout.common + 2 * layout.largest > budget - outside
                and opened[key][0]._refine(key)
            ):
                return self._plan(budget, resume, outside)
        need = sum(r.vram_bytes(layouts[key]) for key, r in paid.items())
        placed = {key: _unpaid(components[key], kept[key], r) for key, r in paid.items()}
        if self.spare and need > budget - outside:
            raise ResidencyRefusal("device_spared", f"{method}() declined: needs {need} B")
        clock = weights.tick()
        for component in components.values():
            let_go(component)  # every set this plan places may move: held again below
        for key, residence in placed.items():
            # The budget's trim takes other components' idle regions, never the sets this
            # stage is about to want: they rank first already.
            component = components[key]
            weights.plane.prioritise(
                component.ws,
                weights.index,
                [
                    *([component.common.index] if component.common is not None else ()),
                    *(component.blocks[i].index for i in residence.resident),
                ],
                priority=plane.PREFIX + clock,
            )
        weights._apply(budget)
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
            arriving.append(
                weights.plane.want(
                    component.ws,
                    weights.index,
                    list(component.resident),
                    priority=plane.PREFIX + clock,
                )
            )
            if streams:
                _stage_stream(weights, component, residence, clock)
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
            # sees real bytes, exactly as when the component was simply on the device. Its
            # blocks' leases are held for the stage: a block then runs with no acquire.
            common = component.common.index if component.common is not None else -1
            component.hold_stream = stream
            for index in component.resident:
                lease = weights.plane.acquire(component.ws, weights.index, index, stream)
                try:
                    bind(weights.torch, component.regions[index], lease)
                except BaseException:
                    lease.release()
                    raise
                if index == common:
                    lease.release()
                else:
                    component.holds[index] = lease
            if residence.streamed and component.observed:
                order = component.order
                start = order.index(resume[key]) if key in resume else 0
                component.schedule = order[start:] + order[:start]
                component.cursor = weights.plane.stream(
                    component.ws,
                    weights.index,
                    component.schedule,
                    residence.window,
                    repeat=0,
                    priority=plane.STREAMED + clock,
                    tag=opened[key][1],
                    # the plane's ring holds the stored bytes; their decoded copies are
                    # counted in the budget beside it
                    ring_bytes=weight_policy.ring_bytes(
                        [component.layout.blocks[i] for i in residence.streamed],
                        residence.window,
                    ),
                )

    def _derived(self) -> int:
        return sum(
            int(t.numel() * t.element_size())
            for c in self.components.values()
            for r in c.regions
            for t in r.derived
        )

    def _refinable(self, component: Component) -> bool:
        """Whether it can move to a finer grain now: never under a block in its forward or a
        held scope (their leases and captured graphs keep its regions), nor under a ring a
        view keeps open (its weight set cannot close, 0.27.3)."""
        layout = component.layout
        return (
            self.refine is not None
            and layout.fine is not None
            and sum(layout.fine) < layout.least
            and component.key not in self.weights.running
            and component.name not in self.stage_holds
            and reap(component)
        )

    def _refine(self, key: str) -> bool:
        """Move the weight set `key` to the sub-block grain at a block boundary, if it can: its
        cursor and leases close with its weight set, its common is leased again for the open
        stage, and a block that chose the old grain runs at the new one (`_regrained`)."""
        name = next((n for n, c in self.components.items() if c.key == key), "")
        component = self.components.get(name)
        if component is None or self.refine is None or not self._refinable(component):
            return False
        staged = component.lease is not None
        self._unhook(name)
        component = self.components[name] = self.refine(name)
        self._hook(component)
        if staged and component.common is not None:  # inside a plan: never its rung 2
            component.lease = self._acquire_common(component, self.weights.stream(), Rung.IDLE)
            bind(self.weights.torch, component.common, component.lease)
        self.note(f"{name}.grain", "sub-block")
        return True

    def _block(
        self, component: Component, region: Region, forward: Callable[..., Any]
    ) -> Callable[..., Any]:
        """The one hook per block: acquire, bind, run, release. Its ops recover one by one in
        the process's dispatch mode. A block whose first run wrote nothing that existed before
        it and changed no attribute (`Effects`) runs with no mode; out of memory there, it gets
        back the generators it drew from and the containers it was handed and runs again with
        its ops recovering, as it does from then on (`tight`). One that wrote nothing runs again
        so from any shortage outside its ops, its changed attributes put back too. A group GPU's
        blocks hold their group's collectives: they never run twice, and always recover op by
        op."""
        weights = self.weights

        def run(*args: Any, **kwargs: Any) -> Any:
            try:
                effects = region.effects
                if effects is None:
                    return learn(*args, **kwargs)
                quiet = not (
                    effects.writes
                    or effects.state
                    or effects.banded
                    or region.tight
                    or weights.grouped
                    or _hosts(args, kwargs)
                )
                # A run that wrote nothing runs again, out of memory outside its ops, from what
                # it started with (the owner, 10-05: replays are made safe, not refused). A
                # group GPU's never runs twice.
                restore = (
                    None if effects.writes or weights.grouped else _start(effects, args, kwargs)
                )
                try:
                    if quiet:
                        with self._quiet() if self._modes() else nullcontext():
                            return attempt(True, *args, **kwargs)
                    with weights.recovering():
                        return attempt(True, *args, **kwargs)
                except _Regrained:
                    raise
                except Exception as exc:
                    short = None if restore is None else shortage(exc)
                    if short is None:
                        if effects.writes and not _in_call(weights.torch):
                            # it leaves its writes: unless a group call's journal puts them
                            # back, the process is not served again should the attempt fail
                            self.unjournaled = (
                                f"{component.name}.{region.path} failed after writing"
                            )
                            weights.modes["unjournaled"] = self.unjournaled
                        raise
                    exc.__traceback__ = None  # the failed run's frames free their tensors
                assert restore is not None
                restore()
                region.tight = True
                op, operands = f"{component.name}.{region.path}", (*args, *kwargs.values())
                weights.memory.free_for(short[0] or region.span, op, operands)
                weights.oom_retries += 1
                with weights.recovering():
                    return attempt(False, *args, **kwargs)
            except _Regrained:
                # Placed again from inside its own attempt, the component moved to another
                # grain and closed the weight set this block chose (run 5894): the block runs
                # at the grain the component has now.
                return self._regrained(component, region, forward)(*args, **kwargs)
            except self._poisoned_error as exc:
                raise self._poison(exc) from exc

        def learn(*args: Any, **kwargs: Any) -> Any:
            """Its first run, journaled: its ops recover one by one, it learns its `Effects`,
            and one that fails is put back."""
            journal = Journal()
            rewrites = weights.bands.rewrites if weights.bands is not None else 0
            attributes = module_state.snapshot(region.module)
            try:
                with weights.recovering(journal):
                    out = attempt(True, *args, **kwargs)
            except BaseException:
                with self._quiet():
                    journal.rewind()
                raise
            banded = weights.bands is not None and weights.bands.rewrites > rewrites
            changed = module_state.changed(attributes, module_state.snapshot(region.module))
            region.effects = journal.effects(banded, changed)
            return out

        def attempt(entering: bool, *args: Any, **kwargs: Any) -> Any:
            if entering:  # once per call, never for its replay
                if weights.cell is not None and not weights.running:
                    self._asked({component.key: region.index})
                passes = component.passes
                self._observe(component, region.index)
                if self.read is not None:
                    self.read.setdefault(component.name, set()).add(region.module)
                if component.passes > passes:
                    weights.hold_host()
                if not weights.running:  # a block boundary, before this block leases anything
                    self._squeezing()
                if weights.growing and component.passes == 2 > passes:
                    self._grow()
            memory, op, seen = weights.memory, f"{component.name}.{region.path}", -1
            while True:
                if memory.shrunk and not weights.running:  # the stage gave weights back
                    memory.shrunk = False
                    self._replan({component.key: region.index})
                if self.components.get(component.name) is not component:
                    raise _Regrained
                if component.spared:
                    self._unspare(component)
                stream = weights.stream()
                holds = component.holds
                hold = holds.pop(region.index, None) if component.hold_stream == stream else None
                lease = hold or memory.retry(
                    functools.partial(self._acquire, component, region.index, stream), op
                )
                short: tuple[int, bool] | None = None
                try:
                    # The plane's bind and bake are its own math, outside the block's
                    # recovery: one short of memory leaves the region dirty, and it refills.
                    if region.bound != int(lease.ptr) or region.bakes:
                        try:
                            with self._quiet():
                                baked = bind(weights.torch, region, lease, refill=True)
                                if baked and region.index in component.resident:
                                    self._bake_resident(component)
                        except Exception as exc:
                            if (short := shortage(exc)) is None:
                                raise
                            exc.__traceback__ = None
                    if short is None:
                        weights.running.append(component.key)
                        weights.last = component.key
                        try:
                            return forward(*args, **kwargs)
                        finally:
                            weights.running.pop()
                except _Stale:
                    pass
                finally:
                    # Outside its lease only a stage-resident region stays bound: any other
                    # read finds the stand-in (`Weights.reading`), never a stale address.
                    if region.index not in component.resident:
                        unbind(region)
                    if hold is not None and holds is component.holds:
                        holds[region.index] = hold  # the stage's, until it lets go
                    else:
                        lease.release()
                if short is None:
                    self._restore(component, region)
                elif (seen := memory.again(short[0] or region.span, op, (), seen)) < 0:
                    memory.exhausted(short[0] or region.span, op)

        return run

    def _regrained(
        self, component: Component, region: Region, forward: Callable[..., object]
    ) -> Callable[..., object]:
        """The forward a block runs at its component's grain now: through its module's region
        there, or, when that grain splits the module, its own forward, whose parts are hooked."""
        current = self.components[component.name]
        again = next((r for r in current.blocks if r.module is region.module), None)
        return forward if again is None else self._block(current, again, forward)

    def _acquire(self, component: Component, index: int, stream: int) -> Lease:
        """The block's lease: from the stage's cursor when the block is due within its window
        (an earlier departure skips ahead), else resident access or a demand fill. A demand
        fill ranks above every earlier streamed block and below the stage's prefix, so the
        least recently used block is the one it can displace. On a group's first pass it
        ranks below every idle weight too: the pass streams through its own window, and other
        stages' weights stay mapped (keep hot)."""
        weights = self.weights
        self._arrive(component, index)
        if component.cursor is not None and _due(component, component.cursor, index):
            lease: Lease = component.cursor.acquire(index, stream)
        else:
            streamed = -plane.STREAMED if weights.growing else plane.STREAMED
            priority = None if index in component.resident else streamed + weights.tick()
            lease = weights.plane.acquire(
                component.ws, weights.index, index, stream, priority=priority
            )
        weights.drain()  # what this fill displaced is unbound at once, in every weight set
        return lease

    def _arrive(self, component: Component, index: int) -> None:
        """A region is read only once its objects are here: the paging fill waits on them,
        raised to Now, and never fetches anything itself."""
        fetch = self.weights.fetch
        if fetch is not None and index not in component.landed:
            fetch.wait(component.manifest, component.objects[index])
            component.landed.add(index)

    def fetch_plan(self) -> list[Group]:
        """Its components in the stage order learned here (definition order before that),
        each one's regions in execution order: the plan an attempt re-asserts."""
        stages = [*self.stages, *(m for m in self.scopes if m not in self.stages)]
        # A partial binding leaves components its scopes name unbuilt; they have no plan.
        names = [n for n in stage_components(stages, self.scopes) if n in self.components]
        names += [name for name in self.components if name not in names]
        groups: list[Group] = []
        for name in names:
            c = self.components[name]
            blocks = c.order or [region.index for region in c.blocks]
            first = [c.common.index] if c.common is not None else []
            groups.append((c.manifest, _ordered(c.objects, [*first, *blocks])))
        return groups

    def _lease(self, component: Component, region: Region, op: str) -> Lease:
        """A region's lease for an op that reads it outside its forward (`Weights.reading`)."""
        weights = self.weights
        return weights.memory.retry(
            lambda: self._acquire(component, region.index, weights.stream()),
            f"{op} ({component.name}.{region.path})",
        )

    def _replan(self, resume: Mapping[str, int]) -> None:
        """Plan the open stages again in the budget they have, or the less memory's rung 2
        left them (`Memory.plan`): their cursors start over from the block that runs again
        (`resume`)."""
        weights = self.weights
        opened = [component for _, _, component in weights.opened().values()]
        for component in opened:
            close_cursor(component)
        if opened:
            budget = weights.applied if weights.memory.plan < 0 else weights.memory.plan
            self._place(max(min(budget, weights.applied), weights.floor()), resume)

    def _restore(self, component: Component, region: Region) -> None:
        """Give a block the stored bytes again (its resident ones hold a delta its forward
        does not want, or half of one), and with it every region of the component the same
        forward would find so (a switch between a DiT's turbo and base forwards): one refill
        and one placement, not one per block. The open stages' cursors close (one may hold a
        region in place), the regions drop, and the stages are placed again from this block,
        which bakes the resident set for this forward when it wants that."""
        weights = self.weights
        opened = [row for _, _, row in weights.opened().values()]
        for row in opened:
            close_cursor(row)
        stale = [
            other
            for other in component.regions
            if other is region
            or (
                other is not component.common
                and other.bakes
                and _stale(other, [_bake_key(bake) for bake in other.bakes])
            )
        ]
        weights.refill(component, stale)
        if opened:
            self._place(max(weights.applied, weights.floor()), {component.key: region.index})

    def _bake_resident(self, component: Component) -> None:
        """A forward baked one of the component's stage-resident regions: bake the rest of the
        set now, back to back before its block runs, rather than one per block through the
        pass (a DiT's first turbo forward)."""
        weights = self.weights
        stream = weights.stream()
        for index in component.resident:
            region = component.regions[index]
            if not region.bake or region.baked or region.bound is None:
                continue
            lease = weights.plane.acquire(component.ws, weights.index, index, stream)
            try:
                bind(weights.torch, region, lease)
            finally:
                lease.release()

    def _spare(self, keep: Collection[str]) -> bool:
        """Activations first, past the stage's floor. A stage that scopes several components
        (Anima's `render`: conditioner, text encoder, transformer and VAE) keeps every one's
        common region leased and bound for the whole stage; while memory runs out, those not
        in `keep` give theirs back: leases released, regions unbound and unmapped. A read of
        one meanwhile maps it for that op (`Weights.reading`), and its next block maps its
        common again (`_unspare`); the next block boundary plans the stage again. With nothing
        in `keep` known to be in use, every one gives back. False when nobody else held
        anything."""
        weights = self.weights
        opened = weights.opened()
        if weights.host_only:
            return False
        idle = [
            component
            for key, (_, _, component) in opened.items()
            if key not in keep and not component.spared
        ]
        if not idle:
            return False
        weights.quiesce()  # nothing queued still reads what leaves
        before = max(plane.stats(weights.plane).device(weights.index).committed, 0)
        for component in idle:
            close_cursor(component)
            if component.lease is not None:
                component.lease.release()
                component.lease = None
            regions = [region.index for region in component.regions]
            for region in component.regions:
                if region.bound is not None:
                    unbind(region)
            with suppress(plane.error("LeaseViolation")):
                weights.plane.drop(component.ws, weights.index, regions)
            component.resident = ()
            component.spared = True
        weights.drain()
        # Dropped regions stay mapped as the plane's idle cache until its budget leaves them
        # no room: the plane keeps what the running component streams in, torch the rest.
        weights._apply(min(weights.applied, max(weights.window(), weights.floor())))
        weights.memory.shrunk = True
        return max(plane.stats(weights.plane).device(weights.index).committed, 0) < before

    def _unspare(self, component: Component) -> None:
        """A spared component runs again within its stage: its common region is leased and
        bound for the rest of the stage, and its blocks join the stage's plan."""
        weights = self.weights
        if component.common is not None and component.lease is None:
            component.lease = self._acquire_common(component, weights.stream())
            bind(weights.torch, component.common, component.lease)
            component.resident = (component.common.index,)
        component.spared = False  # only once its common is leased: a failed lease leaves it spared

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
            if component.residence is not None and component.residence.streamed:
                self.weights.memory.shrunk = True  # streamed in order from this block boundary
            return
        if index not in component.seen:
            component.seen.append(index)

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
            component.spared = False
            if not weights.host_only:  # host maps bind once, at load
                for region in component.regions:
                    _drop_copies(region)
                weights.plane.prioritise(component.ws, weights.index, None, priority=idle)
        if failure is not None:
            raise failure

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


def _ordered(regions: Sequence[Sequence[Ref]], order: Iterable[int] | None = None) -> list[Ref]:
    """The objects of `regions` in `order` (default: address order), each once."""
    picked = range(len(regions)) if order is None else order
    return list(dict.fromkeys(ref for index in picked for ref in regions[index]))


def _objects(plan: ReadPlan) -> list[tuple[str, int]]:
    """The objects `plan` reads, `(sha256 hex, length)` in SHA-256 order."""
    found = {
        item["object"].removeprefix("sha256:"): item["object_length"]
        for item in plan.items()
        if item.get("source") == "object"
    }
    return sorted(found.items())


def _stage_stream(
    weights: Weights, component: Component, residence: weight_policy.Residence, clock: int
) -> None:
    """Ask TensorD to stage streamed blocks into its shared CPU buffer, then pin them
    through the executor's plane. Unstaged regions stream from supplied object descriptors
    through the plane's staging buffers before GPU use."""
    if component.staging is None:
        return  # a whole layout: pinned at registration, inside the pinned budget
    plan, tiers = component.staging
    streamed = [component.blocks[i].index for i in residence.streamed]
    missing = [r for r in streamed if r not in component.staged]
    if missing and tiers.stage is not None:
        component.staged.update(tiers.stage(plan, missing))
    streamed = [r for r in streamed if r in component.staged]
    if streamed and not weights.host_only:
        with suppress(plane.error("Shortfall"), plane.error("BelowFloor")):
            weights.plane.want(component.ws, plane.PINNED, streamed, priority=clock)


def _paid(
    component: Component, layout: weight_policy.Layout
) -> tuple[weight_policy.Layout, tuple[int, ...]]:
    """What a stage's budget pays for, and the block numbers the policy may place. A region
    attached from the machine is on the device already, outside the budget, and never
    streamed: only its decoded copies cost (torch's bytes), held beside `common`."""
    if not component.shared:
        return layout, tuple(range(len(layout.blocks)))
    kept = tuple(i for i, b in enumerate(component.blocks) if b.index not in component.shared)
    common = component.common
    fixed = (common.decoded if common.index in component.shared else layout.common) if common else 0
    fixed += sum(b.decoded for b in component.blocks if b.index in component.shared)
    return replace(layout, common=fixed, blocks=tuple(layout.blocks[i] for i in kept)), kept


def _unpaid(
    component: Component, kept: Sequence[int], paid: weight_policy.Residence
) -> weight_policy.Residence:
    """`paid` (over the placed blocks) in the component's block numbers: attached blocks are
    resident, and the ring streams only blocks the budget pays for."""
    if len(kept) == len(component.blocks):
        return paid
    attached = [i for i in range(len(component.blocks)) if i not in set(kept)]
    return replace(
        paid,
        resident=(*(kept[p] for p in paid.resident), *attached),
        streamed=tuple(kept[p] for p in paid.streamed),
    )


def _due(component: Component, cursor: Cursor, index: int) -> bool:
    """Whether block `index` comes before the cursor's window has passed. The cursor serves a
    block further ahead by skipping every position before it, copying each and handing it on
    unread: a module called again right after itself (SeedVR2's shared-weight projections,
    once for video and once for text) skipped a whole pass each time, 300 GB of copies in one
    17-frame window (run 5540)."""
    schedule, home = component.schedule, cursor.home
    position, streamed = cursor.position, 0
    for step in range(len(schedule)):
        at = (position + step) % len(schedule)
        if schedule[at] == index:
            return True
        streamed += not home[at]
        if streamed > cursor.window:
            return False
    return False


def _end_leases(component: Component) -> None:
    """Close TensorD's lease sockets after Runtime stops mapping the shared GPU regions;
    TensorD observes the end of this executor's use through those sockets."""
    for fd in component.leases:
        with suppress(OSError):
            os.close(fd)
    component.leases.clear()


def _unshared(component: Component) -> bool:
    """Resident regions not yet offered to the machine; a leased one cannot be (yet)."""
    leased = {component.common.index} if component.lease and component.common else set()
    return bool(set(component.resident) - component.shared - leased)


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


def _hosts(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    """Whether a tensor input (one level deep) lives in host memory."""
    return bool(spill.HELD) and any(
        spill.hosted(item)
        for value in (*args, *kwargs.values())
        for item in (value if isinstance(value, (tuple, list)) else (value,))
        if type(item).__name__ in ("Tensor", "Parameter")
    )


def _containers(
    args: tuple[object, ...], kwargs: Mapping[str, object]
) -> list[tuple[object, object]]:
    """The lists and dicts a block is handed, with their items now: a forward that writes into
    them (a decoder's feature cache and its index) finds them so again when it runs again."""
    held: list[tuple[object, object]] = []
    for value in (*args, *kwargs.values()):
        if type(value) is list or type(value) is dict:  # never a subclass's own semantics
            held.append((value, value.copy()))
    return held


def _put_back(held: list[tuple[object, object]]) -> None:
    for value, first in held:
        if isinstance(value, list) and isinstance(first, list):
            value[:] = first
        elif isinstance(value, dict) and isinstance(first, dict):
            value.clear()
            value.update(first)
