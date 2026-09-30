"""Executor weights on the TensorFS weight plane (weight-plane.md §3-4).

Construction runs on meta and torch never allocates a weight. The plane holds every weight
byte, one weight set per component, in regions: `common` and each no-split block
(`paging.partition`). A scope is a stage: at `admit` the policy (`weight_policy.stage`) picks
each component's resident prefix and window inside the plane budget, the plane is asked for
the prefix (`want`) and streams the tail through a slot ring (`stream`). Each block runs
inside exactly one acquire/release, which rebinds cached typed views only when the region's
address changed. A torch OOM at a block boundary lowers the budget (unmap only) and runs the
block again; capacity never poisons the executor.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import tensorfs

from cozy_runtime.author._errors import CapabilityError, RuntimeFailure
from cozy_runtime.author._loader import Census, TensorLike, TensorSpec, _members
from cozy_runtime.internal import accel, canonical, plane, weight_policy
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
from cozy_runtime.internal.probe import QualificationResult, observed_capability_records
from cozy_runtime.internal.resolution import ResolvedModelPlan, TensorResolution

_REQUESTED = re.compile(r"Tried to allocate ([0-9.]+) (B|KiB|MiB|GiB)")
_UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}


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
    """One checkpoint tensor: its stored parts in the region and where it binds."""

    row: PlanRow
    route: str
    provider: Any
    #: role -> (offset in the region, torch dtype name, shape)
    parts: dict[str, tuple[int, str, tuple[int, ...]]]
    #: (tensor table, name, parameter?) for every module attribute that names this tensor
    sites: list[tuple[dict[str, Any], str, bool]] = field(default_factory=list)
    #: an `encoded_gemm` weight's installed leaf and the module it replaced (on meta)
    leaf: Any = None
    replaced: Any = None
    #: leaf roles the provider transforms from the stored bytes (recomputed at bind)
    derived_roles: tuple[str, ...] = ()


@dataclass(slots=True)
class Region:
    index: int
    path: str
    module: Any
    weights: list[Weight]
    nbytes: int
    #: address -> this region's typed views there
    views: dict[int, Views] = field(default_factory=dict)
    #: meta placeholders: the writes that unbind it
    unbound: list[tuple[dict[str, Any], str, Any]] = field(default_factory=list)
    bound: int = 0
    #: decoded tensors and derived leaf roles kept alive while bound
    derived: list[Any] = field(default_factory=list)

    @property
    def transforms(self) -> bool:
        return any(w.route == "decoded_float" or w.derived_roles for w in self.weights)


@dataclass(slots=True)
class Component:
    """One weight set: a constructed component's regions in the plane."""

    name: str
    root: Any
    handle: int
    regions: list[Region]
    layout: weight_policy.Layout
    #: block region indexes in observed execution order (address order until observed)
    order: list[int] = field(default_factory=list)
    observed: bool = False
    seen: list[int] = field(default_factory=list)
    #: forward passes of the blocks in the last stage: >1 is a cyclic (denoise) stage
    passes: int = 0
    last_passes: int = 0
    residence: weight_policy.Residence | None = None
    cursor: Any = None
    common: Any = None
    streamed: frozenset[int] = frozenset()
    resident: frozenset[int] = frozenset()


# ------------------------------------------------------------------------ the process


class Weights:
    """This executor process's plane, budgets and every registered weight set (one per rank).

    Every component of every construction shares one plane: least recently used across
    them is the plane's priority order, so a construction switch moves nothing until a
    stage wants the room.
    """

    def __init__(self, torch: Any, device: Any, kind: str) -> None:
        self.torch = torch
        self.device = device
        self.kind = kind
        self.index = int(device.index or 0)
        self.plane = plane.open_plane(self.index)
        #: -1: no Worker budget; each stage derives one from what the driver has free
        self.budget = -1
        self.pinned_budget = -1
        self.applied = -1
        self.clock = 0
        self.components: dict[tuple[str, str], Component] = {}
        self.activation: dict[str, int] = {}
        self.oom_retries = 0
        self.context_bytes = -1
        self.modes: dict[str, str] = {}

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def stream(self) -> int:
        return int(self.torch.cuda.current_stream(self.device).cuda_stream)

    # -- budgets

    def set_budget(self, vram: int, pinned: int = -1) -> int:
        """The Worker's budgets. Lowering unmaps at once; returns the bytes released."""
        freed = 0
        if pinned >= 0 and pinned != self.pinned_budget:
            self.plane.set_pinned_budget(pinned)
            self.pinned_budget = pinned
        self.budget = vram
        if vram >= 0:
            freed = self._apply(vram)
        if vram == 0:
            accel.release_cached(self.torch, self.kind)
        return freed

    def _apply(self, nbytes: int) -> int:
        if nbytes == self.applied:
            return 0
        before = plane.stats(self.plane).committed
        self.plane.set_vram_budget(self.index, max(nbytes, 0))
        self.applied = nbytes
        self.unbind_evicted()
        return max(before - plane.stats(self.plane).committed, 0)

    def vacate(self) -> int:
        """Unmap every unleased weight (an older Worker's `Vacate`) without fixing a budget:
        the next stage derives its own and wants its weights back from the host tier."""
        freed = self._apply(0)
        self.applied = -1
        return freed

    def stage_budget(self, method: str) -> int:
        """The Worker's budget, else what the driver has free beside this process's weights,
        less the activations this method measured (activations first)."""
        if self.budget >= 0:
            return self.budget
        memory = accel.allocation(self.torch, self.kind)
        committed = max(plane.stats(self.plane).committed, 0)
        # torch reuses its own cached segments first; only growth beyond them needs the driver
        slack = max(memory["reserved_bytes"] - memory["allocated_bytes"], 0)
        growth = max(self.activation.get(method, 0) - slack, 0)
        return weight_policy.plane_budget(memory["driver_free_bytes"] + committed, growth)

    def shrink(self, need: int, floor: int) -> int:
        """OOM at a block boundary: give torch `need` more bytes by unmapping plane bytes."""
        current = self.applied if self.applied >= 0 else max(plane.stats(self.plane).committed, 0)
        target = max(current - need - weight_policy.MARGIN, floor)
        if target >= current:
            return 0
        self.oom_retries += 1
        return self._apply(target)

    def unbind_evicted(self) -> None:
        """Point every bound region the plane no longer holds at meta, before any kernel
        could read it. The plane evicts only inside calls this process makes."""
        for component in self.components.values():
            resident = self.plane.resident(component.handle, self.index)
            for region in component.regions:
                if (
                    region.bound
                    and region.index not in resident
                    and region.index not in (component.streamed)
                ):
                    unbind(region)

    # -- facts

    def resident_bytes(self, component: Component) -> int:
        return plane.stats(self.plane).resident.get(str(component.handle), 0)

    def facts(self) -> PlaneFacts:
        held = plane.stats(self.plane)
        resident: dict[str, int] = {}
        for (_, name), component in self.components.items():
            resident[name] = resident.get(name, 0) + held.resident.get(str(component.handle), 0)
        return PlaneFacts(
            budget_bytes=held.budget,
            committed_bytes=held.committed,
            leased_bytes=held.leased,
            pinned_budget_bytes=held.pinned_budget,
            pinned_bytes=held.pinned,
            context_bytes=self.context_bytes,
            resident=resident,
            streamed={
                key[1]: Streamed(
                    blocks=len(component.layout.blocks),
                    resident_blocks=len(component.residence.resident),
                    window=component.residence.window,
                )
                for key, component in self.components.items()
                if component.residence is not None
            },
            h2d_bytes=held.h2d_bytes,
            h2d_gbps=held.h2d_gbps,
            late=held.late,
            stall_ns=held.stall_ns,
            misses=held.misses,
            evictions=held.evictions,
            oom_retries=self.oom_retries,
            modes=dict(self.modes),
        )

    # -- construction lifecycle

    def prefetch(self, construction: str) -> int:
        """Fill the host tier with every region of `construction`, now; returns the bytes."""
        total = 0
        for (key, _), component in self.components.items():
            if key == construction:
                regions = [region.index for region in component.regions]
                self.plane.want(
                    component.handle, plane.pinned(), regions, (plane.IDLE, self.tick())
                )
                total += component.layout.total
        return total

    def forget(self, construction: str) -> None:
        for key in [key for key in self.components if key[0] == construction]:
            component = self.components.pop(key)
            for region in component.regions:
                unbind(region)
            self.plane.unregister(component.handle)


# ---------------------------------------------------------------------------- binding


@dataclass(frozen=True, slots=True)
class Views:
    """A region's typed views at one address: the writes that bind them, and each weight's
    stored parts by role (what decode and derived leaf roles read)."""

    writes: list[tuple[dict[str, Any], str, Any]]
    parts: list[dict[str, Any]]


def bind(torch: Any, region: Region, lease: Any, *, fresh: bool) -> None:
    """Point the region's destinations at the lease. Views are cached per address; decoded
    tensors and derived leaf roles are recomputed when the address or the content moved
    (`fresh`: a slot the ring just refilled)."""
    address = int(lease.address)
    views = region.views.get(address)
    if views is None:
        views = region.views[address] = _views(torch, region, torch.from_dlpack(lease.view()))
    moved = region.bound != address
    if moved:
        for table, name, tensor in views.writes:
            table[name] = tensor
    if region.transforms and (moved or fresh):
        region.derived = _transform(torch, region, views.parts)
    region.bound = address


def unbind(region: Region) -> None:
    for table, name, tensor in region.unbound:
        table[name] = tensor
    region.derived = []
    region.bound = 0


def _views(torch: Any, region: Region, raw: Any) -> Views:
    writes: list[tuple[dict[str, Any], str, Any]] = []
    parts: list[dict[str, Any]] = []
    for weight in region.weights:
        typed = {
            role: raw[offset : offset + _nbytes(shape, dtype)]
            .view(getattr(torch, TORCH_DTYPES[dtype]))
            .view(shape)
            for role, (offset, dtype, shape) in weight.parts.items()
        }
        parts.append(typed)
        if weight.route == "verbatim":
            (tensor,) = typed.values()
            for table, name, parameter in weight.sites:
                value = torch.nn.Parameter(tensor, requires_grad=False) if parameter else tensor
                writes.append((table, name, value))
        elif weight.route == "encoded_gemm":
            writes.extend(
                (weight.leaf._buffers, role, tensor)
                for role, tensor in typed.items()
                if role not in weight.derived_roles
            )
    return Views(writes, parts)


def _transform(torch: Any, region: Region, parts: list[dict[str, Any]]) -> list[Any]:
    """Decode `decoded_float` weights into fresh device tensors and recompute derived leaf
    roles, on the current stream. Returns what must stay alive while the region is bound."""
    kept: list[Any] = []
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
    ) -> None:
        self.weights = weights
        self.torch = weights.torch
        self.construction = construction
        self.checkpoints = dict(checkpoints)
        self.plan = {row.key: row for row in rows}
        self.custody = custody
        self.device = weights.device
        self.kind = weights.kind
        #: Held for the generation's life: the plane re-reads evicted regions from the store.
        self.leases = ReadLeases(self.checkpoints)
        self.state = "open"
        self.expected: dict[str, frozenset[str]] = {}
        self.enqueued: dict[str, TensorLike] = {}
        self.roots: dict[str, Any] = {}
        self.components: dict[str, Component] = {}
        self.derived: dict[str, dict[tuple[Any, str], Any]] = {}
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
            ignored.extend(str(k) for k in fit.get("ignored", ()))  # type: ignore[attr-defined]
            counted: Mapping[str, int] = fit.get("routes", {})  # type: ignore[assignment]
            for route in routes:
                routes[route] += int(counted.get(route, 0))
        if verdict is None:
            return None
        for full in ignored:
            component, _, key = full.partition("/")
            if component in self.expected:
                self.expected[component] -= {f"{component}.{key}"}
        if len(by_checkpoint) > 1:
            verdict = {**verdict, "ignored": ignored, "routes": routes}
        return verdict

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
                if member is None or not hasattr(member, "named_modules"):
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
        shape = tuple(destination.shape)  # type: ignore[attr-defined]
        if shape != tuple(row.shape) or (spec.nbytes and spec.nbytes != row.nbytes):
            raise _refuse(
                "length_mismatch",
                f"{key}: the destination is {shape} and the checkpoint supplies {row.shape} "
                f"({row.nbytes} B)",
                [key],
            )
        if dtype_name(destination.dtype) != dtype_name(row.dtype):  # type: ignore[attr-defined]
            raise _refuse(
                "dtype_mismatch",
                f"{key}: the checkpoint's logical dtype is {row.dtype} and the destination is "
                f"{dtype_name(destination.dtype)}",  # type: ignore[attr-defined]
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
        leaves = {
            key: self._install_leaf(key) for key in order if self._route(key) == "encoded_gemm"
        }
        for name, root in self.roots.items():
            keys = [key for key in order if self.plan[key].component == name]
            if keys:
                component = self._register(name, root, keys, leaves)
                self.components[name] = component
                self.weights.components[(self.construction, name)] = component
        self.stats["register_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self.state = "ready"

    def poison(self, keys: Sequence[str]) -> None:
        """Abort: forget every registered weight set and give the leases back."""
        self.close()
        self.state = "poisoned"

    def close(self) -> None:
        self.weights.forget(self.construction)
        self.components.clear()
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

    def _register(
        self,
        name: str,
        root: Any,
        keys: list[str],
        leaves: Mapping[str, tuple[Any, Any, tuple[str, ...]]],
    ) -> Component:
        """Partition the component into regions, lay out their stored parts, find every
        binding site, and register the weight set."""
        try:
            layout = partition(root)
        except ValueError:
            layout = None  # shared storage across blocks: one region, not streamable
        blocks = [unit.path for unit in layout.blocks] if layout is not None else []
        members: list[list[str]] = [[] for _ in range(len(blocks) + 1)]
        for key in keys:
            path = key[len(name) + 1 :].rpartition(".")[0]
            members[_unit_of(path, blocks)].append(key)
        sites = _sites(root, name)
        regions: list[Region] = []
        specs: list[plane.Region] = []
        for index, group in enumerate(members):
            placed, nbytes = plane.layout(
                [
                    (self.plan[key].role_what(part.role), part.nbytes)
                    for key in group
                    for part in self.plan[key].encoded.parts
                ]
            )
            offsets = iter(placed)
            region_weights: list[Weight] = []
            unbound: list[tuple[dict[str, Any], str, Any]] = []
            for key in group:
                row = self.plan[key]
                weight = Weight(
                    row=row,
                    route=self._route(key),
                    provider=self.selected[key].provider,
                    parts={
                        part.role: (next(offsets).offset, part.dtype, tuple(part.shape))
                        for part in row.encoded.parts
                    },
                    sites=sites.get(key, []),
                )
                if key in leaves:
                    weight.leaf, weight.replaced, weight.derived_roles = leaves[key]
                    unbound.extend(
                        (weight.leaf._buffers, role, t) for role, t in weight.leaf.roles().items()
                    )
                unbound.extend((table, attr, table[attr]) for table, attr, _ in weight.sites)
                region_weights.append(weight)
            module = root.get_submodule(blocks[index - 1]) if index else root
            regions.append(
                Region(
                    index,
                    blocks[index - 1] if index else "",
                    module,
                    region_weights,
                    nbytes,
                    unbound=unbound,
                )
            )
            specs.append(plane.Region(index, placed, nbytes))
        checkpoint = self.checkpoints[name]
        handle = self.weights.plane.register(
            (checkpoint.manifest_id, name, self.execution.digest()),
            specs,
            source=self.leases.acquire(name),
            header=checkpoint.header_bytes,
        )
        sizes = weight_policy.Layout(
            common=regions[0].nbytes, blocks=tuple(region.nbytes for region in regions[1:])
        )
        return Component(
            name=name,
            root=root,
            handle=handle,
            regions=regions,
            layout=sizes,
            order=list(range(1, len(regions))),
        )


def _unit_of(path: str, blocks: Sequence[str]) -> int:
    """The region a module path belongs to: 1 + its block's index, or 0 (common)."""
    for index, block in enumerate(blocks):
        if path == block or path.startswith(block + "."):
            return index + 1
    return 0


def _sites(root: Any, name: str) -> dict[str, list[tuple[dict[str, Any], str, bool]]]:
    """Every module attribute naming each checkpoint tensor, tied names included: a tensor
    shared by two modules binds at both."""
    by_tensor: dict[int, list[tuple[dict[str, Any], str, bool]]] = {}
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


def _derived_values(root: Any) -> dict[tuple[Any, str], Any]:
    """Non-persistent buffers with their constructed values: no checkpoint supplies them."""
    held: dict[tuple[Any, str], Any] = {}
    for _, sub in root.named_modules():
        for local in getattr(sub, "_non_persistent_buffers_set", ()):
            value = sub._buffers.get(local)
            if value is not None and str(value.device) != "meta":
                held[(sub, local)] = value.detach()
    return held


def _restore_derived(held: Mapping[tuple[Any, str], Any], device: Any) -> None:
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
    ) -> None:
        self.weights = weights
        self.components = dict(components)
        #: method -> the components it declares (`component_use`)
        self.scopes = dict(scopes or {})
        #: method -> activation bytes per batch item, measured; and the batch it last ran at
        self.per_item: dict[str, int] = {}
        self.batch: dict[str, int] = {}
        self.hosted: frozenset[str] = frozenset()
        self.active: tuple[str, tuple[str, ...]] | None = None
        self.spare = False
        self.poisoned = ""
        self.last_shortfall: dict[str, Any] | None = None
        self.refusals: list[str] = []
        self.peaks: dict[str, int] = {}
        self.baseline = -1
        #: The scheduler's turn (W4, `stage/1`): called before a stage plans, returns the
        #: plane budget the Worker granted for it, or None to keep the current one.
        self.turn: Callable[[str, tuple[str, ...]], int | None] | None = None
        for component in self.components.values():
            for region in component.regions[1:]:
                module = region.module
                module.forward = self._block(component, region, module.forward)
            if _image_vae(component.root):
                component.root.decode = self._decode(component, component.root.decode)

    # -- the Residency protocol

    def open_attempt(self, *_: object) -> None:
        """A new attempt: its activation peaks start empty. (Older Workers also send a
        placement rung and headroom; the plane has no rungs, so they are ignored.)"""
        self.peaks.clear()
        self.last_shortfall = None

    def admit(self, method: str, components: tuple[str, ...]) -> None:
        self._require_clean(method)
        if self.active is not None:
            self.refusals.append(f"fault scope_overlap: {method} entered inside {self.active[0]}")
            self.active = None
        local = tuple(
            name for name in components if name in self.components and name not in self.hosted
        )
        weights = self.weights
        if self.turn is not None:
            granted = self.turn(method, components)
            if granted is not None:
                weights.set_budget(granted)
        try:
            self._plan(local, weights.stage_budget(method), method)
        except weight_policy.BelowFloor as exc:
            raise self._shortfall(method, exc.needed, exc.available, str(exc)) from None
        stream = weights.stream()
        for name in local:
            component = self.components[name]
            component.passes = 0
            component.common = weights.plane.acquire(component.handle, weights.index, 0, stream)
            bind(weights.torch, component.regions[0], component.common, fresh=False)
        self.baseline = accel.allocated(weights.torch, weights.kind)
        accel.reset_peak(weights.torch, weights.kind)
        self.active = (method, local)

    def release(self, method: str, components: tuple[str, ...]) -> None:
        if self.active is None:
            self.refusals.append(f"fault scope_over_release: {method}")
            return
        weights = self.weights
        stream = weights.stream()
        order = weights.tick()
        for name in self.active[1]:
            component = self.components[name]
            if component.common is not None:
                component.common.release(stream)
                component.common = None
            if component.cursor is not None:
                component.cursor.close()
                component.cursor = None
            component.streamed = frozenset()
            component.last_passes = component.passes
            # Released weights stay mapped and bound, at idle priority: least recently used
            # first when another stage wants the room.
            weights.plane.want(
                component.handle,
                plane.vram(weights.index),
                sorted(component.resident),
                (plane.IDLE, order),
            )
        if self.baseline >= 0:
            peak = max(accel.peak_allocated(weights.torch, weights.kind) - self.baseline, 0)
            self.peaks[method] = max(self.peaks.get(method, 0), peak)
            weights.activation[method] = max(weights.activation.get(method, 0), peak)
            item = peak // max(self.batch.pop(method, 1), 1)
            self.per_item[method] = max(self.per_item.get(method, 0), item)
        self.baseline = -1
        self.active = None

    # -- the executor's calls

    def host(self, name: str) -> None:
        """`name` lives on another GPU of the group: never held here."""
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
        held = set(self.active[1]) if self.active is not None else set()
        self._drop([c for name, c in self.components.items() if name not in held])

    def vacate(self) -> dict[str, Any]:
        """Unmap this construction's weights (an older Worker's `Vacate`). The bytes stay
        in the host tier; the next stage wants them back at link speed."""
        before = plane.stats(self.weights.plane).committed
        self._drop([c for c in self.components.values() if c.common is None])
        freed = max(before - plane.stats(self.weights.plane).committed, 0)
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
        """(allocatable device bytes now, plane bytes a stage of `components` could unmap)."""
        weights = self.weights
        memory = accel.allocation(weights.torch, weights.kind)
        free = memory["driver_free_bytes"] + max(
            memory["reserved_bytes"] - memory["allocated_bytes"], 0
        )
        held = plane.stats(weights.plane)
        floor = sum(
            weight_policy.floor(self.components[name].layout, 1)
            for name in components
            if name in self.components
        )
        return free, max(held.committed - held.leased - floor, 0)

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
            "attempt_activation_peak_bytes": max(self.peaks.values(), default=0),
            "attempt_activation_peaks": dict(self.peaks),
            "poisoned": self.poisoned,
            "refusals": self.refusals[-8:],
        }

    # -- internals

    def _plan(self, local: tuple[str, ...], budget: int, method: str) -> None:
        """Place the stage's components inside `budget`: prefix wanted, tail streamed."""
        weights = self.weights
        components = {name: self.components[name] for name in local}
        placed = weight_policy.stage(
            {name: c.layout for name, c in components.items()},
            budget,
            orders={name: [index - 1 for index in c.order] for name, c in components.items()},
            cyclic={name for name, c in components.items() if c.last_passes != 1},
        )
        if self.spare:
            need = sum(r.vram_bytes(components[n].layout) for n, r in placed.items())
            if need > budget:
                raise ResidencyRefusal("device_spared", f"{method}() declined: needs {need} B")
        weights._apply(budget)
        order = weights.tick()
        for name, residence in placed.items():
            component = components[name]
            if component.cursor is not None:
                component.cursor.close()
            resident = [0, *(index + 1 for index in residence.resident)]
            streamed = [index + 1 for index in residence.streamed]
            weights.plane.want(
                component.handle, plane.vram(weights.index), resident, (plane.ACTIVE, order)
            )
            component.cursor = (
                weights.plane.stream(component.handle, weights.index, streamed, residence.window, 0)
                if streamed
                else None
            )
            component.resident = frozenset(resident)
            component.streamed = frozenset(streamed)
            component.residence = residence
        weights.unbind_evicted()

    def _block(
        self, component: Component, region: Region, forward: Callable[..., Any]
    ) -> Callable[..., Any]:
        """The one hook per block: acquire, bind, run, release. A torch OOM releases, lowers
        the plane budget and runs the block again, while its inputs are untouched."""
        weights = self.weights
        torch = weights.torch
        oom = accel.oom_error(torch, weights.kind)

        def run(*args: Any, **kwargs: Any) -> Any:
            if component.common is None:
                raise CapabilityError(
                    f"{component.name}.{region.path} ran outside a component-use scope: its "
                    "weights are bound only while a declared method holds them (§1.1)",
                    code="no_active_scope",
                )
            self._observe(component, region.index)
            versions = _versions(args, kwargs)
            while True:
                stream = weights.stream()
                streamed = region.index in component.streamed
                lease = (
                    component.cursor.acquire(region.index, stream)
                    if streamed
                    else weights.plane.acquire(
                        component.handle, weights.index, region.index, stream
                    )
                )
                need = 0
                try:
                    bind(torch, region, lease, fresh=streamed)
                    try:
                        return forward(*args, **kwargs)
                    except oom as exc:
                        need = _requested(exc) or region.nbytes
                finally:
                    if streamed:
                        unbind(region)
                    lease.release(stream)
                if _versions(args, kwargs) != versions:
                    raise self._shortfall(
                        self.active[0] if self.active else "",
                        need,
                        0,
                        f"{component.name}.{region.path} ran out of memory after changing its "
                        "inputs in place, so it cannot run again",
                    )
                self._make_room(need)

        return run

    def _decode(self, component: Component, decode: Callable[..., Any]) -> Callable[..., Any]:
        """An image VAE's decode mode, decided before the call from what is free plus what the
        plane may unmap: untiled when its measured activations fit, else tiled with the tile
        clamped to what is free. Unmeasured runs untiled (its blocks retry on OOM); only a
        decode that cannot run untiled even with every idle weight unmapped runs again tiled.
        The mode is recorded in the run's outputs."""
        root = component.root
        weights = self.weights
        oom = accel.oom_error(weights.torch, weights.kind)
        per_pixel: list[int] = []
        scale = 2 ** (len(root.config.block_out_channels) - 1)

        def run(z: Any, *args: Any, **kwargs: Any) -> Any:
            pixels = int(z.shape[0]) * int(z.shape[-2]) * int(z.shape[-1]) * scale * scale
            free, evictable = self.room((component.name,))
            mode = (
                weight_policy.decode_mode(free, evictable, per_pixel[0] * pixels)
                if per_pixel
                else "untiled"
            )
            if mode == "tiled":
                _tile(root, free // max(per_pixel[0], 1), scale)
            else:
                root.disable_tiling()
            baseline = accel.allocated(weights.torch, weights.kind)
            accel.reset_peak(weights.torch, weights.kind)
            failed = False
            try:
                out = decode(z, *args, **kwargs)
            except oom:
                if mode == "tiled":
                    raise
                failed = True
            if failed:  # outside `except`, so the failed call's frames are gone
                mode = "tiled"
                free, _ = self.room((component.name,))
                _tile(root, free // max(per_pixel[0] if per_pixel else 1, 1), scale)
                out = decode(z, *args, **kwargs)
            elif mode == "untiled":
                peak = accel.peak_allocated(weights.torch, weights.kind) - baseline
                per_pixel[:] = [max(peak // max(pixels, 1), *per_pixel)]
            weights.modes[f"{component.name}.decode"] = mode
            return out

        return run

    def _make_room(self, need: int) -> None:
        """Activations first: unmap `need` bytes of weights, then re-place the stage."""
        assert self.active is not None
        method, local = self.active
        weights = self.weights
        floor = sum(weight_policy.floor(self.components[name].layout, 1) for name in local)
        if not weights.shrink(need, floor):
            raise self._shortfall(method, need, 0, "no weight bytes are left to unmap")
        self._plan(local, weights.applied, method)

    def _observe(self, component: Component, index: int) -> None:
        """Count passes and learn the blocks' execution order from the first pass."""
        entry = component.order[0] if component.observed else (component.seen or [index])[0]
        if index == entry:
            component.passes += 1
        if component.observed:
            return
        if index in component.seen:
            unseen = [i for i in range(1, len(component.regions)) if i not in component.seen]
            component.order = [*component.seen, *unseen]
            component.observed = True
        else:
            component.seen.append(index)

    def _drop(self, components: Sequence[Component]) -> None:
        weights = self.weights
        for component in components:
            everything = [region.index for region in component.regions]
            weights.plane.drop(component.handle, plane.vram(weights.index), everything)
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

    def _require_clean(self, what: str) -> None:
        if self.poisoned:
            raise ResidencyRefusal(
                "residency_poisoned", f"{what}() against a poisoned weight plane: {self.poisoned}"
            )


def _versions(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[int, ...]:
    """In-place mutation counters of the block's tensor inputs (one level deep)."""
    out: list[int] = []
    for value in (*args, *kwargs.values()):
        for item in value if isinstance(value, (tuple, list)) else (value,):
            version = getattr(item, "_version", None)
            if isinstance(version, int):
                out.append(version)
    return tuple(out)


def _requested(exc: BaseException) -> int:
    """The allocation torch could not make, from its own message."""
    found = _REQUESTED.search(str(exc))
    return int(float(found.group(1)) * _UNITS[found.group(2)]) if found else 0


def _image_vae(root: Any) -> bool:
    """diffusers' image `AutoencoderKL`: one decode call, one frame, tiling it can switch."""
    return any(
        kind.__name__ == "AutoencoderKL" and kind.__module__.startswith("diffusers.")
        for kind in type(root).__mro__
    )


def _tile(root: Any, pixels: int, scale: int) -> None:
    """Tile an image VAE with the largest square tile whose activations fit `pixels`."""
    side = max(int(pixels**0.5) // 64 * 64, 256)
    root.enable_tiling()
    root.tile_sample_min_size = min(root.tile_sample_min_size, side)
    root.tile_latent_min_size = root.tile_sample_min_size // scale
